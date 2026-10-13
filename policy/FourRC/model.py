"""Wrist-conditioned 4RC Stage2 evaluation with checkpoint-defined windows."""
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from .geometry import (rigid, AlohaKinematics, project_queries,
                       convert_actions, validate_execute_horizon)
from .checkpoint import DEFAULT_CHECKPOINT, read_config


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = dict(model_cfg)
        self.execute_horizon = validate_execute_horizon(
            os.environ.get("FOURRC_EXECUTE_HORIZON", self.cfg.get("execute_horizon", 8)))
        if self.cfg.get("action_type") != "ee" or self.cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("FourRC supports only aloha-agilex with arx_x5 / ee")
        if self.cfg.get("eval_batch", False):
            raise ValueError("FourRC uses one environment per server; disable eval_batch")
        dims = get_robot_action_dim_info(self.cfg["env_cfg_type"])
        if dims["arm_dim"] != [6, 6] or dims["ee_dim"] != [1, 1]:
            raise ValueError("FourRC requires two six-joint arms and two scalar grippers")
        xpl_root = Path(__file__).resolve().parents[2]
        self.robotwin_root = xpl_root.parent
        upstream = Path(os.environ.get("FOURRC_ROOT") or self.cfg.get("fourrc_root")
                        or self.robotwin_root.parent / "4RC").expanduser().resolve()
        if not (upstream / "stage2_sliding_window_inference.py").is_file():
            raise FileNotFoundError(f"4RC source missing at {upstream}; set FOURRC_ROOT")
        sys.path.insert(0, str(upstream))
        self.inference = importlib.import_module("stage2_sliding_window_inference")
        ckpt_cfg = dict(self.cfg)
        if not ckpt_cfg.get("ckpt_name"):
            ckpt_cfg["ckpt_name"] = str(self.robotwin_root / DEFAULT_CHECKPOINT)
        self.checkpoint = resolve_checkpoint_root(ckpt_cfg, self.robotwin_root / "checkpoints", policy_dir=self.robotwin_root)
        config = read_config(self.checkpoint)
        self.history_frames = config["history_frames"]
        self.prediction_horizon = config["prediction_horizon"]
        self.execute_horizon = validate_execute_horizon(self.execute_horizon, self.prediction_horizon)
        # Reject the old cross-frame-only projection before constructing the model.
        from safetensors import safe_open
        with safe_open(str(self.checkpoint / "model.safetensors"), framework="pt", device="cpu") as weights:
            key = "global_encoder.projection.1.weight"
            if key not in weights.keys() or weights.get_slice(key).get_shape() != [config["action_dim"], 3072]:
                raise ValueError("FourRC requires the 3072-channel concat global visual encoder")
        self.device, self.dtype = self.inference.resolve_device_dtype(
            self.cfg.get("device", "auto"), self.cfg.get("dtype", "auto"))
        self.policy, self.training_cfg = self.inference.load_stage2_policy(
            self.checkpoint, self.device, t5_model=os.environ.get("FOURRC_T5_MODEL") or self.cfg.get("t5_model"))
        self.global_feature_mode = self.policy.global_feature_mode
        self.global_visual_input_dim = self.policy.global_encoder.projection[1].in_features
        print(f"[FourRC] checkpoint global_feature_mode={self.global_feature_mode}, "
              f"global_visual_input_dim={self.global_visual_input_dim}", flush=True)
        if not self.policy.use_wrist_condition or self.policy.global_feature_mode != "concat":
            raise ValueError("FourRC requires a wrist-conditioned concat Stage2 policy")
        self.sampling_steps = int(self.cfg.get("sampling_steps") or self.training_cfg.get("sampling_steps", 8))
        self.kinematics = AlohaKinematics(self.robotwin_root)
        self.offsets = self.kinematics.offsets
        self.case = {}
        log_root = Path(os.environ.get("FOURRC_LOG_DIR") or self.robotwin_root / "eval_result/FourRC_predictions")
        log_root.mkdir(parents=True, exist_ok=True)
        self.log_path = log_root / f"predictions_{time.time_ns()}_{os.getpid()}.jsonl"
        from .visualization import create_viewer
        self.viewer = create_viewer(self.cfg, self.inference.make_point_cloud,
                                    history_frames=self.history_frames, prediction_horizon=self.prediction_horizon)
        self.reset()
        self._log({"event": "loaded", "checkpoint": str(self.checkpoint), "history": self.history_frames,
                   "horizon": self.prediction_horizon, "execute": self.execute_horizon, "time_encoding": "index",
                   "use_wrist_condition": True, "action_normalization": "quantile", "action_normalization_version": 1,
                   "global_feature_mode": self.global_feature_mode,
                   "global_visual_input_dim": self.global_visual_input_dim,
                   "timing": "ordered action steps; RoboTwin waypoint execution is variable-duration",
                   "initial_query": "measured_joint_fk", "frame": "robot_base",
                   "viser_url": self.viewer.url if self.viewer is not None else None})

    def _log(self, record):
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _visualize(self, method, *args, **kwargs):
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        try:
            getattr(viewer, method)(*args, **kwargs)
        except Exception as exc:
            # Automatic mode can continue without its optional display. Manual
            # mode must never turn into ungated inference after a UI failure.
            self.viewer = None
            message = f"{type(exc).__name__}: {exc}"
            print(f"[FourRC viser] disabled after display error: {message}", flush=True)
            self._log({"event": "visualization_error", "error": message})
            try:
                viewer.close()
            except Exception:
                pass
            if getattr(viewer, "manual", False):
                self._manual_viewer_error = message
                raise RuntimeError(f"Manual inference UI failed: {message}") from exc

    def inference_ready(self):
        if getattr(self, "_manual_viewer_error", None):
            raise RuntimeError(f"Manual inference UI failed: {self._manual_viewer_error}")
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return True
        expected = 1 if self.previous is None else self.execute_horizon
        return len(self.pending) == expected and viewer.inference_ready()

    def prepare_case(self, case_meta=None):
        self.case = dict(case_meta or {})

    def reset(self):
        self.pending = []
        self.previous = None
        self.observation_count = 0
        self.window = 0
        self.env_idx = None
        self.instruction = None
        self._visualize("reset")

    def update_obs(self, obs):
        env_idx = int(obs.get("env_idx", 0))
        if self.env_idx is not None and env_idx != self.env_idx:
            raise ValueError("One FourRC server cannot interleave environments")
        self.env_idx = env_idx
        camera = obs["vision"]["cam_third_view"]
        rgb = np.asarray(camera["color"])
        if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
            raise ValueError("FourRC requires decoded RGB uint8 [240,320,3]")
        wrists = []
        for name in ("cam_left_wrist", "cam_right_wrist"):
            try:
                image = np.asarray(obs["vision"][name]["color"])
            except KeyError as exc:
                raise ValueError(f"FourRC requires {name} RGB") from exc
            if image.shape != (240, 320, 3) or image.dtype != np.uint8:
                raise ValueError(f"FourRC {name} must be decoded RGB uint8 [240,320,3]")
            wrists.append(image)
        k = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or
                k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("Invalid third-camera intrinsics")
        info = obs["additional_info"]
        if info.get("third_camera_source") != "observer_camera":
            raise ValueError("Enable observer_camera; front_camera is not the training third view")
        text = obs.get("instruction") or next(iter(obs.get("instructions", [])), "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A non-empty language instruction is required")
        if self.instruction is not None and text != self.instruction:
            raise ValueError("Instruction changed without episode reset")
        self.instruction = text
        if len(self.pending) >= (1 if self.previous is None else self.execute_horizon):
            raise ValueError("Unexpected observation count before inference")
        index = info.get("waypoint_index")
        if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
                or index != self.observation_count):
            raise ValueError(f"Expected waypoint_index={self.observation_count}, got {index}")
        try:
            joints = np.stack([np.asarray(obs["state"][f"{side}_arm_real_joint_state"], dtype=np.float64)
                               for side in ("left", "right")])
        except KeyError as exc:
            raise ValueError("FourRC requires measured left/right arm joint states for FK queries") from exc
        tcp_base = self.kinematics.tcp_base(joints)
        frame = {"rgb": rgb.copy(), "wrist_rgb": np.stack(wrists), "k": k.copy(),
                 "cw": rigid(camera["extrinsics_matrix"], "third camera extrinsics"),
                 "wb": rigid(info["T_world_from_robot_base"], "robot root pose"),
                 "index": int(index), "joints": joints.copy(), "tcp_base": tcp_base,
                 "terminal": bool(info.get("episode_end", False))}
        self.pending.append(frame)
        self.observation_count += 1
        self._visualize("observe", frame)

    @staticmethod
    def _prepare(frames, queries):
        images = torch.from_numpy(np.stack([f["rgb"] for f in frames])).permute(0, 3, 1, 2).float() / 255
        images = F.pad(images, (1, 1, 6, 6), mode="reflect").mul_(2).sub_(1).unsqueeze(0)
        k = torch.from_numpy(np.stack([f["k"] for f in frames])).unsqueeze(0)
        k[..., 0, 2] += 1
        k[..., 1, 2] += 6
        q = torch.from_numpy(np.asarray(queries, dtype=np.float32))[None] + torch.tensor([1., 6.])
        return images, k, q

    def get_action(self):
        try:
            return self._get_action()
        except Exception as exc:
            self._log({"event": "error", "case": self.case, "window": self.window,
                       "error": f"{type(exc).__name__}: {exc}"})
            raise

    def _get_action(self):
        expected = 1 if self.previous is None else self.execute_horizon
        if len(self.pending) != expected:
            raise ValueError(f"Expected {expected} new observations, got {len(self.pending)}")
        if getattr(self, "_manual_viewer_error", None):
            raise RuntimeError(f"Manual inference UI failed: {self._manual_viewer_error}")
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            if getattr(viewer, "manual", False):
                viewer.begin_inference()
            else:
                self._visualize("begin_inference")
        if self.previous is None:
            frames = self.pending * self.history_frames
        else:
            frames = (self.previous + self.pending)[-self.history_frames:]
        # Initial queries belong to the FIRST historical RGB, not the latest pose.
        first = frames[0]
        camera_tcp = first["cw"] @ first["wb"] @ first["tcp_base"]
        queries = project_queries(camera_tcp[:, :3, 3], first["k"], clip=True)
        images, intrinsics, query_tensor = self._prepare(frames, queries)
        anchor = frames[-1]
        wrist_images = torch.from_numpy(anchor["wrist_rgb"]).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0)
        seed = (int(self.case.get("seed", self.cfg.get("seed") or 0)) + self.window) % (2**63 - 1)
        started = time.perf_counter()
        # Stage2 encodes historical and future step indices internally. Source
        # timestamps/frequency do not condition the ordered waypoint predictions.
        prediction = self.inference.infer_stage2_window(
            self.policy, images, intrinsics, query_tensor, self.instruction,
            steps=self.sampling_steps, seed=seed, device=self.device, dtype=self.dtype,
            wrist_images=wrist_images)
        if not prediction["success"]:
            raise RuntimeError("FourRC reported prediction failure")
        chunk, base_tcp = convert_actions(prediction, anchor["cw"], anchor["wb"], self.offsets,
                                          prediction_horizon=self.prediction_horizon, execute_horizon=self.execute_horizon)
        if not np.array_equal(prediction["future_step_indices"], np.arange(1, self.prediction_horizon + 1)):
            raise ValueError("Invalid FourRC future_step_indices")
        self._log({"event": "prediction", "case": self.case, "window": self.window, "seed": seed,
                   "anchor": anchor["index"], "execute": self.execute_horizon,
                   "history": self.history_frames, "horizon": self.prediction_horizon,
                   "wrist_waypoint_index": anchor["index"],
                   "history_indices": [frame["index"] for frame in frames],
                   "future_step_indices": prediction["future_step_indices"].tolist(),
                   "query_pixels": queries.tolist(), "query_source": "measured_joint_fk",
                   "query_waypoint_index": first["index"],
                   "T_world_from_robot_base": anchor["wb"].tolist(),
                   "T_camera_from_world": anchor["cw"].tolist(),
                   "tcp_base": base_tcp.tolist(), "gripper": prediction["action_gripper"].tolist(),
                   "inference_seconds": time.perf_counter() - started})
        self._visualize("publish", frames, prediction, base_tcp,
                        case=self.case, instruction=self.instruction, window=self.window,
                        execute_horizon=self.execute_horizon)
        self.previous = frames
        self.pending = []
        self.window += 1
        return chunk

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("FourRC requires eval_batch=false")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("FourRC requires eval_batch=false")

    def on_trial_end(self, result=None):
        self._visualize("finish", result)
        self._log({"event": "trial_end", "case": self.case, "result": result})

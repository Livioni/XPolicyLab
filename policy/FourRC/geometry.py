"""Metric TCP geometry for the aloha-agilex training-label convention."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def rigid(value, name="transform"):
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (3, 4):
        value = np.vstack((value, [0, 0, 0, 1]))
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4x4/3x4 transform")
    r = value[:3, :3]
    if not (np.allclose(value[3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(r.T @ r, np.eye(3), atol=1e-4)
            and np.isclose(np.linalg.det(r), 1, atol=1e-4)):
        raise ValueError(f"{name} is not a rigid transform")
    return value.copy()


class AlohaKinematics:
    """Measured joint angles -> the same fingertip TCP used by training labels."""

    def __init__(self, robotwin_root):
        root = Path(robotwin_root)
        spec = importlib.util.spec_from_file_location("fourrc_training_tcp", root / "convert_robotwin_tcp.py")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)
        asset = root / "assets/embodiments/aloha-agilex"
        cfg = yaml.safe_load((asset / "config.yml").read_text())
        self.robot = self.module.load_robot_model(str(asset / cfg["urdf_path"]))
        if not (np.isclose(cfg["gripper_bias"], .12)
                and np.allclose(cfg["delta_matrix"], np.eye(3))
                and np.allclose(cfg["global_trans_matrix"], np.diag([1, -1, -1]))):
            raise ValueError("FourRC requires the validated aloha-agilex control-frame convention")
        self.offsets = np.stack([
            np.mean([self.robot.joints_by_name[f"{prefix}_joint{i}"].origin_xyz
                     for i in (7, 8)], axis=0) + self.module.ARX5_FINGERTIP_CONTACT_OFFSET_METERS
            for prefix in ("fl", "fr")
        ])

    def tcp_base(self, joint_positions):
        joints = np.asarray(joint_positions, dtype=np.float64)
        if joints.shape != (2, 6) or not np.isfinite(joints).all():
            raise ValueError("Measured arm joint positions must be finite [2,6] radians, left then right")
        state = np.zeros((1, 14), dtype=np.float64)
        state[0, :6], state[0, 7:13] = joints
        return np.stack([self.module.tcp_pose_series(self.robot, state, side)[0]
                         for side in ("left", "right")])


def project_queries(positions, intrinsics, *, clip=False):
    positions = np.asarray(positions)
    if positions.shape != (2, 3) or not np.isfinite(positions).all() or np.any(positions[:, 2] <= 1e-6):
        raise ValueError("TCP query is non-finite or behind the third camera")
    pixels = positions @ np.asarray(intrinsics).T
    uv = pixels[:, :2] / pixels[:, 2:]
    if not np.isfinite(uv).all():
        raise ValueError("Non-finite projected TCP query")
    outside = np.any((uv < 0) | (uv > [319, 239]))
    if outside and not clip:
        raise ValueError("TCP query is outside the third camera image")
    if outside:
        print(f"[FourRC] clipping FK queries: {uv.tolist()}", flush=True)
    return np.clip(uv, [0, 0], [319, 239]).astype(np.float32)


def validate_execute_horizon(value, prediction_horizon=None):
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < 1 or (prediction_horizon is not None and value > prediction_horizon)):
        limit = f" from 1 to {prediction_horizon}" if prediction_horizon is not None else " greater than zero"
        raise ValueError(f"execute_horizon must be an integer{limit}")
    return int(value)


def convert_actions(prediction, camera_from_world, world_from_base, offsets, *, prediction_horizon, execute_horizon=8):
    execute_horizon = validate_execute_horizon(execute_horizon, prediction_horizon)
    positions = np.asarray(prediction["action_position"], dtype=np.float64)
    rotations = np.asarray(prediction["action_rotation"], dtype=np.float64)
    grip = np.asarray(prediction["action_gripper"])
    if (positions.shape != (prediction_horizon, 2, 3) or rotations.shape != (prediction_horizon, 2, 3, 3)
            or grip.shape != (prediction_horizon, 2) or not np.isin(grip, [0, 1]).all()
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()):
        raise ValueError("Invalid FourRC action prediction")
    if (not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3)):
        raise ValueError("Invalid predicted TCP rotation")
    camera_tcp = np.broadcast_to(np.eye(4), (prediction_horizon, 2, 4, 4)).copy()
    camera_tcp[..., :3, :3] = rotations
    camera_tcp[..., :3, 3] = positions
    base_tcp = np.linalg.inv(world_from_base) @ np.linalg.inv(camera_from_world) @ camera_tcp
    base_ee = base_tcp.copy()
    base_ee[..., :3, 3] -= np.einsum("haij,aj->hai", base_tcp[..., :3, :3], offsets)
    # RoboTwin take_action(ee) accepts WORLD poses; its planner handles base IK.
    world_ee = world_from_base @ base_ee
    chunk = []
    for step in range(execute_horizon):
        action = {}
        for arm, side in enumerate(("left", "right")):
            xyzw = Rotation.from_matrix(world_ee[step, arm, :3, :3]).as_quat()
            action[f"{side}_ee_pose"] = np.r_[world_ee[step, arm, :3, 3], xyzw[[3, 0, 1, 2]]].astype(np.float32)
            action[f"{side}_ee_joint_state"] = np.array([grip[step, arm]], dtype=np.float32)
        chunk.append(action)
    return chunk, base_tcp

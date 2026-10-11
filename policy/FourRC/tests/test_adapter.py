import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.FourRC.geometry import (rigid, AlohaKinematics,
                                              project_queries, convert_actions, validate_execute_horizon)
from XPolicyLab.policy.FourRC.model import Model

ROOT = Path(__file__).resolve().parents[4]


def prediction():
    return {"success": True, "action_position": np.broadcast_to([0., 0., 1.], (16, 2, 3)).copy(),
            "action_rotation": np.broadcast_to(np.eye(3), (16, 2, 3, 3)).copy(),
            "action_gripper": np.tile([0, 1], (16, 1)),
            "future_step_indices": np.arange(1, 17),
            "history_position": np.broadcast_to([0., 0., 1.], (8, 2, 3)).copy()}


def observation(index=0):
    return {"env_idx": 0, "instruction": "Click the bell", "vision": {"cam_third_view": {
        "color": np.full((240, 320, 3), index, dtype=np.uint8),
        "intrinsic_matrix": np.array([[100., 0., 160.], [0., 100., 120.], [0., 0., 1.]], dtype=np.float32),
        "extrinsics_matrix": np.eye(4)}}, "additional_info": {
        "frequency": 15, "third_camera_source": "observer_camera", "T_world_from_robot_base": np.eye(4),
        "waypoint_index": index, "episode_end": False}, "state": {
        "left_arm_real_joint_state": np.array([index / 1000, 0, 0, 0, 0, 0]),
        "right_arm_real_joint_state": np.array([-index / 2000, 0, 0, 0, 0, 0]),
        "left_arm_joint_state": np.full(6, 99.), "right_arm_joint_state": np.full(6, 99.)}}



def lightweight_model(execute_horizon=8):
    model = Model.__new__(Model)
    model.cfg = {"seed": 42}
    model.execute_horizon = execute_horizon
    model.case = {}
    model.device, model.dtype = torch.device("cpu"), torch.float32
    def fk(joints):
        poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        poses[:, 0, 3] = joints[:, 0]
        poses[:, 2, 3] = 1
        return poses
    model.kinematics = SimpleNamespace(tcp_base=fk)
    model.offsets = np.zeros((2, 3))
    model.sampling_steps = 8
    model._log = lambda record: None
    model.reset()
    calls = []
    def infer(policy, images, k, q, instruction, *, steps, seed, device, dtype):
        calls.append((images.clone(), k.clone(), q.clone(), instruction,
                      dict(steps=steps, seed=seed, device=device, dtype=dtype)))
        return prediction()
    # No recovery-only API is available: any old bridge call would fail this test.
    model.policy = SimpleNamespace()
    model.inference = SimpleNamespace(infer_stage2_window=infer)
    return model, calls


def test_eight_step_rollout_and_reset_with_measured_joints():
    model, calls = lightweight_model()
    for anchor in (0, 8, 16):
        for i in ([0] if anchor == 0 else range(anchor-7, anchor+1)):
            model.update_obs(observation(i))
        actions = model.get_action()
        assert len(actions) == 8
        assert actions[0]["left_ee_pose"].shape == (7,)
        assert actions[0]["right_ee_joint_state"].tolist() == [1]
        assert calls[-1][0].shape == (1, 8, 3, 252, 322)
        assert calls[-1][3] == "Click the bell"
    np.testing.assert_allclose((calls[2][0][0, :, 0, 10, 10]+1)*127.5, np.arange(9, 17), atol=1e-5)
    np.testing.assert_allclose(calls[0][1][0, 0, :2, 2], [161, 126])
    np.testing.assert_allclose(calls[0][2][0], [[161, 126], [161, 126]])
    assert [c[4]["seed"] for c in calls] == [42, 43, 44]
    model.reset()
    assert model.previous is None and not model.pending and model.observation_count == 0
    model.update_obs(observation())
    model.get_action()



@pytest.mark.parametrize("horizon", range(1, 17))
def test_configurable_horizon_uses_first_frame_measured_fk(horizon):
    model, calls = lightweight_model(horizon)
    logs = []
    model._log = logs.append
    for anchor in range(0, 4 * horizon, horizon):
        for index in ([0] if anchor == 0 else range(anchor - horizon + 1, anchor + 1)):
            model.update_obs(observation(index))
        actions = model.get_action()
        assert len(actions) == horizon
        expected = np.maximum(0, np.arange(anchor - 7, anchor + 1))
        ids = ((calls[-1][0][0, :, 0, 10, 10] + 1) * 127.5).round().long().numpy()
        np.testing.assert_array_equal(ids, expected)
        # Deliberately differs between the first and latest observation, and between
        # arms. Predicted history_position and commanded joint targets are unrelated.
        first = expected[0]
        np.testing.assert_allclose(calls[-1][2][0],
                                   [[161 + first / 10, 126], [161 - first / 20, 126]], atol=2e-5)
        assert calls[-1][3] == "Click the bell"
        assert logs[-1]["query_source"] == "measured_joint_fk"
        assert logs[-1]["query_waypoint_index"] == first
        assert logs[-1]["history_indices"] == expected.tolist()
        assert logs[-1]["future_step_indices"] == list(range(1, 17))
    assert model.observation_count == 3 * horizon + 1
    for frame in model.previous:
        index = frame["index"]
        np.testing.assert_array_equal(frame["rgb"], observation(index)["vision"]["cam_third_view"]["color"])
        np.testing.assert_allclose(frame["joints"][:, 0], [index / 1000, -index / 2000])
    model.reset()
    assert not model.pending and model.previous is None
    model.update_obs(observation())
    assert len(model.get_action()) == horizon


@pytest.mark.parametrize("frequency", [None, 15, 30])
def test_index_inference_does_not_require_frequency_or_timestamps(frequency):
    model, calls = lightweight_model(4)
    for index in range(5):
        obs = observation(index)
        if frequency is None:
            del obs["additional_info"]["frequency"]
        else:
            obs["additional_info"]["frequency"] = frequency
        model.update_obs(obs)
        if index in (0, 4):
            assert len(model.get_action()) == 4
    # The strict inference signature rejects old positional times and frequency_hz.
    assert [call[3] for call in calls] == ["Click the bell"] * 2
    assert [call[4]["seed"] for call in calls] == [42, 43]
    ids = ((calls[-1][0][0, :, 0, 10, 10] + 1) * 127.5).round().long().numpy()
    np.testing.assert_array_equal(ids, [0, 0, 0, 0, 1, 2, 3, 4])


@pytest.mark.parametrize("horizon", [1, 4, 8, 16])
def test_execute_horizon_returns_first_n_actions(horizon):
    pred = prediction()
    pred["action_position"][:, :, 0] = np.arange(16)[:, None] / 100
    chunk, trajectory = convert_actions(pred, np.eye(4), np.eye(4), np.zeros((2, 3)),
                                        execute_horizon=horizon)
    assert len(chunk) == horizon and trajectory.shape == (16, 2, 4, 4)
    np.testing.assert_allclose([a["left_ee_pose"][0] for a in chunk], np.arange(horizon) / 100, atol=1e-8)


@pytest.mark.parametrize("value", [0, 17, -1, 4.5, True, None, "", "4.0", "abc"])
def test_invalid_execute_horizon_is_rejected_before_model_loading(value, monkeypatch):
    monkeypatch.delenv("FOURRC_EXECUTE_HORIZON", raising=False)
    with pytest.raises(ValueError, match="execute_horizon"):
        Model({"execute_horizon": value})
    with pytest.raises(ValueError, match="execute_horizon"):
        convert_actions(prediction(), np.eye(4), np.eye(4), np.zeros((2, 3)), execute_horizon=value)


def test_execute_horizon_env_overrides_yaml_before_model_loading(monkeypatch):
    monkeypatch.setenv("FOURRC_EXECUTE_HORIZON", "4")
    model = Model.__new__(Model)
    # Stop at the robot guard, after config resolution and before checkpoint loading.
    with pytest.raises(ValueError, match="aloha-agilex"):
        model.__init__({"execute_horizon": 16})
    assert model.execute_horizon == 4
    monkeypatch.delenv("FOURRC_EXECUTE_HORIZON")
    with pytest.raises(ValueError, match="aloha-agilex"):
        model.__init__({"execute_horizon": 16})
    assert model.execute_horizon == 16
    with pytest.raises(ValueError, match="aloha-agilex"):
        model.__init__({})
    assert model.execute_horizon == 8
    assert validate_execute_horizon("1") == 1

def test_rgb_padding_matches_upstream_convention():
    frame = observation()["vision"]["cam_third_view"]
    frame["color"][..., 0] = 255
    model, calls = lightweight_model()
    obs = observation();obs["vision"]["cam_third_view"] = frame
    model.update_obs(obs);model.get_action()
    assert torch.all(calls[0][0][:, :, 0] == 1)
    assert torch.all(calls[0][0][:, :, 1:] == -1)


def test_full_camera_base_control_roundtrip():
    offsets = np.array([[.147335, -.0000015, -.000713515]]*2)
    wb = np.eye(4);wb[:3, :3] = Rotation.from_euler("xyz", [.1, -.2, 1.2]).as_matrix();wb[:3, 3] = [.3, -.65, .05]
    cw = np.eye(4);cw[:3, :3] = Rotation.from_euler("xyz", [-.3, .5, -.8]).as_matrix();cw[:3, 3] = [-.2, .3, 1.1]
    base_tcp = np.broadcast_to(np.eye(4), (16, 2, 4, 4)).copy()
    base_tcp[..., :3, 3] = [.2, .3, .8]
    base_tcp[..., :3, :3] = Rotation.from_euler("xyz", [.3, -.1, .4]).as_matrix()
    camera_tcp = cw @ wb @ base_tcp
    pred = prediction();pred.update(action_position=camera_tcp[..., :3, 3], action_rotation=camera_tcp[..., :3, :3])
    chunk, recovered = convert_actions(pred, cw, wb, offsets)
    np.testing.assert_allclose(recovered, base_tcp, atol=1e-12)
    for action in chunk:
        for arm, side in enumerate(("left", "right")):
            ee = action[f"{side}_ee_pose"]
            r = Rotation.from_quat(ee[[4, 5, 6, 3]]).as_matrix()
            np.testing.assert_allclose(ee[:3] + r @ offsets[arm], (wb @ base_tcp)[0, arm, :3, 3], atol=1e-7)



def test_distinct_arms_reach_correct_robotwin_action_slots():
    # Different positions, rotations, grippers and temporal paths are essential:
    # a symmetric roundtrip cannot catch a left/right swap at either boundary.
    from collections.abc import Mapping
    geometry = AlohaKinematics(ROOT)
    home, offsets = geometry.tcp_base(np.zeros((2, 6))), geometry.offsets
    wb = np.eye(4)
    wb[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    wb[:3, 3] = [0, -.65, .04]
    cw = np.eye(4)
    cw[:3, :3] = Rotation.from_euler("xyz", [.8, -.2, 2.9]).as_matrix()
    cw[:3, 3] = [.1, -.3, 1.4]
    tcp = np.broadcast_to(home, (16, 2, 4, 4)).copy()
    tcp[:, 0, :3, :3] = Rotation.from_euler("xyz", [.3, -.6, .1]).as_matrix()
    tcp[:, 1, :3, :3] = Rotation.from_euler("xyz", [-.4, .2, -.7]).as_matrix()
    tcp[:, 0, 0, 3] += np.arange(16) * .002
    tcp[:, 1, 2, 3] -= np.arange(16) * .003
    camera_tcp = cw @ wb @ tcp
    pred = prediction()
    pred.update(action_position=camera_tcp[..., :3, 3],
                action_rotation=camera_tcp[..., :3, :3])
    chunk, _ = convert_actions(pred, cw, wb, offsets)
    namespace = dict(np=np, Mapping=Mapping)
    for name in ("normalize_robotwin_action_type", "first_present", "action_array",
                 "action_gripper", "current_ee_pose", "current_gripper",
                 "xpolicylab_action_to_robotwin"):
        load_eval_function(name, namespace)
    convert = namespace["xpolicylab_action_to_robotwin"]
    for step, action in enumerate(chunk):
        flat, kind = convert(action, action_type="ee")
        assert kind == "ee" and flat.shape == (16,)
        assert flat[7] == 0 and flat[15] == 1
        for arm, slot in enumerate((0, 8)):
            ee = flat[slot:slot + 7]
            rotation = Rotation.from_quat(ee[[4, 5, 6, 3]]).as_matrix()
            expected = wb @ tcp[step, arm]
            np.testing.assert_allclose(rotation, expected[:3, :3], atol=1e-7)
            np.testing.assert_allclose(ee[:3] + rotation @ offsets[arm],
                                       expected[:3, 3], atol=1e-7)

def test_measured_fk_and_projection():
    geometry = AlohaKinematics(ROOT)
    home, offsets = geometry.tcp_base(np.zeros((2, 6))), geometry.offsets
    np.testing.assert_allclose(offsets, [[.147335, -.0000015, -.000713515]]*2, atol=1e-10)
    np.testing.assert_allclose(home[:, :3, 3], [[.483501526, .300868847, .941286473], [.484527564, -.304961319, .940286473]], atol=1e-8)
    k = observation()["vision"]["cam_third_view"]["intrinsic_matrix"]
    wb = np.eye(4); wb[:3, 3] = [.02, .03, .04]
    cw = np.eye(4); cw[:3, 3] = [.1, -.1, .5]
    q = project_queries((cw @ wb @ home)[:, :3, 3], k)
    point = (cw @ wb @ home)[..., :3, 3]
    uv = point @ k.T
    np.testing.assert_allclose(q, uv[:, :2]/uv[:, 2:], atol=1e-5)
    moved = np.zeros((2, 6)); moved[1] = [.1, .2, -.3, .1, .2, .3]
    moved_tcp = geometry.tcp_base(moved)
    np.testing.assert_allclose(moved_tcp[0], home[0])
    assert np.linalg.norm(moved_tcp[1, :3, 3] - home[1, :3, 3]) > .01
    with pytest.raises(ValueError, match="joint positions"):
        geometry.tcp_base(np.full((2, 6), np.nan))


def test_invalid_input_and_model_failure():
    model, _ = lightweight_model()
    obs = observation();obs["additional_info"]["third_camera_source"] = "front_camera"
    with pytest.raises(ValueError, match="observer_camera"): model.update_obs(obs)
    model.reset();model.update_obs(observation())
    with pytest.raises(ValueError, match="observation count"): model.update_obs(observation())
    model.inference.infer_stage2_window = lambda *a, **kw: dict(prediction(), success=False)
    with pytest.raises(RuntimeError, match="failure"): model.get_action()
    with pytest.raises(ValueError, match="behind"): project_queries(np.zeros((2, 3)), np.eye(3))
    with pytest.raises(ValueError): rigid(np.zeros((4, 4)))
    invalid = prediction();invalid["action_rotation"][0, 0] = 0
    with pytest.raises(ValueError, match="rotation"): convert_actions(invalid, np.eye(4), np.eye(4), np.zeros((2, 3)))


def load_eval_function(name, namespace):
    tree = ast.parse((ROOT / "scripts/eval_policy_xpolicylab.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "eval_policy_xpolicylab.py", "exec"), namespace)
    return namespace[name]


def test_policy_failure_before_first_action_counts_as_failed_episode():
    import traceback
    class Env:
        eval_video_path = None
        render_freq = 0
        def setup_demo(self, **kw): pass
        def close_env(self, **kw): pass
        def set_instruction(self, instruction): self.instruction = instruction
        def get_instruction(self): return self.instruction
        def get_obs(self): return {}
    class Client:
        def call(self, func_name, **kwargs):
            if func_name == "get_action": raise RuntimeError("bad prediction")
    ns = dict(np=np, traceback=traceback, wait_for_policy_inference=lambda *args: None, UnStableError=type("UnStableError", (Exception,), {}),
              parse_bool=bool, build_instruction=lambda *args: "bell", prepare_policy_case=lambda *args: None,
              reset_policy=lambda *args: None, is_episode_end=lambda env: False,
              robotwin_obs_to_xpolicylab=lambda *a, **kw: {}, normalize_action_chunk=lambda x:x,
              notify_trial_end=lambda *args: None, safe_close_env=lambda *a, **kw: None)
    evaluate = load_eval_function("eval_remote_policy", ns)
    args = dict(task_name="click_bell", policy_name="FourRC", render_freq=0, clear_cache_freq=5,
                task_config="4rc_clean", ckpt_setting="test")
    env = Env()
    seed, success = evaluate("click_bell", env, args, Client(), 100000,
                             usr_args={"expert_check":False,"action_type":"ee"}, test_num=1)
    assert (seed, success, env.test_num) == (100001, 0, 1)
    env.get_obs = lambda: (_ for _ in ()).throw(RuntimeError("renderer failure"))
    with pytest.raises(RuntimeError, match="5 consecutive env errors"):
        evaluate("click_bell", env, args, Client(), 100000,
                 usr_args={"expert_check":False,"action_type":"ee"}, test_num=1)


@pytest.mark.parametrize("horizon", [4, 8, 16])
@pytest.mark.parametrize("encoded", [False, True])
def test_websocket_rgb_transport(encoded, horizon):
    import asyncio
    import socket
    import sys
    import threading
    sys.path.insert(0, str(ROOT / "XPolicyLab"))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from client_server.ws.model_client import WsModelClient
    from XPolicyLab.utils.process_data import encode_image_bit
    model, calls = lightweight_model(horizon)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    stop = threading.Event()
    errors = []
    async def serve():
        server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=port))
        task = asyncio.create_task(server.serve_forever())
        try:
            while not stop.is_set():
                if task.done():
                    task.result()
                await asyncio.sleep(.02)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    def run():
        try: asyncio.run(serve())
        except Exception as exc: errors.append(exc)
    thread = threading.Thread(target=run, daemon=True);thread.start()
    try:
        with WsModelClient(url=f"ws://127.0.0.1:{port}", evaluation_id="test", trial_id="test",
                           action_case_id="test", max_connect_attempts=20, connect_retry_delay_s=.05,
                           request_timeout_s=10) as client:
            client.call(func_name="prepare_case", obs={"seed":123})
            client.call(func_name="reset")
            obs = observation()
            rgb = obs["vision"]["cam_third_view"]["color"]
            rgb[..., 0] = 240
            rgb[..., 1] = 40
            rgb[..., 2] = 10
            if encoded:
                obs["vision"]["cam_third_view"]["color"] = encode_image_bit(rgb)
            client.call(func_name="update_obs", obs=obs)
            chunk = client.call(func_name="get_action")
            assert len(chunk) == horizon
            assert chunk[0]["left_ee_pose"].shape == (7,)
            assert calls[0][4]["seed"] == 123
            channels = calls[0][0][0, 0, :, 10, 10].numpy()
            assert channels[0] > channels[1] > channels[2]
            for index in range(1, horizon + 1):
                obs = observation(index)
                if encoded:
                    camera = obs["vision"]["cam_third_view"]
                    camera["color"] = encode_image_bit(camera["color"])
                client.call(func_name="update_obs", obs=obs)
            assert len(client.call(func_name="get_action")) == horizon
            assert calls[-1][0].shape == (1, 8, 3, 252, 322)
            client.call(func_name="trial_end", obs={"success": False})
    finally:
        stop.set();thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors


def test_observation_metadata_uses_actual_robot_root():
    wb = np.eye(4, dtype=np.float32)
    wb[:3, 3] = [.4, -.2, .1]
    robot = SimpleNamespace(is_dual_arm=True, left_entity=SimpleNamespace(
        get_root_pose=lambda: SimpleNamespace(to_transformation_matrix=lambda: wb)))
    converter = load_eval_function('robotwin_obs_to_xpolicylab', dict(
        np=np, convert_vision=lambda obs: {}, convert_state=lambda obs, **kwargs: {}))
    obs = converter({'observation': {'observer_camera': {}}}, frequency=15,
                    task_env=SimpleNamespace(robot=robot))
    np.testing.assert_array_equal(obs['additional_info']['T_world_from_robot_base'], wb)
    assert obs['additional_info']['third_camera_source'] == 'observer_camera'
    legacy = converter({'observation': {'front_camera': {}}})
    assert 'third_camera_source' not in legacy['additional_info']
    assert 'T_world_from_robot_base' not in legacy['additional_info']


def test_real_joint_fields_are_not_command_targets():
    converter = load_eval_function("convert_state", dict(np=np))
    robot = SimpleNamespace(
        get_left_arm_real_jointState=lambda: [.1, .2, .3, .4, .5, .6, 1.],
        get_right_arm_real_jointState=lambda: [-.1, -.2, -.3, -.4, -.5, -.6, 0.])
    state = converter({"joint_action": {"left_arm": [9.] * 6, "right_arm": [8.] * 6}},
                      task_env=SimpleNamespace(robot=robot))
    np.testing.assert_allclose(state["left_arm_real_joint_state"], [.1, .2, .3, .4, .5, .6])
    np.testing.assert_allclose(state["right_arm_real_joint_state"], [-.1, -.2, -.3, -.4, -.5, -.6])
    np.testing.assert_array_equal(state["left_arm_joint_state"], [9.] * 6)


def test_missing_real_joints_and_skipped_waypoints_fail_without_fallback():
    model, _ = lightweight_model(4)
    obs = observation(); del obs["state"]["left_arm_real_joint_state"]
    with pytest.raises(ValueError, match="measured"):
        model.update_obs(obs)
    model.reset(); model.update_obs(observation()); model.get_action()
    with pytest.raises(ValueError, match="waypoint_index=1"):
        model.update_obs(observation(2))


@pytest.mark.parametrize("horizon,limit,success", [(4, 3, True), (4, 4, False), (4, 9, False), (16, 17, True)])
def test_eval_receives_every_waypoint_including_terminal(horizon, limit, success):
    import traceback
    model, calls = lightweight_model(horizon)
    received = []
    logs = []
    model._log = logs.append
    class Env:
        eval_video_path = None
        render_freq = 0
        def setup_demo(self, **kw):
            self.take_action_cnt = 0
            self.eval_success = False
        def close_env(self, **kw): pass
        def set_instruction(self, instruction): self.instruction = instruction
        def get_instruction(self): return self.instruction
        def get_obs(self):
            obs = observation(self.take_action_cnt)
            obs["additional_info"]["episode_end"] = self.take_action_cnt >= limit
            return obs
        def take_action(self, action, **kw):
            self.take_action_cnt += 1
            self.eval_success = success and self.take_action_cnt >= limit
    class Client:
        def call(self, func_name, obs=None):
            method = getattr(model, func_name)
            result = method() if obs is None else method(obs)
            if func_name == "update_obs":
                received.append(model.pending[-1])
            return result
    ns = dict(np=np, traceback=traceback, wait_for_policy_inference=lambda *args: None, UnStableError=type("UnStableError", (Exception,), {}),
              parse_bool=bool, build_instruction=lambda *args: "Click the bell",
              prepare_policy_case=lambda client, task, seed, *args: model.prepare_case({"task_name":task,"seed":seed}),
              reset_policy=lambda client: model.reset(),
              is_episode_end=lambda env: env.take_action_cnt >= limit,
              robotwin_obs_to_xpolicylab=lambda obs, **kw: obs,
              xpolicylab_action_to_robotwin=lambda action, **kw: (action, "ee"),
              normalize_action_chunk=lambda x:x,
              notify_trial_end=lambda client, task, seed, succ: model.on_trial_end({"success":succ}),
              safe_close_env=lambda *args, **kw: None)
    evaluate = load_eval_function("eval_remote_policy", ns)
    args = dict(task_name="click_bell", policy_name="FourRC", render_freq=0, clear_cache_freq=5,
                task_config="4rc_clean", ckpt_setting="test")
    env = Env()
    _, succeeded = evaluate("click_bell", env, args, Client(), 100000,
                            usr_args={"expert_check":False,"action_type":"ee"}, test_num=1)
    assert succeeded == int(success)
    assert [frame["index"] for frame in received] == list(range(limit + 1))
    assert received[-1]["terminal"] and not any(frame["terminal"] for frame in received[:-1])
    assert len(calls) == (limit + horizon - 1) // horizon
    assert model.observation_count == limit + 1
    assert logs[-1]["result"]["success"] == success
    np.testing.assert_array_equal(received[-1]["rgb"], observation(limit)["vision"]["cam_third_view"]["color"])

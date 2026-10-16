"""Observation-buffer checks without loading policy weights or a simulator."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from XPolicyLab.policy.FourRC.model import Model


def observation(index):
    def camera(value):
        return {"color": np.full((240, 320, 3), value, dtype=np.uint8)}

    third = camera(index)
    third.update(intrinsic_matrix=np.array([[100., 0., 160.], [0., 100., 120.], [0., 0., 1.]]),
                 extrinsics_matrix=np.eye(4))
    return {
        "instruction": "adjust the bottle",
        "vision": {"cam_third_view": third, "cam_left_wrist": camera(index),
                   "cam_right_wrist": camera(index + 100)},
        "state": {"left_arm_real_joint_state": np.full(6, index / 100),
                  "right_arm_real_joint_state": np.full(6, -index / 100),
                  "left_ee_joint_state": np.array([.25]),
                  "right_ee_joint_state": np.array([.75]),
                  "left_arm_joint_state": np.full(6, 999.),
                  "right_arm_joint_state": np.full(6, 999.)},
        "additional_info": {"third_camera_source": "observer_camera", "waypoint_index": index,
                            "T_world_from_robot_base": np.eye(4)},
    }


@pytest.fixture
def model():
    model = Model.__new__(Model)
    model.history_frames = 4
    model.prediction_horizon = 40
    model.execute_horizon = 8
    model.cfg = {}
    model.case = {}
    model.viewer = None
    model.policy = object()
    model.device, model.dtype = torch.device("cpu"), torch.float32
    model.sampling_steps = 1
    model.offsets = np.zeros((2, 3))

    def tcp_base(joints):
        tcp = np.repeat(np.eye(4)[None], 2, axis=0)
        tcp[:, 0, 3] = joints[:, 0] + [0., .1]
        tcp[:, 2, 3] = 1.
        return tcp

    model.kinematics = SimpleNamespace(tcp_base=tcp_base)
    model.inference = SimpleNamespace(infer_stage2_window=Mock(return_value={
        "success": True,
        "action_position": np.zeros((40, 2, 3)),
        "action_rotation": np.broadcast_to(np.eye(3), (40, 2, 3, 3)).copy(),
        "action_gripper": np.ones((40, 2)),
        "future_step_indices": np.arange(1, 41),
    }))
    model.records = []
    model._log = model.records.append
    model.reset()
    return model


def assert_window(model, indices):
    chunk = model.get_action()
    record = model.records[-1]
    assert len(chunk) == model.execute_horizon
    assert record["history_indices"] == indices
    assert record["query_waypoint_index"] == indices[0]
    assert record["wrist_waypoint_index"] == indices[-1]
    queries = np.array([[160. + indices[0], 120.], [170. - indices[0], 120.]])
    np.testing.assert_allclose(record["query_pixels"], queries)

    call = model.inference.infer_stage2_window.call_args
    images, intrinsics, query_tensor = call.args[1:4]
    assert images.shape == (1, model.history_frames, 3, 252, 322)
    np.testing.assert_allclose(images[0, :, 0, 6, 1], np.array(indices) / 255 * 2 - 1, atol=1e-6)
    np.testing.assert_allclose(query_tensor[0], queries + [1., 6.])
    np.testing.assert_allclose(intrinsics[0, :, :2, 2], np.tile([161., 126.], (len(indices), 1)))
    state = call.kwargs["current_robot_state"]
    assert state.shape == (1, 14)
    assert state.dtype == torch.float32
    np.testing.assert_allclose(state[0], [indices[-1] / 100] * 6 + [.25]
                               + [-indices[-1] / 100] * 6 + [.75])
    wrists = call.kwargs["wrist_images"]
    assert wrists.shape == (1, 2, 3, 240, 320)
    np.testing.assert_allclose(wrists[0, :, 0, 0, 0],
                               np.array([indices[-1], indices[-1] + 100]) / 255, atol=1e-6)


@pytest.mark.parametrize("count, indices", [
    (1, [0, 0, 0, 0]), (2, [0, 0, 0, 1]), (3, [0, 0, 1, 2]), (4, [0, 1, 2, 3]),
])
def test_initial_history_padding(model, count, indices):
    for index in range(count):
        model.update_obs(observation(index))
    assert_window(model, indices)


@pytest.mark.parametrize("history, execute", [(4, 1), (4, 3), (4, 8), (4, 32), (1, 2), (8, 2)])
def test_complete_chunks_keep_latest_history(model, history, execute):
    model.history_frames, model.execute_horizon = history, execute
    model.reset()
    model.update_obs(observation(0))
    assert_window(model, [0] * history)
    for anchor in (execute, 2 * execute):
        for index in range(anchor - execute + 1, anchor + 1):
            model.update_obs(observation(index))
        assert len(model.observation_history) <= history
        assert_window(model, [max(0, index) for index in range(anchor - history + 1, anchor + 1)])


def test_inference_retains_history_after_partial_chunk(model):
    model.update_obs(observation(0))
    assert_window(model, [0, 0, 0, 0])
    model.update_obs(observation(1))
    assert_window(model, [0, 0, 0, 1])
    assert_window(model, [0, 0, 0, 1])


def test_reset_requires_fresh_observation(model):
    model.update_obs(observation(0))
    assert model.inference_ready()
    model.reset()
    assert not model.inference_ready()
    with pytest.raises(RuntimeError, match="before update_obs"):
        model.get_action()
    model.update_obs(observation(0))
    assert_window(model, [0, 0, 0, 0])


def test_viewer_gate_works_without_full_chunk(model):
    ready = Mock(return_value=False)
    model.viewer = SimpleNamespace(inference_ready=ready, observe=Mock())
    assert not model.inference_ready()
    ready.assert_not_called()
    model.update_obs(observation(0))
    assert not model.inference_ready()
    ready.return_value = True
    assert model.inference_ready()


@pytest.mark.parametrize("key, value", [
    ("left_ee_joint_state", None), ("right_ee_joint_state", [float("nan")]),
    ("left_ee_joint_state", [1.1]), ("right_ee_joint_state", [-.1]),
    ("left_ee_joint_state", [0., 1.]), ("left_arm_real_joint_state", [0.] * 5),
    ("right_arm_real_joint_state", [float("inf")] * 6),
])
def test_reject_invalid_current_state(model, key, value):
    obs = observation(0)
    if value is None:
        del obs["state"][key]
    else:
        obs["state"][key] = value
    with pytest.raises(ValueError):
        model.update_obs(obs)
    assert model.observation_count == 0
    assert not model.observation_history


def test_server_decoded_rgb_observation(model):
    from XPolicyLab.utils.process_data import encode_image_bit, decode_obs_images
    obs = observation(0)
    rgb = np.full((240, 320, 3), [210, 90, 30], dtype=np.uint8)
    for camera in obs["vision"].values():
        camera["color"] = encode_image_bit(rgb)
    decode_obs_images(obs)  # Same decode boundary used by the policy server.
    model.update_obs(obs)
    model.get_action()
    frame = model.observation_history[-1]
    np.testing.assert_allclose(frame["rgb"][0, 0], rgb[0, 0], atol=2)
    np.testing.assert_allclose(frame["wrist_rgb"][:, 0, 0], np.tile(rgb[0, 0], (2, 1)), atol=2)

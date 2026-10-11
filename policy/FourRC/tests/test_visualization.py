import ast
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.FourRC.geometry import convert_actions
from XPolicyLab.policy.FourRC.visualization import (LiveViewer, create_viewer, scene_record,
                                                     project_tcp_pixels, tcp_overlay, FK_COLORS)
from XPolicyLab.policy.FourRC.tests.test_adapter import lightweight_model, observation, prediction

ROOT = Path(__file__).resolve().parents[4]


def cloud_builder():
    # Exercise the same upstream projector without loading its torch/model imports.
    path = ROOT.parent / '4RC/stage2_sliding_window_inference.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_point_cloud')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['make_point_cloud']


def visual_prediction():
    pred = prediction()
    pred.update(depth=np.ones((8, 252, 322), dtype=np.float32),
                depth_confidence=np.ones((8, 252, 322), dtype=np.float32),
                history_valid=np.ones((8, 2), dtype=bool))
    return pred


def transform(xyz, angles):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', angles).as_matrix()
    pose[:3, 3] = xyz
    return pose


def test_cloud_unpadding_and_tcp_share_world_frame():
    model, _ = lightweight_model()
    model.update_obs(observation()); model.get_action()
    frame = model.previous[0]
    frame['rgb'][...] = [213, 42, 11]
    frame['cw'] = transform([.2, -.1, 1.], [.1, .3, -.2])
    frame['wb'] = transform([.3, -.4, .2], [-.2, .1, .8])
    frames = [dict(frame, cw=transform([i / 10, -.1, 1.], [.1, .3, -.2]), index=i) for i in range(8)]
    pred = visual_prediction()
    pred['depth'][...] = 0
    pred['depth'][-1, 126, 161] = 2  # original RGB pixel (160,120), the principal point
    pred['depth'][-1, 0, 0] = 1     # padding must not become scene geometry
    pred['history_valid'][2, 1] = False
    _, base_tcp = convert_actions(pred, frames[-1]['cw'], frame['wb'], np.zeros((2, 3)), execute_horizon=4)
    record = scene_record(frames, pred, base_tcp, cloud_builder(), max_points=100,
                          case={'seed': 0}, instruction='test', window=0, execute_horizon=4)
    camera_world = np.linalg.inv(frames[-1]['cw'])
    expected = (camera_world @ [0, 0, 2, 1])[:3]
    np.testing.assert_allclose(record['clouds'][-1]['points'], expected[None], atol=1e-6)
    np.testing.assert_array_equal(record['clouds'][-1]['colors'], [[213, 42, 11]])
    expected_future = (camera_world @ [0, 0, 1, 1])[:3]
    np.testing.assert_allclose(record['future'][..., :3, 3], np.broadcast_to(expected_future, (16, 2, 3)), atol=1e-6)
    for i in range(8):
        np.testing.assert_allclose(record['recovered'][i, 0], (np.linalg.inv(frames[i]['cw']) @ [0, 0, 1, 1])[:3], atol=1e-6)
    assert np.isnan(record['recovered'][2, 1]).all()
    np.testing.assert_allclose(record['measured'][0], frame['wb'] @ frame['tcp_base'], atol=1e-6)
    assert record['executed_indices'] == [7]


def test_live_viewer_tracks_waypoints_and_bounds_memory():
    pytest.importorskip('viser')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    viewer = LiveViewer(cloud_builder(), port=port, max_points=50, max_windows=2, manual=False)
    try:
        model, calls = lightweight_model(4)
        model.viewer = viewer
        model.inference.infer_stage2_window = lambda *args, **kw: visual_prediction()
        model.reset()
        model.update_obs(observation())
        assert len(model.get_action()) == 4
        for i in range(1, 5):
            model.update_obs(observation(i))
        first = viewer.records[0]
        assert first['executed_indices'] == [0, 1, 2, 3, 4]
        assert len(first['clouds'][-1]['points']) == 50
        np.testing.assert_allclose(first['executed'][-1][:, 0, 3], [.004, -.002])
        np.testing.assert_array_equal(first['future'][..., 0, 3], 0)
        viewer.follow.value = False
        assert len(model.get_action()) == 4
        assert viewer.window.value == 0
        for i in range(5, 9):
            model.update_obs(observation(i))
        model.get_action()
        assert [r['window'] for r in viewer.records] == [1, 2]
        assert viewer.window.value == 1  # selected window was evicted
        viewer.follow.value = True
        assert viewer.window.value == 2
        viewer.step.value = 4
        viewer.cloud.value = False
        viewer.recovered.value = True
        viewer.axes.value = False
        model.on_trial_end({'success': False})
        assert all(r['result'] == {'success': False} for r in viewer.records)
        assert 'failed' in viewer.info.content
        assert model.viewer is viewer  # no display error was swallowed
        model.reset()
        assert not viewer.records
    finally:
        viewer.close()


def test_optional_visualization_failure_preserves_actions():
    model, _ = lightweight_model(4)
    logs, closed = [], []
    model._log = logs.append
    def fail(*args, **kwargs):
        raise RuntimeError('display disconnected')
    model.viewer = SimpleNamespace(begin_inference=lambda: None, observe=lambda frame: None, publish=fail, close=lambda: closed.append(True))
    model.update_obs(observation())
    assert len(model.get_action()) == 4
    assert model.viewer is None and closed == [True]
    assert logs[-1]['event'] == 'visualization_error'
    assert model.window == 1 and len(model.previous) == 8


def test_visualization_defaults_off_and_environment_overrides(monkeypatch):
    monkeypatch.delenv('FOURRC_VISER', raising=False)
    assert create_viewer({}, None) is None
    monkeypatch.setenv('FOURRC_VISER', '0')
    assert create_viewer({'viser': True}, None) is None
    monkeypatch.setenv('FOURRC_VISER', 'invalid')
    with pytest.raises(ValueError, match='FOURRC_VISER'):
        create_viewer({}, None)


@pytest.mark.parametrize('downsample', [1, 2, 4])
def test_point_count_downsample_preserves_rays_and_all_history_frames(downsample):
    model, _ = lightweight_model()
    model.update_obs(observation()); model.get_action()
    base = model.previous[0]
    frames = [dict(base, index=i, rgb=np.full((240, 320, 3), i, dtype=np.uint8)) for i in range(8)]
    pred = visual_prediction()
    # Distinct reconstruction for each historical frame.
    for i in range(8):
        pred['depth'][i] *= 1 + i / 10
    _, base_tcp = convert_actions(pred, np.eye(4), np.eye(4), np.zeros((2, 3)))
    record = scene_record(frames, pred, base_tcp, cloud_builder(), max_points=100000,
                          case={}, instruction='test', window=0, execute_horizon=4, downsample=downsample)
    assert len(record['clouds']) == 8 and record['downsample'] == downsample
    for i, cloud in enumerate(record['clouds']):
        assert cloud['points'].shape == (240 * 320 // downsample, 3)
        assert cloud['index'] == i
        np.testing.assert_allclose(cloud['points'][:, 2], 1 + i / 10, atol=1e-6)
        np.testing.assert_array_equal(cloud['colors'], i)
        # Sampled points keep their original pixel coordinates and camera rays.
        expected = np.array([(downsample - 160) / 100, -1.2, 1]) * (1 + i / 10)
        np.testing.assert_allclose(cloud['points'][1], expected, atol=1e-6)
    bad = dict(pred, depth=pred['depth'][-1])
    with pytest.raises(ValueError, match='depth/confidence'):
        scene_record(frames, bad, base_tcp, cloud_builder(), max_points=100,
                     case={}, instruction='test', window=0, execute_horizon=4)


def test_one_click_one_inference_and_history_browsing():
    pytest.importorskip('viser')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    viewer = LiveViewer(cloud_builder(), port=port, max_points=50)
    try:
        model, _ = lightweight_model(4)
        model.viewer = viewer
        calls = []
        def infer(*args, **kw):
            calls.append(kw['seed'])
            # Clicking during inference must not queue the next invocation.
            viewer._request_inference()
            return visual_prediction()
        model.inference.infer_stage2_window = infer
        model.reset(); model.update_obs(observation())
        assert viewer.manual and viewer.point_size.value == .003 and viewer.downsample == 2
        assert not model.inference_ready()
        with pytest.raises(RuntimeError, match='Click'):
            model.get_action()
        assert not calls and model.window == 0
        viewer._request_inference(); viewer._request_inference()
        assert model.inference_ready()
        assert len(model.get_action()) == 4
        assert len(calls) == 1 and viewer.run_button.disabled
        assert not model.inference_ready()
        # Busy clicks, including after publish, cannot queue a new inference.
        viewer._request_inference()
        for i in range(1, 5): model.update_obs(observation(i))
        assert not model.inference_ready() and not viewer.run_button.disabled
        viewer._request_inference(); model.get_action()
        assert len(calls) == 2
        assert [cloud['index'] for cloud in viewer.records[-1]['clouds']] == [0, 0, 0, 0, 1, 2, 3, 4]
        viewer.follow.value = False
        viewer.history_frame.value = 4
        np.testing.assert_array_equal(viewer.rgb.image[:80], observation(1)['vision']['cam_third_view']['color'][:80])
        assert np.any(viewer.rgb.image != 1)  # TCP annotations are overlaid.
        np.testing.assert_array_equal(viewer.records[-1]['clouds'][4]['rgb'], 1)
        viewer._step_history(1)
        np.testing.assert_array_equal(viewer.rgb.image[:80], observation(2)['vision']['cam_third_view']['color'][:80])
        viewer.measured.value = False
        viewer.future.value = False
        viewer.refresh()
        np.testing.assert_array_equal(viewer.rgb.image, observation(2)['vision']['cam_third_view']['color'])
        assert len(calls) == 2  # browsing never calls the model
        for i in range(5, 9): model.update_obs(observation(i))
        assert not model.inference_ready()
        model.on_trial_end({'success': False})
        viewer._request_inference()
        assert viewer.run_button.disabled and not viewer.inference_ready()
        model.reset()
        assert not viewer.records and viewer.history_frame.value == 7
        viewer.close()
        with pytest.raises(RuntimeError, match='closed'):
            viewer.inference_ready()
    finally:
        if not viewer._closed:
            viewer.close()


def test_manual_ui_failure_cannot_fall_back_to_automatic_inference():
    model, _ = lightweight_model(4)
    model.update_obs(observation())
    def fail(*args, **kwargs):
        raise RuntimeError('display failed')
    model.viewer = SimpleNamespace(manual=True, begin_inference=lambda: None,
                                  publish=fail, close=lambda: None)
    with pytest.raises(RuntimeError, match='Manual inference UI failed'):
        model.get_action()
    assert model.window == 0
    with pytest.raises(RuntimeError, match='Manual inference UI failed'):
        model.inference_ready()
    with pytest.raises(RuntimeError, match='Manual inference UI failed'):
        model.get_action()


def test_manual_polling_uses_short_calls_and_preserves_other_policies(monkeypatch):
    from XPolicyLab.policy.FourRC.tests.test_adapter import load_eval_function
    states = iter([False, False, False, True])
    calls, sleeps = [], []
    def call(*, func_name):
        calls.append(func_name)
        return next(states)
    monkeypatch.setattr('time.sleep', sleeps.append)
    wait = load_eval_function('wait_for_policy_inference', {})
    wait(SimpleNamespace(call=call), 'FourRC')
    assert calls == ['inference_ready'] * 4 and sleeps == [.1] * 3
    wait(SimpleNamespace(call=lambda **kw: pytest.fail('Unexpected poll')), 'OtherPolicy')


def test_tcp_projection_uses_selected_camera_and_unpadded_intrinsics():
    k = np.array([[100., 0., 160.], [0., 100., 120.], [0., 0., 1.]])
    cw = transform([.3, -.2, .5], [.2, -.4, .1])
    camera_points = np.array([[.2, -.4, 2.], [0., 0., -1.], [10., 0., 1.], [0., 0., 0.]])
    wc = np.linalg.inv(cw)
    world_points = camera_points @ wc[:3, :3].T + wc[:3, 3]
    uv, valid = project_tcp_pixels(world_points, cw, k, (240, 320, 3))
    np.testing.assert_allclose(uv[0], [170., 100.])
    np.testing.assert_array_equal(valid, [True, False, False, False])
    moved = cw.copy()
    moved[0, 3] += .2
    uv, _ = project_tcp_pixels(world_points[:1], moved, k, (240, 320, 3))
    np.testing.assert_allclose(uv[0], [180., 100.])


def test_tcp_overlay_preserves_rgb_inputs_and_hides_invalid_points():
    rgb = np.full((240, 320, 3), [20, 40, 80], dtype=np.uint8)
    original = rgb.copy()
    k = np.array([[100., 0., 160.], [0., 100., 120.], [0., 0., 1.]])
    points = np.array([[-.4, 0., 1.], [.4, 0., 1.]])
    result = tcp_overlay(rgb, k, np.eye(4), measured=points)
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_allclose(result[120, 120], FK_COLORS[0], atol=3)
    np.testing.assert_allclose(result[120, 200], FK_COLORS[1], atol=3)
    np.testing.assert_array_equal(result[:80], original[:80])
    hidden = tcp_overlay(rgb, k, np.eye(4), measured=np.array([[0, 0, -1], [np.nan, 0, 1]]))
    np.testing.assert_array_equal(hidden, original)

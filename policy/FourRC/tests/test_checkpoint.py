"""V2 loading boundary without allocating the full policy or downloading assets."""
import json
from types import SimpleNamespace
import sys

import pytest
import torch

from XPolicyLab.policy.FourRC.checkpoint import read_config
from XPolicyLab.policy.FourRC.model import Model


@pytest.fixture
def checkpoint(tmp_path):
    config = dict(training_stage=2, action_head_type="groot_tcp_v1",
                  state_normalization_version=1, action_normalization="componentwise",
                  action_normalization_version=2, wrist_encoder_config={"model_type": "dinov2"},
                  history_frames=4, prediction_horizon=32, action_dim=1280, action_depth=12,
                  action_heads=32, condition_dim=2048, action_output_dim=1024, action_dropout=.2,
                  global_token_grid=[14, 19], text_max_length=128, tcp_query_window_size=3,
                  action_gradient_checkpointing=True, t5_model="google-t5/t5-base", sampling_steps=4)
    for name, value in (("config", config), ("action_norm", {"method": "componentwise", "version": 2}),
                        ("state_norm", {"version": 1})):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    return tmp_path


def test_metadata_only_and_required_weights(checkpoint):
    assert read_config(checkpoint, require_weights=False)["action_head_type"] == "groot_tcp_v1"
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        read_config(checkpoint)


@pytest.mark.parametrize("name", ["action_norm.json", "state_norm.json"])
def test_normalization_files_required_even_without_weights(checkpoint, name):
    (checkpoint / name).unlink()
    with pytest.raises(FileNotFoundError, match=name):
        read_config(checkpoint, require_weights=False)


@pytest.mark.parametrize("key, value", [
    ("action_head_type", None), ("action_head_type", "film"),
    ("state_normalization_version", 2), ("history_frames", 0),
    ("prediction_horizon", True), ("action_normalization_version", 1),
])
def test_reject_incompatible_config(checkpoint, key, value):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        read_config(checkpoint, require_weights=False)


def test_model_loads_v2_without_previous_policy_attributes(checkpoint, monkeypatch):
    from XPolicyLab.policy.FourRC import model, visualization
    config = read_config(checkpoint, require_weights=False)
    (checkpoint / "model.safetensors").touch()
    (checkpoint / "stage2_sliding_window_inference.py").touch()
    policy = object()  # No use_wrist_condition or global_feature_mode in V2.
    inference = SimpleNamespace(
        resolve_device_dtype=lambda *args: (torch.device("cpu"), torch.float32),
        load_stage2_policy=lambda *args, **kwargs: (policy, config), make_point_cloud=None)
    monkeypatch.setitem(sys.modules, "stage2_sliding_window_inference", inference)
    monkeypatch.setattr(model, "AlohaKinematics", lambda root: SimpleNamespace(offsets=None))
    monkeypatch.setattr(visualization, "create_viewer", lambda *args, **kwargs: None)
    monkeypatch.setattr(Model, "_log", lambda *args: None)
    for name in ("FOURRC_ROOT", "FOURRC_EXECUTE_HORIZON", "FOURRC_T5_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FOURRC_LOG_DIR", str(checkpoint))
    loaded = Model(dict(ckpt_name=str(checkpoint), fourrc_root=str(checkpoint),
                        env_cfg_type="arx_x5", action_type="ee", sampling_steps=None))
    assert loaded.policy is policy
    assert loaded.sampling_steps == 4
    assert loaded.history_frames == 4
    assert loaded.prediction_horizon == 32

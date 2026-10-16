"""Checkpoint contract for groot_tcp_v1 Stage2 with real robot state."""
import json
from pathlib import Path


def read_config(checkpoint, *, require_weights=True):
    root = Path(checkpoint).expanduser()
    names = ["config.json", "action_norm.json", "state_norm.json"]
    if require_weights:
        names.append("model.safetensors")
    for name in names:
        if not (root / name).is_file():
            raise FileNotFoundError(f"FourRC V2 checkpoint requires {root / name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("training_stage") != 2 or config.get("normalize_geometry", False):
        raise ValueError("FourRC requires a metric-geometry Stage2 checkpoint")
    if config.get("action_head_type") != "groot_tcp_v1":
        raise ValueError("FourRC requires action_head_type=groot_tcp_v1; initialize a new Stage2 run")
    if config.get("state_normalization_version") != 1:
        raise ValueError("FourRC requires state_normalization_version=1")
    architecture = (
        "action_dim", "action_depth", "action_heads", "condition_dim", "action_output_dim",
        "action_dropout", "global_token_grid", "text_max_length", "tcp_query_window_size",
        "action_gradient_checkpointing", "t5_model", "sampling_steps",
    )
    missing = [key for key in architecture if key not in config]
    if missing:
        raise ValueError(f"Incomplete FourRC V2 architecture metadata: {missing}")
    if (config.get("action_normalization") != "componentwise"
            or config.get("action_normalization_version") != 2):
        raise ValueError("FourRC requires componentwise action normalization version 2; retrain from Stage1")
    if not isinstance(config.get("wrist_encoder_config"), dict) or not config["wrist_encoder_config"]:
        raise ValueError("FourRC requires a saved wrist_encoder_config")
    norm = json.loads((root / "action_norm.json").read_text(encoding="utf-8"))
    if (norm.get("method"), norm.get("version")) != ("componentwise", 2):
        raise ValueError("FourRC action_norm.json must use componentwise version 2")
    state_norm = json.loads((root / "state_norm.json").read_text(encoding="utf-8"))
    if state_norm.get("version") != config["state_normalization_version"]:
        raise ValueError("FourRC state_norm.json must use version 1")
    for name in ("history_frames", "prediction_horizon"):
        value = config.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"FourRC {name} must be a positive integer")
    return config

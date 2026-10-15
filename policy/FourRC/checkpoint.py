"""Checkpoint contract for wrist-conditioned FiLM + componentwise-v2 Stage2."""
import json
from pathlib import Path


def read_config(checkpoint, *, require_weights=True):
    root = Path(checkpoint).expanduser()
    names = ["config.json", "action_norm.json"]
    if require_weights:
        names.append("model.safetensors")
    for name in names:
        if not (root / name).is_file():
            raise FileNotFoundError(f"FourRC wrist checkpoint requires {root / name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("training_stage") != 2 or config.get("normalize_geometry", False):
        raise ValueError("FourRC requires a metric-geometry Stage2 checkpoint")
    if config.get("language_conditioning") != "film":
        raise ValueError("FourRC requires language_conditioning=film; retrain from Stage1")
    if (config.get("action_normalization") != "componentwise"
            or config.get("action_normalization_version") != 2):
        raise ValueError("FourRC requires componentwise action normalization version 2; retrain from Stage1")
    if config.get("time_encoding", "index") != "index" or config.get("tcp_temporal_weight", 0) != 0:
        raise ValueError("FourRC requires step-index encoding and tcp_temporal_weight=0")
    if config.get("use_wrist_condition") is not True:
        raise ValueError("FourRC requires use_wrist_condition=true; non-wrist models are unsupported")
    if not isinstance(config.get("wrist_encoder_config"), dict) or not config["wrist_encoder_config"]:
        raise ValueError("FourRC requires a saved wrist_encoder_config")
    norm = json.loads((root / "action_norm.json").read_text(encoding="utf-8"))
    if (norm.get("method"), norm.get("version")) != ("componentwise", 2):
        raise ValueError("FourRC action_norm.json must use componentwise version 2")
    for name in ("history_frames", "prediction_horizon"):
        value = config.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"FourRC {name} must be a positive integer")
    return config

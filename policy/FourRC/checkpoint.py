"""Checkpoint contract for the wrist-conditioned, quantile Stage2 evaluator."""
import json
from pathlib import Path


DEFAULT_CHECKPOINT = "checkpoints/4rc-stage2-action-clean-qity-1-task-wrist-40chunk/checkpoint-10000"


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
    if config.get("use_wrist_condition") is not True:
        raise ValueError("FourRC requires use_wrist_condition=true; non-wrist models are unsupported")
    if not isinstance(config.get("wrist_encoder_config"), dict) or not config["wrist_encoder_config"]:
        raise ValueError("FourRC requires a saved wrist_encoder_config")
    if (config.get("action_normalization") != "quantile"
            or config.get("action_normalization_version") != 1):
        raise ValueError("FourRC requires quantile action normalization version 1")
    for name in ("history_frames", "prediction_horizon"):
        value = config.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"FourRC {name} must be a positive integer")
    return config

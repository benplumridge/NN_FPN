#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required for sweep configs. Install it with: uv pip install PyYAML"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = ROOT / "configs" / "reproduce_1d.yaml"

PARAM_PATHS = {
    "hidden_width": ("1d", "train", "hidden_width"),
    "nn_weight_decay": ("1d", "train", "nn_weight_decay"),
    "batch_size": ("1d", "train", "params", "batch_size"),
    "num_epochs": ("1d", "train", "params", "num_epochs"),
    "learning_rate": ("1d", "train", "params", "learning_rate"),
    "lr_scheduler": ("1d", "train", "params", "lr_scheduler"),
    "lr_warmup_fraction": ("1d", "train", "params", "lr_warmup_fraction"),
    "lr_min_factor": ("1d", "train", "params", "lr_min_factor"),
    "feature_variant": ("1d", "train", "params", "feature_variant"),
    "feature_normalization": ("1d", "train", "params", "feature_normalization"),
    "material_feature_normalization": (
        "1d",
        "train",
        "params",
        "material_feature_normalization",
    ),
    "feature_log_scale": ("1d", "train", "params", "feature_log_scale"),
    "feature_log_clip": ("1d", "train", "params", "feature_log_clip"),
    "feature_eps": ("1d", "train", "params", "feature_eps"),
    "include_material_scale_features": (
        "1d",
        "train",
        "params",
        "include_material_scale_features",
    ),
    "include_material_ratios": ("1d", "train", "params", "include_material_ratios"),
    "aux_moment_loss_weight": ("1d", "train", "params", "aux_moment_loss_weight"),
    "training_data": ("1d", "train", "params", "training_data"),
    "training_time_horizons": ("1d", "train", "params", "training_time_horizons"),
    "relative_loss_eps": ("1d", "train", "params", "relative_loss_eps"),
}


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def set_path(config, path, value):
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def parse_value(raw):
    value = yaml.safe_load(raw)
    return raw if value is None and str(raw).lower() == "none" else value


def parse_sweep_args(items):
    overrides = {}
    i = 0
    while i < len(items):
        token = items[i]
        if token == "--":
            i += 1
            continue
        if not token.startswith("--"):
            raise ValueError(f"Unexpected sweep argument {token!r}")
        item = token[2:]
        if "=" in item:
            key, raw_value = item.split("=", 1)
        else:
            i += 1
            if i >= len(items):
                raise ValueError(f"Missing value for sweep argument {token!r}")
            key, raw_value = item, items[i]
        overrides[key.replace("-", "_")] = parse_value(raw_value)
        i += 1
    return overrides


def parse_horizons(value):
    if isinstance(value, str):
        return [float(item) for item in value.split(",") if item.strip()]
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def apply_sweep_overrides(config, overrides):
    for key, value in overrides.items():
        if key == "feature_log_clip_max":
            set_path(config, PARAM_PATHS["feature_log_clip"], [0.0, float(value)])
            continue
        if key == "training_time_horizons":
            value = parse_horizons(value)
        if key not in PARAM_PATHS:
            raise ValueError(
                f"Unsupported sweep parameter {key!r}. "
                f"Supported parameters: {', '.join(sorted(set(PARAM_PATHS) | {'feature_log_clip_max'}))}"
            )
        set_path(config, PARAM_PATHS[key], value)


def configure_1d_nn_run(config, N, model_tag):
    workflow = config.setdefault("workflow", {})
    workflow.update(
        {
            "target": "1d",
            "ansatz": "nn",
            "phase": "train",
            "train": True,
            "simulate": False,
        }
    )
    wandb_cfg = workflow.setdefault("wandb", {})
    wandb_cfg["enabled"] = True
    wandb_cfg.setdefault("project", "NN_FPN")
    wandb_cfg.setdefault("mode", os.environ.get("WANDB_MODE", "online"))
    tags = list(wandb_cfg.get("tags") or [])
    if f"N{N}" not in tags:
        tags.append(f"N{N}")
    if "sweep" not in tags:
        tags.append("sweep")
    wandb_cfg["tags"] = tags
    wandb_cfg["name_template"] = "1d-nn-sweep-N{N}-{model_tag}"

    dim_cfg = config.setdefault("1d", {})
    dim_cfg["enabled"] = True
    dim_cfg.setdefault("filter_types", {})["nn"] = 1
    train_cfg = dim_cfg.setdefault("train", {})
    train_cfg["Ns"] = [int(N)]
    train_cfg["num_replicates"] = 1
    params = train_cfg.setdefault("params", {})
    params["model_tag"] = model_tag
    params.setdefault("training_data", "augmented")
    params.setdefault("obj_idx", 0)
    params.setdefault("device", "auto")


def write_temp_config(config):
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".yaml", prefix="nn_fpn_1d_sweep_", delete=False
    )
    with handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return Path(handle.name)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one 1D neural-network sweep trial through the main YAML workflow."
    )
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--N", type=int, required=True, choices=[3, 7, 9])
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = parse_sweep_args(unknown)

    config = load_yaml(args.base_config)
    model_tag = args.model_tag or f"sweep_N{args.N}_{uuid.uuid4().hex[:8]}"
    apply_sweep_overrides(config, overrides)
    configure_1d_nn_run(config, args.N, model_tag)

    config_path = write_temp_config(config)
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "-c",
        str(config_path),
        "--target",
        "1d",
        "--ansatz",
        "nn",
        "--phase",
        "train",
    ]

    if args.dry_run:
        print("Generated config:", config_path)
        print("Command:", " ".join(command))
        print(yaml.safe_dump(config, sort_keys=False))
        return 0

    try:
        subprocess.run(command, cwd=str(ROOT), check=True)
    finally:
        try:
            config_path.unlink()
        except FileNotFoundError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

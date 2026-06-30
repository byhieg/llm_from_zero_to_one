from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from trainer.pretrain.pretrain_args import PreTrainArgs


DEFAULT_CONFIG_PATH = Path("configs/pretrain.yaml")


def _substitute_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            expr = match.group(1)
            if ":-" in expr:
                var_name, default = expr.split(":-", 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(expr, match.group(0))

        return re.sub(pattern, replace_var, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    else:
        return value


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path:
        path = Path(config_path)
    else:
        path = DEFAULT_CONFIG_PATH

    if not path.exists():
        if config_path:
            raise FileNotFoundError(f"Config file not found: {path}")
        else:
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Tip: Use --config to specify a config file, or create {path}"
            )

    return path


def _load_yaml_data(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return _substitute_env_vars(yaml.safe_load(f) or {})


def _detect_mode(data: dict[str, Any]) -> str | None:
    yaml_mode = data.get("mode")
    if yaml_mode is None:
        yaml_mode = data.get("experiment", {}).get("mode")
    return yaml_mode


def detect_mode_from_yaml(config_path: str | Path) -> str:
    """只从 YAML 中识别 mode，并限制为 pretrain。"""

    data = _load_yaml_data(config_path)
    mode = _detect_mode(data)
    if mode is None:
        raise ValueError(
            "Training mode not specified. Add 'mode: pretrain' or "
            "'experiment.mode: pretrain' to your YAML."
        )
    if mode != "pretrain":
        raise ValueError(f"Unsupported mode: {mode}. Only 'pretrain' is supported.")
    return mode


def load_pretrain_args_from_yaml(config_path: str | Path | None = None) -> PreTrainArgs:
    """从 YAML 加载 PreTrainArgs。"""

    path = _resolve_config_path(config_path)
    mode = detect_mode_from_yaml(path)
    data = _load_yaml_data(path)
    args = PreTrainArgs.from_dict(data)
    args.set_mode(mode)
    return args

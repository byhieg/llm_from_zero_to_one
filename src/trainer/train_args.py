from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Type

import yaml

from trainer.common_args import TrainingArgs


_ARGS_REGISTRY: dict[str, Type[TrainingArgs]] = {}

DEFAULT_CONFIG_DIR = Path("configs")


def register_args(name: str, args_cls: Type[TrainingArgs]) -> None:
    _ARGS_REGISTRY[name] = args_cls


def get_args_class(mode: str) -> Type[TrainingArgs]:
    if mode not in _ARGS_REGISTRY:
        available = ", ".join(_ARGS_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown mode '{mode}'. Available: {available}")
    return _ARGS_REGISTRY[mode]


def list_modes() -> list[str]:
    return list(_ARGS_REGISTRY.keys())


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


def _resolve_config_path(mode: str, config_path: str | Path | None = None) -> Path:
    if config_path:
        path = Path(config_path)
    else:
        path = DEFAULT_CONFIG_DIR / f"{mode}.yaml"

    if not path.exists():
        if config_path:
            raise FileNotFoundError(f"Config file not found: {path}")
        else:
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Tip: Use --config to specify a config file, or create {path}"
            )

    return path


def load_args_from_yaml(
    mode: str | None = None,
    config_path: str | Path | None = None,
    validate: bool = True,
) -> tuple[TrainingArgs, str]:
    """
    从 YAML 配置文件加载参数

    Args:
        mode: 训练模式 (如 "pretrain")，如果 YAML 中有 mode 字段则可省略
        config_path: YAML 配置文件路径（可选，默认为 configs/{mode}.yaml）
        validate: 是否验证配置

    Returns:
        (参数实例, 实际使用的模式)

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置验证失败或模式未指定
    """
    if config_path is None and mode is None:
        raise ValueError(
            "Either 'mode' or 'config_path' must be specified.\n"
            "Usage: --config path/to/config.yaml OR --mode pretrain"
        )

    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        path = _resolve_config_path(mode, config_path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data = _substitute_env_vars(data or {})

    yaml_mode = data.pop("mode", None)
    if yaml_mode is None:
        yaml_mode = data.get("experiment", {}).get("mode")

    final_mode = mode or yaml_mode
    if final_mode is None:
        raise ValueError(
            "Training mode not specified. Add 'mode: pretrain' to your YAML "
            "or use --mode pretrain"
        )

    args_cls = get_args_class(final_mode)
    args = args_cls.from_dict(data)
    args.set_mode(final_mode)

    if validate:
        errors = args.validate()
        if errors:
            error_msg = "\n  - ".join(["Config validation failed:"] + errors)
            raise ValueError(error_msg)

    return args, final_mode


def generate_default_config(mode: str, output_path: str | Path | None = None) -> Path:
    args_cls = get_args_class(mode)

    if output_path:
        out_path = Path(output_path)
    else:
        out_path = DEFAULT_CONFIG_DIR / f"{mode}.yaml"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    default_args = args_cls()
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(
            default_args.to_config_dict(mode=mode),
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    return out_path

from __future__ import annotations

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, MISSING, asdict
from pathlib import Path
from typing import Optional, Any

import yaml


@dataclass
class TrainingConfig:
    batch_size: int = 16
    seq_len: int = 1024
    epoch_num: int = 1
    learning_rate: float = 3e-4
    warmup_steps: int = 10
    grad_clip: float = 1.0
    accumulation_steps: int = 4
    log_steps: int = 10


@dataclass
class CheckpointConfig:
    save_steps: int = 1000
    checkpoint_dir: str = ""
    resume_from_checkpoint: Optional[str] = None


@dataclass
class DataConfig:
    data_strategy: str = "padding"
    data_path: str = ""
    dataset_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    steps: int = 500
    tokens: int = 100
    topk: int = 100
    temperature: float = 1.0
    prompt: str = "The meaning of life is"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "gpt2"  # 模型名称
    config: dict[str, Any] = field(default_factory=dict)  # 模型特定配置


# ---------------------------------------------------------------------------
# 主配置 dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingArgs:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = field(default="cpu", init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingArgs":
        init_kwargs = {}

        for f in fields(cls):
            if not f.init:
                continue

            field_name = f.name
            field_type = f.type

            if field_name not in data:
                continue

            value = data[field_name]

            field_type_str = (
                str(field_type).replace("typing.", "").replace(" ", "").strip("<>")
            )
            if field_type_str in (
                "TrainingConfig",
                "CheckpointConfig",
                "DataConfig",
                "InferenceConfig",
                "ModelConfig",
            ):
                field_class = eval(field_type_str)
                if isinstance(value, dict):
                    init_kwargs[field_name] = field_class(**value)
                else:
                    init_kwargs[field_name] = value
            else:
                init_kwargs[field_name] = value

        return cls(**init_kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingArgs":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def validate(self) -> list[str]:
        errors = []

        if self.training.epoch_num > 0 and not self.data.data_path:
            errors.append("data.data_path is required when training.epoch_num > 0")

        if self.data.data_strategy not in ('padding', 'megatron'):
            errors.append(
                f"data.data_strategy must be 'padding' or 'megatron', got '{self.data.data_strategy}'"
            )

        if self.data.data_strategy == 'megatron':
            if 'total_token' not in self.data.dataset_config:
                errors.append(
                    "data.dataset_config.total_token is required when using 'megatron' strategy"
                )

        if self.training.batch_size <= 0:
            errors.append(
                f"training.batch_size must be positive, got {self.training.batch_size}"
            )
        if self.training.seq_len <= 0:
            errors.append(
                f"training.seq_len must be positive, got {self.training.seq_len}"
            )
        if self.training.learning_rate <= 0:
            errors.append(
                f"training.learning_rate must be positive, got {self.training.learning_rate}"
            )
        if self.training.warmup_steps < 0:
            errors.append(
                f"training.warmup_steps must be non-negative, got {self.training.warmup_steps}"
            )
        if self.training.grad_clip <= 0:
            errors.append(
                f"training.grad_clip must be positive, got {self.training.grad_clip}"
            )
        if self.training.accumulation_steps <= 0:
            errors.append(
                f"training.accumulation_steps must be positive, got {self.training.accumulation_steps}"
            )

        return errors


@dataclass
class PretrainArgs(TrainingArgs):
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(checkpoint_dir="checkpoints/pretrain")
    )
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> list[str]:
        errors = super().validate()

        # 验证 inference 参数
        if self.inference.steps <= 0:
            errors.append(
                f"inference.steps must be positive, got {self.inference.steps}"
            )
        if self.inference.tokens <= 0:
            errors.append(
                f"inference.tokens must be positive, got {self.inference.tokens}"
            )
        if self.inference.topk <= 0:
            errors.append(f"inference.topk must be positive, got {self.inference.topk}")
        if self.inference.temperature <= 0:
            errors.append(
                f"inference.temperature must be positive, got {self.inference.temperature}"
            )

        return errors


# ---------------------------------------------------------------------------
# 注册表 + 工厂
# ---------------------------------------------------------------------------
_ARGS_REGISTRY: dict[str, type[TrainingArgs]] = {}

# 默认配置目录
DEFAULT_CONFIG_DIR = Path("configs")


def register_args(name: str, args_cls: type[TrainingArgs]) -> None:
    _ARGS_REGISTRY[name] = args_cls


def get_args_class(mode: str) -> type[TrainingArgs]:
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
        path = DEFAULT_config_dir / f"{mode}.yaml"

    if not path.exists():
        if config_path:
            raise FileNotFoundError(f"Config file not found: {path}")
        else:
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Tip: Use --config to specify a config file, or create {path}"
            )

    return path
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
    # 如果没有指定 config_path，必须有 mode
    if config_path is None and mode is None:
        raise ValueError(
            "Either 'mode' or 'config_path' must be specified.\n"
            "Usage: --config path/to/config.yaml OR --mode pretrain"
        )

    # 解析配置文件路径
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        path = _resolve_config_path(mode, config_path)

    # 加载 YAML
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 替换环境变量
    data = _substitute_env_vars(data or {})

    # 从 YAML 中获取模式（如果有的话）
    yaml_mode = data.pop("mode", None)

    # 确定最终使用的模式
    final_mode = mode or yaml_mode
    if final_mode is None:
        raise ValueError(
            "Training mode not specified. Add 'mode: pretrain' to your YAML "
            "or use --mode pretrain"
        )

    args_cls = get_args_class(final_mode)
    args = args_cls.from_dict(data)

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
    data = default_args.to_dict()
    data["mode"] = mode
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"mode: {mode}\n\n")
        other_data = {k: v for k, v in data.items() if k != "mode"}
        yaml.dump(
            other_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
    return out_path


# 注册内置模式
register_args("pretrain", PretrainArgs)

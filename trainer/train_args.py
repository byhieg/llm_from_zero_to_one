from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, MISSING, asdict
from pathlib import Path
from typing import Optional, Any

import yaml


@dataclass
class TrainingArgs:
    batch_size: int = 16
    seq_len: int = 1024
    epoch_num: int = 1
    data_path: str = ""
    learning_rate: float = 3e-4
    warmup_steps: int = 10
    grad_clip: float = 1.0
    accumulation_steps: int = 4
    log_steps: int = 10
    save_steps: int = 1000
    checkpoint_dir: str = ""
    resume_from_checkpoint: Optional[str] = None

    # 运行时由 train.py 注入，不在 CLI 中暴露
    device: str = field(default="cpu", init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """保存到 YAML 文件"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingArgs":
        """从字典创建实例"""
        # 只保留该类定义的字段
        valid_fields = {f.name for f in fields(cls) if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingArgs":
        """从 YAML 文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def validate(self) -> list[str]:
        """
        验证配置参数
        
        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors = []
        
        # 检查 data_path（如果 epoch_num > 0，则必须提供）
        if self.epoch_num > 0 and not self.data_path:
            errors.append("data_path is required when epoch_num > 0")
        
        # 检查数值范围
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.seq_len <= 0:
            errors.append(f"seq_len must be positive, got {self.seq_len}")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        if self.warmup_steps < 0:
            errors.append(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.grad_clip <= 0:
            errors.append(f"grad_clip must be positive, got {self.grad_clip}")
        if self.accumulation_steps <= 0:
            errors.append(f"accumulation_steps must be positive, got {self.accumulation_steps}")
        
        return errors


@dataclass
class PretrainArgs(TrainingArgs):
    data_path: str = ""
    checkpoint_dir: str = "checkpoints/pretrain"
    inference_steps: int = 500
    inference_tokens: int = 100
    inference_topk: int = 100
    inference_temperature: float = 1.0
    inference_prompt: str = "The meaning of life is"

    def validate(self) -> list[str]:
        """验证 Pretrain 特有参数"""
        errors = super().validate()
        
        if self.inference_steps <= 0:
            errors.append(f"inference_steps must be positive, got {self.inference_steps}")
        if self.inference_tokens <= 0:
            errors.append(f"inference_tokens must be positive, got {self.inference_tokens}")
        if self.inference_topk <= 0:
            errors.append(f"inference_topk must be positive, got {self.inference_topk}")
        if self.inference_temperature <= 0:
            errors.append(f"inference_temperature must be positive, got {self.inference_temperature}")
        
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
    """
    递归替换值中的环境变量
    
    支持 ${VAR_NAME} 和 ${VAR_NAME:-default} 格式
    """
    if isinstance(value, str):
        # 匹配 ${VAR_NAME} 或 ${VAR_NAME:-default}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            expr = match.group(1)
            if ':-' in expr:
                var_name, default = expr.split(':-', 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(expr, match.group(0))  # 如果不存在，保留原值
        
        return re.sub(pattern, replace_var, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    else:
        return value


def _resolve_config_path(mode: str, config_path: str | Path | None = None) -> Path:
    """
    解析配置文件路径
    
    如果未指定 config_path，自动查找 configs/{mode}.yaml
    
    Args:
        mode: 训练模式
        config_path: 指定的配置文件路径（可选）
        
    Returns:
        配置文件的完整路径
        
    Raises:
        FileNotFoundError: 配置文件不存在
    """
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
    validate: bool = True
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
    """
    生成默认配置文件
    
    Args:
        mode: 训练模式
        output_path: 输出路径（可选，默认为 configs/{mode}.yaml）
        
    Returns:
        生成的配置文件路径
    """
    args_cls = get_args_class(mode)
    
    if output_path:
        out_path = Path(output_path)
    else:
        out_path = DEFAULT_CONFIG_DIR / f"{mode}.yaml"
    
    # 确保目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建默认实例
    default_args = args_cls()
    
    # 转换为字典并添加 mode 字段
    data = default_args.to_dict()
    data["mode"] = mode
    
    # 保存，将 mode 放在最前面
    with open(out_path, "w", encoding="utf-8") as f:
        # 先写 mode
        f.write(f"mode: {mode}\n\n")
        # 再写其他字段
        other_data = {k: v for k, v in data.items() if k != "mode"}
        yaml.dump(other_data, f, default_flow_style=False, allow_unicode=True)
    
    return out_path


# 注册内置模式
register_args("pretrain", PretrainArgs)

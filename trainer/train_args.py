from __future__ import annotations

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


@dataclass
class PretrainArgs(TrainingArgs):
    data_path: str = ""
    checkpoint_dir: str = "checkpoints/pretrain"
    inference_steps: int = 500
    inference_tokens: int = 100
    inference_topk: int = 100
    inference_temperature: float = 1.0
    inference_prompt: str = "The meaning of life is"


# ---------------------------------------------------------------------------
# 注册表 + 工厂
# ---------------------------------------------------------------------------
_ARGS_REGISTRY: dict[str, type[TrainingArgs]] = {}


def register_args(name: str, args_cls: type[TrainingArgs]) -> None:
    _ARGS_REGISTRY[name] = args_cls


def get_args_class(mode: str) -> type[TrainingArgs]:
    if mode not in _ARGS_REGISTRY:
        available = ", ".join(_ARGS_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown mode '{mode}'. Available: {available}")
    return _ARGS_REGISTRY[mode]


def list_modes() -> list[str]:
    return list(_ARGS_REGISTRY.keys())


def load_args_from_yaml(mode: str, config_path: str | Path) -> TrainingArgs:
    """
    从 YAML 配置文件加载参数
    
    Args:
        mode: 训练模式 (如 "pretrain")
        config_path: YAML 配置文件路径
        
    Returns:
        对应模式的参数实例
    """
    args_cls = get_args_class(mode)
    return args_cls.from_yaml(config_path)


# 注册内置模式
register_args("pretrain", PretrainArgs)

"""模型注册表

所有模型必须在代码中显式注册，不支持从配置文件动态加载。

使用方式：
    >>> # 1. 注册模型（在代码中）
    >>> from models import BaseModel, register_model, register_config
    >>> 
    >>> @register_config("gpt2")
    >>> class GPT2Config(BaseModelConfig):
    >>>     n_layer: int = 12
    >>> 
    >>> @register_model("gpt2")
    >>> class GPT2(BaseModel):
    >>>     def __init__(self, config: GPT2Config):
    >>>         ...
    >>> 
    >>> # 2. 创建模型实例
    >>> from trainer.model_factory import create_model
    >>> model = create_model("gpt2", n_layer=12)
"""
from models.base import BaseModel, BaseModelConfig
from models.registry import (
    register_model,
    register_config,
    get_model_class,
    get_config_class,
    list_models,
    list_configs,
    is_registered,
)

# 自动导入所有模型，触发注册
import models.gpt2  # noqa: F401

__all__ = [
    "BaseModel",
    "BaseModelConfig",
    "register_model",
    "register_config",
    "get_model_class",
    "get_config_class",
    "list_models",
    "list_configs",
    "is_registered",
]

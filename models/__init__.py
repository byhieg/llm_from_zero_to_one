"""模型注册表和工厂"""
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

"""模型注册表"""
from __future__ import annotations

from typing import Dict, Type

from models.base import BaseModel, BaseModelConfig


# 全局注册表
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}
_CONFIG_REGISTRY: Dict[str, Type[BaseModelConfig]] = {}


def register_model(name: str):
    """
    装饰器：注册模型类
    
    Usage:
        @register_model("gpt2")
        class GPT2(BaseModel):
            ...
    """
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        if not issubclass(cls, BaseModel):
            raise TypeError(f"{cls.__name__} must be a subclass of BaseModel")
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_config(name: str):
    """
    装饰器：注册配置类
    
    Usage:
        @register_config("gpt2")
        class GPT2Config(BaseModelConfig):
            ...
    """
    def decorator(cls: Type[BaseModelConfig]) -> Type[BaseModelConfig]:
        if not issubclass(cls, BaseModelConfig):
            raise TypeError(f"{cls.__name__} must be a subclass of BaseModelConfig")
        if name in _CONFIG_REGISTRY:
            raise ValueError(f"Config '{name}' already registered")
        _CONFIG_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> Type[BaseModel]:
    """
    获取模型类
    
    Args:
        name: 模型名称（如 "gpt2"）
        
    Returns:
        模型类
        
    Raises:
        ValueError: 模型未注册
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name]


def get_config_class(name: str) -> Type[BaseModelConfig]:
    """
    获取配置类
    
    Args:
        name: 配置名称（如 "gpt2"）
        
    Returns:
        配置类
        
    Raises:
        ValueError: 配置未注册
    """
    if name not in _CONFIG_REGISTRY:
        available = ", ".join(_CONFIG_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return _CONFIG_REGISTRY[name]


def list_models() -> list[str]:
    """列出所有注册的模型"""
    return list(_MODEL_REGISTRY.keys())


def list_configs() -> list[str]:
    """列出所有注册的配置"""
    return list(_CONFIG_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """检查模型是否已注册"""
    return name in _MODEL_REGISTRY

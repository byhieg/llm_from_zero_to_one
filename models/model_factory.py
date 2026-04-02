"""模型工厂"""
from typing import Any

from models import get_model_class, get_config_class
from models.base import BaseModel, BaseModelConfig


def create_model(
    model_name: str,
    config_dict: dict[str, Any] | None = None,
    **kwargs
) -> BaseModel:
    """
    创建模型实例
    
    所有模型必须先在代码中通过 @register_model 和 @register_config 装饰器注册。
    
    Args:
        model_name: 已注册的模型名称（如 "gpt2"）
        config_dict: 配置字典（可选）
        **kwargs: 配置参数（覆盖 config_dict）
        
    Returns:
        模型实例
        
    Example:
        >>> # 先在代码中注册模型
        >>> @register_config("gpt2")
        >>> class GPT2Config(BaseModelConfig):
        >>>     ...
        >>> 
        >>> @register_model("gpt2")
        >>> class GPT2(BaseModel):
        >>>     ...
        >>> 
        >>> # 然后创建模型
        >>> model = create_model("gpt2", n_layer=12, n_head=12)
        >>> # 或
        >>> model = create_model("gpt2", {"n_layer": 12, "n_head": 12})
    """
    # 合并配置
    config_data = {**(config_dict or {}), **kwargs}
    
    # 获取配置类和模型类
    config_cls = get_config_class(model_name)
    model_cls = get_model_class(model_name)
    
    # 创建配置
    config = config_cls(**config_data)
    
    # 创建模型
    model = model_cls(config)
    
    return model


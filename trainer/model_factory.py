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
    
    Args:
        model_name: 模型名称（如 "gpt2"）
        config_dict: 配置字典
        **kwargs: 其他参数（覆盖 config_dict）
        
    Returns:
        模型实例
        
    Example:
        >>> model = create_model("gpt2", {"n_layer": 12, "n_head": 12})
        >>> model = create_model("gpt2", n_layer=12, n_head=12)
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


def create_model_from_yaml(
    model_name: str,
    yaml_path: str
) -> BaseModel:
    """
    从 YAML 文件创建模型
    
    Args:
        model_name: 模型名称
        yaml_path: YAML 文件路径
        
    Returns:
        模型实例
    """
    import yaml
    from pathlib import Path
    
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 如果 YAML 中有 model 字段，从中提取
    if 'model' in data:
        model_data = data['model']
        if isinstance(model_data, dict) and 'config' in model_data:
            config_dict = model_data['config']
            model_name = model_data.get('name', model_name)
        else:
            config_dict = model_data
    else:
        config_dict = data
    
    return create_model(model_name, config_dict)

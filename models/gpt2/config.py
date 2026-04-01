"""GPT2 模型配置"""
from dataclasses import dataclass

from models.base import BaseModelConfig
from models.registry import register_config


@dataclass
@register_config("gpt2")
class GPT2Config(BaseModelConfig):
    """GPT2 配置
    
    Args:
        block_size: 最大序列长度
        vocab_size: 词汇表大小
        n_layer: Transformer 层数
        n_head: 注意力头数
        n_embd: 嵌入维度
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

"""数据加载策略工厂"""
from typing import Any
from torch.utils.data import Dataset


def create_dataset(
    data_strategy: str,
    tokenizer: Any = None,
    args: Any = None,
    **kwargs
) -> Dataset:
    """
    创建数据集实例
    
    对于预训练任务，支持两种固定的数据加载策略：
    - 'padding': 边训练边分词策略（PretrainPaddingDataset）
    - 'megatron': 预先分词策略（SimpleMegatronDataset）
    
    Args:
        data_strategy: 数据加载策略名称，'padding' 或 'megatron'
        tokenizer: 分词器实例（padding 策略需要）
        args: 训练参数
        **kwargs: 其他参数
        
    Returns:
        数据集实例
        
    Raises:
        ValueError: 未知的策略名称或缺少必需参数
        
    Example:
        >>> # 使用边训练边分词策略
        >>> dataset = create_dataset('padding', tokenizer=tokenizer, args=args)
        >>> 
        >>> # 使用预先分词策略
        >>> dataset = create_dataset('megatron', args=args)
    """
    if data_strategy == 'padding':
        return _create_padding_dataset(tokenizer, args, **kwargs)
    elif data_strategy == 'megatron':
        return _create_megatron_dataset(args, **kwargs)
    else:
        raise ValueError(
            f"未知的数据加载策略: '{data_strategy}'。"
            f"可用的策略: 'padding'（边训练边分词）或 'megatron'（预先分词）"
        )


def _create_padding_dataset(tokenizer: Any, args: Any, **kwargs) -> Dataset:
    """
    创建边训练边分词策略的数据集
    
    Args:
        tokenizer: 分词器实例
        args: 训练参数
        **kwargs: 其他参数
        
    Returns:
        PretrainPaddingDataset 实例
    """
    from datasets import load_dataset
    from .padding_dataset import PretrainPaddingDataset
    
    if tokenizer is None:
        raise ValueError("边训练边分词策略需要提供 tokenizer 实例")
    
    if args is None or not hasattr(args, 'data'):
        raise ValueError("边训练边分词策略需要提供有效的 args 参数")
    
    dataset_config = args.data.dataset_config or {}
    
    hf_dataset = load_dataset(
        args.data.data_path,
        split=dataset_config.get('split', 'train'),
        **kwargs
    )
    
    return PretrainPaddingDataset(
        tokenizer=tokenizer,
        max_seq=dataset_config.get('max_seq', args.training.seq_len),
        dataset=hf_dataset
    )


def _create_megatron_dataset(args: Any, **kwargs) -> Dataset:
    """
    创建预先分词策略的数据集
    
    Args:
        args: 训练参数
        **kwargs: 其他参数
        
    Returns:
        SimpleMegatronDataset 实例
    """
    from .simple_megatron_dataset.simple_dataset import SimpleMegatronDataset
    
    if args is None or not hasattr(args, 'data'):
        raise ValueError("预先分词策略需要提供有效的 args 参数")
    
    dataset_config = args.data.dataset_config or {}
    
    if 'total_token' not in dataset_config:
        raise ValueError(
            "预先分词策略需要在 data.dataset_config 中提供 'total_token' 参数"
        )
    
    data_path = args.data.data_path
    
    return SimpleMegatronDataset(
        idx_path=f"{data_path}.idx",
        bin_path=f"{data_path}.bin",
        seq_len=dataset_config.get('seq_len', args.training.seq_len),
        total_token=dataset_config['total_token'],
        seed=dataset_config.get('seed', 42)
    )

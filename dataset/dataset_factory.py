"""数据加载策略工厂"""
from typing import Any
from torch.utils.data import Dataset


def create_dataset(
    data_strategy: str,
    dataset_config: dict[str,Any] = None,
    **kwargs
) -> Dataset:
    """
    创建数据集实例

    对于预训练任务，支持两种固定的数据加载策略：
    - 'padding': 边训练边分词策略（PretrainPaddingDataset）
    - 'megatron': 预先分词策略（SimpleMegatronDataset）

    Args:
        data_strategy: 数据加载策略名称，'padding' 或 'megatron'
        dataset_config: 数据集参数字典，包含数据路径、分块、序列长度、分词器配置等参数
        **kwargs: 其他参数

    Returns:
        数据集实例

    Raises:
        ValueError: 未知的策略名称或缺少必需参数

    Example:
        >>> # 使用边训练边分词策略
        >>> dataset = create_dataset('padding', dataset_config=dataset_config)

        >>> # 使用预先分词策略
        >>> dataset = create_dataset('megatron', dataset_config=dataset_config)
    """
    if data_strategy == 'padding':
        return _create_padding_dataset(dataset_config, **kwargs)
    elif data_strategy == 'megatron':
        return _create_megatron_dataset(dataset_config, **kwargs)
    else:
        raise ValueError(
            f"未知的数据加载策略: '{data_strategy}'。"
            f"可用的策略: 'padding'（边训练边分词）或 'megatron'（预先分词）"
        )


def _create_padding_dataset(dataset_config: dict[str,Any], **kwargs) -> Dataset:
    """
    创建边训练边分词策略的数据集

    Args:
        dataset_config: 数据集参数字典，包含数据路径、分块、序列长度、分词器配置等参数
        **kwargs: 其他参数

    Returns:
        PretrainPaddingDataset 实例
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from .padding_dataset import PretrainPaddingDataset

    tokenizer_path = dataset_config.get('tokenizer_path')
    if not tokenizer_path:
        raise ValueError("padding 策略需要在 dataset_config 中提供 'tokenizer_path' 参数")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    data_path = dataset_config['data_path']
    data_files = dataset_config.get('data_files')

    if data_files:
        hf_dataset = load_dataset(
            data_path,
            data_files=data_files,
            split=dataset_config.get('split', 'train'),
            **kwargs
        )
    else:
        hf_dataset = load_dataset(
            data_path,
            split=dataset_config.get('split', 'train'),
            **kwargs
        )

    return PretrainPaddingDataset(
        tokenizer=tokenizer,
        max_seq=dataset_config.get('seq_len', 1024),
        dataset=hf_dataset,
        dataset_config=dataset_config
    )


def _create_megatron_dataset(dataset_config: dict[str,Any], **kwargs) -> Dataset:
    """
    创建预先分词策略的数据集

    Args:
        dataset_config: 数据集参数字典，包含数据路径、分块、最大序列长度等参数
        **kwargs: 其他参数

    Returns:
        SimpleMegatronDataset 实例
    """
    from .simple_megatron_dataset.simple_dataset import SimpleMegatronDataset

    if 'total_token' not in dataset_config:
        raise ValueError(
            "预先分词策略需要在 data.dataset_config 中提供 'total_token' 参数"
        )

    data_path = dataset_config['data_path']

    return SimpleMegatronDataset(
        idx_path=f"{data_path}.idx",
        bin_path=f"{data_path}.bin",
        seq_len=dataset_config.get('seq_len', 1024),
        total_token=dataset_config['total_token'],
        seed=dataset_config.get('seed', 42)
    )

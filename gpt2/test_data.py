import pytest
import torch
from pathlib import Path
from gpt2.data import ShardIndexDataset


# 使用真实数据路径
DATA_PATH = Path(__file__).parent / "data"


@pytest.fixture
def dataset():
    """使用真实数据创建 dataset"""
    ds = ShardIndexDataset(str(DATA_PATH), seq_len=1024)
    return ds


class TestShardIndexDataset:
    """ShardIndexDataset 的完整测试套件"""

    def test_init(self, dataset):
        """测试初始化"""
        assert dataset.seq_len == 1024
        assert dataset.length > 0
        assert dataset.total_shard > 0

    def test_len(self, dataset):
        ds = dataset
        length = len(ds)
        assert length > 0
        assert length == ds.length

    def test_getitem_first_sample(self, dataset):
        ds = dataset
        input_ids, labels = ds[1]  # 注意：_get_shard_by_index 要求 index > 0
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert input_ids.shape == (1024,)
        assert labels.shape == (1024,)

    def test_getitem_middle_sample(self, dataset):
        ds = dataset
        mid_idx = len(ds) // 2
        input_ids, labels = ds[mid_idx]
        assert input_ids.shape == (1024,)
        assert labels.shape == (1024,)

    def test_getitem_return_type(self, dataset):
        ds = dataset
        input_ids, labels = ds[1]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_getitem_shape(self, dataset):
        ds = dataset
        input_ids, labels = ds[1]
        assert input_ids.shape == (1024,)
        assert labels.shape == (1024,)

    def test_input_label_offset(self, dataset):
        ds = dataset
        input_ids, labels = ds[1]
        for i in range(min(10, len(input_ids) - 1)):
            assert labels[i] == input_ids[i + 1], (
                f"labels[{i}] != input_ids[{i + 1}]: {labels[i]} != {input_ids[i + 1]}"
            )

    def test_getitem_dtype(self, dataset):
        ds = dataset
        input_ids, labels = ds[1]
        assert input_ids.dtype in [torch.int32, torch.int64]
        assert labels.dtype in [torch.int32, torch.int64]

    def test_index_zero_raises_error(self, dataset):
        ds = dataset
        with pytest.raises(AssertionError):
            _ = ds[0]

    def test_index_negative_raises_error(self, dataset):
        ds = dataset
        with pytest.raises(AssertionError):
            _ = ds[-1]

    def test_index_out_of_range(self, dataset):
        ds = dataset
        with pytest.raises(AssertionError):
            _ = ds[len(ds) + 1]

    def test_get_shard_by_index_valid(self, dataset):
        ds = dataset
        shard_id, local_index = ds._get_shard_by_index(1)
        assert isinstance(shard_id, str)
        assert isinstance(local_index, int)
        assert local_index >= 0

    def test_multiple_samples(self, dataset):
        ds = dataset
        for idx in [1, 10, 100, 1000]:
            if idx < len(ds):
                input_ids, labels = ds[idx]
                assert input_ids.shape == (1024,)
                assert labels.shape == (1024,)

    def test_cross_shard_reading(self, dataset):
        ds = dataset
        if len(ds) > 10000:
            for idx in [9995, 9996, 9997]:
                if idx < len(ds):
                    input_ids, labels = ds[idx]
                    assert input_ids.shape == (1024,)
                    assert labels.shape == (1024,)

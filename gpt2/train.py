from dataclasses import dataclass
import torch
import random
from typing import Iterator
from torch.utils.data import DataLoader, Sampler
from gpt2.data import ShardIndexDataset


@dataclass
class ModelConfig:
    total_tokens: int = 320 * 1024
    batch_tokens: int = 16 * 1024
    batch_size: int = 16
    seq_len: int = 1024
    epoch_num: int = 3


class RandomStartSampler(Sampler):
    """
    自定义采样器：每个 epoch 开始时随机指定一个 start index
    然后从该位置开始顺序采样

    适用于语言模型训练，增加数据随机性
    """

    def __init__(self, data_source, batch_size: int, drop_last: bool = True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(data_source)

        # 计算每个 epoch 可以产生的样本数
        if self.drop_last:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (
                self.num_samples + self.batch_size - 1
            ) // self.batch_size

        # 随机起始索引（会在每个 epoch 重新生成）
        self.start_index = 0

    def set_epoch(self, epoch: int):
        """
        设置当前 epoch，并随机生成起始索引
        这个方法会在每个 epoch 开始时被调用
        """
        # 随机选择一个起始位置（0 到 num_samples - 1）
        self.start_index = random.randint(0, self.num_samples - 1)
        print(f"  [Sampler] Epoch {epoch}: 随机起始索引 = {self.start_index}")

    def __iter__(self) -> Iterator[int]:
        """
        生成索引迭代器
        从 start_index 开始，循环遍历整个数据集
        """
        indices = list(range(self.num_samples))

        # 从 start_index 开始重新排列
        # 例如：start_index=500, num_samples=1000
        # 结果：[500, 501, ..., 999, 0, 1, ..., 499]
        rotated_indices = indices[self.start_index :] + indices[: self.start_index]

        # 如果 drop_last，只保留完整的 batch
        if self.drop_last:
            total_samples = self.num_batches * self.batch_size
            rotated_indices = rotated_indices[:total_samples]

        yield from rotated_indices

    def __len__(self) -> int:
        """返回样本总数"""
        if self.drop_last:
            return self.num_batches * self.batch_size
        return self.num_samples


if __name__ == "__main__":
    gpt2_config = ModelConfig()
    ds = ShardIndexDataset("gpt2/data", seq_len=gpt2_config.seq_len)
    # 使用自定义 RandomStartSampler 替代 shuffle，以实现每个 epoch 的随机起始
    sampler = RandomStartSampler(
        data_source=ds,
        batch_size=gpt2_config.batch_size,
        drop_last=True,
    )
    loader = DataLoader(
        ds, batch_size=gpt2_config.batch_size, sampler=sampler, drop_last=True
    )
    for epoch in range(gpt2_config.epoch_num):
        print(f"current epoch:{epoch}")
        # 每个 epoch 启动前设置随机起始索引
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            x: torch.Tensor
            y: torch.Tensor
            x = batch[0]  # type: torch.Tensor
            y = batch[1]  # type: torch.Tensor
            print(f"Step {step}: x.shape={x.shape}, y.shape={y.shape}")
            if step >= 2:  
                break

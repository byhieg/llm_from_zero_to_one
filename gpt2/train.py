from dataclasses import dataclass, field
import torch
from torch.cuda.amp import autocast, GradScaler
import random
import time
from typing import Iterator, Literal
from torch import cuda
from torch.utils.data import DataLoader, Sampler
from gpt2.data import ShardIndexDataset
from gpt2.model import GPTConfig, GPT2
import swanlab


@dataclass
class TrainConfig:
    batch_size: int = 8
    seq_len: int = 1024
    epoch_num: int = 1
    data_path: str = "/root/autodl-tmp/data"
    use_amp: bool = True  # 是否使用混合精度训练
    amp_dtype: Literal["fp16", "bf16"] = "bf16"  # 混合精度类型


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = TrainConfig()

    # 检查 bf16 支持
    if train_config.use_amp and train_config.amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("警告: 当前设备不支持 bf16，将自动切换到 fp16")
            train_config.amp_dtype = "fp16"

    # 确定混合精度的 dtype
    amp_dtype = None
    if train_config.use_amp:
        amp_dtype = (
            torch.bfloat16 if train_config.amp_dtype == "bf16" else torch.float16
        )
        print(f"使用混合精度训练: {train_config.amp_dtype}")

    # 初始化 GradScaler（仅 fp16 需要）
    scaler = None
    if train_config.use_amp and train_config.amp_dtype == "fp16":
        scaler = GradScaler()

    # 初始化 SwanLab
    swanlab.init(
        project="gpt2-training",
        experiment_name="gpt2-train",
        config={
            "batch_size": train_config.batch_size,
            "seq_len": train_config.seq_len,
            "epoch_num": train_config.epoch_num,
            "use_amp": train_config.use_amp,
            "amp_dtype": train_config.amp_dtype,
        },
    )

    ds = ShardIndexDataset(train_config.data_path, seq_len=train_config.seq_len)
    # 使用自定义 RandomStartSampler 替代 shuffle，以实现每个 epoch 的随机起始
    sampler = RandomStartSampler(
        data_source=ds,
        batch_size=train_config.batch_size,
        drop_last=True,
    )
    loader = DataLoader(
        ds,
        batch_size=train_config.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
    )

    model = GPT2(GPTConfig())
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)
    model = model.to(device)
    # 简单训练优化器，用于输出指标
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    global_step = 0
    for epoch in range(train_config.epoch_num):
        # 每个 epoch 启动前设置随机起始索引
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            start_time = time.time()
            x = batch[0].to(device)
            y = batch[1].to(device)

            # 使用混合精度训练
            if train_config.use_amp and amp_dtype is not None:
                with autocast(dtype=amp_dtype):
                    logits, loss = model(x, y)

                optimizer.zero_grad()

                # FP16 需要 scaler，BF16 不需要
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # 计算梯度范数（在 backward 之后，step 之前）
                    scaler.unscale_(optimizer)
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += float(p.grad.data.norm(2).item()) ** 2
                    grad_norm = grad_norm**0.5
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # BF16 直接反向传播
                    loss.backward()
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += float(p.grad.data.norm(2).item()) ** 2
                    grad_norm = grad_norm**0.5
                    optimizer.step()
            else:
                # 不使用混合精度
                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += float(p.grad.data.norm(2).item()) ** 2
                grad_norm = grad_norm**0.5
                optimizer.step()

            elapsed = time.time() - start_time
            tokens = train_config.batch_size * train_config.seq_len
            throughput = int(tokens / elapsed) if elapsed > 0 else 0

            # 打印指标
            print(
                f"epoch: {epoch} | step: {step} | loss: {loss.item():.2f} | "
                f"grad_norm: {grad_norm:.2f} | tokens: {tokens} | "
                f"time: {elapsed:.2f}s | throughput: {throughput} tokens/s"
            )

            # 记录指标到 SwanLab
            swanlab.log(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/throughput": throughput,
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/global_step": global_step,
                }
            )

            global_step += 1
            if step == 2:
                break

    # 结束 SwanLab 记录
    swanlab.finish()

from dataclasses import dataclass, field
import torch
from torch.cuda.amp import autocast, GradScaler
import time
import math
from typing import Literal
from torch.utils.data import DataLoader
from gpt2.data import ShardIndexDataset
from gpt2.model import GPTConfig, GPT2
import swanlab


@dataclass
class TrainConfig:
    batch_size: int = 8
    seq_len: int = 1024
    epoch_num: int = 1
    data_path: str = "/root/autodl-tmp/data"
    use_amp: bool = True
    amp_dtype: Literal["fp16", "bf16"] = "bf16"
    learning_rate: float = 3e-4
    warmup_steps: int = 10
    grad_clip: float = 1.0


def get_lr(it, warmup_steps, max_steps, learning_rate):
    if it < warmup_steps:
        return learning_rate * it / warmup_steps
    if it > max_steps:
        return learning_rate * 0.1
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * coeff


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = TrainConfig()

    if train_config.use_amp and train_config.amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("警告: 当前设备不支持 bf16，将自动切换到 fp16")
            train_config.amp_dtype = "fp16"

    amp_dtype = None
    if train_config.use_amp:
        amp_dtype = (
            torch.bfloat16 if train_config.amp_dtype == "bf16" else torch.float16
        )
        print(f"使用混合精度训练: {train_config.amp_dtype}")

    scaler = None
    if train_config.use_amp and train_config.amp_dtype == "fp16":
        scaler = GradScaler()

    swanlab.init(
        project="gpt2-training",
        experiment_name="gpt2-train",
        config={
            "batch_size": train_config.batch_size,
            "seq_len": train_config.seq_len,
            "epoch_num": train_config.epoch_num,
            "use_amp": train_config.use_amp,
            "amp_dtype": train_config.amp_dtype,
            "learning_rate": train_config.learning_rate,
            "warmup_steps": train_config.warmup_steps,
            "grad_clip": train_config.grad_clip,
        },
    )

    ds = ShardIndexDataset(train_config.data_path, seq_len=train_config.seq_len)
    loader = DataLoader(
        ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )

    max_steps = train_config.epoch_num * len(loader)
    print(
        f"Total training steps: {max_steps} (epochs: {train_config.epoch_num}, steps_per_epoch: {len(loader)})"
    )

    model = GPT2(GPTConfig())
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    global_step = 0
    for epoch in range(train_config.epoch_num):
        ds.shuffle_shard(epoch)
        total_steps = len(loader)
        print(f"Epoch {epoch}: Total steps = {total_steps}")
        swanlab.log({"train/total_steps": total_steps, "train/epoch": epoch})

        for step, batch in enumerate(loader):
            lr = get_lr(
                global_step,
                train_config.warmup_steps,
                max_steps,
                train_config.learning_rate,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            start_time = time.time()
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()

            if train_config.use_amp and amp_dtype is not None:
                with autocast(dtype=amp_dtype):
                    logits, loss = model(x, y)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
            else:
                logits, loss = model(x, y)
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            elapsed_ms = (time.time() - start_time) * 1000
            tokens = train_config.batch_size * train_config.seq_len
            throughput = int(tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

            print(
                f"epoch: {epoch} | step: {step} | loss: {loss.item():.2f} | "
                f"grad_norm: {grad_norm:.2f} | tokens: {tokens} | "
                f"time: {elapsed_ms:.2f}ms | throughput: {throughput} tokens/s | lr: {lr:.2e}"
            )

            swanlab.log(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/throughput": throughput,
                    "train/time_ms": elapsed_ms,
                    "train/epoch": epoch,
                    "train/step": step,
                    "train/global_step": global_step,
                    "train/learning_rate": lr,
                }
            )

            global_step += 1
    swanlab.finish()

from dataclasses import dataclass
import torch
import time
import math
import os
from typing import Optional
from torch.utils.data import DataLoader
from torch.amp import autocast
from gpt2.data import ShardIndexDataset
from gpt2.model import GPTConfig, GPT2
import tiktoken
import swanlab


@dataclass
class TrainConfig:
    batch_size: int = 64
    seq_len: int = 1024
    epoch_num: int = 1
    data_path: str = "/root/autodl-tmp/data"
    learning_rate: float = 3e-4
    warmup_steps: int = 10
    grad_clip: float = 1.0
    accumulation_steps: int = 4
    log_steps: int = 10
    inference_steps: int = 500
    inference_tokens: int = 100
    inference_topk: int = 100
    inference_temperature: float = 1.0
    inference_prompt: str = "The meaning of life is"
    save_steps: int = 1000
    checkpoint_dir: str = "/root/autodl-tmp/gpt2/checkpoints"
    resume_from_checkpoint: Optional[str] = None


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    block_size: int,
    topk: int = 100,
    temperature: float = 1.0,
    device: str = "cuda",
) -> list[int]:
    model.eval()
    tokens = prompt_tokens.copy()

    for _ in range(max_new_tokens):
        if len(tokens) > block_size:
            context = tokens[-block_size:]
        else:
            context = tokens

        x = torch.tensor([context], dtype=torch.long, device=device)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x, x)

        logits = logits[0, -1, :] / temperature

        topk_values, topk_indices = torch.topk(logits, topk)
        probs = torch.softmax(topk_values, dim=-1)
        idx_next = topk_indices[torch.multinomial(probs, num_samples=1)]
        tokens.append(idx_next.item())

    model.train()
    return tokens


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    config: TrainConfig,
    checkpoint_path: str,
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model_state = (
        model._orig_mod.state_dict()
        if hasattr(model, "_orig_mod")
        else model.state_dict()
    )

    checkpoint = {
        "global_step": global_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "epoch_num": config.epoch_num,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "accumulation_steps": config.accumulation_steps,
        },
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_state = checkpoint["model_state_dict"]
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        checkpoint["global_step"],
        checkpoint["epoch"],
        checkpoint["step_in_epoch"],
    )


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

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前设备不支持 bf16，请使用支持 bf16 的 GPU")

    print("使用 bf16 混合精度训练")

    swanlab.init(
        project="gpt2-training",
        experiment_name="gpt2-train",
        config={
            "batch_size": train_config.batch_size,
            "seq_len": train_config.seq_len,
            "epoch_num": train_config.epoch_num,
            "learning_rate": train_config.learning_rate,
            "warmup_steps": train_config.warmup_steps,
            "grad_clip": train_config.grad_clip,
            "accumulation_steps": train_config.accumulation_steps,
            "log_steps": train_config.log_steps,
            "inference_steps": train_config.inference_steps,
            "inference_tokens": train_config.inference_tokens,
            "inference_topk": train_config.inference_topk,
            "inference_temperature": train_config.inference_temperature,
            "inference_prompt": train_config.inference_prompt,
            "save_steps": train_config.save_steps,
        },
    )

    ds = ShardIndexDataset(train_config.data_path, seq_len=train_config.seq_len)
    loader = DataLoader(
        ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    steps_per_epoch = len(loader) // train_config.accumulation_steps
    max_steps = train_config.epoch_num * steps_per_epoch
    print(
        f"Total training steps: {max_steps} (epochs: {train_config.epoch_num}, "
        f"steps_per_epoch: {steps_per_epoch}, accumulation_steps: {train_config.accumulation_steps})"
    )

    model = GPT2(GPTConfig())
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    global_step = 0
    start_epoch = 0
    start_step_in_epoch = 0

    if train_config.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {train_config.resume_from_checkpoint}")
        global_step, start_epoch, start_step_in_epoch = load_checkpoint(
            train_config.resume_from_checkpoint, model, optimizer
        )
        print(f"Resumed from step {global_step}, epoch {start_epoch}")

    for epoch in range(start_epoch, train_config.epoch_num):
        ds.shuffle_shard(epoch)
        print(f"Epoch {epoch}: steps_per_epoch = {steps_per_epoch}")
        swanlab.log({"train/total_steps": steps_per_epoch, "train/epoch": epoch})

        epoch_iterator = enumerate(loader)
        if epoch == start_epoch and start_step_in_epoch > 0:
            for _ in range(start_step_in_epoch * train_config.accumulation_steps):
                next(epoch_iterator)

        accumulated_loss = torch.tensor(0.0, device=device)
        last_log_time = time.time()
        for step, batch in epoch_iterator:
            lr = get_lr(
                global_step,
                train_config.warmup_steps,
                max_steps,
                train_config.learning_rate,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            batch: torch.Tensor
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)

            is_accumulation_step = (step + 1) % train_config.accumulation_steps != 0

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / train_config.accumulation_steps
            loss.backward()

            accumulated_loss += loss.detach()

            if is_accumulation_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % train_config.log_steps == 0:
                elapsed_ms = (time.time() - last_log_time) * 1000
                last_log_time = time.time()
                tokens = (
                    train_config.batch_size
                    * train_config.seq_len
                    * train_config.accumulation_steps
                    * train_config.log_steps
                )
                throughput = int(tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

                loss_value = accumulated_loss.item() / train_config.accumulation_steps
                grad_norm_value = (
                    grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                )

                print(
                    f"epoch: {epoch} | step: {global_step} | "
                    f"loss: {loss_value:.2f} | grad_norm: {grad_norm_value:.2f} | "
                    f"tokens: {tokens} | time: {elapsed_ms:.2f}ms | throughput: {throughput} tokens/s | lr: {lr:.2e}"
                )

                swanlab.log(
                    {
                        "train/loss": loss_value,
                        "train/grad_norm": grad_norm_value,
                        "train/throughput": throughput,
                        "train/time_ms": elapsed_ms,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/learning_rate": lr,
                    }
                )

            accumulated_loss = torch.tensor(0.0, device=device)

            if global_step % train_config.inference_steps == 0:
                tokenizer = tiktoken.get_encoding("gpt2")
                prompt_tokens = tokenizer.encode(train_config.inference_prompt)
                generated_tokens = generate(
                    model,
                    prompt_tokens,
                    train_config.inference_tokens,
                    GPTConfig().block_size,
                    topk=train_config.inference_topk,
                    temperature=train_config.inference_temperature,
                    device=device,
                )
                generated_text = tokenizer.decode(generated_tokens)
                print(f"\n{'=' * 60}")
                print(f"[Inference @ step {global_step}]")
                print(f"{generated_text}")
                print(f"{'=' * 60}\n")
                swanlab.log(
                    {"inference/text": generated_text, "train/global_step": global_step}
                )

            if global_step % train_config.save_steps == 0:
                checkpoint_path = os.path.join(
                    train_config.checkpoint_dir, f"checkpoint_step_{global_step}.pt"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    epoch,
                    step // train_config.accumulation_steps + 1,
                    train_config,
                    checkpoint_path,
                )

    checkpoint_path = os.path.join(train_config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(
        model,
        optimizer,
        global_step,
        train_config.epoch_num - 1,
        0,
        train_config,
        checkpoint_path,
    )
    swanlab.finish()

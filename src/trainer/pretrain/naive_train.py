from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class NaiveBackendState:
    model: torch.nn.Module
    optimizer: Any
    grad_scaler: torch.amp.GradScaler | None
    accumulation_steps: int
    accumulated_loss: torch.Tensor | None = None


def get_accumulation_steps(backend_state: NaiveBackendState) -> int:
    return backend_state.accumulation_steps


def prepare_backend(
    trainer: Any,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint,
) -> NaiveBackendState:
    accumulation_steps = _resolve_gradient_accumulation_steps(
        trainer.args.train.naive_config.get("accumulation_steps", 4)
    )
    model = model.to(device)
    optimizer = trainer._build_optimizer(model)
    grad_scaler = build_grad_scaler(trainer.args, device)
    if checkpoint and checkpoint.optimizer_state_dict:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    model = maybe_compile_model(model, device)
    if trainer._is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[trainer.rank_info["local_rank"]]
        )
    return NaiveBackendState(
        model=model,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        accumulation_steps=accumulation_steps,
    )


def set_train_mode(backend_state: NaiveBackendState) -> None:
    backend_state.model.train()


def train_batch(
    trainer: Any,
    backend_state: NaiveBackendState,
    x: torch.Tensor,
    y: torch.Tensor,
    global_step: int,
    max_steps: int,
    micro_step: int,
    device: torch.device,
) -> dict[str, Any]:
    lr = get_lr(trainer.args, global_step, max_steps)
    set_optimizer_learning_rate(backend_state.optimizer, lr)
    with _forward_context(trainer.args, device):
        _, loss = backend_state.model(x, y)
    accumulation_steps = backend_state.accumulation_steps
    log_loss = (loss / accumulation_steps).detach()
    if backend_state.accumulated_loss is None:
        backend_state.accumulated_loss = log_loss
    else:
        backend_state.accumulated_loss += log_loss
    backward_loss = loss / accumulation_steps
    if backend_state.grad_scaler is not None:
        backend_state.grad_scaler.scale(backward_loss).backward()
    else:
        backward_loss.backward()
    if micro_step % accumulation_steps != 0:
        return None
    grad_norm = clip_naive_grad_norm(
        trainer.args,
        backend_state.model,
        backend_state.optimizer,
        backend_state.grad_scaler,
    )
    if backend_state.grad_scaler is not None:
        backend_state.grad_scaler.step(backend_state.optimizer)
        backend_state.grad_scaler.update()
    else:
        backend_state.optimizer.step()
    backend_state.optimizer.zero_grad()
    total_log_loss = backend_state.accumulated_loss
    backend_state.accumulated_loss = None
    return {
        "log_loss": total_log_loss,
        "did_update": True,
        "grad_norm": grad_norm,
        "lr": lr,
    }


def save_checkpoint_if_needed(
    trainer: Any,
    backend_state: NaiveBackendState,
    global_step: int,
    epoch: int,
    micro_step_in_epoch: int,
    dataloader_length: int,
) -> None:
    return


def save_training_checkpoint(
    trainer: Any,
    backend_state: NaiveBackendState,
    global_step: int,
    epoch: int,
    micro_step_in_epoch: int,
    dataloader_length: int,
) -> None:
    return


def get_lr(args, step: int, max_steps: int) -> float:
    warmup_steps = args.train.naive_config.get("warmup_steps", 10)
    learning_rate = args.train.naive_config.get("learning_rate", 3e-4)
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    if step > max_steps:
        return learning_rate * 0.1
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * coeff


def maybe_compile_model(
    model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    if not hasattr(torch, "compile"):
        return model
    if device.type not in ("cuda", "mps"):
        return model
    try:
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        return torch.compile(model)
    except Exception:
        return model


def is_amp_enabled(args, device: torch.device) -> bool:
    return device.type == "cuda" and args.train.naive_config.get("amp", False)


def get_amp_dtype(args) -> torch.dtype:
    amp_dtype = args.train.naive_config.get("amp_dtype", "bf16")
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def build_grad_scaler(args, device: torch.device) -> torch.amp.GradScaler | None:
    if not is_amp_enabled(args, device):
        return None
    if args.train.naive_config.get("amp_dtype", "bf16") != "fp16":
        return None
    return torch.amp.GradScaler("cuda")


def set_optimizer_learning_rate(optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def clip_naive_grad_norm(
    args,
    model: torch.nn.Module,
    optimizer,
    grad_scaler: torch.amp.GradScaler | None,
) -> torch.Tensor:
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(args.train.naive_config.get("grad_clip", 1.0)),
    )


def _resolve_gradient_accumulation_steps(accumulation_steps: Any) -> int:
    resolved_steps = int(accumulation_steps or 1)
    if resolved_steps <= 0:
        raise ValueError("gradient accumulation steps must be greater than 0")
    return resolved_steps


def _forward_context(args, device: torch.device):
    if not is_amp_enabled(args, device):
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=get_amp_dtype(args),
    )

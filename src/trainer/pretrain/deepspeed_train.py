from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
import deepspeed
import torch

from .pretrain_args import PreTrainArgs


class DeepSpeedPretrainRuntime:
    def __init__(self, args: PreTrainArgs):
        self.args = args
        self.deepspeed = self._import_deepspeed()
        self.config = self._resolve_config()
        self.engine: deepspeed.DeepSpeedEngine | None = None
        self.optimizer = None
        self._accumulated_loss: torch.Tensor | None = None
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            self.init_distributed()

    def _import_deepspeed(self):
        try:
            return import_module("deepspeed")
        except ImportError as exc:
            raise ImportError(
                "train.backend=deepspeed but deepspeed is not installed, please install it first."
            ) from exc

    def _resolve_config(self) -> str | dict[str, Any]:
        config = self.args.train.deepspeed_config
        if not config:
            raise ValueError("train.deepspeed_config must be configured")
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"DeepSpeed config file not found: {config_path}")
        return str(config_path)

    def init_distributed(self) -> None:
        self.deepspeed.init_distributed()

    def get_accumulation_steps(self) -> int:
        if self.engine is None:
            raise RuntimeError("DeepSpeed runtime engine is not initialized")
        resolved_steps = int(self.engine.gradient_accumulation_steps())
        if resolved_steps <= 0:
            raise ValueError(
                "DeepSpeed gradient accumulation steps must be greater than 0"
            )
        return resolved_steps

    def prepare(self, model: torch.nn.Module) -> "DeepSpeedPretrainRuntime":
        engine, engine_optimizer, _, _ = self.deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=self.config,
        )
        optimizer = engine_optimizer or getattr(engine, "optimizer", None)
        if optimizer is None:
            raise RuntimeError(
                "DeepSpeed initialize did not return an optimizer. "
                "Please configure optimizer in train.deepspeed_config."
            )
        self.engine = engine
        self.optimizer = optimizer
        self._accumulated_loss = None
        return self

    def set_train_mode(self) -> None:
        if self.engine is None:
            raise RuntimeError("DeepSpeed runtime engine is not initialized")
        self.engine.train()

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> dict[str, Any] | None:
        if self.engine is None:
            raise RuntimeError("DeepSpeed runtime engine is not initialized")
        _, loss = self.engine(x, y)
        log_loss = (loss / self.get_accumulation_steps()).detach()
        if self._accumulated_loss is None:
            self._accumulated_loss = log_loss
        else:
            self._accumulated_loss += log_loss
        self.engine.backward(loss)
        self.engine.step()
        if not bool(self.engine.is_gradient_accumulation_boundary()):
            return None
        total_log_loss = self._accumulated_loss
        self._accumulated_loss = None
        return {
            "log_loss": total_log_loss,
            "grad_norm": self._get_grad_norm(device),
            "lr": self._get_optimizer_learning_rate(),
        }

    def _get_grad_norm(self, device: torch.device) -> torch.Tensor:
        if self.engine is None:
            raise RuntimeError("DeepSpeed runtime engine is not initialized")
        grad_norm_getter = getattr(self.engine, "get_global_grad_norm", None)
        if callable(grad_norm_getter):
            grad_norm = grad_norm_getter()
            if isinstance(grad_norm, torch.Tensor):
                return grad_norm
            if grad_norm is None:
                return torch.tensor(0.0, device=device)
            return torch.tensor(float(grad_norm), device=device)
        return torch.tensor(0.0, device=device)

    def _get_optimizer_learning_rate(self) -> float | None:
        param_groups = getattr(self.optimizer, "param_groups", None)
        if not param_groups:
            return None
        return float(param_groups[0].get("lr", 0.0))

    def save_checkpoint(self, checkpoint_dir: str, client_state: any) -> None:
        self.engine.save_checkpoint(save_dir=checkpoint_dir, client_state=client_state)

    def load_checkpoint(
        self, resume_checkpoint_dir: str, resume_tag: str
    ) -> tuple[str, any]:
        return self.engine.load_checkpoint(
            load_dir=resume_checkpoint_dir, tag=resume_tag
        )

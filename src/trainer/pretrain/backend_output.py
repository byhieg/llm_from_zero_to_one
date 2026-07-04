from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainStepOutput:
    """训练后端单次优化步输出。

    多个训练后端在完成一次真实参数更新后，统一返回该结构给主训练循环。
    如果当前 micro step 仍处于梯度累积阶段，后端应返回 ``None``。
    """

    log_loss: torch.Tensor
    grad_norm: torch.Tensor | None = None
    lr: float | None = None
    did_update: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

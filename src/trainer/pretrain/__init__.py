from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pretrain import PreTrainTrainer

__all__ = ["PreTrainTrainer"]


def __getattr__(name: str) -> Any:
    if name == "PreTrainTrainer":
        from .pretrain import PreTrainTrainer

        return PreTrainTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

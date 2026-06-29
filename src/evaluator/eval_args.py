from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from trainer.common_args import ModelConfig, TrainingArgs


@dataclass
class EvalConfig:
    steps: int = 0
    dataset_path: str = ""
    dataset_name: str = ""
    data_files: dict[str, str] = field(default_factory=dict)
    split: str = "test"
    text_column: str = "text"
    tokenizer_path: str = ""
    max_samples: int = 256
    batch_size: int = 8
    add_bos_id: bool = False
    add_eos_id: bool = True
    checkpoint_step: Optional[int] = None


@dataclass
class EvalTrainingConfig:
    seq_len: int = 1024


@dataclass
class EvalCheckpointConfig:
    save_steps: int = 1000
    checkpoint_dir: str = "checkpoints/pretrain"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class EvalArgs(TrainingArgs):
    name: str = ""
    training: EvalTrainingConfig = field(default_factory=EvalTrainingConfig)
    checkpoint: EvalCheckpointConfig = field(default_factory=EvalCheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalArgs":
        return cls(
            name=data.get("name", ""),
            training=EvalTrainingConfig(**data.get("training", {})),
            checkpoint=EvalCheckpointConfig(**data.get("checkpoint", {})),
            eval=EvalConfig(**data.get("eval", {})),
            model=ModelConfig(**data.get("model", {})),
        )

    def validate(self) -> list[str]:
        errors = []

        if not self.model.name:
            errors.append("model.name is required")
        if not self.checkpoint.checkpoint_dir:
            errors.append("checkpoint.checkpoint_dir is required")
        if not self.eval.dataset_path:
            errors.append("eval.dataset_path is required")
        if not self.eval.text_column:
            errors.append("eval.text_column is required")
        if self.eval.max_samples <= 0:
            errors.append(
                f"eval.max_samples must be positive, got {self.eval.max_samples}"
            )
        if self.eval.batch_size <= 0:
            errors.append(
                f"eval.batch_size must be positive, got {self.eval.batch_size}"
            )
        if self.eval.checkpoint_step is not None and self.eval.checkpoint_step < 0:
            errors.append("eval.checkpoint_step must be non-negative when provided")
        if not self.eval.tokenizer_path:
            errors.append("eval.tokenizer_path is required")

        return errors

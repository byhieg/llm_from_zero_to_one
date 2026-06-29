from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional, Type

from trainer.common_args import TrainingArgs, ModelConfig, DataConfig, CheckpointConfig


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
class EvalArgs(TrainingArgs):
    training: EvalTrainingConfig = field(default_factory=EvalTrainingConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(checkpoint_dir="checkpoints/pretrain")
    )
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalArgs":
        init_kwargs = {}

        for f in fields(cls):
            if not f.init:
                continue

            field_name = f.name
            field_type = f.type

            if field_name not in data:
                continue

            value = data[field_name]

            field_type_str = (
                str(field_type).replace("typing.", "").replace(" ", "").strip("<>")
            )

            if field_type_str in cls._FIELD_TYPE_MAP:
                field_class = cls._FIELD_TYPE_MAP[field_type_str]
                if isinstance(value, dict):
                    init_kwargs[field_name] = field_class(**value)
                else:
                    init_kwargs[field_name] = value
            else:
                init_kwargs[field_name] = value

        return cls(**init_kwargs)

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


EvalArgs._FIELD_TYPE_MAP = {
    "EvalTrainingConfig": EvalTrainingConfig,
    "EvalConfig": EvalConfig,
    "ModelConfig": ModelConfig,
    "DataConfig": DataConfig,
    "CheckpointConfig": CheckpointConfig,
}
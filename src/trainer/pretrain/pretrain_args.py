from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional, Type

from trainer.common_args import (
    TrainingArgs,
    ModelConfig,
    ExperimentConfig,
    DataConfig,
    CheckpointConfig,
)


@dataclass
class PretrainTrainingConfig:
    epoch_num: int = 1
    batch_size: int = 16
    seq_len: int = 1024
    seed: int = 42
    log_steps: int = 10

    backend: str = "naive"
    naive_config: dict[str, Any] = field(default_factory=dict)
    deepspeed_config: dict[str, Any] = field(default_factory=dict)
    megatron_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PretrainEvalConfig:
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
class PretrainOptimizerConfig:
    name: str = "adamw"
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class PretrainArgs(TrainingArgs):
    training: PretrainTrainingConfig = field(default_factory=PretrainTrainingConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(checkpoint_dir="checkpoints/pretrain")
    )
    data: DataConfig = field(default_factory=DataConfig)
    eval: PretrainEvalConfig = field(default_factory=PretrainEvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: PretrainOptimizerConfig = field(default_factory=PretrainOptimizerConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PretrainArgs":
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

        if self.training.epoch_num > 0 and not self.data.dataset_config.get(
            "dataset_path"
        ):
            errors.append(
                "data.dataset_config.dataset_path is required when training.epoch_num > 0"
            )

        if self.data.data_strategy not in ("padding", "megatron"):
            errors.append(
                f"data.data_strategy must be 'padding' or 'megatron', got '{self.data.data_strategy}'"
            )

        if self.data.data_strategy == "megatron":
            if "total_token" not in self.data.dataset_config:
                errors.append(
                    "data.dataset_config.total_token is required when using 'megatron' strategy"
                )

        dataloader_num_workers = self.data.dataloader_config.get("num_workers", 0)
        if dataloader_num_workers < 0:
            errors.append(
                f"data.dataloader_config.num_workers must be non-negative, got {dataloader_num_workers}"
            )

        if self.training.batch_size <= 0:
            errors.append(
                f"training.batch_size must be positive, got {self.training.batch_size}"
            )
        if self.training.seq_len <= 0:
            errors.append(
                f"training.seq_len must be positive, got {self.training.seq_len}"
            )
        if self.training.log_steps <= 0:
            errors.append(
                f"training.log_steps must be positive, got {self.training.log_steps}"
            )

        if self.training.backend not in ("naive", "deepspeed", "megatron"):
            errors.append(
                f"training.backend must be 'naive', 'deepspeed', or 'megatron', got '{self.training.backend}'"
            )

        if self.training.backend == "naive":
            lr = self.training.naive_config.get("learning_rate")
            if lr is not None and lr <= 0:
                errors.append(
                    f"training.naive_config.learning_rate must be positive, got {lr}"
                )
        elif self.training.backend == "deepspeed":
            if not self.training.deepspeed_config.get("train_batch_size"):
                errors.append(
                    "training.deepspeed_config.train_batch_size is required when backend='deepspeed'"
                )
        elif self.training.backend == "megatron":
            if not self.training.megatron_config.get("global_batch_size"):
                errors.append(
                    "training.megatron_config.global_batch_size is required when backend='megatron'"
                )

        if self.eval.steps < 0:
            errors.append(f"eval.steps must be non-negative, got {self.eval.steps}")
        if self.eval.steps > 0:
            if not self.eval.dataset_path:
                errors.append("eval.dataset_path is required when eval.steps > 0")
            if not self.eval.text_column:
                errors.append("eval.text_column is required when eval.steps > 0")
            if self.eval.max_samples <= 0:
                errors.append(
                    f"eval.max_samples must be positive, got {self.eval.max_samples}"
                )
            if self.eval.batch_size <= 0:
                errors.append(
                    f"eval.batch_size must be positive, got {self.eval.batch_size}"
                )
            if not self.eval.tokenizer_path:
                errors.append("eval.tokenizer_path is required when eval.steps > 0")

        if self.optimizer.weight_decay < 0:
            errors.append(
                f"optimizer.weight_decay must be non-negative, got {self.optimizer.weight_decay}"
            )
        if self.optimizer.name not in ("adamw", "adam"):
            errors.append(
                f"optimizer.name must be 'adamw' or 'adam', got '{self.optimizer.name}'"
            )
        if len(self.optimizer.betas) != 2:
            errors.append(
                f"optimizer.betas must contain exactly 2 values, got {self.optimizer.betas}"
            )

        if self.experiment.enabled:
            if not self.experiment.project:
                errors.append(
                    "experiment.project is required when experiment.enabled is true"
                )
            if not self.experiment.experiment_name:
                errors.append(
                    "experiment.experiment_name is required when experiment.enabled is true"
                )

        return errors


PretrainArgs._FIELD_TYPE_MAP = {
    "PretrainTrainingConfig": PretrainTrainingConfig,
    "PretrainEvalConfig": PretrainEvalConfig,
    "PretrainOptimizerConfig": PretrainOptimizerConfig,
    "ModelConfig": ModelConfig,
    "ExperimentConfig": ExperimentConfig,
    "DataConfig": DataConfig,
    "CheckpointConfig": CheckpointConfig,
}
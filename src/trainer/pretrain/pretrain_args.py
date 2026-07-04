from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from trainer.common_args import (
    ExperimentConfig,
    ModelConfig,
    SwanLabConfig,
    TrainingArgs,
)


@dataclass
class PreTrainTrainConfig:
    """预训练 train 模块配置。"""

    epoch_num: int = 1
    batch_size: int = 16
    seq_len: int = 1024
    seed: int = 42
    log_steps: int = 10
    backend: str = "naive"
    naive_config: dict[str, Any] = field(default_factory=dict)
    deepspeed_config: str | dict[str, Any] = field(default_factory=dict)
    megatron_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreTrainCheckpointConfig:
    """预训练 checkpoint 模块配置。"""

    save_step: int = -1
    save_checkpoint_dir: str = None
    resume_checkpoint_dir: str = None
    resume_tag: str = None


@dataclass
class PreTrainTrainDataConfig:
    """预训练训练数据配置。"""

    data_strategy: str = "padding"
    dataset_config: dict[str, Any] = field(default_factory=dict)
    dataloader_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreTrainDataConfig:
    """预训练 data 模块配置。"""

    train: PreTrainTrainDataConfig = field(default_factory=PreTrainTrainDataConfig)


@dataclass
class PreTrainArgs(TrainingArgs):
    """预训练模式配置。"""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: PreTrainTrainConfig = field(default_factory=PreTrainTrainConfig)
    checkpoint: PreTrainCheckpointConfig = field(
        default_factory=PreTrainCheckpointConfig
    )
    data: PreTrainDataConfig = field(default_factory=PreTrainDataConfig)

    def set_mode(self, mode: str) -> None:
        """同步 experiment.mode。"""

        self.experiment.mode = mode

    def to_config_dict(self, mode: str | None = None) -> dict[str, Any]:
        """渲染为 demo_config.yaml 风格的配置结构。"""

        experiment = asdict(self.experiment)
        experiment["mode"] = mode or self.experiment.mode or "pretrain"
        return {
            "experiment": experiment,
            "model": asdict(self.model),
            "train": asdict(self.train),
            "checkpoint": asdict(self.checkpoint),
            "data": asdict(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreTrainArgs":
        """从 YAML 字典构造预训练配置。"""

        experiment_dict = data.get("experiment", {})
        swanlab_dict = experiment_dict.get("swanlab", {})
        data_dict = data.get("data", {})
        checkpoint_dict = dict(data.get("checkpoint", {}))
        if "save_steps" in checkpoint_dict and "save_step" not in checkpoint_dict:
            checkpoint_dict["save_step"] = checkpoint_dict.pop("save_steps")
        if (
            "checkpoint_dir" in checkpoint_dict
            and "save_checkpoint_dir" not in checkpoint_dict
        ):
            checkpoint_dict["save_checkpoint_dir"] = checkpoint_dict.pop(
                "checkpoint_dir"
            )
        if (
            "resume_from_checkpoint" in checkpoint_dict
            and "resume_checkpoint_dir" not in checkpoint_dict
        ):
            checkpoint_dict["resume_checkpoint_dir"] = checkpoint_dict.pop(
                "resume_from_checkpoint"
            )

        return cls(
            experiment=ExperimentConfig(
                mode=experiment_dict.get("mode", ""),
                name=experiment_dict.get("name", ""),
                swanlab=SwanLabConfig(
                    enabled=swanlab_dict.get("enabled", False),
                    project=swanlab_dict.get("project", "llm-training"),
                    tags=list(swanlab_dict.get("tags", [])),
                ),
            ),
            model=ModelConfig(**data.get("model", {})),
            train=PreTrainTrainConfig(**data.get("train", {})),
            checkpoint=PreTrainCheckpointConfig(**checkpoint_dict),
            data=PreTrainDataConfig(
                train=PreTrainTrainDataConfig(**data_dict.get("train", {})),
            ),
        )

    def validate(self) -> list[str]:
        """校验预训练配置。"""

        errors = []
        train_dataset_config = self.data.train.dataset_config
        dataloader_config = self.data.train.dataloader_config

        if self.train.epoch_num > 0 and not train_dataset_config.get("dataset_path"):
            errors.append(
                "data.train.dataset_config.dataset_path is required when train.epoch_num > 0"
            )

        if self.data.train.data_strategy not in ("padding", "megatron"):
            errors.append(
                "data.train.data_strategy must be 'padding' or 'megatron', "
                f"got '{self.data.train.data_strategy}'"
            )

        if (
            self.data.train.data_strategy == "megatron"
            and "total_token" not in train_dataset_config
        ):
            errors.append(
                "data.train.dataset_config.total_token is required when using 'megatron' strategy"
            )

        dataloader_num_workers = dataloader_config.get("num_workers", 0)
        if dataloader_num_workers < 0:
            errors.append(
                "data.train.dataloader_config.num_workers must be non-negative, "
                f"got {dataloader_num_workers}"
            )

        if self.train.batch_size <= 0:
            errors.append(
                f"train.batch_size must be positive, got {self.train.batch_size}"
            )
        if self.train.seq_len <= 0:
            errors.append(f"train.seq_len must be positive, got {self.train.seq_len}")
        if self.train.log_steps <= 0:
            errors.append(
                f"train.log_steps must be positive, got {self.train.log_steps}"
            )

        if self.train.backend not in ("naive", "deepspeed", "megatron"):
            errors.append(
                "train.backend must be 'naive', 'deepspeed', or 'megatron', "
                f"got '{self.train.backend}'"
            )

        learning_rate = self.train.naive_config.get("learning_rate")
        if (
            self.train.backend == "naive"
            and learning_rate is not None
            and learning_rate <= 0
        ):
            errors.append(
                "train.naive_config.learning_rate must be positive, "
                f"got {learning_rate}"
            )
        if self.train.backend == "deepspeed" and not self.train.deepspeed_config:
            errors.append("train.deepspeed_config is required when backend='deepspeed'")
        if self.train.backend == "megatron" and not self.train.megatron_config.get(
            "global_batch_size"
        ):
            errors.append(
                "train.megatron_config.global_batch_size is required when backend='megatron'"
            )

        if self.experiment.swanlab.enabled:
            if not self.experiment.swanlab.project:
                errors.append(
                    "experiment.swanlab.project is required when experiment.swanlab.enabled is true"
                )
            if not self.experiment.name:
                errors.append(
                    "experiment.name is required when experiment.swanlab.enabled is true"
                )

        return errors

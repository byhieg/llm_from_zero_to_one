from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional, Any


@dataclass
class ModelConfig:
    name: str = "gpt2"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    enabled: bool = False
    project: str = "llm-training"
    experiment_name: str = "default"
    tags: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    data_strategy: str = "padding"
    dataset_config: dict[str, Any] = field(default_factory=dict)
    dataloader_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    save_steps: int = 1000
    checkpoint_dir: str = ""
    resume_from_checkpoint: Optional[str] = None


@dataclass
class TrainingArgs:
    name: str = ""
    device: str = field(default="cpu", init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingArgs":
        init_kwargs = {}
        for f in fields(cls):
            if not f.init:
                continue
            if f.name in data:
                init_kwargs[f.name] = data[f.name]
        return cls(**init_kwargs)

    def validate(self) -> list[str]:
        return []
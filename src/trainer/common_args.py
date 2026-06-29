from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """模型公共配置。"""

    name: str = "gpt2"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SwanLabConfig:
    """SwanLab 实验跟踪公共配置。"""

    enabled: bool = False
    project: str = "llm-training"
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """实验公共配置。"""

    mode: str = ""
    name: str = ""
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)


@dataclass
class TrainingArgs:
    """训练参数基类，提供通用序列化能力。"""

    device: str = field(default="cpu", init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """将参数对象序列化为 Python 字典。"""

        return asdict(self)

    def to_config_dict(self, mode: str | None = None) -> dict[str, Any]:
        """将参数对象渲染为配置文件结构。"""

        data = self.to_dict()
        if mode and "mode" not in data:
            data["mode"] = mode
        return data

    def to_yaml(self, path: str | Path, mode: str | None = None) -> None:
        """将参数对象写入 YAML 文件。"""

        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_config_dict(mode=mode),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingArgs":
        """从字典构造参数对象。"""

        init_kwargs = {}
        for current_field in fields(cls):
            if not current_field.init:
                continue
            if current_field.name in data:
                init_kwargs[current_field.name] = data[current_field.name]
        return cls(**init_kwargs)

    def set_mode(self, mode: str) -> None:
        """记录当前模式，默认无额外行为。"""

    def validate(self) -> list[str]:
        """校验配置，返回错误列表。"""

        return []

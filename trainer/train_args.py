from __future__ import annotations

from dataclasses import dataclass, field, fields, MISSING
from typing import Optional


@dataclass
class TrainingArgs:
    batch_size: int = 16
    seq_len: int = 1024
    epoch_num: int = 1
    data_path: str = ""
    learning_rate: float = 3e-4
    warmup_steps: int = 10
    grad_clip: float = 1.0
    accumulation_steps: int = 4
    log_steps: int = 10
    save_steps: int = 1000
    checkpoint_dir: str = ""
    resume_from_checkpoint: Optional[str] = None

    # 运行时由 train.py 注入，不在 CLI 中暴露
    device: str = field(default="cpu", init=False, repr=False)


@dataclass
class PretrainArgs(TrainingArgs):
    data_path: str = ""
    checkpoint_dir: str = "checkpoints/pretrain"
    inference_steps: int = 500
    inference_tokens: int = 100
    inference_topk: int = 100
    inference_temperature: float = 1.0
    inference_prompt: str = "The meaning of life is"


# ---------------------------------------------------------------------------
# 注册表 + 工厂
# ---------------------------------------------------------------------------
_ARGS_REGISTRY: dict[str, type[TrainingArgs]] = {}


def register_args(name: str, args_cls: type[TrainingArgs]) -> None:
    _ARGS_REGISTRY[name] = args_cls


def get_args_class(mode: str) -> type[TrainingArgs]:
    if mode not in _ARGS_REGISTRY:
        available = ", ".join(_ARGS_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown mode '{mode}'. Available: {available}")
    return _ARGS_REGISTRY[mode]


def list_modes() -> list[str]:
    return list(_ARGS_REGISTRY.keys())


def parse_args(mode: str, argv: list[str] | None = None) -> TrainingArgs:
    """
    将命令行参数解析为对应 mode 的 dataclass 实例。
    支持 ``--batch_size 32`` 和 ``--batch-size 32`` 两种写法。
    """
    args_cls = get_args_class(mode)
    import argparse
    from typing import get_type_hints

    parser = argparse.ArgumentParser(description=f"Training mode: {mode}")

    # 获取实际的类型提示（解决字符串注解问题）
    type_hints = get_type_hints(args_cls)

    for f in fields(args_cls):
        if not f.init:
            continue

        # 支持两种写法: --batch-size 和 --batch_size
        cli_name = f.name.replace("_", "-")

        # 构建 argparse 参数
        kwargs: dict = {}

        # 获取实际的类型
        actual_type = type_hints.get(f.name, f.type)

        # 处理不同类型
        if actual_type is bool:
            # bool 类型特殊处理
            # 获取默认值
            default_val = False
            if f.default is not MISSING:
                default_val = f.default
            elif f.default_factory is not MISSING:  # type: ignore
                default_val = f.default_factory()  # type: ignore

            # 如果默认值是 False，使用 store_true
            # 如果默认值是 True，使用 store_false
            if default_val is False:
                kwargs["action"] = "store_true"
            else:
                kwargs["action"] = "store_false"
        else:
            # 非布尔类型
            # 对于 Optional[type]，需要提取内部类型
            actual_type_str = str(actual_type)
            if actual_type_str.startswith("typing.Optional") or actual_type_str.startswith("Optional"):
                # 提取 Optional 中的实际类型
                import typing
                origin = typing.get_origin(actual_type)
                if origin is typing.Union:
                    # Optional[X] 实际上是 Union[X, None]
                    args = typing.get_args(actual_type)
                    # 找到非 None 的类型
                    actual_type = next((arg for arg in args if arg is not type(None)), str)
            
            kwargs["type"] = actual_type

            # 处理默认值
            if f.default is not MISSING:
                kwargs["default"] = f.default
                kwargs["help"] = f"(default: {f.default})"
            elif f.default_factory is not MISSING:  # type: ignore
                default_val = f.default_factory()  # type: ignore
                kwargs["default"] = default_val
                kwargs["help"] = f"(default: {default_val})"

        parser.add_argument(f"--{cli_name}", f"--{f.name}", **kwargs)

    parsed = parser.parse_args(argv)

    # 将解析结果转换为字典，处理 bool 类型的特殊情况
    result_dict = {}
    for f in fields(args_cls):
        if not f.init:
            continue
        value = getattr(parsed, f.name, None)
        if value is not None:
            result_dict[f.name] = value

    return args_cls(**result_dict)


# 注册内置模式
register_args("pretrain", PretrainArgs)

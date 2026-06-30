"""项目统一日志封装。

仅在标准库 ``logging`` 之上增加两点能力：

1. 统一 ``llm.*`` 命名空间，避免各模块各自散落配置。
2. 控制台支持按分布式 rank 过滤，仅输出指定 rank 的日志。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import cast

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        if fmt is None:
            fmt = "%(asctime)s | %(color)s%(levelname)-8s%(reset)s | %(name)s | %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        record.color = _LEVEL_COLORS.get(record.levelno, "")  # type: ignore[attr-defined]
        record.reset = _RESET  # type: ignore[attr-defined]
        return super().format(record)


class _PlainFormatter(logging.Formatter):
    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)


class _RankFilter(logging.Filter):
    """仅允许目标 rank 的日志通过。"""

    def __init__(self, rank: int = 0) -> None:
        super().__init__()
        self.target_rank: int = rank

    def filter(self, record: logging.LogRecord) -> bool:
        current_rank = _get_rank()
        record.rank = current_rank  # type: ignore[attr-defined]
        return current_rank == self.target_rank


class _InjectRankFilter(logging.Filter):
    """为日志记录补充当前 rank 信息。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = _get_rank()  # type: ignore[attr-defined]
        return True


def _get_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except (ImportError, RuntimeError):
        pass
    for env_var in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        val = os.environ.get(env_var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return 0


def _detect_color_support() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if not hasattr(sys.stderr, "isatty"):
        return False
    return sys.stderr.isatty()


def _get_ranked_log_file_path(log_file: str, rank: int) -> str:
    return f"{log_file}.rank{rank}"


_ROOT_NAME = "llm"

_global_handler_configured: bool = False
NewLogger = logging.Logger


def get_logger(name: str = "llm") -> logging.Logger:
    """获取项目日志对象。"""

    full_name = name if name.startswith(_ROOT_NAME) else f"{_ROOT_NAME}.{name}"
    return cast(logging.Logger, logging.getLogger(full_name))


def init_logger(
    level: str | int = "INFO",
    *,
    log_file: str | None = None,
    log_file_level: str | int | None = None,
    rank: int | None = None,
    fmt: str | None = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    color: bool | None = None,
) -> None:
    """初始化项目根日志器。"""
    global _global_handler_configured
    if _global_handler_configured:
        return

    if rank is None:
        rank = _get_rank()
    current_rank = _get_rank()

    numeric_level = level if isinstance(level, int) else getattr(logging, level.upper())
    file_level = (
        numeric_level
        if log_file_level is None
        else (
            log_file_level
            if isinstance(log_file_level, int)
            else getattr(logging, log_file_level.upper())
        )
    )

    use_color = color if color is not None else _detect_color_support()

    root_logger = get_logger("llm")
    root_logger.setLevel(logging.DEBUG)
    root_logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    console.addFilter(_RankFilter(rank))
    if use_color:
        console.setFormatter(_ColorFormatter(fmt=fmt, datefmt=datefmt))
    else:
        console.setFormatter(_PlainFormatter(fmt=fmt, datefmt=datefmt))
    root_logger.addHandler(console)

    if log_file is not None:
        file_fmt = (
            fmt
            or "%(asctime)s | %(levelname)-8s | rank=%(rank)s | %(name)s | %(message)s"
        )
        file_handler = logging.FileHandler(
            _get_ranked_log_file_path(log_file, current_rank),
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.addFilter(_InjectRankFilter())
        file_handler.setFormatter(_PlainFormatter(fmt=file_fmt, datefmt=datefmt))
        root_logger.addHandler(file_handler)

    _global_handler_configured = True


def reset_logger() -> None:
    """重置项目日志配置，供测试使用。"""

    global _global_handler_configured
    logger_dict = logging.Logger.manager.loggerDict
    llm_logger_names = [
        name
        for name in logger_dict
        if name == _ROOT_NAME or name.startswith(f"{_ROOT_NAME}.")
    ]
    for logger_name in [_ROOT_NAME, *llm_logger_names]:
        current_logger = cast(logging.Logger, logging.getLogger(logger_name))
        for handler in list(current_logger.handlers):
            current_logger.removeHandler(handler)
            handler.close()
        current_logger.setLevel(logging.NOTSET)
        current_logger.propagate = True
    _global_handler_configured = False


logger = get_logger("llm")

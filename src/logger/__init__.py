"""项目统一日志封装。

仅管理本项目 ``llm.*`` 命名空间下的日志：

1. 项目日志统一从 ``llm`` 根日志器输出。
2. ``llm`` 日志不向 Python root logger 传播，避免影响第三方库。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
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
            fmt = "%(asctime)s | %(color)s%(levelname)-8s%(reset)s | rank=%(rank)s | %(name)s | %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        record.color = _LEVEL_COLORS.get(record.levelno, "")  # type: ignore[attr-defined]
        record.reset = _RESET  # type: ignore[attr-defined]
        record.rank = _get_rank()  # type: ignore[attr-defined]
        return super().format(record)


class _PlainFormatter(logging.Formatter):
    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        if fmt is None:
            fmt = (
                "%(asctime)s | %(levelname)-8s | rank=%(rank)s | %(name)s | %(message)s"
            )
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        record.rank = _get_rank()  # type: ignore[attr-defined]
        return super().format(record)


def _get_rank() -> int:
    val = os.environ.get("RANK")
    if val is not None:
        try:
            return int(val)
        except ValueError:
            return 0
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
    path = Path(log_file)
    suffix = path.suffix or ".log"
    return str(path.with_name(f"{path.stem}_rank{rank}{suffix}"))


_ROOT_NAME = "llm"
_VERBOSITY_ENV = "LLM_VERBOSITY"

_global_handler_configured: bool = False
NewLogger = logging.Logger


def _resolve_log_level(level: str | int | None = None) -> int:
    """解析日志等级，默认读取 ``LLM_VERBOSITY``。"""

    if level is None:
        level = os.environ.get(_VERBOSITY_ENV, "INFO")
    if isinstance(level, int):
        return level
    level_name = level.upper()
    if level_name not in logging._nameToLevel:
        raise ValueError(f"Unknown logging level: {level}")
    return logging._nameToLevel[level_name]


def get_logger(name: str = "llm") -> logging.Logger:
    """获取项目日志对象。"""

    full_name = name if name.startswith(_ROOT_NAME) else f"{_ROOT_NAME}.{name}"
    return cast(logging.Logger, logging.getLogger(full_name))


def init_logger(
    level: str | int | None = None,
    *,
    log_file: str | None = None,
) -> None:
    """初始化项目根日志器。"""
    global _global_handler_configured
    if _global_handler_configured:
        return

    current_rank = _get_rank()

    numeric_level = _resolve_log_level(level)

    use_color = _detect_color_support()

    root_logger = get_logger("llm")
    root_logger.setLevel(logging.DEBUG)
    root_logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    if use_color:
        console.setFormatter(_ColorFormatter())
    else:
        console.setFormatter(_PlainFormatter())
    root_logger.addHandler(console)

    if log_file is not None:
        file_fmt = (
            "%(asctime)s | %(levelname)-8s | rank=%(rank)s | %(name)s | %(message)s"
        )
        file_handler = logging.FileHandler(
            _get_ranked_log_file_path(log_file, current_rank),
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(_PlainFormatter(fmt=file_fmt))
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

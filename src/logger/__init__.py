"""
Production-grade logging module for LLM training.

Inspired by HuggingFace Transformers, vLLM, and Megatron-LM patterns:
- Rank-aware: only rank 0 logs by default in distributed training
- Colorized console output with structured formatting
- log_once / log_every_n / log_rank utilities
- File handler support for distributed runs
- Zero external dependencies (stdlib only)

Usage:
    from logger import get_logger, init_logger

    init_logger(level="DEBUG", log_file="train.log")
    logger = get_logger("train")

    logger.info("Training started")
    logger.log_rank("Rank %d checkpoint saved", 0, rank=0)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any, cast

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
    def __init__(self, rank: int = 0) -> None:
        super().__init__()
        self.rank: int = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = getattr(record, "rank", self.rank)  # type: ignore[attr-defined]
        return getattr(record, "rank", 0) == self.rank


_log_once_set: set[int] = set()
_log_once_lock = threading.Lock()

_log_every_n_timestamps: dict[int, float] = {}
_log_every_n_lock = threading.Lock()


class NewLogger(logging.Logger):
    """
    Drop-in replacement for ``logging.Logger`` with training utilities:

    - ``log_rank(msg, rank)``       — only emit on a specific rank
    - ``log_once(msg, level)``      — emit exactly once across the entire run
    - ``log_every_n(msg, n, level)`` — emit at most once every *n* seconds
    """

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)

    def log_rank(
        self,
        msg: str,
        *args: Any,
        rank: int = 0,
        level: int = logging.INFO,
        **kwargs: Any,
    ) -> None:
        """Log *msg* only when the current process rank equals *rank*."""
        extra: dict[str, Any] = kwargs.pop("extra", None) or {}
        extra["rank"] = rank
        if _get_rank() == rank:
            self._log(level, msg, args, extra=extra)

    def log_once(
        self,
        msg: str,
        *args: Any,
        level: int = logging.INFO,
        **kwargs: Any,
    ) -> None:
        """Log *msg* exactly once (identity is the string hash)."""
        key = hash((self.name, msg))
        with _log_once_lock:
            if key in _log_once_set:
                return
            _log_once_set.add(key)
        self._log(level, msg, args, **kwargs)

    def log_every_n(
        self,
        msg: str,
        n: float = 10.0,
        *args: Any,
        level: int = logging.INFO,
        **kwargs: Any,
    ) -> None:
        """Log *msg* at most once every *n* seconds."""
        key = hash((self.name, msg))
        now = time.monotonic()
        with _log_every_n_lock:
            last = _log_every_n_timestamps.get(key, 0.0)
            if now - last < n:
                return
            _log_every_n_timestamps[key] = now
        self._log(level, msg, args, **kwargs)


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


_ROOT_NAME = "llm"

_loggers: dict[str, NewLogger] = {}
_global_handler_configured: bool = False


def get_logger(name: str = "llm") -> NewLogger:
    full_name = name if name.startswith(_ROOT_NAME) else f"{_ROOT_NAME}.{name}"
    if full_name in _loggers:
        return _loggers[full_name]

    logging.setLoggerClass(NewLogger)
    raw = logging.getLogger(full_name)
    if isinstance(raw, NewLogger):
        lg = raw
    else:
        raw.__class__ = NewLogger
        lg = cast(NewLogger, raw)
    _loggers[full_name] = lg
    return lg


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
    """
    Configure the root ``"llm"`` logger and all subsequently created loggers.

    Parameters
    ----------
    level:
        Console log level. Accepts ``"DEBUG"``, ``"INFO"``, etc. or int.
    log_file:
        Optional path to a log file. All ranks write to this file.
    log_file_level:
        Level for the file handler (defaults to *level*).
    rank:
        Process rank. Only this rank's messages appear on console.
        Auto-detected via PyTorch / env vars if ``None``.
    fmt:
        Custom format string. Supports ``%(color)s`` / ``%(reset)s`` placeholders.
    datefmt:
        Date format string (default ``"%Y-%m-%d %H:%M:%S"``).
    color:
        Force color on/off. ``None`` = auto-detect.
    """
    global _global_handler_configured
    if _global_handler_configured:
        return

    if rank is None:
        rank = _get_rank()

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

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    console.addFilter(_RankFilter(rank=rank))
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
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(_PlainFormatter(fmt=file_fmt, datefmt=datefmt))
        root_logger.addHandler(file_handler)

    _global_handler_configured = True


def reset_logger() -> None:
    global _global_handler_configured
    for lg in _loggers.values():
        lg.handlers.clear()
    _loggers.clear()
    _global_handler_configured = False
    with _log_once_lock:
        _log_once_set.clear()
    with _log_every_n_lock:
        _log_every_n_timestamps.clear()


logger = get_logger("llm")

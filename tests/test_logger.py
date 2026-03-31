import logging
import os
import time
from unittest.mock import patch

import pytest

from logger import NewLogger, get_logger, init_logger, reset_logger


@pytest.fixture(autouse=True)
def _clean_logger():
    yield
    reset_logger()


class TestGetLogger:
    def test_returns_new_logger_instance(self):
        lg = get_logger("train")
        assert isinstance(lg, NewLogger)

    def test_same_name_returns_same_instance(self):
        a = get_logger("train")
        b = get_logger("train")
        assert a is b

    def test_default_name_is_root(self):
        lg = get_logger()
        assert lg.name == "llm"

    def test_child_name_prefixed(self):
        lg = get_logger("data")
        assert lg.name == "llm.data"

    def test_explicit_root_name(self):
        lg = get_logger("llm")
        assert lg.name == "llm"


class TestInitLogger:
    def test_basic_output(self, capsys):
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.info("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.err

    def test_level_filtering(self, capsys):
        init_logger(level="WARNING", color=False)
        lg = get_logger("train")
        lg.debug("should not appear")
        lg.info("should not appear")
        lg.warning("should appear")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.err
        assert "should appear" in captured.err

    def test_init_twice_is_noop(self, capsys):
        init_logger(level="INFO", color=False)
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.debug("should not appear because second init is ignored")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.err

    def test_numeric_level(self, capsys):
        init_logger(level=logging.ERROR, color=False)
        lg = get_logger("train")
        lg.warning("hidden")
        lg.error("visible")
        captured = capsys.readouterr()
        assert "hidden" not in captured.err
        assert "visible" in captured.err

    def test_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        init_logger(level="DEBUG", log_file=str(log_file), color=False)
        lg = get_logger("train")
        lg.info("file output")
        assert log_file.exists()
        assert "file output" in log_file.read_text()

    def test_log_file_level(self, tmp_path):
        log_file = tmp_path / "test.log"
        init_logger(
            level="DEBUG",
            log_file=str(log_file),
            log_file_level="WARNING",
            color=False,
        )
        lg = get_logger("train")
        lg.info("console only")
        lg.warning("both")
        content = log_file.read_text()
        assert "console only" not in content
        assert "both" in content

    def test_custom_fmt(self, capsys):
        init_logger(level="INFO", fmt="%(message)s", datefmt="%H", color=False)
        lg = get_logger("train")
        lg.info("raw message")
        captured = capsys.readouterr()
        assert captured.err.strip() == "raw message"

    def test_color_formatter_injects_ansi(self, capsys):
        init_logger(level="INFO", color=True)
        lg = get_logger("train")
        lg.info("colored")
        captured = capsys.readouterr()
        assert "\033[" in captured.err


class TestLogRank:
    def test_rank_zero_logs_by_default(self, capsys):
        init_logger(level="DEBUG", rank=0, color=False)
        lg = get_logger("train")
        lg.log_rank("rank0 message", rank=0)
        captured = capsys.readouterr()
        assert "rank0 message" in captured.err

    def test_non_matching_rank_suppressed(self, capsys):
        init_logger(level="DEBUG", rank=0, color=False)
        lg = get_logger("train")
        lg.log_rank("rank1 message", rank=1)
        captured = capsys.readouterr()
        assert "rank1 message" not in captured.err

    def test_format_args(self, capsys):
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.log_rank("saved at rank %d", 0, rank=0)
        captured = capsys.readouterr()
        assert "saved at rank 0" in captured.err

    def test_env_rank(self, capsys):
        with patch.dict(os.environ, {"RANK": "3"}):
            reset_logger()
            init_logger(level="DEBUG", color=False)
            lg = get_logger("train")
            lg.log_rank("from rank 3", rank=3)
            lg.log_rank("from rank 0", rank=0)
            captured = capsys.readouterr()
            assert "from rank 3" in captured.err
            assert "from rank 0" not in captured.err


class TestLogOnce:
    def test_prints_exactly_once(self, capsys):
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.log_once("once")
        lg.log_once("once")
        lg.log_once("once")
        captured = capsys.readouterr()
        assert captured.err.count("once") == 1

    def test_different_messages_both_print(self, capsys):
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.log_once("msg_a")
        lg.log_once("msg_b")
        captured = capsys.readouterr()
        assert "msg_a" in captured.err
        assert "msg_b" in captured.err

    def test_reset_clears_once_state(self, capsys):
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.log_once("once")
        reset_logger()
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.log_once("once")
        captured = capsys.readouterr()
        assert captured.err.count("once") == 2


class TestLogEveryN:
    def test_first_call_always_prints(self, capsys):
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.log_every_n("throttled", 60.0)
        captured = capsys.readouterr()
        assert "throttled" in captured.err

    def test_suppressed_within_interval(self, capsys):
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.log_every_n("throttled", 60.0)
        lg.log_every_n("throttled", 60.0)
        captured = capsys.readouterr()
        assert captured.err.count("throttled") == 1

    def test_prints_after_interval(self, capsys):
        import logger as logger_mod

        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.log_every_n("interval", 0.0)
        logger_mod._log_every_n_timestamps.clear()
        lg.log_every_n("interval", 0.0)
        captured = capsys.readouterr()
        assert captured.err.count("interval") == 2


class TestResetLogger:
    def test_reset_allows_reinit(self, capsys):
        init_logger(level="WARNING", color=False)
        lg = get_logger("train")
        lg.debug("hidden")
        captured = capsys.readouterr()
        assert "hidden" not in captured.err

        reset_logger()
        init_logger(level="DEBUG", color=False)
        lg = get_logger("train")
        lg.debug("visible now")
        captured = capsys.readouterr()
        assert "visible now" in captured.err


class TestModuleLogger:
    def test_module_level_logger_exists(self):
        from logger import logger as mod_logger

        assert isinstance(mod_logger, NewLogger)
        assert mod_logger.name == "llm"

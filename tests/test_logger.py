import logging
import os
from unittest.mock import patch

import pytest

from logger import (
    NewLogger,
    _get_ranked_log_file_path,
    get_logger,
    init_logger,
    reset_logger,
)


@pytest.fixture(autouse=True)
def _clean_logger():
    yield
    reset_logger()


class TestGetLogger:
    def test_returns_standard_logger_instance(self):
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
        ranked_log_file = log_file.with_name(f"{log_file.name}.rank0")
        assert ranked_log_file.exists()
        assert "file output" in ranked_log_file.read_text()

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
        ranked_log_file = log_file.with_name(f"{log_file.name}.rank0")
        content = ranked_log_file.read_text()
        assert "console only" not in content
        assert "both" in content

    def test_log_file_writes_to_ranked_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        with patch.dict(os.environ, {"RANK": "3"}):
            init_logger(level="DEBUG", log_file=str(log_file), color=False)
            lg = get_logger("train")
            lg.info("ranked output")
        ranked_log_file = tmp_path / "test.log.rank3"
        assert ranked_log_file.exists()
        assert "rank=3" in ranked_log_file.read_text()
        assert _get_ranked_log_file_path(str(log_file), 3) == str(ranked_log_file)

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
    def test_matching_rank_logs(self, capsys):
        with patch.dict(os.environ, {"RANK": "3"}):
            reset_logger()
            init_logger(level="DEBUG", rank=3, color=False)
            lg = get_logger("train")
            lg.info("from rank 3")
            captured = capsys.readouterr()
            assert "from rank 3" in captured.err

    def test_non_matching_rank_suppressed(self, capsys):
        with patch.dict(os.environ, {"RANK": "3"}):
            reset_logger()
            init_logger(level="DEBUG", rank=0, color=False)
            lg = get_logger("train")
            lg.info("hidden")
            captured = capsys.readouterr()
            assert "hidden" not in captured.err

    def test_env_rank_used_by_default(self, capsys):
        with patch.dict(os.environ, {"RANK": "3"}):
            reset_logger()
            init_logger(level="DEBUG", color=False)
            lg = get_logger("train")
            lg.info("visible on env rank")
            captured = capsys.readouterr()
            assert "visible on env rank" in captured.err


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

import logging
import os
from unittest.mock import patch

import pytest

from logger import (
    NewLogger,
    _get_rank,
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
        init_logger(level="DEBUG")
        lg = get_logger("train")
        lg.info("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.err

    def test_level_filtering(self, capsys):
        init_logger(level="WARNING")
        lg = get_logger("train")
        lg.debug("should not appear")
        lg.info("should not appear")
        lg.warning("should appear")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.err
        assert "should appear" in captured.err

    def test_init_twice_is_noop(self, capsys):
        init_logger(level="INFO")
        init_logger(level="DEBUG")
        lg = get_logger("train")
        lg.debug("should not appear because second init is ignored")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.err

    def test_numeric_level(self, capsys):
        init_logger(level=logging.ERROR)
        lg = get_logger("train")
        lg.warning("hidden")
        lg.error("visible")
        captured = capsys.readouterr()
        assert "hidden" not in captured.err
        assert "visible" in captured.err

    def test_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        init_logger(level="DEBUG", log_file=str(log_file))
        lg = get_logger("train")
        lg.info("file output")
        ranked_log_file = tmp_path / "test_rank0.log"
        assert ranked_log_file.exists()
        assert "file output" in ranked_log_file.read_text()

    def test_log_file_writes_to_ranked_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        with patch.dict(os.environ, {"RANK": "3"}):
            init_logger(level="DEBUG", log_file=str(log_file))
            lg = get_logger("train")
            lg.info("ranked output")
        ranked_log_file = tmp_path / "test_rank3.log"
        assert ranked_log_file.exists()
        assert "rank=3" in ranked_log_file.read_text()
        assert _get_ranked_log_file_path(str(log_file), 3) == str(ranked_log_file)

    def test_level_can_be_read_from_env(self, capsys):
        with patch.dict(os.environ, {"LLM_VERBOSITY": "ERROR"}):
            init_logger()
            lg = get_logger("train")
            lg.warning("hidden by env")
            lg.error("visible by env")
        captured = capsys.readouterr()
        assert "hidden by env" not in captured.err
        assert "visible by env" in captured.err

    def test_project_logger_does_not_propagate_to_root(self):
        records = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        root_handler = CaptureHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(root_handler)
        try:
            init_logger(level="INFO")
            get_logger("train").info("project only")
        finally:
            root_logger.removeHandler(root_handler)

        assert records == []

    def test_external_logger_is_not_captured_by_project_logger(self, capsys):
        init_logger(level="INFO")

        external_logger = logging.getLogger("external_library")
        external_logger.propagate = False
        null_handler = logging.NullHandler()
        external_logger.addHandler(null_handler)
        try:
            external_logger.warning("external message")
        finally:
            external_logger.removeHandler(null_handler)

        captured = capsys.readouterr()
        assert "external message" not in captured.err

    def test_get_rank_only_reads_rank_env(self):
        with patch.dict(os.environ, {"RANK": "3", "LOCAL_RANK": "0"}):
            assert _get_rank() == 3

        with patch.dict(os.environ, {"LOCAL_RANK": "2"}, clear=True):
            assert _get_rank() == 0

        with patch.dict(os.environ, {"RANK": "invalid"}):
            assert _get_rank() == 0


class TestResetLogger:
    def test_reset_allows_reinit(self, capsys):
        init_logger(level="WARNING")
        lg = get_logger("train")
        lg.debug("hidden")
        captured = capsys.readouterr()
        assert "hidden" not in captured.err

        reset_logger()
        init_logger(level="DEBUG")
        lg = get_logger("train")
        lg.debug("visible now")
        captured = capsys.readouterr()
        assert "visible now" in captured.err


class TestModuleLogger:
    def test_module_level_logger_exists(self):
        from logger import logger as mod_logger

        assert isinstance(mod_logger, NewLogger)
        assert mod_logger.name == "llm"

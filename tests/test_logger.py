# Checks the shared logger in src/logger/logger.py. Run with: pytest -q tests/test_logger.py
import logging

from src.logger.logger import logger


def test_logger_is_a_logger_with_handlers():
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) >= 1


def test_logger_has_console_and_file_handler():
    kinds = {type(handler) for handler in logger.handlers}
    assert logging.StreamHandler in kinds
    assert logging.FileHandler in kinds


def test_logger_does_not_add_handlers_twice():
    # Importing the module again must not add more handlers
    import importlib
    import src.logger.logger as logger_module
    count = len(logger.handlers)
    importlib.reload(logger_module)
    assert len(logging.getLogger("churn").handlers) >= 1
    assert logger_module.logger is logging.getLogger("churn")
    assert count >= 1

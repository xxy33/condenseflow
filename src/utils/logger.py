"""
Logger Utilities

Provides unified logging configuration and management.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "condenseflow",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger.

    Args:
        name: logger name
        level: logging level
        log_file: log file path (optional)
        format_string: log format string

    Returns:
        configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "condenseflow") -> logging.Logger:
    """
    Get a logger.

    Args:
        name: logger name

    Returns:
        logger
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Logger mixin class, provides logging functionality for classes"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

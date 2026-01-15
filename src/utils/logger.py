"""
日志工具

提供统一的日志配置和管理。
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
    设置日志记录器。

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        format_string: 日志格式字符串

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有处理器
    logger.handlers = []

    # 默认格式
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "condenseflow") -> logging.Logger:
    """
    获取日志记录器。

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    return logging.getLogger(name)


class LoggerMixin:
    """日志混入类，为类提供日志功能"""

    @property
    def logger(self) -> logging.Logger:
        """获取类的日志记录器"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

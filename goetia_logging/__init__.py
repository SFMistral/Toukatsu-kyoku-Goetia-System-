# -*- coding: utf-8 -*-
"""
Logging 模块 - 统一日志管理中心

提供训练过程中的日志记录、指标追踪与日志分析功能。
支持多后端输出、分布式日志聚合、结构化日志格式。
"""

from .logger import Logger, LogLevel, get_logger, setup_logging, shutdown_logging
from .formatters import PlainFormatter, ColorFormatter, JsonFormatter, CompactFormatter
from .handlers import (
    StreamHandler, FileHandler, RotatingHandler, 
    BufferedHandler, AsyncHandler, MultiHandler, FilterHandler
)
from .writers import (
    WriterBase, FileWriter, ConsoleWriter, JsonWriter, create_writer
)
from .utils import LogParser, LogAggregator, parse_log_file, aggregate_metrics

# 装饰器
from .logger import log_function_call, log_execution_time

# 便捷函数
from .logger import log_metrics, log_hyperparams

__version__ = "1.0.0"
__author__ = "AI Training Framework Team"

__all__ = [
    # 核心类
    "Logger", "LogLevel", 
    
    # 工厂函数
    "get_logger", "setup_logging", "shutdown_logging",
    
    # 格式化器
    "PlainFormatter", "ColorFormatter", "JsonFormatter", "CompactFormatter",
    
    # 处理器
    "StreamHandler", "FileHandler", "RotatingHandler", 
    "BufferedHandler", "AsyncHandler", "MultiHandler", "FilterHandler",
    
    # 写入器
    "WriterBase", "FileWriter", "ConsoleWriter", "JsonWriter", "create_writer",
    
    # 工具类
    "LogParser", "LogAggregator", "parse_log_file", "aggregate_metrics",
    
    # 装饰器
    "log_function_call", "log_execution_time",
    
    # 便捷函数
    "log_metrics", "log_hyperparams",
]
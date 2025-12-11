# -*- coding: utf-8 -*-
"""
Logging Utils 模块 - 日志工具

提供日志解析、聚合、分析等工具功能。
"""

from .log_parser import LogParser
from .log_aggregator import LogAggregator

__all__ = [
    "LogParser",
    "LogAggregator", 
    "parse_log_file",
    "aggregate_metrics",
]


def parse_log_file(file_path: str, **kwargs):
    """解析日志文件的便捷函数"""
    parser = LogParser()
    return parser.parse_file(file_path, **kwargs)


def aggregate_metrics(log_dir: str, **kwargs):
    """聚合指标的便捷函数"""
    aggregator = LogAggregator()
    return aggregator.aggregate_directory(log_dir, **kwargs)
# -*- coding: utf-8 -*-
"""
Writers 模块 - 多后端写入器

提供各种日志写入后端，包括文件、控制台、JSON等格式的输出。
"""

from .base import WriterBase
from .file_writer import FileWriter
from .console_writer import ConsoleWriter
from .json_writer import JsonWriter

__all__ = [
    "WriterBase",
    "FileWriter", 
    "ConsoleWriter",
    "JsonWriter",
    "create_writer",
]


def create_writer(writer_type: str, config: dict) -> WriterBase:
    """创建写入器的工厂函数"""
    writers = {
        'file': FileWriter,
        'console': ConsoleWriter,
        'json': JsonWriter,
    }
    
    writer_class = writers.get(writer_type.lower())
    if writer_class is None:
        raise ValueError(f"Unknown writer type: {writer_type}")
        
    return writer_class(config)
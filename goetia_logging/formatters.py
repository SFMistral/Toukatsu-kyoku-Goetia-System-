# -*- coding: utf-8 -*-
"""
日志格式化器模块

定义各种日志输出格式，包括纯文本、彩色、JSON等格式。
"""

import json
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class FormatterBase(ABC):
    """格式化器基类"""
    
    def __init__(self, 
                 timestamp_format: str = "%Y-%m-%d %H:%M:%S",
                 include_fields: Optional[list] = None,
                 exclude_fields: Optional[list] = None):
        self.timestamp_format = timestamp_format
        self.include_fields = include_fields or []
        self.exclude_fields = exclude_fields or []
        
    @abstractmethod
    def format(self, record: Dict[str, Any]) -> str:
        """格式化日志记录"""
        pass
        
    def _format_timestamp(self, timestamp: float) -> str:
        """格式化时间戳"""
        return datetime.fromtimestamp(timestamp).strftime(self.timestamp_format)
        
    def _filter_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """过滤字段"""
        if self.include_fields:
            record = {k: v for k, v in record.items() if k in self.include_fields}
        if self.exclude_fields:
            record = {k: v for k, v in record.items() if k not in self.exclude_fields}
        return record
        
    def _sanitize_message(self, message: str, max_length: int = 1000) -> str:
        """清理和截断消息"""
        if len(message) > max_length:
            message = message[:max_length] + "..."
        return message


class PlainFormatter(FormatterBase):
    """纯文本格式化器"""
    
    def __init__(self, 
                 format_string: str = "{timestamp} [{level}] {logger_name}: {message}",
                 **kwargs):
        super().__init__(**kwargs)
        self.format_string = format_string
        
    def format(self, record: Dict[str, Any]) -> str:
        """格式化为纯文本"""
        record = self._filter_fields(record.copy())
        
        # 格式化时间戳
        if 'timestamp' in record:
            record['timestamp'] = self._format_timestamp(record['timestamp'])
            
        # 清理消息
        if 'message' in record:
            record['message'] = self._sanitize_message(record['message'])
            
        # 处理异常信息
        if 'exc_info' in record and record['exc_info']:
            record['message'] += f"\n{record['exc_info']}"
            
        try:
            return self.format_string.format(**record)
        except KeyError as e:
            # 如果格式字符串中的字段不存在，使用默认格式
            return f"{record.get('timestamp', '')} [{record.get('level', 'INFO')}] {record.get('message', '')}"


class ColorFormatter(PlainFormatter):
    """带ANSI颜色的格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[37m',       # 白色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[31;1m', # 红色加粗
        'RESET': '\033[0m'        # 重置
    }
    
    def __init__(self, use_colors: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_colors = use_colors and self._supports_color()
        
    def _supports_color(self) -> bool:
        """检测终端是否支持颜色"""
        import os
        import sys
        
        # 检查NO_COLOR环境变量
        if os.environ.get('NO_COLOR'):
            return False
            
        # 检查是否为TTY
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False
            
        # 检查TERM环境变量
        term = os.environ.get('TERM', '')
        if term in ('dumb', 'unknown'):
            return False
            
        return True
        
    def format(self, record: Dict[str, Any]) -> str:
        """格式化为带颜色的文本"""
        formatted = super().format(record)
        
        if not self.use_colors:
            return formatted
            
        level = record.get('level', 'INFO')
        color = self.COLORS.get(level, self.COLORS['INFO'])
        reset = self.COLORS['RESET']
        
        return f"{color}{formatted}{reset}"


class JsonFormatter(FormatterBase):
    """JSON格式化器"""
    
    def __init__(self, 
                 indent: Optional[int] = None,
                 ensure_ascii: bool = False,
                 timestamp_format: str = "iso",
                 **kwargs):
        super().__init__(**kwargs)
        self.indent = indent
        self.ensure_ascii = ensure_ascii
        self.timestamp_format = timestamp_format
        
    def format(self, record: Dict[str, Any]) -> str:
        """格式化为JSON"""
        record = self._filter_fields(record.copy())
        
        # 格式化时间戳
        if 'timestamp' in record:
            timestamp = record['timestamp']
            if self.timestamp_format == "iso":
                record['timestamp'] = datetime.fromtimestamp(timestamp).isoformat()
            elif self.timestamp_format == "unix":
                record['timestamp'] = int(timestamp)
            elif self.timestamp_format == "unix_ms":
                record['timestamp'] = int(timestamp * 1000)
            else:
                record['timestamp'] = self._format_timestamp(timestamp)
                
        # 确保所有值都可以JSON序列化
        record = self._make_json_serializable(record)
        
        try:
            return json.dumps(record, 
                            indent=self.indent, 
                            ensure_ascii=self.ensure_ascii,
                            separators=(',', ':') if self.indent is None else None)
        except (TypeError, ValueError) as e:
            # 如果序列化失败，返回错误信息
            error_record = {
                'timestamp': time.time(),
                'level': 'ERROR',
                'message': f'JSON serialization failed: {e}',
                'original_record': str(record)
            }
            return json.dumps(error_record)
            
    def _make_json_serializable(self, obj):
        """确保对象可以JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)


class CompactFormatter(FormatterBase):
    """精简格式化器"""
    
    def __init__(self, 
                 show_timestamp: bool = True,
                 show_level: bool = True,
                 show_logger: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.show_timestamp = show_timestamp
        self.show_level = show_level
        self.show_logger = show_logger
        
    def format(self, record: Dict[str, Any]) -> str:
        """格式化为精简文本"""
        parts = []
        
        if self.show_timestamp and 'timestamp' in record:
            timestamp = self._format_timestamp(record['timestamp'])
            # 只显示时间部分
            time_part = timestamp.split(' ')[-1]
            parts.append(time_part)
            
        if self.show_level and 'level' in record:
            level = record['level']
            # 使用缩写
            level_abbr = {
                'DEBUG': 'D',
                'INFO': 'I', 
                'WARNING': 'W',
                'ERROR': 'E',
                'CRITICAL': 'C'
            }.get(level, level[0])
            parts.append(f"[{level_abbr}]")
            
        if self.show_logger and 'logger_name' in record:
            logger_name = record['logger_name']
            # 只显示最后一部分
            short_name = logger_name.split('.')[-1]
            parts.append(f"{short_name}:")
            
        if 'message' in record:
            message = self._sanitize_message(record['message'], max_length=500)
            parts.append(message)
            
        return ' '.join(parts)


class TableFormatter(FormatterBase):
    """表格格式化器（用于指标汇总）"""
    
    def __init__(self, 
                 columns: list = None,
                 column_widths: Dict[str, int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns or ['step', 'loss', 'accuracy', 'lr']
        self.column_widths = column_widths or {}
        self.default_width = 12
        
    def format(self, record: Dict[str, Any]) -> str:
        """格式化为表格行"""
        if not isinstance(record, dict):
            return str(record)
            
        parts = []
        for col in self.columns:
            width = self.column_widths.get(col, self.default_width)
            value = record.get(col, '-')
            
            # 格式化数值
            if isinstance(value, float):
                if abs(value) < 0.001:
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)
                
            # 截断或填充到指定宽度
            if len(value_str) > width:
                value_str = value_str[:width-3] + "..."
            else:
                value_str = value_str.ljust(width)
                
            parts.append(value_str)
            
        return ' | '.join(parts)
        
    def format_header(self) -> str:
        """格式化表格头"""
        parts = []
        for col in self.columns:
            width = self.column_widths.get(col, self.default_width)
            header = col.upper().ljust(width)
            parts.append(header)
            
        header_line = ' | '.join(parts)
        separator_line = '-' * len(header_line)
        
        return f"{header_line}\n{separator_line}"


# 工厂函数
def create_formatter(formatter_type: str, **kwargs) -> FormatterBase:
    """创建格式化器"""
    formatters = {
        'plain': PlainFormatter,
        'color': ColorFormatter,
        'json': JsonFormatter,
        'compact': CompactFormatter,
        'table': TableFormatter,
    }
    
    formatter_class = formatters.get(formatter_type.lower())
    if formatter_class is None:
        raise ValueError(f"Unknown formatter type: {formatter_type}")
        
    return formatter_class(**kwargs)
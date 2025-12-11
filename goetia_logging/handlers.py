# -*- coding: utf-8 -*-
"""
日志处理器模块

管理日志路由与分发，提供各种处理策略如过滤、缓冲、异步等。
"""

import os
import sys
import time
import queue
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from collections import defaultdict, deque
from enum import Enum

from .formatters import FormatterBase, PlainFormatter


class LogLevel(Enum):
    """日志级别枚举（handlers模块内部使用）"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class HandlerBase(ABC):
    """处理器基类"""
    
    def __init__(self, 
                 level: str = "INFO",
                 formatter: Optional[FormatterBase] = None):
        self.level = getattr(LogLevel, level.upper())
        self.formatter = formatter or PlainFormatter()
        self.filters: List[Callable] = []
        
    def add_filter(self, filter_func: Callable[[Dict[str, Any]], bool]):
        """添加过滤函数"""
        self.filters.append(filter_func)
        
    def should_handle(self, record: Dict[str, Any]) -> bool:
        """判断是否应该处理该记录"""
        # 检查日志级别
        record_level = getattr(LogLevel, record.get('level', 'INFO'))
        if record_level.value < self.level.value:
            return False
            
        # 应用过滤器
        for filter_func in self.filters:
            if not filter_func(record):
                return False
                
        return True
        
    @abstractmethod
    def handle(self, record: Dict[str, Any]):
        """处理日志记录"""
        pass
        
    def close(self):
        """关闭处理器"""
        pass


class StreamHandler(HandlerBase):
    """流处理器"""
    
    def __init__(self, 
                 stream=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.stream = stream or sys.stdout
        self._lock = threading.Lock()
        
    def handle(self, record: Dict[str, Any]):
        """输出到流"""
        if not self.should_handle(record):
            return
            
        try:
            message = self.formatter.format(record)
            with self._lock:
                self.stream.write(message + '\n')
                self.stream.flush()
        except Exception as e:
            print(f"StreamHandler error: {e}", file=sys.stderr)


class FileHandler(HandlerBase):
    """文件处理器"""
    
    def __init__(self, 
                 filename: str,
                 mode: str = 'a',
                 encoding: str = 'utf-8',
                 **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.file = None
        self._lock = threading.Lock()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    def _open_file(self):
        """打开文件"""
        if self.file is None:
            self.file = open(self.filename, self.mode, encoding=self.encoding)
            
    def handle(self, record: Dict[str, Any]):
        """写入文件"""
        if not self.should_handle(record):
            return
            
        try:
            message = self.formatter.format(record)
            with self._lock:
                self._open_file()
                self.file.write(message + '\n')
                self.file.flush()
        except Exception as e:
            print(f"FileHandler error: {e}", file=sys.stderr)
            
    def close(self):
        """关闭文件"""
        with self._lock:
            if self.file:
                self.file.close()
                self.file = None


class RotatingHandler(FileHandler):
    """滚动文件处理器"""
    
    def __init__(self, 
                 filename: str,
                 max_size: int = 10 * 1024 * 1024,  # 10MB
                 max_files: int = 5,
                 rotation_type: str = 'size',  # 'size', 'time', 'both'
                 **kwargs):
        super().__init__(filename, **kwargs)
        self.max_size = max_size
        self.max_files = max_files
        self.rotation_type = rotation_type
        self.current_size = 0
        
        # 如果文件已存在，获取当前大小
        if os.path.exists(filename):
            self.current_size = os.path.getsize(filename)
            
    def _should_rotate(self) -> bool:
        """判断是否需要滚动"""
        if self.rotation_type in ('size', 'both'):
            if self.current_size >= self.max_size:
                return True
                
        if self.rotation_type in ('time', 'both'):
            # 简单的按日滚动实现
            if self.file:
                file_time = os.path.getctime(self.filename)
                current_day = time.strftime('%Y%m%d')
                file_day = time.strftime('%Y%m%d', time.localtime(file_time))
                if current_day != file_day:
                    return True
                    
        return False
        
    def _rotate_files(self):
        """执行文件滚动"""
        if self.file:
            self.file.close()
            self.file = None
            
        # 移动现有文件
        for i in range(self.max_files - 1, 0, -1):
            old_name = f"{self.filename}.{i}"
            new_name = f"{self.filename}.{i + 1}"
            if os.path.exists(old_name):
                if i == self.max_files - 1:
                    os.remove(old_name)  # 删除最老的文件
                else:
                    os.rename(old_name, new_name)
                    
        # 移动当前文件
        if os.path.exists(self.filename):
            os.rename(self.filename, f"{self.filename}.1")
            
        self.current_size = 0
        
    def handle(self, record: Dict[str, Any]):
        """处理日志记录，支持滚动"""
        if not self.should_handle(record):
            return
            
        try:
            message = self.formatter.format(record)
            message_size = len(message.encode(self.encoding)) + 1  # +1 for newline
            
            with self._lock:
                # 检查是否需要滚动
                if self._should_rotate():
                    self._rotate_files()
                    
                self._open_file()
                self.file.write(message + '\n')
                self.file.flush()
                self.current_size += message_size
                
        except Exception as e:
            print(f"RotatingHandler error: {e}", file=sys.stderr)


class BufferedHandler(HandlerBase):
    """缓冲处理器"""
    
    def __init__(self, 
                 target_handler: HandlerBase,
                 buffer_size: int = 100,
                 flush_interval: float = 5.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer: List[Dict[str, Any]] = []
        self.last_flush = time.time()
        self._lock = threading.Lock()
        
    def handle(self, record: Dict[str, Any]):
        """缓冲日志记录"""
        if not self.should_handle(record):
            return
            
        with self._lock:
            self.buffer.append(record)
            
            # 检查是否需要刷新
            should_flush = (
                len(self.buffer) >= self.buffer_size or
                time.time() - self.last_flush >= self.flush_interval
            )
            
            if should_flush:
                self._flush()
                
    def _flush(self):
        """刷新缓冲区"""
        if not self.buffer:
            return
            
        for record in self.buffer:
            try:
                self.target_handler.handle(record)
            except Exception as e:
                print(f"BufferedHandler flush error: {e}", file=sys.stderr)
                
        self.buffer.clear()
        self.last_flush = time.time()
        
    def close(self):
        """关闭处理器"""
        with self._lock:
            self._flush()
        self.target_handler.close()


class AsyncHandler(HandlerBase):
    """异步处理器"""
    
    def __init__(self, 
                 target_handler: HandlerBase,
                 queue_size: int = 1000,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_handler = target_handler
        self.queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self._start_worker()
        
    def _start_worker(self):
        """启动工作线程"""
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    def _worker(self):
        """工作线程主循环"""
        while not self.shutdown_event.is_set():
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:  # 关闭信号
                    break
                self.target_handler.handle(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"AsyncHandler worker error: {e}", file=sys.stderr)
                
    def handle(self, record: Dict[str, Any]):
        """异步处理日志记录"""
        if not self.should_handle(record):
            return
            
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            print("AsyncHandler queue full, dropping log record", file=sys.stderr)
            
    def close(self):
        """关闭异步处理器"""
        self.shutdown_event.set()
        self.queue.put(None)  # 发送关闭信号
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            
        self.target_handler.close()


class MultiHandler(HandlerBase):
    """多路分发处理器"""
    
    def __init__(self, handlers: List[HandlerBase] = None, **kwargs):
        super().__init__(**kwargs)
        self.handlers = handlers or []
        
    def add_handler(self, handler: HandlerBase):
        """添加处理器"""
        self.handlers.append(handler)
        
    def remove_handler(self, handler: HandlerBase):
        """移除处理器"""
        if handler in self.handlers:
            self.handlers.remove(handler)
            
    def handle(self, record: Dict[str, Any]):
        """分发到所有处理器"""
        for handler in self.handlers:
            try:
                handler.handle(record)
            except Exception as e:
                print(f"MultiHandler error: {e}", file=sys.stderr)
                
    def close(self):
        """关闭所有处理器"""
        for handler in self.handlers:
            try:
                handler.close()
            except Exception as e:
                print(f"MultiHandler close error: {e}", file=sys.stderr)


class FilterHandler(HandlerBase):
    """带过滤条件的处理器"""
    
    def __init__(self, 
                 target_handler: HandlerBase,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_handler = target_handler
        self.keyword_filters: List[str] = []
        self.regex_filters: List = []
        self.level_filter: Optional[str] = None
        self.logger_name_filter: Optional[str] = None
        
    def add_keyword_filter(self, keyword: str):
        """添加关键词过滤"""
        self.keyword_filters.append(keyword)
        
    def add_regex_filter(self, pattern: str):
        """添加正则表达式过滤"""
        import re
        self.regex_filters.append(re.compile(pattern))
        
    def set_level_filter(self, level: str):
        """设置级别过滤"""
        self.level_filter = level
        
    def set_logger_name_filter(self, name_prefix: str):
        """设置logger名称过滤"""
        self.logger_name_filter = name_prefix
        
    def should_handle(self, record: Dict[str, Any]) -> bool:
        """应用自定义过滤规则"""
        if not super().should_handle(record):
            return False
            
        message = record.get('message', '')
        
        # 关键词过滤
        for keyword in self.keyword_filters:
            if keyword in message:
                return False
                
        # 正则表达式过滤
        for regex in self.regex_filters:
            if regex.search(message):
                return False
                
        # Logger名称过滤
        if self.logger_name_filter:
            logger_name = record.get('logger_name', '')
            if not logger_name.startswith(self.logger_name_filter):
                return False
                
        return True
        
    def handle(self, record: Dict[str, Any]):
        """处理过滤后的记录"""
        if self.should_handle(record):
            self.target_handler.handle(record)
            
    def close(self):
        """关闭目标处理器"""
        self.target_handler.close()


class RateLimitHandler(HandlerBase):
    """限流处理器"""
    
    def __init__(self, 
                 target_handler: HandlerBase,
                 max_rate: float = 10.0,  # 每秒最大记录数
                 window_size: float = 1.0,  # 时间窗口大小
                 **kwargs):
        super().__init__(**kwargs)
        self.target_handler = target_handler
        self.max_rate = max_rate
        self.window_size = window_size
        self.timestamps = deque()
        self._lock = threading.Lock()
        
    def handle(self, record: Dict[str, Any]):
        """限流处理"""
        if not self.should_handle(record):
            return
            
        current_time = time.time()
        
        with self._lock:
            # 清理过期的时间戳
            while self.timestamps and current_time - self.timestamps[0] > self.window_size:
                self.timestamps.popleft()
                
            # 检查是否超过限制
            if len(self.timestamps) >= self.max_rate * self.window_size:
                return  # 丢弃记录
                
            self.timestamps.append(current_time)
            
        self.target_handler.handle(record)
        
    def close(self):
        """关闭目标处理器"""
        self.target_handler.close()
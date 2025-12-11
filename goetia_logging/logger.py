# -*- coding: utf-8 -*-
"""
核心日志器模块

提供统一的日志记录接口，支持分级日志、指标记录、上下文管理等功能。
"""

import os
import sys
import time
import threading
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from contextlib import contextmanager
from functools import wraps

from .handlers import MultiHandler
from .formatters import ColorFormatter, PlainFormatter
from .writers import ConsoleWriter, FileWriter, JsonWriter


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class Logger:
    """核心日志器类"""
    
    def __init__(self, name: str = "root"):
        self.name = name
        self.level = LogLevel.INFO
        self.handlers: List = []
        self.writers: List = []
        
        # 上下文状态
        self._context_stack = []
        self._current_epoch = None
        self._current_step = None
        self._current_phase = None
        
        # 分布式相关
        self._rank = 0
        self._world_size = 1
        self._log_all_ranks = False
        
        # 线程安全
        self._lock = threading.RLock()
        
    def set_level(self, level: Union[LogLevel, str, int]):
        """设置日志级别"""
        if isinstance(level, str):
            level = getattr(LogLevel, level.upper())
        elif isinstance(level, int):
            for log_level in LogLevel:
                if log_level.value == level:
                    level = log_level
                    break
        self.level = level
        
    def add_handler(self, handler):
        """添加日志处理器"""
        with self._lock:
            self.handlers.append(handler)
            
    def add_writer(self, writer):
        """添加日志写入器"""
        with self._lock:
            self.writers.append(writer)
            
    def remove_handler(self, handler):
        """移除日志处理器"""
        with self._lock:
            if handler in self.handlers:
                self.handlers.remove(handler)
                
    def remove_writer(self, writer):
        """移除日志写入器"""
        with self._lock:
            if writer in self.writers:
                self.writers.remove(writer)
                
    def set_distributed_config(self, rank: int = 0, world_size: int = 1, log_all_ranks: bool = False):
        """设置分布式配置"""
        self._rank = rank
        self._world_size = world_size
        self._log_all_ranks = log_all_ranks
        
    def _should_log(self, level: LogLevel) -> bool:
        """判断是否应该记录日志"""
        # 检查日志级别
        if level.value < self.level.value:
            return False
            
        # 分布式环境下，默认只有主进程记录
        if not self._log_all_ranks and self._rank != 0:
            return False
            
        return True
        
    def _format_message(self, message: str, **kwargs) -> str:
        """格式化日志消息"""
        # 添加上下文信息
        context_parts = []
        
        if self._current_phase:
            context_parts.append(f"[{self._current_phase}]")
            
        if self._current_epoch is not None:
            context_parts.append(f"Epoch {self._current_epoch}")
            
        if self._current_step is not None:
            context_parts.append(f"Step {self._current_step}")
            
        if self._rank > 0:
            context_parts.append(f"Rank {self._rank}")
            
        context_str = " ".join(context_parts)
        if context_str:
            message = f"{context_str} | {message}"
            
        return message
        
    def _log(self, level: LogLevel, message: str, **kwargs):
        """内部日志记录方法"""
        if not self._should_log(level):
            return
            
        with self._lock:
            formatted_message = self._format_message(message, **kwargs)
            
            # 创建日志记录
            record = {
                'timestamp': time.time(),
                'level': level.name,
                'logger_name': self.name,
                'message': formatted_message,
                'rank': self._rank,
                'epoch': self._current_epoch,
                'step': self._current_step,
                'phase': self._current_phase,
                **kwargs
            }
            
            # 分发到处理器
            for handler in self.handlers:
                try:
                    handler.handle(record)
                except Exception as e:
                    print(f"Handler error: {e}", file=sys.stderr)
                    
            # 分发到写入器
            for writer in self.writers:
                try:
                    writer.write_text(record)
                except Exception as e:
                    print(f"Writer error: {e}", file=sys.stderr)
                    
    # 基础日志方法
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self._log(LogLevel.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self._log(LogLevel.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self._log(LogLevel.WARNING, message, **kwargs)
        
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """记录错误日志"""
        if exc_info:
            import traceback
            kwargs['exc_info'] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self._log(LogLevel.CRITICAL, message, **kwargs)
        
    # 指标记录方法
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """记录标量指标"""
        if not self._should_log(LogLevel.INFO):
            return
            
        if step is None:
            step = self._current_step
            
        with self._lock:
            for writer in self.writers:
                try:
                    writer.write_scalar(tag, value, step)
                except Exception as e:
                    print(f"Writer error in log_scalar: {e}", file=sys.stderr)
                    
    def log_scalars(self, tag_value_dict: Dict[str, float], step: Optional[int] = None):
        """记录多个标量指标"""
        if not self._should_log(LogLevel.INFO):
            return
            
        if step is None:
            step = self._current_step
            
        with self._lock:
            for writer in self.writers:
                try:
                    writer.write_scalars(tag_value_dict, step)
                except Exception as e:
                    print(f"Writer error in log_scalars: {e}", file=sys.stderr)
                    
    def log_histogram(self, tag: str, values: List[float], step: Optional[int] = None):
        """记录直方图数据"""
        if not self._should_log(LogLevel.INFO):
            return
            
        if step is None:
            step = self._current_step
            
        with self._lock:
            for writer in self.writers:
                try:
                    writer.write_histogram(tag, values, step)
                except Exception as e:
                    print(f"Writer error in log_histogram: {e}", file=sys.stderr)
                    
    def log_hyperparams(self, params: Dict[str, Any]):
        """记录超参数"""
        if not self._should_log(LogLevel.INFO):
            return
            
        with self._lock:
            for writer in self.writers:
                try:
                    writer.write_hyperparams(params)
                except Exception as e:
                    print(f"Writer error in log_hyperparams: {e}", file=sys.stderr)
                    
    # 上下文管理器
    @contextmanager
    def train(self):
        """训练阶段上下文"""
        old_phase = self._current_phase
        self._current_phase = "train"
        try:
            yield
        finally:
            self._current_phase = old_phase
            
    @contextmanager
    def val(self):
        """验证阶段上下文"""
        old_phase = self._current_phase
        self._current_phase = "val"
        try:
            yield
        finally:
            self._current_phase = old_phase
            
    @contextmanager
    def test(self):
        """测试阶段上下文"""
        old_phase = self._current_phase
        self._current_phase = "test"
        try:
            yield
        finally:
            self._current_phase = old_phase
            
    @contextmanager
    def epoch(self, epoch_num: int):
        """Epoch上下文"""
        old_epoch = self._current_epoch
        self._current_epoch = epoch_num
        try:
            yield
        finally:
            self._current_epoch = old_epoch
            
    @contextmanager
    def step(self, step_num: int):
        """Step上下文"""
        old_step = self._current_step
        self._current_step = step_num
        try:
            yield
        finally:
            self._current_step = old_step
            
    @contextmanager
    def timer(self, name: str):
        """计时上下文"""
        start_time = time.time()
        self.info(f"Timer '{name}' started")
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.info(f"Timer '{name}' finished: {elapsed:.3f}s")
            
    def flush(self):
        """刷新所有写入器"""
        with self._lock:
            for writer in self.writers:
                try:
                    writer.flush()
                except Exception as e:
                    print(f"Writer flush error: {e}", file=sys.stderr)
                    
    def close(self):
        """关闭日志器"""
        with self._lock:
            for writer in self.writers:
                try:
                    writer.close()
                except Exception as e:
                    print(f"Writer close error: {e}", file=sys.stderr)


# 全局日志器管理
_loggers: Dict[str, Logger] = {}
_default_logger: Optional[Logger] = None
_setup_done = False


def get_logger(name: Optional[str] = None) -> Logger:
    """获取命名日志器"""
    global _loggers, _default_logger
    
    if name is None:
        if _default_logger is None:
            _default_logger = Logger("root")
        return _default_logger
        
    if name not in _loggers:
        _loggers[name] = Logger(name)
        
    return _loggers[name]


def setup_logging(config: Dict[str, Any]):
    """初始化日志系统"""
    global _setup_done
    
    if _setup_done:
        return
        
    logger = get_logger()
    
    # 设置日志级别
    if 'level' in config:
        logger.set_level(config['level'])
        
    # 设置分布式配置
    if 'distributed' in config:
        dist_config = config['distributed']
        logger.set_distributed_config(
            rank=dist_config.get('rank', 0),
            world_size=dist_config.get('world_size', 1),
            log_all_ranks=dist_config.get('log_all_ranks', False)
        )
        
    # 配置控制台输出
    if config.get('console', {}).get('enabled', True):
        console_config = config.get('console', {})
        console_writer = ConsoleWriter(console_config)
        logger.add_writer(console_writer)
        
    # 配置文件输出
    if config.get('file', {}).get('enabled', False):
        file_config = config.get('file', {})
        file_writer = FileWriter(file_config)
        logger.add_writer(file_writer)
        
    # 配置JSON输出
    if config.get('json', {}).get('enabled', False):
        json_config = config.get('json', {})
        json_writer = JsonWriter(json_config)
        logger.add_writer(json_writer)
        
    _setup_done = True


def shutdown_logging():
    """关闭日志系统"""
    global _loggers, _default_logger, _setup_done
    
    if _default_logger:
        _default_logger.close()
        
    for logger in _loggers.values():
        logger.close()
        
    _loggers.clear()
    _default_logger = None
    _setup_done = False


# 装饰器
def log_function_call(logger_name: Optional[str] = None):
    """记录函数调用的装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            logger.debug(f"Calling function: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_execution_time(logger_name: Optional[str] = None):
    """记录函数执行时间的装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"Function {func.__name__} executed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator


# 便捷函数
def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """快捷记录指标"""
    logger = get_logger()
    logger.log_scalars(metrics, step)


def log_hyperparams(params: Dict[str, Any]):
    """快捷记录超参数"""
    logger = get_logger()
    logger.log_hyperparams(params)
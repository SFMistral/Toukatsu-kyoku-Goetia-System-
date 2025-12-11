# -*- coding: utf-8 -*-
"""
写入器基类

定义所有写入器的通用接口和基础功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class WriterBase(ABC):
    """写入器抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_open = False
        
    @abstractmethod
    def open(self):
        """打开写入器"""
        pass
        
    @abstractmethod
    def close(self):
        """关闭写入器"""
        pass
        
    @abstractmethod
    def write_text(self, record: Dict[str, Any]):
        """写入文本日志"""
        pass
        
    @abstractmethod
    def write_scalar(self, tag: str, value: float, step: int):
        """写入标量指标"""
        pass
        
    @abstractmethod
    def write_scalars(self, tag_value_dict: Dict[str, float], step: int):
        """写入多个标量指标"""
        pass
        
    @abstractmethod
    def write_histogram(self, tag: str, values: List[float], step: int):
        """写入直方图数据"""
        pass
        
    @abstractmethod
    def write_hyperparams(self, params: Dict[str, Any]):
        """写入超参数"""
        pass
        
    @abstractmethod
    def flush(self):
        """刷新缓冲区"""
        pass
        
    def __enter__(self):
        """上下文管理器入口"""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
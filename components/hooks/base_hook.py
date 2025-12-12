# -*- coding: utf-8 -*-
"""
钩子基类模块

定义所有钩子的抽象基类，提供生命周期接口和辅助方法。
"""

from abc import ABC
from enum import IntEnum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class HookPriority(IntEnum):
    """钩子优先级（数值越小优先级越高）"""
    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 30
    ABOVE_NORMAL = 40
    NORMAL = 50
    BELOW_NORMAL = 60
    LOW = 70
    VERY_LOW = 90
    LOWEST = 100


class BaseHook(ABC):
    """
    所有钩子的抽象基类
    
    定义钩子生命周期接口，提供默认空实现。
    子类可以选择性地覆盖需要的方法。
    
    Attributes:
        priority: 钩子优先级，数值越小越先执行
        runner: 训练器引用
        
    Example:
        >>> class MyHook(BaseHook):
        ...     def after_train_iter(self, runner):
        ...         print(f"Iter {runner.iter} completed")
    """
    
    priority: int = HookPriority.NORMAL
    
    def __init__(self, priority: Optional[int] = None):
        """
        初始化钩子
        
        Args:
            priority: 钩子优先级，None 则使用类默认值
        """
        if priority is not None:
            self.priority = priority
        self._runner = None
        
    @property
    def runner(self) -> 'Any':
        """获取训练器引用"""
        return self._runner
        
    @runner.setter
    def runner(self, value: 'Any'):
        """设置训练器引用"""
        self._runner = value
        
    # ============== 生命周期方法 ==============
    
    def before_run(self, runner: 'Any') -> None:
        """
        训练开始前调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def after_run(self, runner: 'Any') -> None:
        """
        训练结束后调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def before_epoch(self, runner: 'Any') -> None:
        """
        每个 epoch 开始前调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def after_epoch(self, runner: 'Any') -> None:
        """
        每个 epoch 结束后调用
        
        Args:
            runner: 训练器实例
        """
        pass

    def before_train_epoch(self, runner: 'Any') -> None:
        """
        训练 epoch 开始前调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def after_train_epoch(self, runner: 'Any') -> None:
        """
        训练 epoch 结束后调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def before_val_epoch(self, runner: 'Any') -> None:
        """
        验证 epoch 开始前调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def after_val_epoch(self, runner: 'Any') -> None:
        """
        验证 epoch 结束后调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def before_train_iter(self, runner: 'Any') -> None:
        """
        训练迭代开始前调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def after_train_iter(self, runner: 'Any') -> None:
        """
        训练迭代结束后调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def before_val_iter(self, runner: 'Any') -> None:
        """
        验证迭代开始前调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def after_val_iter(self, runner: 'Any') -> None:
        """
        验证迭代结束后调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def before_save_checkpoint(self, runner: 'Any') -> None:
        """
        保存检查点前调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    def after_load_checkpoint(self, runner: 'Any') -> None:
        """
        加载检查点后调用
        
        Args:
            runner: 训练器实例
        """
        pass
        
    # ============== 辅助方法 ==============
    
    def every_n_epochs(self, runner: 'Any', n: int) -> bool:
        """
        判断是否每 n 个 epoch
        
        Args:
            runner: 训练器实例
            n: 间隔 epoch 数
            
        Returns:
            是否满足条件
        """
        return (runner.epoch + 1) % n == 0 if n > 0 else False
        
    def every_n_iters(self, runner: 'Any', n: int) -> bool:
        """
        判断是否每 n 个迭代
        
        Args:
            runner: 训练器实例
            n: 间隔迭代数
            
        Returns:
            是否满足条件
        """
        return (runner.iter + 1) % n == 0 if n > 0 else False
        
    def is_last_epoch(self, runner: 'Any') -> bool:
        """
        判断是否最后一个 epoch
        
        Args:
            runner: 训练器实例
            
        Returns:
            是否是最后一个 epoch
        """
        return runner.epoch + 1 == runner.max_epochs
        
    def is_last_iter(self, runner: 'Any') -> bool:
        """
        判断是否最后一个迭代
        
        Args:
            runner: 训练器实例
            
        Returns:
            是否是最后一个迭代
        """
        return runner.iter + 1 == runner.max_iters
        
    def get_triggered_stages(self) -> List[str]:
        """
        获取钩子触发的阶段列表
        
        Returns:
            触发阶段名称列表
        """
        stages = []
        base_methods = set(dir(BaseHook))
        
        for name in dir(self):
            if name.startswith('before_') or name.startswith('after_'):
                method = getattr(self, name)
                # 检查方法是否被覆盖
                if callable(method) and name in base_methods:
                    base_method = getattr(BaseHook, name, None)
                    if base_method is not None and method.__func__ is not base_method:
                        stages.append(name)
                        
        return stages
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(priority={self.priority})"

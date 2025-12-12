# -*- coding: utf-8 -*-
"""
早停钩子模块

提供早停机制，监控指标变化并在满足条件时停止训练。
"""

import math
from typing import Dict, Any, Optional, TYPE_CHECKING

import torch

from .base_hook import BaseHook, HookPriority
from registry import HOOKS

if TYPE_CHECKING:
    from typing import Any as RunnerType


class EarlyStoppingHook(BaseHook):
    """
    早停钩子
    
    监控指标变化，判断是否停止训练，支持 patience 机制。
    
    Args:
        monitor: 监控的指标名
        patience: 容忍次数
        min_delta: 最小改善量
        mode: 模式 ('min', 'max', 'auto')
        baseline: 基准值
        restore_best_weights: 是否恢复最佳权重
        check_finite: 检查是否为有限值
        stopping_threshold: 达到阈值立即停止
        divergence_threshold: 发散阈值立即停止
        
    Example:
        >>> hook = EarlyStoppingHook(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     mode='min'
        ... )
    """
    
    priority = HookPriority.VERY_LOW
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'auto',
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        
        # 状态追踪
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.wait_count: int = 0
        self.stopped_epoch: int = 0
        self._best_weights: Optional[Dict[str, torch.Tensor]] = None
        
        # 自动推断模式
        self._mode_func = self._get_mode_func()
        
    def _get_mode_func(self):
        """获取模式比较函数"""
        if self.mode == 'auto':
            # 根据指标名自动推断
            if any(x in self.monitor.lower() for x in ['loss', 'error', 'mse', 'mae']):
                self.mode = 'min'
            else:
                self.mode = 'max'
                
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:  # max
            return lambda current, best: current > best + self.min_delta
            
    def before_run(self, runner: 'RunnerType') -> None:
        """训练开始前初始化"""
        self.best_score = self.baseline
        self.best_epoch = 0
        self.wait_count = 0
        self.stopped_epoch = 0
        self._best_weights = None
        
    def after_val_epoch(self, runner: 'RunnerType') -> None:
        """验证结束后检查是否早停"""
        # 获取当前指标值
        current_score = self._get_metric_value(runner)
        if current_score is None:
            return
            
        # 检查是否为有限值
        if self.check_finite and not self._is_finite(current_score):
            print(f"Early stopping: {self.monitor} is not finite ({current_score})")
            self._stop_training(runner)
            return
            
        # 检查发散阈值
        if self._check_divergence(current_score):
            print(f"Early stopping: {self.monitor} diverged ({current_score})")
            self._stop_training(runner)
            return
            
        # 检查停止阈值
        if self._check_stopping_threshold(current_score):
            print(f"Early stopping: {self.monitor} reached threshold ({current_score})")
            self._stop_training(runner)
            return
            
        # 检查是否改善
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.best_epoch = runner.epoch
            self.wait_count = 0
            
            # 保存最佳权重
            if self.restore_best_weights:
                self._save_best_weights(runner)
        else:
            self.wait_count += 1
            
        # 检查是否超过 patience
        if self.wait_count >= self.patience:
            print(f"Early stopping triggered at epoch {runner.epoch + 1}")
            print(f"Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch + 1}")
            self._stop_training(runner)
            
    def _get_metric_value(self, runner: 'RunnerType') -> Optional[float]:
        """获取指标值"""
        # 尝试从 val_outputs 获取
        if hasattr(runner, 'val_outputs') and runner.val_outputs:
            value = runner.val_outputs.get(self.monitor)
            if value is not None:
                return float(value)
                
        # 尝试从 log_buffer 获取
        if hasattr(runner, 'log_buffer'):
            value = runner.log_buffer.get(self.monitor)
            if value is not None:
                return float(value)
                
        # 尝试直接从 runner 获取
        value = getattr(runner, self.monitor, None)
        if value is not None:
            return float(value)
            
        return None
        
    def _is_finite(self, value: float) -> bool:
        """检查值是否有限"""
        return not (math.isnan(value) or math.isinf(value))
        
    def _is_improvement(self, current: float) -> bool:
        """判断是否有改善"""
        if self.best_score is None:
            return True
        return self._mode_func(current, self.best_score)
        
    def _check_divergence(self, current: float) -> bool:
        """检查是否发散"""
        if self.divergence_threshold is None:
            return False
            
        if self.mode == 'min':
            return current >= self.divergence_threshold
        else:
            return current <= self.divergence_threshold
            
    def _check_stopping_threshold(self, current: float) -> bool:
        """检查是否达到停止阈值"""
        if self.stopping_threshold is None:
            return False
            
        if self.mode == 'min':
            return current <= self.stopping_threshold
        else:
            return current >= self.stopping_threshold
            
    def _save_best_weights(self, runner: 'RunnerType') -> None:
        """保存最佳权重"""
        self._best_weights = {
            k: v.cpu().clone() for k, v in runner.model.state_dict().items()
        }
        
    def _restore_best_weights(self, runner: 'RunnerType') -> None:
        """恢复最佳权重"""
        if self._best_weights is not None:
            runner.model.load_state_dict(self._best_weights)
            print(f"Restored best weights from epoch {self.best_epoch + 1}")
            
    def _stop_training(self, runner: 'RunnerType') -> None:
        """停止训练"""
        self.stopped_epoch = runner.epoch
        runner.should_stop = True
        
        if self.restore_best_weights:
            self._restore_best_weights(runner)
            
    def state_dict(self) -> Dict[str, Any]:
        """获取钩子状态"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载钩子状态"""
        self.best_score = state_dict.get('best_score')
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.wait_count = state_dict.get('wait_count', 0)
        self.stopped_epoch = state_dict.get('stopped_epoch', 0)


# 注册钩子（如果尚未注册）
if not HOOKS.contains('EarlyStoppingHook'):
    HOOKS.register('EarlyStoppingHook', EarlyStoppingHook, 
                   priority=HookPriority.VERY_LOW, category='control')
else:
    HOOKS._components['EarlyStoppingHook'].cls = EarlyStoppingHook

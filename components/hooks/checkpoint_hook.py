# -*- coding: utf-8 -*-
"""
检查点钩子模块

提供检查点保存与管理功能，支持定期保存、最佳模型保存、断点续训。
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING

import torch

from .base_hook import BaseHook, HookPriority
from registry import HOOKS

if TYPE_CHECKING:
    from typing import Any as RunnerType


class CheckpointHook(BaseHook):
    """
    检查点保存钩子
    
    定期保存模型检查点，管理检查点数量，支持保存最佳模型。
    
    Args:
        interval: 保存间隔（epoch 或 iter）
        by_epoch: 按 epoch 还是 iter 保存
        save_optimizer: 是否保存优化器状态
        save_scheduler: 是否保存调度器状态
        max_keep_ckpts: 最大保留检查点数量
        save_last: 是否保存最后一个检查点
        save_best: 是否保存最佳模型
        best_metric: 最佳模型判断指标
        rule: 比较规则 ('greater' 或 'less')
        out_dir: 保存目录，None 则使用工作目录
        filename_tmpl: 文件名模板
        
    Example:
        >>> hook = CheckpointHook(
        ...     interval=1,
        ...     max_keep_ckpts=3,
        ...     save_best=True,
        ...     best_metric='accuracy'
        ... )
    """
    
    priority = HookPriority.BELOW_NORMAL
    
    def __init__(
        self,
        interval: int = 1,
        by_epoch: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        max_keep_ckpts: int = 5,
        save_last: bool = True,
        save_best: bool = True,
        best_metric: str = 'accuracy',
        rule: str = 'greater',
        out_dir: Optional[str] = None,
        filename_tmpl: str = 'epoch_{}.pth',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.save_best = save_best
        self.best_metric = best_metric
        self.rule = rule
        self.out_dir = out_dir
        self.filename_tmpl = filename_tmpl
        
        # 状态追踪
        self._saved_ckpts: List[str] = []
        self._best_score: Optional[float] = None
        self._best_ckpt: Optional[str] = None
        
        # 验证规则
        if rule not in ('greater', 'less'):
            raise ValueError(f"rule must be 'greater' or 'less', got {rule}")
            
    def before_run(self, runner: 'RunnerType') -> None:
        """训练开始前初始化保存目录"""
        if self.out_dir is None:
            self.out_dir = getattr(runner, 'work_dir', './checkpoints')
        os.makedirs(self.out_dir, exist_ok=True)
        
    def after_train_epoch(self, runner: 'RunnerType') -> None:
        """训练 epoch 结束后保存检查点"""
        if not self.by_epoch:
            return
            
        if self.every_n_epochs(runner, self.interval) or self.is_last_epoch(runner):
            self._save_checkpoint(runner)
            
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """训练迭代结束后保存检查点（按 iter 模式）"""
        if self.by_epoch:
            return
            
        if self.every_n_iters(runner, self.interval) or self.is_last_iter(runner):
            self._save_checkpoint(runner)
            
    def after_val_epoch(self, runner: 'RunnerType') -> None:
        """验证结束后检查是否保存最佳模型"""
        if not self.save_best:
            return
            
        # 获取当前指标值
        current_score = self._get_metric_value(runner)
        if current_score is None:
            return
            
        # 判断是否是最佳
        is_best = self._is_better(current_score)
        if is_best:
            self._best_score = current_score
            self._save_best_checkpoint(runner)
            
    def before_save_checkpoint(self, runner: 'RunnerType') -> None:
        """保存检查点前的回调"""
        pass

    def _save_checkpoint(self, runner: 'RunnerType') -> None:
        """保存检查点"""
        # 调用保存前回调
        self.before_save_checkpoint(runner)
        
        # 构建检查点内容
        checkpoint = self._build_checkpoint(runner)
        
        # 生成文件名
        if self.by_epoch:
            filename = self.filename_tmpl.format(runner.epoch + 1)
        else:
            filename = f'iter_{runner.iter + 1}.pth'
            
        filepath = os.path.join(self.out_dir, filename)
        
        # 保存检查点
        torch.save(checkpoint, filepath)
        self._saved_ckpts.append(filepath)
        
        # 保存 last 链接
        if self.save_last:
            last_path = os.path.join(self.out_dir, 'last.pth')
            if os.path.exists(last_path):
                os.remove(last_path)
            # 复制而不是软链接，确保跨平台兼容
            torch.save(checkpoint, last_path)
            
        # 清理旧检查点
        self._cleanup_checkpoints()
        
    def _save_best_checkpoint(self, runner: 'RunnerType') -> None:
        """保存最佳检查点"""
        checkpoint = self._build_checkpoint(runner)
        checkpoint['best_metric'] = self.best_metric
        checkpoint['best_score'] = self._best_score
        
        best_path = os.path.join(self.out_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        self._best_ckpt = best_path
        
        print(f"Saved best checkpoint with {self.best_metric}={self._best_score:.4f}")
        
    def _build_checkpoint(self, runner: 'RunnerType') -> Dict[str, Any]:
        """构建检查点字典"""
        checkpoint = {
            'meta': {
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': runner.epoch + 1,
                'iter': runner.iter + 1,
            },
            'state_dict': runner.model.state_dict(),
        }
        
        # 保存优化器状态
        if self.save_optimizer and hasattr(runner, 'optimizer'):
            checkpoint['optimizer'] = runner.optimizer.state_dict()
            
        # 保存调度器状态
        if self.save_scheduler and hasattr(runner, 'scheduler'):
            if runner.scheduler is not None:
                checkpoint['scheduler'] = runner.scheduler.state_dict()
                
        # 保存 EMA 状态（如果存在）
        if hasattr(runner, 'ema_model') and runner.ema_model is not None:
            checkpoint['ema_state_dict'] = runner.ema_model.state_dict()
            
        return checkpoint
        
    def _cleanup_checkpoints(self) -> None:
        """清理旧检查点"""
        if self.max_keep_ckpts <= 0:
            return
            
        # 排除 best 和 last
        regular_ckpts = [
            ckpt for ckpt in self._saved_ckpts
            if not ckpt.endswith('best.pth') and not ckpt.endswith('last.pth')
        ]
        
        while len(regular_ckpts) > self.max_keep_ckpts:
            oldest = regular_ckpts.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)
            if oldest in self._saved_ckpts:
                self._saved_ckpts.remove(oldest)
                
    def _get_metric_value(self, runner: 'RunnerType') -> Optional[float]:
        """获取指标值"""
        # 尝试从 runner 的 log_buffer 或 outputs 获取
        if hasattr(runner, 'log_buffer'):
            return runner.log_buffer.get(self.best_metric)
        if hasattr(runner, 'outputs'):
            return runner.outputs.get(self.best_metric)
        return getattr(runner, self.best_metric, None)
        
    def _is_better(self, current: float) -> bool:
        """判断当前值是否更好"""
        if self._best_score is None:
            return True
            
        if self.rule == 'greater':
            return current > self._best_score
        else:
            return current < self._best_score
            
    def state_dict(self) -> Dict[str, Any]:
        """获取钩子状态"""
        return {
            'saved_ckpts': self._saved_ckpts.copy(),
            'best_score': self._best_score,
            'best_ckpt': self._best_ckpt,
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载钩子状态"""
        self._saved_ckpts = state_dict.get('saved_ckpts', [])
        self._best_score = state_dict.get('best_score')
        self._best_ckpt = state_dict.get('best_ckpt')


# 注册钩子（如果尚未注册）
if not HOOKS.contains('CheckpointHook'):
    HOOKS.register('CheckpointHook', CheckpointHook, 
                   priority=HookPriority.BELOW_NORMAL, 
                   category='checkpoint')
else:
    # 更新为新的实现
    HOOKS._components['CheckpointHook'].cls = CheckpointHook

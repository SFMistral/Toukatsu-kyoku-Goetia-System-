# -*- coding: utf-8 -*-
"""
EMA 钩子模块

提供指数移动平均模型功能，用于提升模型泛化能力。
"""

import copy
from typing import Dict, Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from .base_hook import BaseHook, HookPriority
from registry import HOOKS

if TYPE_CHECKING:
    from typing import Any as RunnerType


class EMAHook(BaseHook):
    """
    指数移动平均钩子
    
    维护 EMA 模型，在验证时使用 EMA 权重。
    
    EMA 更新公式:
        ema_weight = momentum * ema_weight + (1 - momentum) * current_weight
        
    动态动量（预热阶段）:
        current_momentum = min(momentum, (1 + iter) / (warm_up + iter))
    
    Args:
        momentum: EMA 动量
        interval: 更新间隔（iter）
        warm_up: 预热迭代数
        resume_from: 恢复 EMA 的检查点路径
        update_buffers: 是否更新 buffer
        
    Example:
        >>> hook = EMAHook(momentum=0.9999, warm_up=100)
    """
    
    priority = HookPriority.ABOVE_NORMAL
    
    def __init__(
        self,
        momentum: float = 0.9999,
        interval: int = 1,
        warm_up: int = 100,
        resume_from: Optional[str] = None,
        update_buffers: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.interval = interval
        self.warm_up = warm_up
        self.resume_from = resume_from
        self.update_buffers = update_buffers
        
        # EMA 模型
        self._ema_model: Optional[nn.Module] = None
        self._original_model: Optional[nn.Module] = None
        self._is_ema_active: bool = False
        
    def before_run(self, runner: 'RunnerType') -> None:
        """训练开始前初始化 EMA 模型"""
        # 创建 EMA 模型（深拷贝）
        self._ema_model = copy.deepcopy(runner.model)
        self._ema_model.eval()
        
        # 禁用 EMA 模型的梯度
        for param in self._ema_model.parameters():
            param.requires_grad = False
            
        # 从检查点恢复
        if self.resume_from is not None:
            self._load_ema_checkpoint(self.resume_from)
            
        # 保存原始模型引用
        self._original_model = runner.model
        
        # 将 EMA 模型附加到 runner
        runner.ema_model = self._ema_model
        
    def after_train_iter(self, runner: 'RunnerType') -> None:
        """训练迭代后更新 EMA 权重"""
        if not self.every_n_iters(runner, self.interval):
            return
            
        # 计算当前动量（考虑预热）
        current_iter = runner.iter + 1
        momentum = self._get_current_momentum(current_iter)
        
        # 更新 EMA 权重
        self._update_ema(runner.model, momentum)
        
    def before_val_epoch(self, runner: 'RunnerType') -> None:
        """验证开始前切换到 EMA 模型"""
        if self._ema_model is not None:
            self._swap_to_ema(runner)
            
    def after_val_epoch(self, runner: 'RunnerType') -> None:
        """验证结束后切换回原始模型"""
        if self._is_ema_active:
            self._swap_to_original(runner)
            
    def before_save_checkpoint(self, runner: 'RunnerType') -> None:
        """保存检查点前确保 EMA 状态被保存"""
        # EMA 状态会通过 runner.ema_model 自动保存
        pass
        
    def after_load_checkpoint(self, runner: 'RunnerType') -> None:
        """加载检查点后恢复 EMA 状态"""
        if hasattr(runner, 'ema_state_dict') and runner.ema_state_dict is not None:
            self._ema_model.load_state_dict(runner.ema_state_dict)
            
    def _get_current_momentum(self, current_iter: int) -> float:
        """获取当前动量（考虑预热）"""
        if self.warm_up <= 0:
            return self.momentum
            
        # 动态动量：预热阶段线性增加
        dynamic_momentum = (1 + current_iter) / (self.warm_up + current_iter)
        return min(self.momentum, dynamic_momentum)
        
    def _update_ema(self, model: nn.Module, momentum: float) -> None:
        """更新 EMA 权重"""
        with torch.no_grad():
            # 更新参数
            for ema_param, model_param in zip(
                self._ema_model.parameters(), 
                model.parameters()
            ):
                ema_param.data.mul_(momentum).add_(
                    model_param.data, alpha=1 - momentum
                )
                
            # 更新 buffer（如果需要）
            if self.update_buffers:
                for ema_buf, model_buf in zip(
                    self._ema_model.buffers(),
                    model.buffers()
                ):
                    ema_buf.data.copy_(model_buf.data)
                    
    def _swap_to_ema(self, runner: 'RunnerType') -> None:
        """切换到 EMA 模型"""
        runner.model = self._ema_model
        self._is_ema_active = True
        
    def _swap_to_original(self, runner: 'RunnerType') -> None:
        """切换回原始模型"""
        runner.model = self._original_model
        self._is_ema_active = False
        
    def _load_ema_checkpoint(self, checkpoint_path: str) -> None:
        """从检查点加载 EMA 状态"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema_state_dict' in checkpoint:
            self._ema_model.load_state_dict(checkpoint['ema_state_dict'])
            print(f"Loaded EMA state from {checkpoint_path}")
            
    def get_ema_model(self) -> Optional[nn.Module]:
        """获取 EMA 模型"""
        return self._ema_model
        
    def state_dict(self) -> Dict[str, Any]:
        """获取钩子状态"""
        state = {}
        if self._ema_model is not None:
            state['ema_state_dict'] = self._ema_model.state_dict()
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载钩子状态"""
        if 'ema_state_dict' in state_dict and self._ema_model is not None:
            self._ema_model.load_state_dict(state_dict['ema_state_dict'])


# 注册钩子（如果尚未注册）
if not HOOKS.contains('EMAHook'):
    HOOKS.register('EMAHook', EMAHook, 
                   priority=HookPriority.ABOVE_NORMAL, category='optimization')
else:
    HOOKS._components['EMAHook'].cls = EMAHook

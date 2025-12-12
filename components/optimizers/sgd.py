# -*- coding: utf-8 -*-
"""
SGD 优化器模块

提供 SGD 及其变体的封装。
"""

from typing import Iterator, Optional, Tuple
import torch
from torch.optim import Optimizer
from registry import OPTIMIZERS, ComponentSource


class SGD(torch.optim.SGD):
    """
    标准 SGD 优化器 (PyTorch 封装)
    
    Args:
        params: 模型参数
        lr: 学习率
        momentum: 动量 (默认 0.9)
        weight_decay: 权重衰减 L2 (默认 0.0)
        dampening: 动量抑制 (默认 0.0)
        nesterov: 是否使用 Nesterov 动量 (默认 False)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov
        )


class SGDP(Optimizer):
    """
    SGD with Decoupled Weight Decay
    
    解耦权重衰减，类似 AdamW 的处理方式。
    权重衰减不通过梯度应用，更稳定的正则化效果。
    
    Args:
        params: 模型参数
        lr: 学习率
        momentum: 动量 (默认 0.9)
        weight_decay: 权重衰减 (默认 0.0)
        dampening: 动量抑制 (默认 0.0)
        nesterov: 是否使用 Nesterov 动量 (默认 False)
        delta: 投影阈值 (默认 0.1)
        wd_ratio: 权重衰减比率 (默认 0.1)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            delta=delta,
            wd_ratio=wd_ratio
        )
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad
                
                # 动量
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                        
                # 解耦权重衰减
                if group['weight_decay'] != 0:
                    # 投影检查
                    if self._should_project(p, d_p, group['delta'], group['wd_ratio']):
                        wd = group['weight_decay']
                    else:
                        wd = group['weight_decay'] * group['wd_ratio']
                    p.mul_(1 - group['lr'] * wd)
                    
                # 参数更新
                p.add_(d_p, alpha=-group['lr'])
                
        return loss
        
    def _should_project(
        self,
        p: torch.Tensor,
        d_p: torch.Tensor,
        delta: float,
        wd_ratio: float
    ) -> bool:
        """判断是否应用完整权重衰减"""
        if p.dim() <= 1:
            return False
            
        p_norm = p.norm()
        d_p_norm = d_p.norm()
        
        if p_norm == 0 or d_p_norm == 0:
            return False
            
        # 计算投影
        cos_sim = (p * d_p).sum() / (p_norm * d_p_norm)
        return cos_sim.abs() < delta


class NesterovSGD(SGD):
    """
    Nesterov 动量 SGD
    
    预先计算动量方向，通常收敛更快。
    
    Args:
        params: 模型参数
        lr: 学习率
        momentum: 动量 (默认 0.9)
        weight_decay: 权重衰减 (默认 0.0)
        dampening: 动量抑制 (默认 0.0)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=True
        )


# 注册优化器
OPTIMIZERS.register('SGD', SGD, category='basic', source=ComponentSource.BUILTIN, force=True)
OPTIMIZERS.register('SGDP', SGDP, category='basic', source=ComponentSource.BUILTIN)
OPTIMIZERS.register('NesterovSGD', NesterovSGD, category='basic', source=ComponentSource.BUILTIN)

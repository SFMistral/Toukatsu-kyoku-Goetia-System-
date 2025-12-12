# -*- coding: utf-8 -*-
"""
Lion 优化器模块

Google 最新优化器，只使用动量的符号更新，内存效率更高。
"""

from typing import Iterator, Tuple
import torch
from torch.optim import Optimizer
from registry import OPTIMIZERS, ComponentSource


class Lion(Optimizer):
    """
    Lion 优化器 (EvoLved Sign Momentum)
    
    Google 通过程序搜索发现的优化器，特点：
    - 只使用动量的符号（sign）更新
    - 内存效率更高（无二阶矩）
    - 更新幅度统一
    
    更新规则：
        update = sign(beta1 * momentum + (1 - beta1) * grad)
        param = param - lr * (update + weight_decay * param)
        momentum = beta2 * momentum + (1 - beta2) * grad
    
    调优建议：
    - 学习率通常为 AdamW 的 3-10 倍小
    - batch size 较大时效果更好
    - 权重衰减可适当增大
    
    Args:
        params: 模型参数
        lr: 学习率 (默认 0.0001)
        betas: 动量参数 (默认 (0.9, 0.99))
        weight_decay: 权重衰减 (默认 0.0)
    """
    
    def __init__(
        self,
        params: Iterator,
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单步优化
        
        Args:
            closure: 可选的闭包，用于重新计算损失
            
        Returns:
            损失值（如果提供了 closure）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                    
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    
                exp_avg = state['exp_avg']
                
                # Lion 更新
                # 1. 计算更新方向 (使用当前动量和梯度的插值的符号)
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                update = update.sign()
                
                # 2. 应用权重衰减 (解耦方式)
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)
                    
                # 3. 参数更新
                p.add_(update, alpha=-lr)
                
                # 4. 更新动量 (用于下一步)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
        return loss
        
    def get_memory_usage(self) -> dict:
        """
        获取优化器内存使用情况
        
        Returns:
            内存使用统计字典
        """
        total_params = 0
        total_state_memory = 0
        
        for group in self.param_groups:
            for p in group['params']:
                total_params += p.numel()
                if p in self.state:
                    state = self.state[p]
                    if 'exp_avg' in state:
                        total_state_memory += state['exp_avg'].numel() * state['exp_avg'].element_size()
                        
        return {
            'total_params': total_params,
            'state_memory_bytes': total_state_memory,
            'state_memory_mb': total_state_memory / (1024 * 1024),
            # Lion 只需要一阶矩，相比 Adam 节省约 50% 状态内存
            'memory_ratio_vs_adam': 0.5
        }


# 注册优化器
OPTIMIZERS.register('Lion', Lion, category='efficient', source=ComponentSource.BUILTIN)

# -*- coding: utf-8 -*-
"""
优化器注册器模块

管理优化器组件，支持参数组自动构建和状态管理。
"""

from typing import Dict, Any, Optional, List, Type, Iterator
from .registry import Registry, ComponentSource


class OptimizerRegistry(Registry):
    """优化器注册器，支持参数组构建"""
    
    def build(
        self,
        config: Dict[str, Any],
        params: Optional[Iterator] = None,
        default_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        构建优化器实例
        
        Args:
            config: 优化器配置
            params: 模型参数迭代器
            default_args: 默认参数
            
        Returns:
            优化器实例
        """
        config = config.copy()
        optimizer_type = config.pop('type', None)
        
        if optimizer_type is None:
            raise KeyError("Config must contain 'type' field")
            
        cls = self.get(optimizer_type)
        
        # 合并默认参数
        if default_args:
            for key, value in default_args.items():
                config.setdefault(key, value)
                
        # 处理参数组配置
        param_groups = config.pop('param_groups', None)
        
        if params is None:
            raise ValueError("params must be provided for optimizer")
            
        if param_groups:
            # 使用参数组配置
            params = self._build_param_groups(params, param_groups, config)
        else:
            params = list(params)
            
        return cls(params, **config)
        
    def _build_param_groups(
        self,
        params: Iterator,
        param_groups_config: List[Dict[str, Any]],
        base_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        构建参数组
        
        Args:
            params: 模型参数
            param_groups_config: 参数组配置
            base_config: 基础配置
            
        Returns:
            参数组列表
        """
        # 将参数转换为字典形式 {name: param}
        named_params = dict(params) if hasattr(params, '__iter__') else {}
        
        param_groups = []
        used_params = set()
        
        for group_config in param_groups_config:
            group_config = group_config.copy()
            pattern = group_config.pop('pattern', None)
            
            if pattern:
                import re
                group_params = []
                for name, param in named_params.items():
                    if re.match(pattern, name) and name not in used_params:
                        group_params.append(param)
                        used_params.add(name)
                        
                if group_params:
                    param_groups.append({
                        'params': group_params,
                        **group_config
                    })
                    
        # 添加剩余参数到默认组
        remaining_params = [
            param for name, param in named_params.items() 
            if name not in used_params
        ]
        
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                **base_config
            })
            
        return param_groups


# 创建优化器注册器单例
OPTIMIZERS = OptimizerRegistry('optimizers', base_class=None)


def _register_builtin_optimizers():
    """注册内置优化器"""
    try:
        import torch.optim as optim
        
        # 基础优化器
        OPTIMIZERS.register('SGD', optim.SGD, 
                           category='basic', source=ComponentSource.THIRD_PARTY)
        OPTIMIZERS.register('Adam', optim.Adam,
                           category='adaptive', source=ComponentSource.THIRD_PARTY)
        OPTIMIZERS.register('AdamW', optim.AdamW,
                           category='adaptive', source=ComponentSource.THIRD_PARTY)
        OPTIMIZERS.register('RMSprop', optim.RMSprop,
                           category='adaptive', source=ComponentSource.THIRD_PARTY)
        OPTIMIZERS.register('Adagrad', optim.Adagrad,
                           category='adaptive', source=ComponentSource.THIRD_PARTY)
        OPTIMIZERS.register('Adadelta', optim.Adadelta,
                           category='adaptive', source=ComponentSource.THIRD_PARTY)
        
        # 高级优化器
        if hasattr(optim, 'NAdam'):
            OPTIMIZERS.register('NAdam', optim.NAdam,
                               category='adaptive', source=ComponentSource.THIRD_PARTY)
        if hasattr(optim, 'RAdam'):
            OPTIMIZERS.register('RAdam', optim.RAdam,
                               category='adaptive', source=ComponentSource.THIRD_PARTY)
                               
    except ImportError:
        pass


# 自定义优化器
class LARS:
    """Layer-wise Adaptive Rate Scaling optimizer"""
    
    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8
    ):
        import torch
        
        self.defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'trust_coefficient': trust_coefficient,
            'eps': eps
        }
        
        self.param_groups = []
        if isinstance(params, dict):
            self.param_groups.append(params)
        else:
            self.param_groups.append({'params': list(params), **self.defaults})
            
        self.state = {}
        
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
                    
    def step(self):
        import torch
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                # 权重衰减
                if group.get('weight_decay', 0) > 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                    
                # LARS缩放
                p_norm = p.data.norm()
                g_norm = grad.norm()
                
                if p_norm > 0 and g_norm > 0:
                    local_lr = group['trust_coefficient'] * p_norm / (
                        g_norm + group['weight_decay'] * p_norm + group['eps']
                    )
                else:
                    local_lr = 1.0
                    
                # 动量
                if group.get('momentum', 0) > 0:
                    if p not in self.state:
                        self.state[p] = {'momentum_buffer': torch.zeros_like(p.data)}
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad, alpha=local_lr * group['lr'])
                    p.data.add_(buf, alpha=-1)
                else:
                    p.data.add_(grad, alpha=-local_lr * group['lr'])
                    
    def state_dict(self):
        return {'state': self.state, 'param_groups': self.param_groups}
        
    def load_state_dict(self, state_dict):
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']


class LAMB:
    """Layer-wise Adaptive Moments optimizer for Batch training"""
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        trust_ratio_clip: float = 10.0
    ):
        import torch
        
        self.defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'trust_ratio_clip': trust_ratio_clip
        }
        
        self.param_groups = []
        if isinstance(params, dict):
            self.param_groups.append(params)
        else:
            self.param_groups.append({'params': list(params), **self.defaults})
            
        self.state = {}
        
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
                    
    def step(self):
        import torch
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                if p not in self.state:
                    self.state[p] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(p.data),
                        'exp_avg_sq': torch.zeros_like(p.data)
                    }
                    
                state = self.state[p]
                state['step'] += 1
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # 更新一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Adam更新
                adam_update = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group['eps'])
                
                # 权重衰减
                if group['weight_decay'] > 0:
                    adam_update = adam_update.add(p.data, alpha=group['weight_decay'])
                    
                # LAMB信任比率
                p_norm = p.data.norm()
                update_norm = adam_update.norm()
                
                if p_norm > 0 and update_norm > 0:
                    trust_ratio = p_norm / update_norm
                    trust_ratio = min(trust_ratio, group['trust_ratio_clip'])
                else:
                    trust_ratio = 1.0
                    
                p.data.add_(adam_update, alpha=-group['lr'] * trust_ratio)
                
    def state_dict(self):
        return {'state': self.state, 'param_groups': self.param_groups}
        
    def load_state_dict(self, state_dict):
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']


# 注册自定义优化器
OPTIMIZERS.register('LARS', LARS, category='large_batch', source=ComponentSource.BUILTIN)
OPTIMIZERS.register('LAMB', LAMB, category='large_batch', source=ComponentSource.BUILTIN)


# 初始化时注册内置优化器
_register_builtin_optimizers()

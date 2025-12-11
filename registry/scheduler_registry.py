# -*- coding: utf-8 -*-
"""
学习率调度器注册器模块

管理学习率调度器组件，支持Warmup包装和调度器组合。
"""

from typing import Dict, Any, Optional, List, Type, Union
from .registry import Registry, ComponentSource


class SchedulerRegistry(Registry):
    """学习率调度器注册器"""
    
    def build(
        self,
        config: Dict[str, Any],
        optimizer = None,
        default_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        构建调度器实例
        
        Args:
            config: 调度器配置
            optimizer: 优化器实例
            default_args: 默认参数
            
        Returns:
            调度器实例
        """
        config = config.copy()
        scheduler_type = config.pop('type', None)
        
        if scheduler_type is None:
            raise KeyError("Config must contain 'type' field")
            
        # 处理Warmup配置
        warmup_config = config.pop('warmup', None)
        
        cls = self.get(scheduler_type)
        
        # 合并默认参数
        if default_args:
            for key, value in default_args.items():
                config.setdefault(key, value)
                
        if optimizer is None:
            raise ValueError("optimizer must be provided for scheduler")
            
        # 构建主调度器
        scheduler = cls(optimizer, **config)
        
        # 包装Warmup
        if warmup_config:
            scheduler = WarmupScheduler(
                scheduler=scheduler,
                optimizer=optimizer,
                **warmup_config
            )
            
        return scheduler
        
    def build_combined(
        self,
        configs: List[Dict[str, Any]],
        optimizer,
        milestones: List[int]
    ) -> 'CombinedScheduler':
        """
        构建组合调度器
        
        Args:
            configs: 调度器配置列表
            optimizer: 优化器
            milestones: 切换里程碑
            
        Returns:
            组合调度器实例
        """
        schedulers = [self.build(cfg, optimizer) for cfg in configs]
        return CombinedScheduler(schedulers, milestones)


# 创建调度器注册器单例
SCHEDULERS = SchedulerRegistry('schedulers', base_class=None)


class WarmupScheduler:
    """Warmup包装器"""
    
    def __init__(
        self,
        scheduler,
        optimizer,
        warmup_epochs: int = 5,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.1,
        warmup_type: str = 'linear'
    ):
        """
        Args:
            scheduler: 主调度器
            optimizer: 优化器
            warmup_epochs: warmup的epoch数
            warmup_steps: warmup的step数（优先于epochs）
            warmup_ratio: 初始学习率比例
            warmup_type: warmup类型 ('linear', 'constant', 'exponential')
        """
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_type = warmup_type
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        self.warmup_finished = False
        
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        self.current_step += 1
        
        # 判断是否在warmup阶段
        if self.warmup_steps:
            in_warmup = self.current_step <= self.warmup_steps
            warmup_progress = self.current_step / self.warmup_steps
        else:
            in_warmup = epoch is not None and epoch < self.warmup_epochs
            warmup_progress = (epoch + 1) / self.warmup_epochs if epoch is not None else 1.0
            
        if in_warmup:
            # 计算warmup学习率
            if self.warmup_type == 'linear':
                factor = self.warmup_ratio + (1 - self.warmup_ratio) * warmup_progress
            elif self.warmup_type == 'constant':
                factor = self.warmup_ratio
            elif self.warmup_type == 'exponential':
                factor = self.warmup_ratio ** (1 - warmup_progress)
            else:
                factor = 1.0
                
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * factor
        else:
            if not self.warmup_finished:
                # 恢复初始学习率
                for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group['lr'] = base_lr
                self.warmup_finished = True
                
            # 使用主调度器
            self.scheduler.step()
            
    def get_last_lr(self) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def state_dict(self):
        return {
            'scheduler': self.scheduler.state_dict(),
            'current_step': self.current_step,
            'warmup_finished': self.warmup_finished
        }
        
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.current_step = state_dict['current_step']
        self.warmup_finished = state_dict['warmup_finished']


class CombinedScheduler:
    """组合调度器，在不同阶段使用不同调度器"""
    
    def __init__(self, schedulers: List, milestones: List[int]):
        """
        Args:
            schedulers: 调度器列表
            milestones: 切换里程碑（epoch数）
        """
        if len(schedulers) != len(milestones) + 1:
            raise ValueError("Number of schedulers must be len(milestones) + 1")
            
        self.schedulers = schedulers
        self.milestones = milestones
        self.current_epoch = 0
        
    def step(self):
        """更新学习率"""
        # 确定当前使用哪个调度器
        scheduler_idx = 0
        for i, milestone in enumerate(self.milestones):
            if self.current_epoch >= milestone:
                scheduler_idx = i + 1
                
        self.schedulers[scheduler_idx].step()
        self.current_epoch += 1
        
    def get_last_lr(self) -> List[float]:
        """获取当前学习率"""
        scheduler_idx = 0
        for i, milestone in enumerate(self.milestones):
            if self.current_epoch >= milestone:
                scheduler_idx = i + 1
        return self.schedulers[scheduler_idx].get_last_lr()
        
    def state_dict(self):
        return {
            'schedulers': [s.state_dict() for s in self.schedulers],
            'current_epoch': self.current_epoch
        }
        
    def load_state_dict(self, state_dict):
        for s, sd in zip(self.schedulers, state_dict['schedulers']):
            s.load_state_dict(sd)
        self.current_epoch = state_dict['current_epoch']


# 自定义调度器
class LinearWarmupCosineDecay:
    """线性Warmup + 余弦衰减调度器"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        
    def step(self):
        import math
        
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # 线性warmup
            factor = self.current_step / self.warmup_steps
        else:
            # 余弦衰减
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.min_lr + (base_lr - self.min_lr) * factor
            
    def get_last_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def state_dict(self):
        return {'current_step': self.current_step}
        
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']


class PolynomialDecay:
    """多项式衰减调度器"""
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        
        factor = (1 - self.current_step / self.total_steps) ** self.power
        factor = max(factor, 0)
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.min_lr + (base_lr - self.min_lr) * factor
            
    def get_last_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def state_dict(self):
        return {'current_step': self.current_step}
        
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']


def _register_builtin_schedulers():
    """注册内置调度器"""
    try:
        from torch.optim import lr_scheduler
        
        SCHEDULERS.register('StepLR', lr_scheduler.StepLR,
                           category='step', source=ComponentSource.THIRD_PARTY)
        SCHEDULERS.register('MultiStepLR', lr_scheduler.MultiStepLR,
                           category='step', source=ComponentSource.THIRD_PARTY)
        SCHEDULERS.register('ExponentialLR', lr_scheduler.ExponentialLR,
                           category='exponential', source=ComponentSource.THIRD_PARTY)
        SCHEDULERS.register('CosineAnnealingLR', lr_scheduler.CosineAnnealingLR,
                           category='cosine', source=ComponentSource.THIRD_PARTY)
        SCHEDULERS.register('CosineAnnealingWarmRestarts', lr_scheduler.CosineAnnealingWarmRestarts,
                           category='cosine', source=ComponentSource.THIRD_PARTY)
        SCHEDULERS.register('ReduceLROnPlateau', lr_scheduler.ReduceLROnPlateau,
                           category='adaptive', source=ComponentSource.THIRD_PARTY)
        SCHEDULERS.register('OneCycleLR', lr_scheduler.OneCycleLR,
                           category='cycle', source=ComponentSource.THIRD_PARTY)
        SCHEDULERS.register('LambdaLR', lr_scheduler.LambdaLR,
                           category='custom', source=ComponentSource.THIRD_PARTY)
    except ImportError:
        pass


# 注册自定义调度器
SCHEDULERS.register('LinearWarmupCosineDecay', LinearWarmupCosineDecay,
                   category='warmup', source=ComponentSource.BUILTIN)
SCHEDULERS.register('PolynomialDecay', PolynomialDecay,
                   category='polynomial', source=ComponentSource.BUILTIN)
SCHEDULERS.register('WarmupScheduler', WarmupScheduler,
                   category='warmup', source=ComponentSource.BUILTIN)

# 初始化时注册内置调度器
_register_builtin_schedulers()

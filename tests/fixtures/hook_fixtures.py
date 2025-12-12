# -*- coding: utf-8 -*-
"""
Hooks 模块测试夹具

提供 hooks 测试所需的 fixtures 和 mock 对象。
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional


# ============== Mock Runner ==============

class MockRunner:
    """
    模拟训练器
    
    用于测试钩子的 runner 模拟对象。
    """
    
    def __init__(
        self,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        max_epochs: int = 10,
        max_iters: int = 100,
        work_dir: str = None,
    ):
        self.model = model or self._create_simple_model()
        self.optimizer = optimizer
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.max_epochs = max_epochs
        self.max_iters = max_iters
        self.work_dir = work_dir or tempfile.mkdtemp()
        
        # 状态
        self.epoch = 0
        self.iter = 0
        self.inner_iter = 0
        self.should_stop = False
        self.is_best = False
        
        # 输出
        self.outputs: Dict[str, Any] = {}
        self.val_outputs: Dict[str, Any] = {}
        self.log_buffer: Dict[str, Any] = {}
        self.data_batch: Dict[str, Any] = {}
        
        # EMA
        self.ema_model = None
        
        # 初始化优化器
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            
    def _create_simple_model(self) -> nn.Module:
        """创建简单模型"""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
    def set_epoch(self, epoch: int):
        """设置当前 epoch"""
        self.epoch = epoch
        
    def set_iter(self, iter_num: int):
        """设置当前迭代"""
        self.iter = iter_num
        self.inner_iter = iter_num % (self.max_iters // self.max_epochs)
        
    def set_outputs(self, outputs: Dict[str, Any]):
        """设置输出"""
        self.outputs = outputs
        
    def set_val_outputs(self, outputs: Dict[str, Any]):
        """设置验证输出"""
        self.val_outputs = outputs
        
    def val(self):
        """模拟验证"""
        self.val_outputs = {
            'val_loss': 0.5,
            'accuracy': 0.9
        }


class MockDataLoader:
    """模拟数据加载器"""
    
    def __init__(self, num_batches: int = 10, batch_size: int = 4):
        self.num_batches = num_batches
        self.batch_size = batch_size
        
    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                'inputs': torch.randn(self.batch_size, 10),
                'targets': torch.randint(0, 10, (self.batch_size,))
            }
            
    def __len__(self):
        return self.num_batches


# ============== Fixtures ==============

@pytest.fixture
def simple_model():
    """简单测试模型"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )


@pytest.fixture
def conv_model():
    """卷积测试模型"""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10)
    )


@pytest.fixture
def optimizer(simple_model):
    """SGD 优化器"""
    return torch.optim.SGD(simple_model.parameters(), lr=0.1)


@pytest.fixture
def adam_optimizer(simple_model):
    """Adam 优化器"""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def mock_runner(simple_model, optimizer, temp_dir):
    """Mock Runner fixture"""
    return MockRunner(
        model=simple_model,
        optimizer=optimizer,
        max_epochs=10,
        max_iters=100,
        work_dir=temp_dir
    )


@pytest.fixture
def mock_dataloader():
    """Mock DataLoader fixture"""
    return MockDataLoader(num_batches=10, batch_size=4)


@pytest.fixture
def hook_config_checkpoint():
    """检查点钩子配置"""
    return {
        'type': 'CheckpointHook',
        'interval': 1,
        'max_keep_ckpts': 3,
        'save_best': True,
        'best_metric': 'accuracy',
        'rule': 'greater'
    }


@pytest.fixture
def hook_config_logger():
    """日志钩子配置"""
    return {
        'type': 'LoggerHook',
        'interval': 10,
        'by_epoch': True
    }


@pytest.fixture
def hook_config_eval():
    """评估钩子配置"""
    return {
        'type': 'EvalHook',
        'interval': 1,
        'metric': 'accuracy',
        'rule': 'greater'
    }


@pytest.fixture
def hook_config_early_stopping():
    """早停钩子配置"""
    return {
        'type': 'EarlyStoppingHook',
        'monitor': 'val_loss',
        'patience': 5,
        'mode': 'min'
    }


@pytest.fixture
def hook_config_ema():
    """EMA 钩子配置"""
    return {
        'type': 'EMAHook',
        'momentum': 0.9999,
        'warm_up': 10
    }


@pytest.fixture
def hook_config_timer():
    """计时钩子配置"""
    return {
        'type': 'IterTimerHook',
        'interval': 10,
        'log_eta': True
    }


@pytest.fixture
def hook_configs_default():
    """默认钩子配置列表"""
    return [
        {'type': 'IterTimerHook', 'interval': 10},
        {'type': 'LoggerHook', 'interval': 10},
        {'type': 'CheckpointHook', 'interval': 1, 'max_keep_ckpts': 3},
    ]


@pytest.fixture
def sample_checkpoint(simple_model, optimizer):
    """示例检查点"""
    return {
        'meta': {
            'time': '2024-01-01 00:00:00',
            'epoch': 5,
            'iter': 50,
        },
        'state_dict': simple_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

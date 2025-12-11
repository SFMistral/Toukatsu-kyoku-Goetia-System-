# -*- coding: utf-8 -*-
"""
Pytest配置和共享fixtures
"""

import os
import sys
import tempfile
import shutil
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # 清理
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def temp_log_file(temp_dir):
    """创建临时日志文件路径"""
    return os.path.join(temp_dir, "test.log")


@pytest.fixture
def sample_log_records():
    """示例日志记录"""
    import time
    return [
        {
            'timestamp': time.time(),
            'level': 'INFO',
            'logger_name': 'test',
            'message': 'Test message 1',
            'step': 1
        },
        {
            'timestamp': time.time(),
            'level': 'WARNING',
            'logger_name': 'test',
            'message': 'Test warning',
            'step': 2
        },
        {
            'timestamp': time.time(),
            'level': 'ERROR',
            'logger_name': 'test',
            'message': 'Test error',
            'step': 3
        }
    ]


@pytest.fixture
def sample_metrics():
    """示例指标数据"""
    return {
        'loss': 0.5,
        'accuracy': 0.95,
        'learning_rate': 0.001
    }


@pytest.fixture
def sample_hyperparams():
    """示例超参数"""
    return {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'model': 'ResNet50'
    }


@pytest.fixture
def logging_config(temp_dir):
    """日志配置"""
    return {
        'level': 'DEBUG',
        'console': {
            'enabled': True,
            'level': 'INFO',
            'use_colors': False
        },
        'file': {
            'enabled': True,
            'log_dir': temp_dir,
            'filename': 'test.log',
            'level': 'DEBUG'
        },
        'json': {
            'enabled': True,
            'log_dir': temp_dir,
            'filename': 'test.jsonl',
            'separate_metrics': True
        }
    }
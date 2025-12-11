"""
Pytest 配置和共享 fixtures
"""

import os
import sys
import tempfile
import pytest

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config():
    """示例配置字典"""
    return {
        "model": {
            "type": "ImageClassifier",
            "backbone": {
                "type": "ResNet",
                "depth": 50,
                "pretrained": True,
            },
            "head": {
                "type": "LinearClsHead",
                "num_classes": 1000,
            },
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 0.001,
        },
    }


@pytest.fixture
def sample_yaml_content():
    """示例 YAML 内容"""
    return """
model:
  type: ImageClassifier
  backbone:
    type: ResNet
    depth: 50
training:
  epochs: 100
  batch_size: 32
"""

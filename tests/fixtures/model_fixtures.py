# -*- coding: utf-8 -*-
"""
模型测试夹具

提供模型测试所需的fixtures和mock对象。
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple


# ============== 基础数据 Fixtures ==============

@pytest.fixture
def device():
    """获取测试设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_image_batch():
    """示例图像批次 (B, C, H, W)"""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_image_batch_small():
    """小尺寸图像批次"""
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def sample_image_batch_large():
    """大尺寸图像批次"""
    return torch.randn(2, 3, 512, 512)


@pytest.fixture
def sample_feature_maps():
    """示例多尺度特征图（ResNet50输出）"""
    return [
        torch.randn(2, 256, 56, 56),   # C2
        torch.randn(2, 512, 28, 28),   # C3
        torch.randn(2, 1024, 14, 14),  # C4
        torch.randn(2, 2048, 7, 7),    # C5
    ]


@pytest.fixture
def sample_feature_maps_small():
    """小尺寸特征图"""
    return [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 16, 16),
        torch.randn(2, 256, 8, 8),
        torch.randn(2, 512, 4, 4),
    ]


@pytest.fixture
def sample_labels():
    """示例分类标签"""
    return torch.randint(0, 10, (2,))


@pytest.fixture
def sample_labels_100():
    """100类分类标签"""
    return torch.randint(0, 100, (2,))


# ============== 配置 Fixtures ==============

@pytest.fixture
def norm_cfg_bn():
    """BatchNorm配置"""
    return dict(type='BN', requires_grad=True)


@pytest.fixture
def norm_cfg_gn():
    """GroupNorm配置"""
    return dict(type='GN', num_groups=32)


@pytest.fixture
def norm_cfg_ln():
    """LayerNorm配置"""
    return dict(type='LN')


@pytest.fixture
def act_cfg_relu():
    """ReLU激活配置"""
    return dict(type='ReLU', inplace=True)


@pytest.fixture
def act_cfg_silu():
    """SiLU激活配置"""
    return dict(type='SiLU', inplace=True)


@pytest.fixture
def act_cfg_gelu():
    """GELU激活配置"""
    return dict(type='GELU')


# ============== 模型配置 Fixtures ==============

@pytest.fixture
def resnet18_config():
    """ResNet-18配置"""
    return dict(
        type='ResNet',
        depth=18,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
    )


@pytest.fixture
def resnet50_config():
    """ResNet-50配置"""
    return dict(
        type='ResNet',
        depth=50,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
    )


@pytest.fixture
def vit_base_config():
    """ViT-Base配置"""
    return dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        with_cls_token=True,
    )


@pytest.fixture
def fpn_config():
    """FPN配置"""
    return dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    )


@pytest.fixture
def linear_cls_head_config():
    """线性分类头配置"""
    return dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
    )


@pytest.fixture
def image_classifier_config():
    """图像分类器配置"""
    return dict(
        type='ImageClassifier',
        backbone=dict(type='ResNet', depth=18),
        head=dict(type='LinearClsHead', num_classes=10),
    )


@pytest.fixture
def yolo_config():
    """YOLO配置"""
    return dict(
        type='YOLO',
        backbone=dict(type='CSPDarknet'),
        neck=dict(
            type='YOLONeck',
            in_channels=[256, 512, 1024],
            out_channels=[256, 512, 1024],
        ),
        head=dict(
            type='YOLOHead',
            num_classes=80,
            in_channels=[256, 512, 1024],
        ),
    )


# ============== Mock 对象 ==============

@pytest.fixture
def mock_backbone():
    """Mock骨干网络"""
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_channels = [256, 512, 1024, 2048]
            self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            
        def forward(self, x):
            return [
                torch.randn(x.shape[0], 256, 56, 56),
                torch.randn(x.shape[0], 512, 28, 28),
                torch.randn(x.shape[0], 1024, 14, 14),
                torch.randn(x.shape[0], 2048, 7, 7),
            ]
    return MockBackbone()


@pytest.fixture
def mock_neck():
    """Mock Neck"""
    class MockNeck(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return [torch.randn(x[0].shape[0], 256, s, s) for s in [80, 40, 20, 10, 5]]
    return MockNeck()


@pytest.fixture
def mock_head():
    """Mock Head"""
    class MockHead(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.fc = nn.Linear(2048, num_classes)
            
        def forward(self, x):
            if isinstance(x, list):
                x = x[-1]
            if x.dim() == 4:
                x = x.mean(dim=[2, 3])
            return self.fc(x)
            
        def loss(self, x, labels):
            logits = self.forward(x)
            loss = nn.functional.cross_entropy(logits, labels)
            return {'loss': loss}
            
        def predict(self, x):
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            scores, labels = probs.max(dim=1)
            return {'pred_label': labels, 'pred_score': scores}
            
    return MockHead()


# ============== 检测相关 Fixtures ==============

@pytest.fixture
def sample_det_targets():
    """示例检测目标"""
    return [
        {
            'boxes': torch.tensor([[10, 20, 100, 150], [50, 60, 200, 250]], dtype=torch.float32),
            'labels': torch.tensor([1, 2]),
        },
        {
            'boxes': torch.tensor([[30, 40, 120, 180]], dtype=torch.float32),
            'labels': torch.tensor([3]),
        },
    ]


@pytest.fixture
def sample_anchors():
    """示例Anchors"""
    return [
        torch.randn(2, 9, 80, 80, 4),
        torch.randn(2, 9, 40, 40, 4),
        torch.randn(2, 9, 20, 20, 4),
    ]


# ============== 分割相关 Fixtures ==============

@pytest.fixture
def sample_seg_targets():
    """示例分割目标"""
    return torch.randint(0, 21, (2, 224, 224))


@pytest.fixture
def sample_seg_masks():
    """示例分割掩码"""
    return torch.randint(0, 2, (2, 1, 224, 224), dtype=torch.float32)


# ============== 权重相关 Fixtures ==============

@pytest.fixture
def sample_state_dict():
    """示例state_dict"""
    return {
        'conv1.weight': torch.randn(64, 3, 7, 7),
        'conv1.bias': torch.randn(64),
        'bn1.weight': torch.randn(64),
        'bn1.bias': torch.randn(64),
        'bn1.running_mean': torch.zeros(64),
        'bn1.running_var': torch.ones(64),
        'fc.weight': torch.randn(10, 512),
        'fc.bias': torch.randn(10),
    }


@pytest.fixture
def sample_state_dict_with_prefix():
    """带前缀的state_dict"""
    return {
        'module.conv1.weight': torch.randn(64, 3, 7, 7),
        'module.conv1.bias': torch.randn(64),
        'module.bn1.weight': torch.randn(64),
        'module.bn1.bias': torch.randn(64),
    }

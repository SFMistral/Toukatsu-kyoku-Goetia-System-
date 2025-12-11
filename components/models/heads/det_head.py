# -*- coding: utf-8 -*-
"""
检测任务头

提供Anchor-based、Anchor-free检测头。
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import HEADS
from ..layers import ConvModule


@HEADS.register(name='AnchorHead')
class AnchorHead(nn.Module):
    """
    基于Anchor的检测头
    
    Args:
        num_classes: 类别数
        in_channels: 输入通道数
        feat_channels: 特征通道数
        stacked_convs: 堆叠卷积层数
        num_anchors: 每个位置的anchor数量
        norm_cfg: 归一化层配置
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        num_anchors: int = 9,
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 分类分支
        cls_convs = []
        for i in range(stacked_convs):
            cls_convs.append(ConvModule(
                in_channels if i == 0 else feat_channels,
                feat_channels, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.cls_out = nn.Conv2d(feat_channels, num_anchors * num_classes, 3, padding=1)

        # 回归分支
        reg_convs = []
        for i in range(stacked_convs):
            reg_convs.append(ConvModule(
                in_channels if i == 0 else feat_channels,
                feat_channels, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
        self.reg_convs = nn.Sequential(*reg_convs)
        self.reg_out = nn.Conv2d(feat_channels, num_anchors * 4, 3, padding=1)
        
    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """前向传播"""
        cls_scores = []
        bbox_preds = []
        
        for feat in feats:
            cls_feat = self.cls_convs(feat)
            reg_feat = self.reg_convs(feat)
            cls_scores.append(self.cls_out(cls_feat))
            bbox_preds.append(self.reg_out(reg_feat))
            
        return cls_scores, bbox_preds
        
    def loss(self, cls_scores, bbox_preds, targets) -> Dict[str, torch.Tensor]:
        """计算损失（简化版）"""
        # 实际实现需要anchor生成、正负样本分配等
        return {'loss_cls': torch.tensor(0.0), 'loss_bbox': torch.tensor(0.0)}


@HEADS.register(name='YOLOHead')
class YOLOHead(nn.Module):
    """
    YOLO检测头
    
    Args:
        num_classes: 类别数
        in_channels: 输入各层通道数
        num_anchors: 每层anchor数量
        strides: 各层步长
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: List[int] = [256, 512, 1024],
        num_anchors: int = 3,
        strides: List[int] = [8, 16, 32],
        norm_cfg: Dict[str, Any] = dict(type='BN'),
        act_cfg: Dict[str, Any] = dict(type='SiLU', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.strides = strides
        self.num_outputs = 5 + num_classes  # x, y, w, h, obj, cls...
        
        self.heads = nn.ModuleList([
            nn.Conv2d(ch, num_anchors * self.num_outputs, 1)
            for ch in in_channels
        ])
        
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播"""
        outputs = []
        for feat, head in zip(feats, self.heads):
            outputs.append(head(feat))
        return outputs
        
    def loss(self, outputs, targets) -> Dict[str, torch.Tensor]:
        """计算损失"""
        return {'loss_cls': torch.tensor(0.0), 'loss_bbox': torch.tensor(0.0), 'loss_obj': torch.tensor(0.0)}


@HEADS.register(name='FCOSHead')
class FCOSHead(nn.Module):
    """
    FCOS无锚框检测头
    
    Args:
        num_classes: 类别数
        in_channels: 输入通道数
        feat_channels: 特征通道数
        stacked_convs: 堆叠卷积层数
        strides: 各层步长
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        strides: List[int] = [8, 16, 32, 64, 128],
        norm_cfg: Dict[str, Any] = dict(type='GN', num_groups=32),
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides
        
        # 共享卷积
        cls_convs = []
        reg_convs = []
        for i in range(stacked_convs):
            cls_convs.append(ConvModule(
                in_channels if i == 0 else feat_channels,
                feat_channels, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
            reg_convs.append(ConvModule(
                in_channels if i == 0 else feat_channels,
                feat_channels, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)
        
        # 输出层
        self.cls_out = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        self.reg_out = nn.Conv2d(feat_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(feat_channels, 1, 3, padding=1)
        
        # 可学习的scale
        self.scales = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in strides])
        
    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], ...]:
        """前向传播"""
        cls_scores = []
        bbox_preds = []
        centernesses = []
        
        for feat, scale in zip(feats, self.scales):
            cls_feat = self.cls_convs(feat)
            reg_feat = self.reg_convs(feat)
            
            cls_scores.append(self.cls_out(cls_feat))
            bbox_preds.append(F.relu(scale(self.reg_out(reg_feat))))
            centernesses.append(self.centerness(reg_feat))
            
        return cls_scores, bbox_preds, centernesses
        
    def loss(self, cls_scores, bbox_preds, centernesses, targets) -> Dict[str, torch.Tensor]:
        """计算损失"""
        return {'loss_cls': torch.tensor(0.0), 'loss_bbox': torch.tensor(0.0), 'loss_centerness': torch.tensor(0.0)}

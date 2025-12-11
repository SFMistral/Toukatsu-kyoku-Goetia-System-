# -*- coding: utf-8 -*-
"""
实例分割任务头

提供Mask R-CNN等实例分割头。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import HEADS
from ..layers import ConvModule


@HEADS.register(name='MaskRCNNHead')
class MaskRCNNHead(nn.Module):
    """
    Mask R-CNN掩码头
    
    Args:
        num_classes: 类别数
        in_channels: 输入通道数
        num_convs: 卷积层数量
        roi_feat_size: RoI特征尺寸
        norm_cfg: 归一化层配置
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        num_convs: int = 4,
        roi_feat_size: int = 14,
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Dict[str, Any] = dict(type='ReLU', inplace=True),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.roi_feat_size = roi_feat_size
        
        # 卷积层
        convs = []
        for i in range(num_convs):
            convs.append(ConvModule(
                in_channels, in_channels, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
        self.convs = nn.Sequential(*convs)
        
        # 上采样
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        # 输出
        self.conv_logits = nn.Conv2d(in_channels, num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RoI特征 (N, C, H, W)
            
        Returns:
            mask_pred: 掩码预测 (N, num_classes, 2H, 2W)
        """
        x = self.convs(x)
        x = self.relu(self.upsample(x))
        return self.conv_logits(x)
        
    def loss(self, mask_pred: torch.Tensor, mask_targets: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算掩码损失"""
        # 选择对应类别的预测
        num_rois = mask_pred.shape[0]
        indices = torch.arange(num_rois, device=mask_pred.device)
        mask_pred = mask_pred[indices, labels]
        
        loss_mask = F.binary_cross_entropy_with_logits(mask_pred, mask_targets.float())
        return {'loss_mask': loss_mask}

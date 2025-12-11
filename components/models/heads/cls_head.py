# -*- coding: utf-8 -*-
"""
分类任务头

提供线性分类头、多层MLP分类头、ViT分类头等。
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.model_registry import HEADS


@HEADS.register(name='LinearClsHead')
class LinearClsHead(nn.Module):
    """
    线性分类头
    
    Args:
        num_classes: 类别数
        in_channels: 输入通道数
        dropout: Dropout概率
        loss: 损失函数配置
        topk: Top-K精度计算
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        dropout: float = 0.0,
        loss: Optional[Dict[str, Any]] = None,
        topk: Tuple[int, ...] = (1, 5),
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.topk = topk
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)
        
        # 损失函数
        if loss is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = self._build_loss(loss)

    def _build_loss(self, loss_cfg: Dict[str, Any]) -> nn.Module:
        loss_type = loss_cfg.get('type', 'CrossEntropyLoss')
        if loss_type == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss(label_smoothing=loss_cfg.get('label_smoothing', 0.0))
        return nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回logits"""
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
        
    def loss(self, x: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失"""
        logits = self.forward(x)
        loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        pred_scores, pred_labels = probs.max(dim=1)
        return {'pred_label': pred_labels, 'pred_score': pred_scores, 'pred_probs': probs}


@HEADS.register(name='StackedLinearClsHead')
class StackedLinearClsHead(nn.Module):
    """多层MLP分类头"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dims: List[int] = [512],
        dropout: float = 0.0,
        loss: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        layers = []
        prev_dim = in_channels
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
        
    def loss(self, x: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        return {'loss': self.loss_fn(logits, labels), 'logits': logits}
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        pred_scores, pred_labels = probs.max(dim=1)
        return {'pred_label': pred_labels, 'pred_score': pred_scores}


@HEADS.register(name='VisionTransformerClsHead')
class VisionTransformerClsHead(nn.Module):
    """ViT专用分类头"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        loss: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.pre_logits = nn.Identity()
        if hidden_dim is not None:
            self.pre_logits = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.Tanh())
            in_channels = hidden_dim
            
        self.head = nn.Linear(in_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if x.dim() == 3:
            x = x[:, 0]  # CLS token
        x = self.pre_logits(x)
        x = self.dropout(x)
        return self.head(x)
        
    def loss(self, x: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        return {'loss': self.loss_fn(logits, labels), 'logits': logits}
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        pred_scores, pred_labels = probs.max(dim=1)
        return {'pred_label': pred_labels, 'pred_score': pred_scores}

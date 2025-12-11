# -*- coding: utf-8 -*-
"""
DETR检测器
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from registry.model_registry import MODELS, BACKBONES
from .base_detector import BaseDetector
from ..layers import TransformerEncoderLayer, TransformerDecoderLayer, SinusoidalPositionEncoding


@MODELS.register(name='DETR')
class DETR(BaseDetector):
    """
    DETR检测器
    
    Args:
        backbone: CNN骨干网络配置
        num_classes: 类别数
        num_queries: Object Queries数量
        hidden_dim: Transformer隐藏维度
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
    """
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        num_classes: int = 80,
        num_queries: int = 100,
        hidden_dim: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        train_cfg: Optional[Dict[str, Any]] = None,
        test_cfg: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(init_cfg)
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Backbone
        self.backbone = BACKBONES.build(backbone)
        
        # 输入投影
        backbone_out_channels = self.backbone.out_channels[-1] if hasattr(self.backbone, 'out_channels') else 2048
        self.input_proj = nn.Conv2d(backbone_out_channels, hidden_dim, 1)
        
        # 位置编码
        self.pos_encoding = SinusoidalPositionEncoding(hidden_dim // 2)
        
        # Transformer
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads) for _ in range(num_decoder_layers)
        ])
        
        # Object Queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 输出头
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(img)
        
    def forward_train(self, img: torch.Tensor, targets: Any) -> Dict[str, torch.Tensor]:
        outputs = self._forward(img)
        # 匈牙利匹配 + 损失计算（简化）
        return {'loss_cls': torch.tensor(0.0), 'loss_bbox': torch.tensor(0.0)}
        
    def forward_test(self, img: torch.Tensor) -> List[Any]:
        return self._forward(img)
        
    def _forward(self, img: torch.Tensor):
        # Backbone
        feats = self.backbone(img)
        feat = feats[-1]
        
        # 投影
        src = self.input_proj(feat)
        B, C, H, W = src.shape
        
        # 位置编码
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=src.device)
        pos = self.pos_encoding(mask)
        
        # Flatten
        src = src.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        pos = pos.flatten(2).permute(2, 0, 1)
        
        # Encoder
        memory = src
        for layer in self.encoder:
            memory = layer(memory.permute(1, 0, 2)).permute(1, 0, 2)
            
        # Decoder
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        for layer in self.decoder:
            query = layer(query.permute(1, 0, 2), memory.permute(1, 0, 2)).permute(1, 0, 2)
            
        # 输出
        query = query.permute(1, 0, 2)  # (B, num_queries, C)
        outputs_class = self.class_embed(query)
        outputs_bbox = self.bbox_embed(query).sigmoid()
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_bbox}

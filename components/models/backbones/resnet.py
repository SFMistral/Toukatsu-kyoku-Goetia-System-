# -*- coding: utf-8 -*-
"""
ResNet系列骨干网络

支持ResNet-18/34/50/101/152、ResNeXt、Res2Net等变体。
"""

from typing import Dict, Any, Optional, List, Tuple, Type
import torch
import torch.nn as nn

from registry.model_registry import BACKBONES
from ..layers import ConvModule, BasicBlock, Bottleneck
from ..utils import init_weights


# ResNet配置
RESNET_CONFIGS = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


@BACKBONES.register(name='ResNet')
class ResNet(nn.Module):
    """
    ResNet骨干网络
    
    Args:
        depth: 网络深度 18/34/50/101/152
        num_stages: 使用的stage数量（1-4）
        out_indices: 输出特征层索引
        frozen_stages: 冻结的stage数量
        norm_cfg: 归一化层配置
        style: 风格 'pytorch' / 'caffe'
        deep_stem: 是否使用深层stem
        avg_down: 是否使用平均池化下采样
        base_channels: 基础通道数
        init_cfg: 初始化配置
    """
    
    def __init__(
        self,
        depth: int = 50,
        num_stages: int = 4,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        norm_cfg: Dict[str, Any] = dict(type='BN', requires_grad=True),
        style: str = 'pytorch',
        deep_stem: bool = False,
        avg_down: bool = False,
        base_channels: int = 64,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        if depth not in RESNET_CONFIGS:
            raise ValueError(f"Invalid depth {depth}, must be one of {list(RESNET_CONFIGS.keys())}")

        self.depth = depth
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.init_cfg = init_cfg
        
        block, stage_blocks = RESNET_CONFIGS[depth]
        self.block = block
        self.expansion = block.expansion
        
        # 构建stem
        self.in_channels = base_channels
        if deep_stem:
            self.stem = nn.Sequential(
                ConvModule(3, base_channels // 2, 3, stride=2, padding=1, norm_cfg=norm_cfg),
                ConvModule(base_channels // 2, base_channels // 2, 3, padding=1, norm_cfg=norm_cfg),
                ConvModule(base_channels // 2, base_channels, 3, padding=1, norm_cfg=norm_cfg),
            )
        else:
            self.stem = ConvModule(3, base_channels, 7, stride=2, padding=3, norm_cfg=norm_cfg)
            
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建stages
        self.stages = nn.ModuleList()
        self.out_channels = []
        
        for i in range(num_stages):
            num_blocks = stage_blocks[i]
            stride = 1 if i == 0 else 2
            out_channels = base_channels * (2 ** i)
            
            stage = self._make_stage(block, out_channels, num_blocks, stride, norm_cfg)
            self.stages.append(stage)
            self.out_channels.append(out_channels * self.expansion)
            self.in_channels = out_channels * self.expansion
            
        self._freeze_stages()
        
    def _make_stage(
        self,
        block: Type[nn.Module],
        out_channels: int,
        num_blocks: int,
        stride: int,
        norm_cfg: Dict[str, Any],
    ) -> nn.Sequential:
        """构建一个stage"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = ConvModule(
                self.in_channels, out_channels * block.expansion, 1,
                stride=stride, norm_cfg=norm_cfg, act_cfg=None,
            )
            
        layers = [block(self.in_channels, out_channels, stride=stride, downsample=downsample, norm_cfg=norm_cfg)]
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, norm_cfg=norm_cfg))
            
        return nn.Sequential(*layers)
        
    def _freeze_stages(self):
        """冻结指定的stages"""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
                
        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False
                
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播，返回多尺度特征"""
        x = self.stem(x)
        x = self.maxpool(x)
        
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
                
        return outs
        
    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()


@BACKBONES.register(name='ResNeXt')
class ResNeXt(ResNet):
    """ResNeXt骨干网络"""
    
    def __init__(
        self,
        depth: int = 50,
        groups: int = 32,
        width_per_group: int = 4,
        **kwargs,
    ):
        self.groups = groups
        self.width_per_group = width_per_group
        super().__init__(depth=depth, **kwargs)


@BACKBONES.register(name='Res2Net')
class Res2Net(ResNet):
    """Res2Net骨干网络"""
    
    def __init__(
        self,
        depth: int = 50,
        scales: int = 4,
        base_width: int = 26,
        **kwargs,
    ):
        self.scales = scales
        self.base_width = base_width
        super().__init__(depth=depth, **kwargs)

# -*- coding: utf-8 -*-
"""
Registry 模块 - 全局组件注册中心

提供统一的组件注册、发现与实例化机制。
支持装饰器注册、延迟加载、依赖注入、组件自动扫描。
"""

# 核心注册器
from .registry import (
    Registry,
    ComponentMeta,
    ComponentSource,
    ConflictPolicy,
    build_from_config
)

# 模型注册器
from .model_registry import (
    ModelRegistry,
    MODELS,
    BACKBONES,
    NECKS,
    HEADS
)

# 数据集注册器
from .dataset_registry import (
    DatasetRegistry,
    DATASETS,
    SAMPLERS,
    COLLATE_FNS
)

# 损失函数注册器
from .loss_registry import (
    LossRegistry,
    CombinedLoss,
    BaseLoss,
    FocalLoss,
    LabelSmoothingLoss,
    DiceLoss,
    IoULoss,
    LOSSES
)

# 评估指标注册器
from .metric_registry import (
    MetricRegistry,
    MetricGroup,
    BaseMetric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MeanIoU,
    ConfusionMatrix,
    METRICS
)

# 数据增强注册器
from .augmentation_registry import (
    AugmentationRegistry,
    TransformPipeline,
    BaseTransform,
    Resize,
    RandomCrop,
    RandomFlip,
    RandomRotate,
    Normalize,
    ColorJitter,
    Mixup,
    CutMix,
    TRANSFORMS,
    BATCH_TRANSFORMS
)

# 优化器注册器
from .optimizer_registry import (
    OptimizerRegistry,
    LARS,
    LAMB,
    OPTIMIZERS
)

# 学习率调度器注册器
from .scheduler_registry import (
    SchedulerRegistry,
    WarmupScheduler,
    CombinedScheduler,
    LinearWarmupCosineDecay,
    PolynomialDecay,
    SCHEDULERS
)

# 训练钩子注册器
from .hook_registry import (
    HookRegistry,
    HookPriority,
    BaseHook,
    LoggerHook,
    ProgressBarHook,
    CheckpointHook,
    BestCheckpointHook,
    EvalHook,
    GradientClipHook,
    EMAHook,
    EarlyStoppingHook,
    HOOKS
)

# 模型导出器注册器
from .exporter_registry import (
    ExporterRegistry,
    BaseExporter,
    ONNXExporter,
    TorchScriptExporter,
    OpenVINOExporter,
    TensorRTExporter,
    EXPORTERS
)

# 组件扫描器
from .component_scanner import (
    ComponentScanner,
    ScanMode,
    ScanResult,
    register_marker,
    create_scanner_from_config
)


__version__ = "1.0.0"
__author__ = "AI Training Framework Team"

__all__ = [
    # 核心类
    "Registry",
    "ComponentMeta",
    "ComponentSource",
    "ConflictPolicy",
    "build_from_config",
    
    # 模型相关
    "ModelRegistry",
    "MODELS",
    "BACKBONES", 
    "NECKS",
    "HEADS",
    
    # 数据集相关
    "DatasetRegistry",
    "DATASETS",
    "SAMPLERS",
    "COLLATE_FNS",
    
    # 损失函数相关
    "LossRegistry",
    "CombinedLoss",
    "BaseLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiceLoss",
    "IoULoss",
    "LOSSES",
    
    # 评估指标相关
    "MetricRegistry",
    "MetricGroup",
    "BaseMetric",
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "MeanIoU",
    "ConfusionMatrix",
    "METRICS",
    
    # 数据增强相关
    "AugmentationRegistry",
    "TransformPipeline",
    "BaseTransform",
    "Resize",
    "RandomCrop",
    "RandomFlip",
    "RandomRotate",
    "Normalize",
    "ColorJitter",
    "Mixup",
    "CutMix",
    "TRANSFORMS",
    "BATCH_TRANSFORMS",
    
    # 优化器相关
    "OptimizerRegistry",
    "LARS",
    "LAMB",
    "OPTIMIZERS",
    
    # 调度器相关
    "SchedulerRegistry",
    "WarmupScheduler",
    "CombinedScheduler",
    "LinearWarmupCosineDecay",
    "PolynomialDecay",
    "SCHEDULERS",
    
    # 钩子相关
    "HookRegistry",
    "HookPriority",
    "BaseHook",
    "LoggerHook",
    "ProgressBarHook",
    "CheckpointHook",
    "BestCheckpointHook",
    "EvalHook",
    "GradientClipHook",
    "EMAHook",
    "EarlyStoppingHook",
    "HOOKS",
    
    # 导出器相关
    "ExporterRegistry",
    "BaseExporter",
    "ONNXExporter",
    "TorchScriptExporter",
    "OpenVINOExporter",
    "TensorRTExporter",
    "EXPORTERS",
    
    # 扫描器相关
    "ComponentScanner",
    "ScanMode",
    "ScanResult",
    "register_marker",
    "create_scanner_from_config",
]

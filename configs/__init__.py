"""
Configs 模块 - 基于 Hydra 的配置管理系统

提供配置的加载、组合、校验、覆盖与插值功能。
支持多层级配置继承、运行时动态覆盖、类型安全校验。
"""

from configs.hydra import (
    ConfigLoader,
    ConfigComposer,
    ConfigValidator,
    OverrideParser,
    InterpolationResolver,
    load_config,
    validate_config,
    merge_configs,
    dump_config,
    resolve_interpolation,
)

from configs.schemas import (
    get_schema,
    validate_full_config,
    ModelSchema,
    DatasetSchema,
    TrainingSchema,
    AugmentationSchema,
    ExportSchema,
)

__all__ = [
    # 核心类
    "ConfigLoader",
    "ConfigComposer",
    "ConfigValidator",
    "OverrideParser",
    "InterpolationResolver",
    # 便捷函数
    "load_config",
    "validate_config",
    "merge_configs",
    "dump_config",
    "resolve_interpolation",
    # Schema
    "get_schema",
    "validate_full_config",
    "ModelSchema",
    "DatasetSchema",
    "TrainingSchema",
    "AugmentationSchema",
    "ExportSchema",
]

__version__ = "1.0.0"

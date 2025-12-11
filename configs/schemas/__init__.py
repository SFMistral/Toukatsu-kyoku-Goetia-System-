"""
配置校验 Schema 模块

提供各领域配置的校验规则。
"""

from configs.schemas.model_schema import ModelSchema
from configs.schemas.dataset_schema import DatasetSchema
from configs.schemas.training_schema import TrainingSchema
from configs.schemas.augmentation_schema import AugmentationSchema
from configs.schemas.export_schema import ExportSchema

# Schema 注册表
_SCHEMAS = {
    "model": ModelSchema,
    "dataset": DatasetSchema,
    "training": TrainingSchema,
    "augmentation": AugmentationSchema,
    "export": ExportSchema,
}


def get_schema(config_type: str):
    """
    获取指定类型的 Schema
    
    Args:
        config_type: 配置类型名称
    
    Returns:
        Schema 类
    """
    if config_type not in _SCHEMAS:
        raise ValueError(f"未知的配置类型: {config_type}")
    return _SCHEMAS[config_type]


def validate_full_config(config: dict) -> dict:
    """
    校验完整配置
    
    Args:
        config: 完整配置字典
    
    Returns:
        校验结果字典，包含各部分的校验结果
    """
    results = {}
    
    for key, schema_cls in _SCHEMAS.items():
        if key in config:
            schema = schema_cls()
            results[key] = schema.validate(config[key])
    
    return results


__all__ = [
    "get_schema",
    "validate_full_config",
    "ModelSchema",
    "DatasetSchema",
    "TrainingSchema",
    "AugmentationSchema",
    "ExportSchema",
]

"""
数据增强配置校验 Schema
"""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class AugmentationOp(BaseModel):
    """单个增强操作"""
    type: str
    prob: float = 1.0
    
    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("prob 必须在 0-1 之间")
        return v


class BatchAugmentConfig(BaseModel):
    """Batch 级增强配置"""
    type: Literal["Mixup", "CutMix", "Mosaic"]
    alpha: float = 1.0
    prob: float = 0.5
    
    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("prob 必须在 0-1 之间")
        return v


class PipelineConfig(BaseModel):
    """增强 Pipeline 配置"""
    pipeline: list[dict] = Field(default_factory=list)


class AugmentationSchema(BaseModel):
    """数据增强配置 Schema"""
    train: PipelineConfig | None = None
    val: PipelineConfig | None = None
    batch_augments: list[BatchAugmentConfig] | None = None
    
    def validate(self, config: dict) -> dict:
        """校验配置"""
        errors = []
        warnings = []
        
        # 校验 pipeline 中的每个操作
        for split in ["train", "val"]:
            if split in config and "pipeline" in config[split]:
                for i, op in enumerate(config[split]["pipeline"]):
                    if "type" not in op:
                        errors.append(f"{split}.pipeline[{i}] 缺少 type 字段")
                    if "prob" in op:
                        if not isinstance(op["prob"], (int, float)):
                            errors.append(f"{split}.pipeline[{i}].prob 必须是数字")
                        elif not 0 <= op["prob"] <= 1:
                            errors.append(f"{split}.pipeline[{i}].prob 必须在 0-1 之间")
        
        return {"errors": errors, "warnings": warnings, "is_valid": len(errors) == 0}

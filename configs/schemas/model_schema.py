"""
模型配置校验 Schema
"""

from typing import Any
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class BackboneConfig(BaseModel):
    """骨干网络配置"""
    type: str
    depth: int | None = None
    variant: str | None = None
    out_indices: list[int] | None = None
    frozen_stages: int = -1
    pretrained: bool | str = False
    
    @field_validator("frozen_stages")
    @classmethod
    def validate_frozen_stages(cls, v):
        if v < -1:
            raise ValueError("frozen_stages 必须 >= -1")
        return v


class NeckConfig(BaseModel):
    """Neck 配置"""
    type: str
    in_channels: list[int] | None = None
    out_channels: int | None = None


class LossConfig(BaseModel):
    """损失函数配置"""
    type: str
    weight: float = 1.0
    reduction: Literal["mean", "sum", "none"] = "mean"


class HeadConfig(BaseModel):
    """任务头配置"""
    type: str
    num_classes: int | None = None
    in_channels: int | list[int] | None = None
    loss: LossConfig | None = None
    
    @field_validator("num_classes")
    @classmethod
    def validate_num_classes(cls, v):
        if v is not None and v < 1:
            raise ValueError("num_classes 必须是正整数")
        return v


class PretrainedConfig(BaseModel):
    """预训练权重配置"""
    enabled: bool = False
    source: Literal["torchvision", "timm", "url", "local"] = "torchvision"
    path: str | None = None


class ModelSchema(BaseModel):
    """模型配置 Schema"""
    type: str | None = None
    backbone: BackboneConfig | None = None
    neck: NeckConfig | None = None
    head: HeadConfig | None = None
    pretrained: PretrainedConfig | bool | str | None = None
    
    def validate(self, config: dict) -> dict:
        """校验配置"""
        errors = []
        warnings = []
        
        # backbone 必须有 type
        if "backbone" in config:
            if isinstance(config["backbone"], dict):
                if "type" not in config["backbone"]:
                    errors.append("backbone 必须指定 type")
        
        # head 的 num_classes 校验
        if "head" in config:
            head = config["head"]
            if isinstance(head, dict) and "num_classes" in head:
                nc = head["num_classes"]
                if not isinstance(nc, int) or nc < 1:
                    errors.append("head.num_classes 必须是正整数")
        
        return {"errors": errors, "warnings": warnings, "is_valid": len(errors) == 0}

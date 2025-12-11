"""
数据集配置校验 Schema
"""

from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class DataSplitConfig(BaseModel):
    """数据分割配置"""
    ann_file: str | None = None
    data_prefix: str | None = None
    pipeline: list[dict] | None = None


class DataSourceConfig(BaseModel):
    """数据源配置"""
    type: str
    data_root: str
    weight: float = 1.0


class DatasetSchema(BaseModel):
    """数据集配置 Schema"""
    type: str | None = None
    data_root: str | None = None
    data_roots: list[str] | None = None
    num_classes: int | None = None
    classes_file: str | None = None
    train: DataSplitConfig | None = None
    val: DataSplitConfig | None = None
    test: DataSplitConfig | None = None
    sources: list[DataSourceConfig] | None = None
    
    @field_validator("num_classes")
    @classmethod
    def validate_num_classes(cls, v):
        if v is not None and v < 1:
            raise ValueError("num_classes 必须是正整数")
        return v
    
    @model_validator(mode="after")
    def validate_data_path(self):
        if not self.data_root and not self.data_roots and not self.sources:
            pass  # 允许为空，由上层校验
        return self
    
    def validate(self, config: dict) -> dict:
        """校验配置"""
        errors = []
        warnings = []
        
        if "num_classes" in config:
            if not isinstance(config["num_classes"], int) or config["num_classes"] < 1:
                errors.append("num_classes 必须是正整数")
        
        return {"errors": errors, "warnings": warnings, "is_valid": len(errors) == 0}

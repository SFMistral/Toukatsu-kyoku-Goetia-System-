"""
模型导出配置校验 Schema
"""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class InputSpecConfig(BaseModel):
    """输入规格配置"""
    name: str = "input"
    shape: list[int]
    dtype: Literal["float32", "float16", "int32", "int64"] = "float32"


class ONNXConfig(BaseModel):
    """ONNX 导出配置"""
    type: Literal["onnx"] = "onnx"
    opset_version: int = 13
    dynamic_axes: dict | None = None
    
    @field_validator("opset_version")
    @classmethod
    def validate_opset(cls, v):
        if v < 9 or v > 20:
            raise ValueError("opset_version 应在 9-20 之间")
        return v


class TensorRTConfig(BaseModel):
    """TensorRT 导出配置"""
    type: Literal["tensorrt"] = "tensorrt"
    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    workspace_size: int = 1 << 30  # 1GB
    calibration: dict | None = None


class ValidationConfig(BaseModel):
    """导出验证配置"""
    enabled: bool = True
    num_samples: int = 10
    tolerance: float = 1e-5
    
    @field_validator("num_samples")
    @classmethod
    def validate_num_samples(cls, v):
        if v < 1:
            raise ValueError("num_samples 必须是正整数")
        return v


class CompressionConfig(BaseModel):
    """压缩配置"""
    quantization: dict | None = None
    pruning: dict | None = None
    distillation: dict | None = None


class ExportSchema(BaseModel):
    """导出配置 Schema"""
    enabled: bool = False
    formats: list[dict] = Field(default_factory=list)
    input_spec: InputSpecConfig | None = None
    validation: ValidationConfig | None = None
    compression: CompressionConfig | None = None
    
    def validate(self, config: dict) -> dict:
        """校验配置"""
        errors = []
        warnings = []
        
        valid_formats = {"onnx", "tensorrt", "openvino", "coreml", "ncnn"}
        
        if "formats" in config:
            for i, fmt in enumerate(config["formats"]):
                if isinstance(fmt, dict):
                    if "type" not in fmt:
                        errors.append(f"formats[{i}] 缺少 type 字段")
                    elif fmt["type"] not in valid_formats:
                        errors.append(f"formats[{i}].type 不支持: {fmt['type']}")
        
        return {"errors": errors, "warnings": warnings, "is_valid": len(errors) == 0}

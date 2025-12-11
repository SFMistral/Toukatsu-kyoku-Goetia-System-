"""
训练配置校验 Schema
"""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class DistributedConfig(BaseModel):
    """分布式训练配置"""
    enabled: bool = False
    backend: Literal["nccl", "gloo", "mpi"] | None = None
    world_size: int | str = "auto"


class MixedPrecisionConfig(BaseModel):
    """混合精度配置"""
    enabled: bool = False
    dtype: Literal["fp16", "bf16"] = "fp16"


class GradientClipConfig(BaseModel):
    """梯度裁剪配置"""
    enabled: bool = False
    max_norm: float = 1.0
    norm_type: float = 2.0


class EarlyStoppingConfig(BaseModel):
    """早停配置"""
    enabled: bool = False
    monitor: str = "val_loss"
    patience: int = 10
    mode: Literal["min", "max"] = "min"
    
    @field_validator("patience")
    @classmethod
    def validate_patience(cls, v):
        if v < 1:
            raise ValueError("patience 必须是正整数")
        return v


class EMAConfig(BaseModel):
    """EMA 配置"""
    enabled: bool = False
    decay: float = 0.9999


class CheckpointConfig(BaseModel):
    """检查点配置"""
    save_interval: int = 1
    save_best: bool = True
    max_keep: int = 5


class LoggingConfig(BaseModel):
    """日志配置"""
    interval: int = 50
    tensorboard: bool = True
    wandb: bool = False


class TrainingSchema(BaseModel):
    """训练配置 Schema"""
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    distributed: DistributedConfig | None = None
    mixed_precision: MixedPrecisionConfig | None = None
    gradient_accumulation: dict | None = None
    gradient_clip: GradientClipConfig | None = None
    early_stopping: EarlyStoppingConfig | None = None
    ema: EMAConfig | None = None
    checkpoint: CheckpointConfig | None = None
    logging: LoggingConfig | None = None
    
    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, v):
        if v < 1 or v > 10000:
            raise ValueError("epochs 必须在 1-10000 之间")
        return v
    
    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1 or v > 1024:
            raise ValueError("batch_size 必须在 1-1024 之间")
        return v
    
    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, v):
        if v < 0 or v > 64:
            raise ValueError("num_workers 必须在 0-64 之间")
        return v
    
    def validate(self, config: dict) -> dict:
        """校验配置"""
        errors = []
        warnings = []
        
        # epochs 范围
        if "epochs" in config:
            e = config["epochs"]
            if not isinstance(e, int) or e < 1 or e > 10000:
                errors.append("epochs 必须是 1-10000 的整数")
        
        # batch_size 范围
        if "batch_size" in config:
            bs = config["batch_size"]
            if not isinstance(bs, int) or bs < 1 or bs > 1024:
                errors.append("batch_size 必须是 1-1024 的整数")
        
        # 分布式依赖检查
        if "distributed" in config:
            dist = config["distributed"]
            if isinstance(dist, dict) and dist.get("enabled"):
                if "backend" not in dist:
                    errors.append("启用分布式训练时必须指定 backend")
        
        return {"errors": errors, "warnings": warnings, "is_valid": len(errors) == 0}

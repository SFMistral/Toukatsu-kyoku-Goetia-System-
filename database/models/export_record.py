"""
导出记录模型
"""
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, ForeignKey, BigInteger, Float, Boolean, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from .base import BaseModel


class ExportFormat(Enum):
    """导出格式枚举"""
    ONNX = "onnx"              # ONNX格式
    TORCHSCRIPT = "torchscript"  # TorchScript格式
    OPENVINO = "openvino"      # OpenVINO格式
    TENSORRT = "tensorrt"      # TensorRT格式
    COREML = "coreml"          # CoreML格式
    OTHER = "other"            # 其他格式


class ExportStatus(Enum):
    """导出状态枚举"""
    PENDING = "pending"      # 等待中
    RUNNING = "running"      # 运行中
    SUCCESS = "success"      # 成功
    FAILED = "failed"        # 失败


class ExportRecord(BaseModel):
    """导出记录模型"""
    __tablename__ = 'export_records'
    
    # 基本信息
    name = Column(String(255), nullable=False, comment='导出名称')
    export_format = Column(SQLEnum(ExportFormat), nullable=False, comment='导出格式')
    export_type = Column(String(100), nullable=False, comment='导出类型')
    status = Column(SQLEnum(ExportStatus), default=ExportStatus.PENDING, nullable=False, comment='导出状态')
    
    # 来源信息
    task_id = Column(Integer, ForeignKey('tasks.id'), comment='来源任务ID')
    checkpoint_id = Column(Integer, ForeignKey('checkpoints.id'), comment='来源检查点ID')
    
    # 输入输出规格
    input_spec = Column(Text, comment='输入规格JSON')
    export_config = Column(Text, comment='导出配置JSON')
    
    # 输出信息
    output_path = Column(String(512), comment='输出路径')
    file_path = Column(String(512), nullable=False, comment='导出文件路径')
    file_size = Column(BigInteger, comment='文件大小(字节)')
    
    # 验证信息
    validation_passed = Column(Boolean, comment='验证是否通过')
    validation_report = Column(Text, comment='验证报告JSON')
    
    # 错误信息
    error_message = Column(Text, comment='错误信息')
    export_duration = Column(Float, comment='导出耗时(秒)')
    
    # 关联信息
    user_id = Column(Integer, ForeignKey('users.id'), comment='创建用户ID')
    
    # 额外信息
    description = Column(Text, comment='描述')
    meta_data = Column(Text, comment='元数据JSON')
    
    # 关系
    task = relationship("Task")
    checkpoint = relationship("Checkpoint")
    user = relationship("User")
    
    # 索引
    __table_args__ = (
        Index('idx_task_id', 'task_id'),
        Index('idx_checkpoint_id', 'checkpoint_id'),
        Index('idx_export_format', 'export_format'),
    )
    
    def __repr__(self):
        return f"<ExportRecord(id={self.id}, name='{self.name}', format='{self.export_format.value}', status='{self.status.value}')>"
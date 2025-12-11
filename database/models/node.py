"""
节点模型
"""
import uuid
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, BigInteger, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship
from .base import BaseModel


class NodeStatus(Enum):
    """节点状态枚举"""
    ONLINE = "online"        # 在线
    OFFLINE = "offline"      # 离线
    BUSY = "busy"           # 忙碌
    MAINTENANCE = "maintenance"  # 维护中
    ERROR = "error"         # 错误


class NodeType(Enum):
    """节点类型枚举"""
    MASTER = "master"    # 主节点
    WORKER = "worker"    # 工作节点


class Node(BaseModel):
    """计算节点模型"""
    __tablename__ = 'nodes'
    
    # 基本信息
    node_id = Column(String(64), unique=True, default=lambda: str(uuid.uuid4()), comment='节点唯一标识')
    name = Column(String(255), nullable=False, unique=True, comment='节点名称')
    host = Column(String(255), nullable=False, comment='节点主机地址')
    port = Column(Integer, nullable=False, comment='节点端口')
    status = Column(SQLEnum(NodeStatus), default=NodeStatus.OFFLINE, nullable=False, comment='节点状态')
    node_type = Column(SQLEnum(NodeType), default=NodeType.WORKER, nullable=False, comment='节点类型')
    
    # 节点信息
    description = Column(Text, comment='节点描述')
    tags = Column(Text, comment='节点标签JSON')
    capabilities = Column(Text, comment='节点能力标签JSON')
    
    # 硬件信息
    cpu_cores = Column(Integer, comment='CPU核心数')
    memory_total = Column(BigInteger, comment='总内存(字节)')
    memory_gb = Column(Float, comment='内存大小(GB)')
    disk_total = Column(BigInteger, comment='总磁盘(字节)')
    gpu_count = Column(Integer, default=0, comment='GPU数量')
    gpu_info = Column(Text, comment='GPU详细信息JSON')
    
    # 状态信息
    cpu_usage = Column(Float, default=0.0, comment='CPU使用率')
    memory_usage = Column(Float, default=0.0, comment='内存使用率')
    gpu_usage = Column(Float, default=0.0, comment='GPU使用率')
    
    # 任务信息
    current_task_id = Column(Integer, comment='当前执行任务ID')
    max_concurrent_tasks = Column(Integer, default=1, comment='最大并发任务数')
    
    # 心跳信息
    last_heartbeat = Column(DateTime, comment='最后心跳时间')
    
    # 配置
    is_active = Column(Boolean, default=True, comment='是否激活')
    
    # 关系
    tasks = relationship("Task", back_populates="node", foreign_keys="Task.node_id")
    experiments = relationship("Experiment", back_populates="node")
    
    def __repr__(self):
        return f"<Node(id={self.id}, node_id='{self.node_id}', name='{self.name}', status='{self.status.value}')>"
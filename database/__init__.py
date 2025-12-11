"""
数据库模块
"""
from .connection import (
    DatabaseConnection, 
    init_database, 
    get_db_connection, 
    get_db_session,
    Base
)
from .cloud_connection import CloudDatabaseConnection, init_cloud_database

# 导入所有模型
from .models import *

# 导入所有仓库
from .repositories import *

__version__ = "1.0.0"

__all__ = [
    # 连接管理
    'DatabaseConnection',
    'CloudDatabaseConnection', 
    'init_database',
    'init_cloud_database',
    'get_db_connection',
    'get_db_session',
    'Base',
    
    # 模型
    'BaseModel',
    'Task', 'TaskStatus',
    'Node', 'NodeStatus', 
    'Experiment', 'ExperimentStatus',
    'MetricRecord',
    'Checkpoint',
    'Artifact',
    'EnvironmentSnapshot',
    'ConfigSnapshot',
    'ComparisonRecord',
    'ExportRecord',
    'Report',
    'User',
    'SystemLog',
    
    # 仓库
    'TaskRepository',
    'NodeRepository',
    'ExperimentRepository',
    'MetricRepository',
    'CheckpointRepository',
    'ArtifactRepository',
    'ConfigRepository',
    'ExportRepository',
    'ReportRepository',
    'UserRepository',
    'CSVExporter'
]
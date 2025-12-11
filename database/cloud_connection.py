"""
云数据库连接模块
"""
import logging
from typing import Optional, Dict, Any
from database.connection import DatabaseConnection

logger = logging.getLogger(__name__)

class CloudDatabaseConnection(DatabaseConnection):
    """云数据库连接管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cloud_config = config.get('cloud', {})
        
    def _build_database_url(self) -> str:
        """构建云数据库连接URL"""
        # 如果配置了云数据库，使用云数据库配置
        if self.cloud_config.get('enabled', False):
            return self._build_cloud_url()
        else:
            # 否则使用本地数据库配置
            return super()._build_database_url()
    
    def _build_cloud_url(self) -> str:
        """构建云数据库URL"""
        provider = self.cloud_config.get('provider', 'aws')
        
        if provider == 'aws':
            return self._build_aws_rds_url()
        elif provider == 'gcp':
            return self._build_gcp_sql_url()
        elif provider == 'azure':
            return self._build_azure_sql_url()
        else:
            raise ValueError(f"不支持的云数据库提供商: {provider}")
    
    def _build_aws_rds_url(self) -> str:
        """构建AWS RDS连接URL"""
        host = self.cloud_config['host']
        port = self.cloud_config.get('port', 3306)
        database = self.cloud_config['database']
        username = self.cloud_config['username']
        password = self.cloud_config['password']
        
        # AWS RDS MySQL连接
        return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    
    def _build_gcp_sql_url(self) -> str:
        """构建GCP Cloud SQL连接URL"""
        # GCP Cloud SQL连接配置
        connection_name = self.cloud_config['connection_name']
        database = self.cloud_config['database']
        username = self.cloud_config['username']
        password = self.cloud_config['password']
        
        # 使用Unix socket连接
        socket_path = f"/cloudsql/{connection_name}"
        return f"mysql+pymysql://{username}:{password}@/{database}?unix_socket={socket_path}"
    
    def _build_azure_sql_url(self) -> str:
        """构建Azure SQL连接URL"""
        server = self.cloud_config['server']
        database = self.cloud_config['database']
        username = self.cloud_config['username']
        password = self.cloud_config['password']
        
        # Azure SQL Server连接
        return f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    
    def initialize(self) -> None:
        """初始化云数据库连接"""
        try:
            # 设置云数据库特定的连接参数
            if self.cloud_config.get('enabled', False):
                self._setup_cloud_connection_args()
            
            super().initialize()
            
            if self.cloud_config.get('enabled', False):
                logger.info(f"云数据库连接初始化成功: {self.cloud_config.get('provider', 'unknown')}")
            
        except Exception as e:
            logger.error(f"云数据库连接初始化失败: {e}")
            raise
    
    def _setup_cloud_connection_args(self) -> None:
        """设置云数据库连接参数"""
        provider = self.cloud_config.get('provider', 'aws')
        
        if provider == 'aws':
            # AWS RDS特定配置
            self.config.setdefault('connect_args', {}).update({
                'connect_timeout': 60,
                'read_timeout': 60,
                'write_timeout': 60
            })
        elif provider == 'gcp':
            # GCP Cloud SQL特定配置
            self.config.setdefault('connect_args', {}).update({
                'connect_timeout': 60
            })
        elif provider == 'azure':
            # Azure SQL特定配置
            self.config.setdefault('connect_args', {}).update({
                'timeout': 60,
                'login_timeout': 60
            })

def init_cloud_database(config: Dict[str, Any]) -> CloudDatabaseConnection:
    """初始化云数据库连接"""
    cloud_db = CloudDatabaseConnection(config)
    cloud_db.initialize()
    return cloud_db
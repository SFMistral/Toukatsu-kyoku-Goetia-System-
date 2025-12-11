"""
数据库连接管理模块
"""
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
import pymysql

logger = logging.getLogger(__name__)

# 创建基础模型类
Base = declarative_base()

class DatabaseConnection:
    """数据库连接管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        
    def initialize(self) -> None:
        """初始化数据库连接"""
        try:
            # 构建数据库URL
            db_url = self._build_database_url()
            
            # 创建引擎
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.config.get('pool_size', 10),
                max_overflow=self.config.get('max_overflow', 20),
                pool_timeout=self.config.get('pool_timeout', 30),
                pool_recycle=self.config.get('pool_recycle', 3600),
                echo=self.config.get('echo', False),
                connect_args=self.config.get('connect_args', {})
            )
            
            # 创建会话工厂
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            db_info = f"{self.config['type']}"
            if 'host' in self.config and 'port' in self.config:
                db_info += f": {self.config['host']}:{self.config['port']}"
            elif 'name' in self.config:
                db_info += f": {self.config['name']}"
            logger.info(f"数据库连接初始化成功: {db_info}")
            
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise
    
    def _build_database_url(self) -> str:
        """构建数据库连接URL"""
        db_type = self.config['type']
        
        if db_type == 'sqlite':
            name = self.config['name']
            return f"sqlite:///{name}"
        elif db_type == 'mysql':
            host = self.config['host']
            port = self.config['port']
            name = self.config['name']
            user = self.config['user']
            password = self.config['password']
            charset = self.config.get('charset', 'utf8mb4')
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset={charset}"
        elif db_type == 'postgresql':
            host = self.config['host']
            port = self.config['port']
            name = self.config['name']
            user = self.config['user']
            password = self.config['password']
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
    
    @contextmanager
    def get_session(self):
        """获取数据库会话上下文管理器"""
        if not self.SessionLocal:
            raise RuntimeError("数据库连接未初始化")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self) -> None:
        """创建所有表"""
        if not self.engine:
            raise RuntimeError("数据库连接未初始化")
        
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("数据库表创建成功")
        except Exception as e:
            logger.error(f"数据库表创建失败: {e}")
            raise
    
    def drop_tables(self) -> None:
        """删除所有表"""
        if not self.engine:
            raise RuntimeError("数据库连接未初始化")
        
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("数据库表删除成功")
        except Exception as e:
            logger.error(f"数据库表删除失败: {e}")
            raise
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("数据库连接已关闭")

# 全局数据库连接实例
db_connection: Optional[DatabaseConnection] = None

def init_database(config: Dict[str, Any]) -> DatabaseConnection:
    """初始化数据库连接"""
    global db_connection
    db_connection = DatabaseConnection(config)
    db_connection.initialize()
    return db_connection

def get_db_connection() -> DatabaseConnection:
    """获取数据库连接实例"""
    if not db_connection:
        raise RuntimeError("数据库连接未初始化")
    return db_connection

def get_db_session():
    """获取数据库会话"""
    return get_db_connection().get_session()
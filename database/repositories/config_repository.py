"""
配置仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from database.models.config_snapshot import ConfigSnapshot
from database.models.environment_snapshot import EnvironmentSnapshot
from database.connection import get_db_session

class ConfigRepository:
    """配置数据访问层"""
    
    def create_config_snapshot(self, config_data: Dict[str, Any]) -> ConfigSnapshot:
        """创建配置快照"""
        with get_db_session() as session:
            config = ConfigSnapshot(**config_data)
            session.add(config)
            session.flush()
            session.refresh(config)
            return config
    
    def create_environment_snapshot(self, env_data: Dict[str, Any]) -> EnvironmentSnapshot:
        """创建环境快照"""
        with get_db_session() as session:
            env = EnvironmentSnapshot(**env_data)
            session.add(env)
            session.flush()
            session.refresh(env)
            return env
    
    def get_config_by_id(self, config_id: int) -> Optional[ConfigSnapshot]:
        """根据ID获取配置快照"""
        with get_db_session() as session:
            return session.query(ConfigSnapshot).filter(ConfigSnapshot.id == config_id).first()
    
    def get_environment_by_id(self, env_id: int) -> Optional[EnvironmentSnapshot]:
        """根据ID获取环境快照"""
        with get_db_session() as session:
            return session.query(EnvironmentSnapshot).filter(EnvironmentSnapshot.id == env_id).first()
    
    def get_configs_by_experiment(self, experiment_id: int) -> List[ConfigSnapshot]:
        """根据实验ID获取配置快照"""
        with get_db_session() as session:
            return session.query(ConfigSnapshot).filter(
                ConfigSnapshot.experiment_id == experiment_id
            ).order_by(desc(ConfigSnapshot.created_at)).all()
    
    def get_environments_by_experiment(self, experiment_id: int) -> List[EnvironmentSnapshot]:
        """根据实验ID获取环境快照"""
        with get_db_session() as session:
            return session.query(EnvironmentSnapshot).filter(
                EnvironmentSnapshot.experiment_id == experiment_id
            ).order_by(desc(EnvironmentSnapshot.created_at)).all()
    
    def get_config_by_type(self, experiment_id: int, config_type: str) -> List[ConfigSnapshot]:
        """根据配置类型获取快照"""
        with get_db_session() as session:
            return session.query(ConfigSnapshot).filter(
                and_(
                    ConfigSnapshot.experiment_id == experiment_id,
                    ConfigSnapshot.config_type == config_type
                )
            ).order_by(desc(ConfigSnapshot.created_at)).all()
    
    def get_config_by_hash(self, file_hash: str) -> Optional[ConfigSnapshot]:
        """根据文件哈希获取配置"""
        with get_db_session() as session:
            return session.query(ConfigSnapshot).filter(
                ConfigSnapshot.file_hash == file_hash
            ).first()
    
    def update_config(self, config_id: int, update_data: Dict[str, Any]) -> Optional[ConfigSnapshot]:
        """更新配置快照"""
        with get_db_session() as session:
            config = session.query(ConfigSnapshot).filter(ConfigSnapshot.id == config_id).first()
            if config:
                for key, value in update_data.items():
                    setattr(config, key, value)
                session.flush()
                session.refresh(config)
            return config
    
    def update_environment(self, env_id: int, update_data: Dict[str, Any]) -> Optional[EnvironmentSnapshot]:
        """更新环境快照"""
        with get_db_session() as session:
            env = session.query(EnvironmentSnapshot).filter(EnvironmentSnapshot.id == env_id).first()
            if env:
                for key, value in update_data.items():
                    setattr(env, key, value)
                session.flush()
                session.refresh(env)
            return env
    
    def delete_config(self, config_id: int) -> bool:
        """删除配置快照"""
        with get_db_session() as session:
            config = session.query(ConfigSnapshot).filter(ConfigSnapshot.id == config_id).first()
            if config:
                session.delete(config)
                return True
            return False
    
    def delete_environment(self, env_id: int) -> bool:
        """删除环境快照"""
        with get_db_session() as session:
            env = session.query(EnvironmentSnapshot).filter(EnvironmentSnapshot.id == env_id).first()
            if env:
                session.delete(env)
                return True
            return False
    
    def get_config_types(self, experiment_id: int) -> List[str]:
        """获取实验的所有配置类型"""
        with get_db_session() as session:
            result = session.query(ConfigSnapshot.config_type).filter(
                ConfigSnapshot.experiment_id == experiment_id
            ).distinct().all()
            return [row[0] for row in result]
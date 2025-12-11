"""
工件仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from database.models.artifact import Artifact
from database.connection import get_db_session

class ArtifactRepository:
    """工件数据访问层"""
    
    def create(self, artifact_data: Dict[str, Any]) -> Artifact:
        """创建工件"""
        with get_db_session() as session:
            artifact = Artifact(**artifact_data)
            session.add(artifact)
            session.flush()
            session.refresh(artifact)
            return artifact
    
    def get_by_id(self, artifact_id: int) -> Optional[Artifact]:
        """根据ID获取工件"""
        with get_db_session() as session:
            return session.query(Artifact).filter(Artifact.id == artifact_id).first()
    
    def get_by_experiment_id(self, experiment_id: int) -> List[Artifact]:
        """根据实验ID获取所有工件"""
        with get_db_session() as session:
            return session.query(Artifact).filter(
                Artifact.experiment_id == experiment_id
            ).order_by(desc(Artifact.created_at)).all()
    
    def get_by_type(self, experiment_id: int, artifact_type: str) -> List[Artifact]:
        """根据类型获取工件"""
        with get_db_session() as session:
            return session.query(Artifact).filter(
                and_(
                    Artifact.experiment_id == experiment_id,
                    Artifact.artifact_type == artifact_type
                )
            ).order_by(desc(Artifact.created_at)).all()
    
    def get_by_name(self, experiment_id: int, name: str) -> Optional[Artifact]:
        """根据名称获取工件"""
        with get_db_session() as session:
            return session.query(Artifact).filter(
                and_(
                    Artifact.experiment_id == experiment_id,
                    Artifact.name == name
                )
            ).first()
    
    def update(self, artifact_id: int, update_data: Dict[str, Any]) -> Optional[Artifact]:
        """更新工件"""
        with get_db_session() as session:
            artifact = session.query(Artifact).filter(Artifact.id == artifact_id).first()
            if artifact:
                for key, value in update_data.items():
                    setattr(artifact, key, value)
                session.flush()
                session.refresh(artifact)
            return artifact
    
    def delete(self, artifact_id: int) -> bool:
        """删除工件"""
        with get_db_session() as session:
            artifact = session.query(Artifact).filter(Artifact.id == artifact_id).first()
            if artifact:
                session.delete(artifact)
                return True
            return False
    
    def delete_by_experiment_id(self, experiment_id: int) -> int:
        """删除实验的所有工件"""
        with get_db_session() as session:
            count = session.query(Artifact).filter(
                Artifact.experiment_id == experiment_id
            ).count()
            session.query(Artifact).filter(
                Artifact.experiment_id == experiment_id
            ).delete()
            return count
    
    def get_artifact_types(self, experiment_id: int) -> List[str]:
        """获取实验的所有工件类型"""
        with get_db_session() as session:
            result = session.query(Artifact.artifact_type).filter(
                Artifact.experiment_id == experiment_id
            ).distinct().all()
            return [row[0] for row in result]
    
    def search_by_tags(self, experiment_id: int, tags: List[str]) -> List[Artifact]:
        """根据标签搜索工件"""
        with get_db_session() as session:
            # 这里简化处理，实际应该解析JSON标签
            query = session.query(Artifact).filter(
                Artifact.experiment_id == experiment_id
            )
            
            for tag in tags:
                query = query.filter(Artifact.tags.contains(tag))
            
            return query.order_by(desc(Artifact.created_at)).all()
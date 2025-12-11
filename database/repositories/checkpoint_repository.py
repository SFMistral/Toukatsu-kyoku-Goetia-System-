"""
检查点仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc
from database.models.checkpoint import Checkpoint
from database.connection import get_db_session

class CheckpointRepository:
    """检查点数据访问层"""
    
    def create(self, checkpoint_data: Dict[str, Any]) -> Checkpoint:
        """创建检查点"""
        with get_db_session() as session:
            checkpoint = Checkpoint(**checkpoint_data)
            session.add(checkpoint)
            session.flush()
            session.refresh(checkpoint)
            return checkpoint
    
    def get_by_id(self, checkpoint_id: int) -> Optional[Checkpoint]:
        """根据ID获取检查点"""
        with get_db_session() as session:
            return session.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
    
    def get_by_experiment_id(self, experiment_id: int) -> List[Checkpoint]:
        """根据实验ID获取所有检查点"""
        with get_db_session() as session:
            return session.query(Checkpoint).filter(
                Checkpoint.experiment_id == experiment_id
            ).order_by(desc(Checkpoint.epoch)).all()
    
    def get_best_checkpoint(self, experiment_id: int) -> Optional[Checkpoint]:
        """获取最佳检查点"""
        with get_db_session() as session:
            return session.query(Checkpoint).filter(
                and_(
                    Checkpoint.experiment_id == experiment_id,
                    Checkpoint.is_best == True
                )
            ).first()
    
    def get_latest_checkpoint(self, experiment_id: int) -> Optional[Checkpoint]:
        """获取最新检查点"""
        with get_db_session() as session:
            return session.query(Checkpoint).filter(
                Checkpoint.experiment_id == experiment_id
            ).order_by(desc(Checkpoint.epoch), desc(Checkpoint.step)).first()
    
    def get_by_epoch(self, experiment_id: int, epoch: int) -> List[Checkpoint]:
        """根据轮次获取检查点"""
        with get_db_session() as session:
            return session.query(Checkpoint).filter(
                and_(
                    Checkpoint.experiment_id == experiment_id,
                    Checkpoint.epoch == epoch
                )
            ).order_by(desc(Checkpoint.step)).all()
    
    def update(self, checkpoint_id: int, update_data: Dict[str, Any]) -> Optional[Checkpoint]:
        """更新检查点"""
        with get_db_session() as session:
            checkpoint = session.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
            if checkpoint:
                for key, value in update_data.items():
                    setattr(checkpoint, key, value)
                session.flush()
                session.refresh(checkpoint)
            return checkpoint
    
    def set_as_best(self, checkpoint_id: int) -> Optional[Checkpoint]:
        """设置为最佳检查点"""
        with get_db_session() as session:
            # 先取消其他最佳检查点
            checkpoint = session.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
            if checkpoint:
                session.query(Checkpoint).filter(
                    and_(
                        Checkpoint.experiment_id == checkpoint.experiment_id,
                        Checkpoint.is_best == True
                    )
                ).update({'is_best': False})
                
                # 设置当前为最佳
                checkpoint.is_best = True
                session.flush()
                session.refresh(checkpoint)
            return checkpoint
    
    def delete(self, checkpoint_id: int) -> bool:
        """删除检查点"""
        with get_db_session() as session:
            checkpoint = session.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
            if checkpoint:
                session.delete(checkpoint)
                return True
            return False
    
    def delete_by_experiment_id(self, experiment_id: int) -> int:
        """删除实验的所有检查点"""
        with get_db_session() as session:
            count = session.query(Checkpoint).filter(
                Checkpoint.experiment_id == experiment_id
            ).count()
            session.query(Checkpoint).filter(
                Checkpoint.experiment_id == experiment_id
            ).delete()
            return count
    
    def get_checkpoints_by_score_range(self, experiment_id: int, 
                                     min_score: float = None, 
                                     max_score: float = None) -> List[Checkpoint]:
        """根据得分范围获取检查点"""
        with get_db_session() as session:
            query = session.query(Checkpoint).filter(
                Checkpoint.experiment_id == experiment_id
            )
            
            if min_score is not None:
                query = query.filter(Checkpoint.score >= min_score)
            if max_score is not None:
                query = query.filter(Checkpoint.score <= max_score)
            
            return query.order_by(desc(Checkpoint.score)).all()
    
    def cleanup_old_checkpoints(self, experiment_id: int, keep_count: int = 5) -> int:
        """清理旧检查点，保留指定数量"""
        with get_db_session() as session:
            # 获取所有检查点，按轮次降序排列
            checkpoints = session.query(Checkpoint).filter(
                Checkpoint.experiment_id == experiment_id
            ).order_by(desc(Checkpoint.epoch), desc(Checkpoint.step)).all()
            
            if len(checkpoints) <= keep_count:
                return 0
            
            # 保留最佳检查点和最新的几个检查点
            best_checkpoint = session.query(Checkpoint).filter(
                and_(
                    Checkpoint.experiment_id == experiment_id,
                    Checkpoint.is_best == True
                )
            ).first()
            
            to_keep = set()
            if best_checkpoint:
                to_keep.add(best_checkpoint.id)
            
            # 保留最新的检查点
            for checkpoint in checkpoints[:keep_count]:
                to_keep.add(checkpoint.id)
            
            # 删除其他检查点
            deleted_count = 0
            for checkpoint in checkpoints:
                if checkpoint.id not in to_keep:
                    session.delete(checkpoint)
                    deleted_count += 1
            
            return deleted_count
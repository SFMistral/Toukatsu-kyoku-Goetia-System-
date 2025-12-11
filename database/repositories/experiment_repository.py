"""
实验仓库
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from database.models.experiment import Experiment, ExperimentStatus
from database.connection import get_db_session

class ExperimentRepository:
    """实验数据访问层"""
    
    def create(self, experiment_data: Dict[str, Any]) -> Experiment:
        """创建实验"""
        with get_db_session() as session:
            experiment = Experiment(**experiment_data)
            session.add(experiment)
            session.flush()
            session.refresh(experiment)
            session.expunge(experiment)
            return experiment
    
    def get_by_id(self, experiment_id: int) -> Optional[Experiment]:
        """根据ID获取实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    def get_by_name(self, name: str) -> Optional[Experiment]:
        """根据名称获取实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(Experiment.name == name).first()
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Experiment]:
        """获取所有实验"""
        with get_db_session() as session:
            return session.query(Experiment).order_by(desc(Experiment.created_at)).offset(offset).limit(limit).all()
    
    def get_by_task_id(self, task_id: int) -> List[Experiment]:
        """根据任务ID获取实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(Experiment.task_id == task_id).all()
    
    def get_by_node_id(self, node_id: int) -> List[Experiment]:
        """根据节点ID获取实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(Experiment.node_id == node_id).all()
    
    def get_by_user_id(self, user_id: int) -> List[Experiment]:
        """根据用户ID获取实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(Experiment.user_id == user_id).all()
    
    def get_by_status(self, status: ExperimentStatus) -> List[Experiment]:
        """根据状态获取实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(Experiment.status == status).all()
    
    def get_running_experiments(self) -> List[Experiment]:
        """获取运行中的实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(Experiment.status == ExperimentStatus.RUNNING).all()
    
    def get_completed_experiments(self, limit: int = 50) -> List[Experiment]:
        """获取已完成的实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(
                Experiment.status == ExperimentStatus.COMPLETED
            ).order_by(desc(Experiment.updated_at)).limit(limit).all()
    
    def get_best_experiments(self, limit: int = 10) -> List[Experiment]:
        """获取最佳实验（按最终得分排序）"""
        with get_db_session() as session:
            return session.query(Experiment).filter(
                and_(
                    Experiment.status == ExperimentStatus.COMPLETED,
                    Experiment.final_score.isnot(None)
                )
            ).order_by(desc(Experiment.final_score)).limit(limit).all()
    
    def update(self, experiment_id: int, update_data: Dict[str, Any]) -> Optional[Experiment]:
        """更新实验"""
        with get_db_session() as session:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                for key, value in update_data.items():
                    setattr(experiment, key, value)
                session.flush()
                session.refresh(experiment)
                session.expunge(experiment)
            return experiment
    
    def update_status(self, experiment_id: int, status: ExperimentStatus) -> Optional[Experiment]:
        """更新实验状态"""
        return self.update(experiment_id, {'status': status})
    
    def update_score(self, experiment_id: int, final_score: float, best_epoch: int = None) -> Optional[Experiment]:
        """更新实验得分"""
        update_data = {'final_score': final_score}
        if best_epoch is not None:
            update_data['best_epoch'] = best_epoch
        return self.update(experiment_id, update_data)
    
    def delete(self, experiment_id: int) -> bool:
        """删除实验"""
        with get_db_session() as session:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                session.delete(experiment)
                return True
            return False
    
    def search(self, keyword: str, limit: int = 50) -> List[Experiment]:
        """搜索实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(
                or_(
                    Experiment.name.contains(keyword),
                    Experiment.description.contains(keyword)
                )
            ).limit(limit).all()
    
    def count_by_status(self) -> Dict[str, int]:
        """按状态统计实验数量"""
        with get_db_session() as session:
            result = {}
            for status in ExperimentStatus:
                count = session.query(Experiment).filter(Experiment.status == status).count()
                result[status.value] = count
            return result
    
    def get_experiments_by_date_range(self, start_date, end_date) -> List[Experiment]:
        """根据日期范围获取实验"""
        with get_db_session() as session:
            return session.query(Experiment).filter(
                and_(
                    Experiment.created_at >= start_date,
                    Experiment.created_at <= end_date
                )
            ).order_by(desc(Experiment.created_at)).all()
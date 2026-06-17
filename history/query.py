import json
import csv
from datetime import datetime, timedelta
from typing import Any

from .storage import HistoryStorage


class HistoryQuery:
    """训练历史记录查询与筛选。"""

    def __init__(self, storage: HistoryStorage):
        self._storage = storage

    def list_runs(self, status: str | None = None, limit: int = 50,
                  offset: int = 0) -> list[dict]:
        """列出训练运行，可选按状态筛选。"""
        with self._storage._get_conn() as conn:
            if status:
                rows = conn.execute(
                    """SELECT id, task_name, start_time, end_time, status,
                              dataset_name, model_type, total_epochs, current_epoch,
                              best_loss, best_accuracy, device
                       FROM training_runs
                       WHERE status = ?
                       ORDER BY start_time DESC LIMIT ? OFFSET ?""",
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, task_name, start_time, end_time, status,
                              dataset_name, model_type, total_epochs, current_epoch,
                              best_loss, best_accuracy, device
                       FROM training_runs
                       ORDER BY start_time DESC LIMIT ? OFFSET ?""",
                    (limit, offset),
                ).fetchall()
            return [dict(r) for r in rows]

    def filter_runs(self, *, dataset_name: str | None = None,
                    model_type: str | None = None,
                    date_from: str | None = None,
                    date_to: str | None = None,
                    status: str | None = None,
                    limit: int = 50) -> list[dict]:
        """按多条件组合筛选训练运行。"""
        conditions = []
        params = []

        if dataset_name:
            conditions.append("dataset_name LIKE ?")
            params.append(f"%{dataset_name}%")
        if model_type:
            conditions.append("model_type LIKE ?")
            params.append(f"%{model_type}%")
        if date_from:
            conditions.append("start_time >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("start_time <= ?")
            params.append(date_to)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"""SELECT id, task_name, start_time, end_time, status,
                           dataset_name, model_type, total_epochs, current_epoch,
                           best_loss, best_accuracy, device
                    FROM training_runs
                    WHERE {where}
                    ORDER BY start_time DESC LIMIT ?"""
        params.append(limit)

        with self._storage._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_full_config(self, run_id: int) -> dict | None:
        """获取某次运行的完整配置。"""
        run = self._storage.get_run(run_id)
        if run and run.get("config_json"):
            return json.loads(run["config_json"])
        return None

    def get_metrics_summary(self, run_id: int) -> list[dict]:
        """获取某次运行的指标摘要（每个 epoch 的均值）。"""
        with self._storage._get_conn() as conn:
            rows = conn.execute(
                """SELECT epoch, phase,
                          AVG(loss) as avg_loss,
                          AVG(accuracy) as avg_accuracy,
                          AVG(precision) as avg_precision,
                          AVG(recall) as avg_recall
                   FROM training_metrics
                   WHERE run_id = ?
                   GROUP BY epoch, phase
                   ORDER BY epoch""",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def export_metrics_csv(self, run_id: int, output_path: str):
        """将某次运行的指标导出为 CSV。"""
        metrics = self._storage.get_run_metrics(run_id)
        if not metrics:
            return
        headers = ["epoch", "batch", "phase", "loss", "accuracy",
                    "precision", "recall", "lr", "timestamp"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metrics)

    def get_statistics(self) -> dict:
        """获取历史统计概览。"""
        with self._storage._get_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM training_runs"
            ).fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM training_runs WHERE status='completed'"
            ).fetchone()[0]
            running = conn.execute(
                "SELECT COUNT(*) FROM training_runs WHERE status='running'"
            ).fetchone()[0]
            best = conn.execute(
                "SELECT MIN(best_loss) FROM training_runs WHERE best_loss IS NOT NULL"
            ).fetchone()[0]
            return {
                "total_runs": total,
                "completed_runs": completed,
                "running_runs": running,
                "best_loss_overall": best,
            }

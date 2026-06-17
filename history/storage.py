import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from contextlib import contextmanager


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name       TEXT    NOT NULL,
    start_time      TEXT    NOT NULL,
    end_time        TEXT,
    status          TEXT    NOT NULL DEFAULT 'running',
    dataset_name    TEXT,
    dataset1_path   TEXT,
    dataset2_path   TEXT,
    model_type      TEXT    DEFAULT 'ResUnet3D_Physics',
    config_json     TEXT,
    total_epochs    INTEGER DEFAULT 0,
    current_epoch   INTEGER DEFAULT 0,
    best_loss       REAL,
    best_accuracy   REAL,
    best_precision  REAL,
    best_recall     REAL,
    device          TEXT,
    log_dir         TEXT,
    checkpoint_path TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS training_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    epoch       INTEGER NOT NULL,
    batch       INTEGER,
    phase       TEXT    NOT NULL DEFAULT 'train',
    loss        REAL,
    accuracy    REAL,
    precision   REAL,
    recall      REAL,
    lr          REAL,
    timestamp   TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_epoch
    ON training_metrics(run_id, epoch);
CREATE INDEX IF NOT EXISTS idx_runs_start_time
    ON training_runs(start_time);
CREATE INDEX IF NOT EXISTS idx_runs_status
    ON training_runs(status);
"""


class HistoryStorage:
    """SQLite 训练历史存储。"""

    def __init__(self, db_path: str = "./history.db"):
        self._db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构。"""
        with self._get_conn() as conn:
            conn.executescript(DB_SCHEMA)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    def create_run(self, task_name: str, config: dict, device: str,
                   dataset1: str = "", dataset2: str = "",
                   log_dir: str = "", model_type: str = "ResUnet3D_Physics") -> int:
        """创建新的训练记录，返回 run_id。"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO training_runs
                   (task_name, start_time, status, dataset_name,
                    dataset1_path, dataset2_path, model_type,
                    config_json, device, log_dir)
                   VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_name,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    Path(dataset1).name if dataset1 else "",
                    dataset1, dataset2,
                    model_type,
                    json.dumps(config, ensure_ascii=False),
                    device, log_dir,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_run_status(self, run_id: int, status: str, **kwargs):
        """更新训练运行状态和可选的最终指标。"""
        fields = ["status = ?"]
        values = [status]
        if "end_time" not in kwargs:
            kwargs["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for key, value in kwargs.items():
            if value is not None:
                fields.append(f"{key} = ?")
                values.append(value)
        values.append(run_id)
        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE training_runs SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()

    def save_metrics_batch(self, run_id: int, metrics_rows: list[dict]):
        """批量保存指标数据。"""
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT INTO training_metrics
                   (run_id, epoch, batch, phase, loss, accuracy, precision, recall, lr, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id, row["epoch"], row.get("batch"), row.get("phase", "train"),
                        row.get("loss"), row.get("accuracy"), row.get("precision"),
                        row.get("recall"), row.get("lr"),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    for row in metrics_rows
                ],
            )
            conn.commit()

    def save_metric(self, run_id: int, epoch: int, phase: str, **metrics):
        """保存单条指标记录。"""
        with self._get_conn() as conn:
            # 确保 run_id 存在（处理并发/竞态导致的外键缺失）
            existing = conn.execute(
                "SELECT id FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not existing:
                return
            conn.execute(
                """INSERT INTO training_metrics
                   (run_id, epoch, phase, loss, accuracy, precision, recall, lr, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, epoch, phase,
                    metrics.get("loss"), metrics.get("accuracy"),
                    metrics.get("precision"), metrics.get("recall"),
                    metrics.get("lr"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()

    def get_run(self, run_id: int) -> dict | None:
        """获取单条训练运行记录。"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_run_metrics(self, run_id: int) -> list[dict]:
        """获取某次运行的所有指标。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM training_metrics WHERE run_id = ? ORDER BY epoch, batch",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_run(self, run_id: int):
        """删除训练运行及其关联指标。"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM training_metrics WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))
            conn.commit()

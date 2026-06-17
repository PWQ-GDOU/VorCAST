import os
import csv
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Any


class LogManager:
    """管理训练任务的日志文件夹。

    每次训练创建独立文件夹: logs/YYYY-MM-DD_HH-MM-SS_任务名/
    包含: training.log, metrics.csv, config_copy.yaml, checkpoints/ 链接
    """

    def __init__(self, log_dir: str = "./logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def create_task_dir(self, task_name: str | None = None) -> Path:
        """创建新的任务日志文件夹，返回路径。"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_{task_name}" if task_name else timestamp
        task_dir = self._log_dir / folder_name
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def save_config(self, task_dir: Path, config: dict):
        """保存配置副本到任务文件夹。"""
        import yaml
        config_path = task_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    def save_metrics_csv(self, task_dir: Path, metrics: dict[str, list]):
        """将指标列表写入 CSV 文件。"""
        if not metrics:
            return
        csv_path = task_dir / "metrics.csv"
        headers = list(metrics.keys())
        rows = zip(*metrics.values())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def link_checkpoint(self, task_dir: Path, checkpoint_dir: str):
        """创建 checkpoint 目录的符号链接（或复制路径引用）。"""
        link_path = task_dir / "checkpoints"
        if os.name == "nt":
            # Windows: 写一个路径引用文件
            link_path.write_text(checkpoint_dir, encoding="utf-8")
        else:
            if link_path.exists():
                link_path.unlink()
            os.symlink(checkpoint_dir, link_path, target_is_directory=True)

    def list_task_dirs(self) -> list[Path]:
        """列出所有任务日志文件夹（按时间排序，最新在前）。"""
        dirs = [d for d in self._log_dir.iterdir() if d.is_dir()]
        dirs.sort(key=lambda d: d.name, reverse=True)
        return dirs

    def clean_old_logs(self, keep_days: int = 30):
        """清理超过指定天数的旧日志文件夹。"""
        import time
        now = time.time()
        cutoff = now - keep_days * 86400
        for d in self._log_dir.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d)

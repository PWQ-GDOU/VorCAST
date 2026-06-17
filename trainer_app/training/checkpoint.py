"""Checkpoint 保存与加载。"""
import os
import torch
from pathlib import Path
from datetime import datetime
from typing import Any


class CheckpointManager:
    """训练检查点管理器。

    功能：
    - 定期保存模型、优化器、调度器状态
    - 保存训练进度（epoch, best_loss）
    - 最佳模型独立保存
    - 支持恢复训练
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints",
                 save_best_only: bool = True, max_keep: int = 5):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.max_keep = max_keep
        self._history: list[dict] = []

    def save(self, filename: str, **state) -> str:
        """保存 checkpoint。

        Args:
            filename: 文件名（不含路径）
            **state: 要保存的状态 dict（model, optimizer, epoch, metrics 等）

        Returns:
            保存的完整路径
        """
        path = self._dir / filename
        save_dict = {
            "timestamp": datetime.now().isoformat(),
            **state,
        }
        torch.save(save_dict, path)
        self._history.append({"path": str(path), "metrics": state.get("metrics", {})})

        # 清理旧 checkpoint
        if not self.save_best_only and len(self._history) > self.max_keep:
            old = self._history.pop(0)
            old_path = Path(old["path"])
            if old_path.exists():
                old_path.unlink()

        return str(path)

    def save_best(self, metrics: dict, **state) -> str:
        """保存最佳模型（覆盖）。"""
        path = self._dir / "best_model.pt"
        save_dict = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            **state,
        }
        torch.save(save_dict, path)
        return str(path)

    def load(self, path: str, map_location: str = "cpu") -> dict:
        """加载 checkpoint。"""
        return torch.load(path, map_location=map_location, weights_only=False)

    def load_latest(self, map_location: str = "cpu") -> dict | None:
        """加载最新的 checkpoint。接受 .pt 和 .pth 格式。"""
        files = sorted(self._dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if not files:
            files = sorted(self._dir.glob("*.pth"), key=os.path.getmtime, reverse=True)
        if not files:
            # 尝试加载 best_model (优先 .pt，其次 .pth)
            for ext in (".pt", ".pth"):
                best = self._dir / f"best_model{ext}"
                if best.exists():
                    return self.load(str(best), map_location)
            return None
        return self.load(str(files[0]), map_location)

    def list_checkpoints(self) -> list[str]:
        """列出所有 checkpoint 文件 (.pt 和 .pth)。"""
        files = sorted(self._dir.glob("*.pt"))
        files.extend(sorted(self._dir.glob("*.pth")))
        return sorted([str(p) for p in files])

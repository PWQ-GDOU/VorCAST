"""实时训练指标监控。

负责计算和广播训练/验证过程中的所有关键指标：
损失值、准确率、精确率、召回率等。
"""
import time
import torch
import numpy as np
from collections import defaultdict
from typing import Any


class TrainingMonitor:
    """训练指标实时追踪器。

    维护各指标的运行统计，支持按 epoch/batch 粒度查询。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计。"""
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._current: dict[str, dict[str, float]] = {}
        self._running: dict[str, list] = defaultdict(list)
        self._start_time = time.time()

    def update(self, phase: str, metrics: dict[str, float],
               epoch: int, batch: int | None = None):
        """记录一批指标。

        Args:
            phase: 'train' | 'val' | 'test'
            metrics: {name: value} 字典
            epoch: 当前 epoch
            batch: 当前 batch（可选）
        """
        record = {
            "epoch": epoch,
            "batch": batch,
            "phase": phase,
            "timestamp": time.time(),
            **metrics,
        }
        self._history[phase].append(record)
        self._current[phase] = metrics

        for k, v in metrics.items():
            self._running[f"{phase}_{k}"].append(v)

    def get_latest(self, phase: str = "train") -> dict[str, float]:
        """获取最新指标。"""
        return self._current.get(phase, {})

    def get_running_avg(self, phase: str, metric: str, window: int = 100) -> float:
        """获取运行窗口内的均值。"""
        key = f"{phase}_{metric}"
        values = self._running[key][-window:]
        return float(np.mean(values)) if values else 0.0

    def get_epoch_summary(self, phase: str = "train", epoch: int | None = None) -> dict:
        """获取指定 epoch 的汇总（均值）。"""
        records = self._history.get(phase, [])
        if epoch is not None:
            records = [r for r in records if r["epoch"] == epoch]
        if not records:
            return {}

        summary = {}
        for k in records[0].keys():
            if k in ("epoch", "batch", "phase", "timestamp"):
                continue
            vals = [r[k] for r in records if r.get(k) is not None]
            if vals:
                summary[k] = float(np.mean(vals))
        return summary

    def get_elapsed_time(self) -> float:
        """获取训练已用时间（秒）。"""
        return time.time() - self._start_time

    def to_dict(self) -> dict:
        """导出所有历史数据。"""
        return {
            "history": dict(self._history),
            "elapsed": self.get_elapsed_time(),
        }


def compute_binary_metrics(pred: torch.Tensor, target: torch.Tensor,
                           threshold: float | None = None) -> dict[str, float]:
    """Compute binary classification metrics for vorticity > threshold detection.

    Also returns zero_baseline_acc (accuracy if we predict all-zeros)
    and pos_ratio (fraction of target pixels > threshold) to diagnose
    class-imbalance issues. The zero-baseline is the accuracy floor
    the model must beat.

    Args:
        pred: predicted values (B, ...)
        target: true values (B, ...)
        threshold: positive-class threshold, defaults to 0.001
    """
    with torch.no_grad():
        if threshold is None:
            threshold = 0.001
        pred_bin = (pred > threshold).float()
        target_bin = (target > threshold).float()

        tp = (pred_bin * target_bin).sum().item()
        fp = (pred_bin * (1 - target_bin)).sum().item()
        fn = ((1 - pred_bin) * target_bin).sum().item()
        tn = ((1 - pred_bin) * (1 - target_bin)).sum().item()

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / (total + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Dice & IoU — ignore background (TN), directly measure overlap on rare positives
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        # Zero baseline: accuracy if model predicts all zeros
        zero_baseline_acc = (target <= threshold).float().mean().item()

        pos_ratio = (tp + fn) / (total + 1e-8)  # true fraction > threshold

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "csi": iou,  # CSI = IoU (Critical Success Index)
        "zero_baseline_acc": zero_baseline_acc,
        "pos_ratio": pos_ratio,
    }

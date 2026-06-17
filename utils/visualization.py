"""训练结果可视化模块。

训练完成时自动生成效果图表，保存到指定文件夹：
- loss 曲线（训练 + 验证）
- 准确率/精确率/召回率曲线
- 学习率变化曲线
- 预测 vs 真实涡度场对比
- 涡度场二维切片
- 综合指标仪表盘
"""
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# 抑制 matplotlib 在无 GUI 环境下的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
                                     "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _get_output_dir(task_dir: str | Path, subdir: str = "plots") -> Path:
    """获取/创建输出子目录。"""
    out = Path(task_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_loss_curves(monitor_data: dict, output_dir: str | Path,
                     dpi: int = 150) -> str:
    """绘制训练/验证 Loss 曲线。

    Args:
        monitor_data: TrainingMonitor.to_dict() 返回的字典
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    out = _get_output_dir(output_dir)
    history = monitor_data.get("history", {})

    fig, ax = plt.subplots(figsize=(10, 5))

    for phase, color, ls in [("train", "#2196F3", "-"), ("val", "#FF5722", "--")]:
        records = history.get(phase, [])
        if not records:
            continue
        epochs = sorted(set(r["epoch"] for r in records))
        losses = []
        for ep in epochs:
            ep_records = [r for r in records if r["epoch"] == ep]
            avg_loss = np.mean([r.get("loss", 0) for r in ep_records])
            losses.append(avg_loss)
        ax.plot(epochs, losses, color=color, linestyle=ls, linewidth=1.5,
                label=f"{phase.upper()} Loss", marker=".")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练 / 验证 Loss 曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    path = str(out / "loss_curves.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metrics_dashboard(monitor_data: dict, output_dir: str | Path,
                           dpi: int = 150) -> str:
    """绘制准确率/精确率/召回率面板。

    Returns:
        保存的文件路径
    """
    out = _get_output_dir(output_dir)
    history = monitor_data.get("history", {})

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    metric_names = ["accuracy", "precision", "recall", "f1", "iou", "dice"]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score", "IoU (CSI)", "Dice"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    for ax, metric, title, color in zip(axes, metric_names, titles, colors):
        for phase, ls in [("train", "-"), ("val", "--")]:
            records = history.get(phase, [])
            if not records:
                continue
            epochs = sorted(set(r["epoch"] for r in records))
            vals = []
            for ep in epochs:
                ep_records = [r for r in records if r["epoch"] == ep]
                avg = np.mean([r.get(metric, 0) for r in ep_records])
                vals.append(avg)
            ax.plot(epochs, vals, color=color, linestyle=ls, linewidth=1.5,
                    label=phase.upper(), marker=".")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric in ("accuracy", "precision", "recall", "f1"):
            ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("训练指标面板", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = str(out / "metrics_dashboard.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_csi_fss_curves(monitor_data: dict, output_dir: str | Path,
                         dpi: int = 150) -> str:
    """绘制 CSI Loss 和 FSS Loss 变化曲线。"""
    out = _get_output_dir(output_dir)
    history = monitor_data.get("history", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for phase, color in [("train", "#2196F3"), ("val", "#FF5722")]:
        records = history.get(phase, [])
        if records:
            epochs = [r["epoch"] for r in records]
            csi_vals = [r.get("csi_loss", 0) for r in records]
            fcsi_vals = [r.get("fcsi_loss", 0) for r in records]

            avg_csi, avg_fcsi = [], []
            for ep in sorted(set(epochs)):
                ep_csi = [csi_vals[i] for i, e in enumerate(epochs) if e == ep]
                ep_fcsi = [fcsi_vals[i] for i, e in enumerate(epochs) if e == ep]
                avg_csi.append(np.mean(ep_csi) if ep_csi else 0)
                avg_fcsi.append(np.mean(ep_fcsi) if ep_fcsi else 0)

            unique_eps = sorted(set(epochs))
            axes[0].plot(unique_eps, avg_csi, color=color, label=phase, alpha=0.8)
            axes[1].plot(unique_eps, avg_fcsi, color=color, label=phase, alpha=0.8)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CSI Loss")
    axes[0].set_title("CSI Loss (Soft IoU)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("FSS Loss")
    axes[1].set_title("FSS Loss (Multi-scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = str(out / "csi_fss_curves.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lpips_auc_curves(monitor_data: dict, output_dir: str | Path,
                           dpi: int = 150) -> str:
    """绘制 LPIPS 和 AUC 损失变化曲线。"""
    out = _get_output_dir(output_dir)
    history = monitor_data.get("history", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for phase, color in [("train", "#2196F3"), ("val", "#FF5722")]:
        records = history.get(phase, [])
        if records:
            epochs = [r["epoch"] for r in records]
            lpips_vals = [r.get("lpips_loss", 0) for r in records]
            auc_vals = [r.get("auc_loss", 0) for r in records]

            avg_lpips, avg_auc = [], []
            for ep in sorted(set(epochs)):
                ep_lpips = [lpips_vals[i] for i, e in enumerate(epochs) if e == ep]
                ep_auc = [auc_vals[i] for i, e in enumerate(epochs) if e == ep]
                avg_lpips.append(np.mean(ep_lpips) if ep_lpips else 0)
                avg_auc.append(np.mean(ep_auc) if ep_auc else 0)

            unique_eps = sorted(set(epochs))
            axes[0].plot(unique_eps, avg_lpips, color=color, label=phase, alpha=0.8)
            axes[1].plot(unique_eps, avg_auc, color=color, label=phase, alpha=0.8)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("LPIPS Distance")
    axes[0].set_title("LPIPS Perceptual Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC Loss")
    axes[1].set_title("AUC Ranking Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = str(out / "lpips_auc_curves.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lr_curve(monitor_data: dict, output_dir: str | Path,
                  dpi: int = 150) -> str:
    """绘制学习率变化曲线。"""
    out = _get_output_dir(output_dir)
    history = monitor_data.get("history", {})

    fig, ax = plt.subplots(figsize=(8, 4))

    records = history.get("train", [])
    if records:
        epochs = sorted(set(r["epoch"] for r in records))
        lrs = []
        for ep in epochs:
            ep_records = [r for r in records if r["epoch"] == ep]
            lr_vals = [r.get("lr", 0) for r in ep_records if r.get("lr")]
            lrs.append(lr_vals[0] if lr_vals else 0)
        ax.plot(epochs, lrs, color="#9C27B0", linewidth=1.5, marker=".")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_title("学习率变化曲线")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    path = str(out / "lr_curve.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vorticity_slices(predictions: np.ndarray, output_dir: str | Path,
                          targets: np.ndarray | None = None,
                          dpi: int = 150) -> list[str]:
    """绘制涡度场二维切片对比图。

    Args:
        predictions: (T, H, W, L) 预测涡度
        targets: (T, H, W, L) 真实涡度，可选
        output_dir: 输出目录

    Returns:
        保存的文件路径列表
    """
    out = _get_output_dir(output_dir)
    paths = []

    T, H, W, L = predictions.shape
    mid_t = T // 2
    mid_l = L // 2

    # 取中间时间步、中间高度层
    pred_slice = predictions[mid_t, :, :, mid_l]
    target_slice = targets[mid_t, :, :, mid_l] if targets is not None else None

    if target_slice is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        im0 = axes[0].imshow(target_slice, cmap="RdBu_r", origin="lower",
                              aspect="auto")
        axes[0].set_title(f"真实涡度 (t={mid_t}, z={mid_l})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(pred_slice, cmap="RdBu_r", origin="lower",
                              aspect="auto")
        axes[1].set_title(f"预测涡度 (t={mid_t}, z={mid_l})")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        diff = pred_slice - target_slice
        vmax = max(abs(diff.min()), abs(diff.max()))
        im2 = axes[2].imshow(diff, cmap="coolwarm", origin="lower", aspect="auto",
                              vmin=-vmax, vmax=vmax)
        axes[2].set_title("预测误差 (Pred - True)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        fig.tight_layout()
        path = str(out / "vorticity_comparison.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pred_slice, cmap="RdBu_r", origin="lower", aspect="auto")
        ax.set_title(f"预测涡度 (t={mid_t}, z={mid_l})")
        plt.colorbar(im, ax=ax, fraction=0.046)
        path = str(out / "vorticity_prediction.png")

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # 时间演化: 取中心点涡度随时间变化
    fig, ax = plt.subplots(figsize=(10, 4))
    center_h, center_w = H // 2, W // 2
    pred_series = predictions[:, center_h, center_w, mid_l]
    ax.plot(range(T), pred_series, color="#E91E63", linewidth=1.5, marker=".",
            label="预测涡度")
    if targets is not None:
        target_series = targets[:, center_h, center_w, mid_l]
        ax.plot(range(T), target_series, color="#2196F3", linewidth=1.5,
                linestyle="--", marker=".", label="真实涡度")
    ax.set_xlabel("时间步 (5min/步)")
    ax.set_ylabel("垂直涡度 (1/s)")
    ax.set_title(f"中心格点涡度时间演化 (H/2, W/2, z={mid_l})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    path2 = str(out / "vorticity_timeseries.png")
    fig.savefig(path2, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path2)

    return paths


def plot_summary_card(best_loss: float, best_epoch: int, total_epochs: int,
                      elapsed: float, device: str, output_dir: str | Path,
                      dpi: int = 150) -> str:
    """生成训练总结卡片。"""
    out = _get_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    lines = [
        ("训练总结", 16, "bold"),
        ("", 8, "normal"),
        (f"最佳 Loss:           {best_loss:.6f}", 13, "normal"),
        (f"最佳 Epoch:          {best_epoch} / {total_epochs}", 13, "normal"),
        (f"训练用时:            {h:02d}:{m:02d}:{s:02d}", 13, "normal"),
        (f"训练设备:            {device}", 13, "normal"),
        ("", 8, "normal"),
        ("龙卷风垂直涡度预测训练器", 10, "normal"),
        ("基于 Nowcast3D 灰盒物理模型", 10, "normal"),
    ]

    y = 0.95
    for text, size, style in lines:
        weight = "bold" if style == "bold" else "normal"
        ax.text(0.5, y, text, transform=ax.transAxes, fontsize=size,
                fontweight=weight, ha="center", va="top",
                fontfamily="monospace" if style != "bold" else "sans-serif")
        y -= 0.09

    path = str(out / "summary_card.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    plt.close(fig)
    return path


def generate_all_plots(monitor_data: dict, output_dir: str | Path,
                       predictions: np.ndarray | None = None,
                       targets: np.ndarray | None = None,
                       best_loss: float = 0.0, best_epoch: int = 0,
                       total_epochs: int = 0, elapsed: float = 0.0,
                       device: str = "cpu") -> dict[str, str]:
    """生成全套训练效果图。

    Args:
        monitor_data: monitor.to_dict() 返回值
        output_dir: 输出根目录（会在其下创建 plots/ 子目录）
        predictions: 预测涡度 (T, H, W, L)，可选
        targets: 真实涡度 (T, H, W, L)，可选
        best_loss: 最佳验证 Loss
        best_epoch: 最佳 epoch
        total_epochs: 总 epoch 数
        elapsed: 训练总用时（秒）
        device: 设备名称

    Returns:
        {图类型: 文件路径} 字典
    """
    out = _get_output_dir(output_dir)
    results = {}

    _log = logging.getLogger(__name__)

    try:
        results["loss"] = plot_loss_curves(monitor_data, out)
    except Exception as e:
        _log.warning("Failed to plot loss curves: %s", e)

    try:
        results["metrics"] = plot_metrics_dashboard(monitor_data, out)
    except Exception as e:
        _log.warning("Failed to plot metrics dashboard: %s", e)

    try:
        results["csi_fss"] = plot_csi_fss_curves(monitor_data, out)
    except Exception as e:
        _log.warning("Failed to plot CSI/FSS curves: %s", e)

    try:
        results["lpips_auc"] = plot_lpips_auc_curves(monitor_data, out)
    except Exception as e:
        _log.warning("Failed to plot LPIPS/AUC curves: %s", e)

    try:
        results["lr"] = plot_lr_curve(monitor_data, out)
    except Exception as e:
        _log.warning("Failed to plot LR curve: %s", e)

    if predictions is not None:
        try:
            paths = plot_vorticity_slices(predictions, out, targets)
            results["vorticity"] = paths[0]
            results["timeseries"] = paths[1]
        except Exception as e:
            _log.warning("Failed to plot vorticity slices: %s", e)

    try:
        results["summary"] = plot_summary_card(
            best_loss, best_epoch, total_epochs, elapsed, device, out,
        )
    except Exception as e:
        _log.warning("Failed to plot summary card: %s", e)

    # 保存指标原始数据 CSV
    try:
        _save_metrics_csv(monitor_data, out)
        results["csv"] = str(out / "metrics_data.csv")
    except Exception as e:
        _log.warning("Failed to save metrics CSV: %s", e)

    return results


def _save_metrics_csv(monitor_data: dict, output_dir: Path):
    """导出指标 CSV 到 plots 目录。"""
    import csv

    history = monitor_data.get("history", {})
    csv_path = output_dir / "metrics_data.csv"

    all_rows = []
    for phase in ["train", "val"]:
        for r in history.get(phase, []):
            all_rows.append({
                "phase": phase,
                "epoch": r["epoch"],
                "batch": r.get("batch", ""),
                "loss": r.get("loss", ""),
                "total_loss": r.get("total_loss", ""),
                "accuracy": r.get("accuracy", ""),
                "precision": r.get("precision", ""),
                "recall": r.get("recall", ""),
                "f1": r.get("f1", ""),
                "dice": r.get("dice", ""),
                "iou": r.get("iou", ""),
                "csi": r.get("csi", ""),
                "grad_loss": r.get("grad_loss", ""),
                "csi_loss": r.get("csi_loss", ""),
                "fcsi_loss": r.get("fcsi_loss", ""),
                "lpips_loss": r.get("lpips_loss", ""),
                "auc_loss": r.get("auc_loss", ""),
                "continuity_loss": r.get("continuity_loss", ""),
                "pred_std": r.get("pred_std", ""),
                "target_std": r.get("target_std", ""),
                "pos_ratio": r.get("pos_ratio", ""),
                "lr": r.get("lr", ""),
            })

    if not all_rows:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

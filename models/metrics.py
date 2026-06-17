"""气象行业标准指标模块。

训练时可微指标（集成到损失函数）：
  - SoftIoU Loss (= differentiable CSI)
  - Soft FSS Loss (多尺度邻域分数技巧评分)
  - LPIPS Loss (感知相似度，每5 batch采样)
  - Soft AUC Loss (排序能力代理损失，HIP安全版)

推理时精确指标（numpy 实现，不可微但精确）：
  - CSI = IoU (Critical Success Index)
  - fCSI / FSS (Fractions Skill Score)
  - LPIPS
  - AUC-ROC

参考：
  - Roberts & Lean (2008) "Scale-Selective Verification..."
  - Mittermaier & Roberts (2010) "Intercomparison of Spatial Forecast Verification Methods"
  - Zhang et al. (2018) "The Unreasonable Effectiveness of Deep Features..."
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 可微损失函数（用于训练）
# ═══════════════════════════════════════════════════════════════


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor,
                  threshold: float = 0.001,
                  temperature: float = 0.1) -> torch.Tensor:
    """可微 Soft IoU Loss = 1 - soft_CSI。

    用 sigmoid 代替硬阈值实现可微计算：
    pred_soft = σ((pred - τ) / T)
    IoU = Σ(pred_soft · target_bin) / (Σ pred_soft + Σ target_bin - Σ pred_soft · target_bin)

    Args:
        pred: 预测涡度 (B, T, H, W, L) 或任意形状
        target: 真实涡度，同形状
        threshold: 正类涡度阈值 (1/s)
        temperature: 软阈值温度（越低越接近硬 CSI，梯度越稀疏）

    Returns:
        L_csi = 1 - soft_iou (标量)
    """
    target_bin = (target > threshold).float().detach()
    pred_soft = torch.sigmoid((pred - threshold) / temperature)

    # 确保连续，避免 HIP 潜在问题
    pred_soft = pred_soft.contiguous()
    target_bin = target_bin.contiguous()

    intersection = (pred_soft * target_bin).sum()
    union = pred_soft.sum() + target_bin.sum() - intersection

    soft_iou = intersection / (union + 1e-8)
    return 1.0 - soft_iou


def soft_fss_loss(pred: torch.Tensor, target: torch.Tensor,
                  scales: list[int] = [1, 3, 5, 11, 21],
                  threshold: float = 0.001,
                  temperature: float = 0.1) -> torch.Tensor:
    """可微多尺度 FSS (Fractions Skill Score) 损失。

    对每个邻域尺度 n，用 average_pool2d 计算邻域覆盖比例，
    然后计算预报与观测比例的 MSE。

    FSS_n = 1 - MSE(fractions_obs, fractions_pred) / MSE(fractions_obs, fractions_ref)
    L_fss = mean(1 - FSS_n over scales)

    Args:
        pred: 预测涡度 (B, T, H, W, L) 或 (B, H, W) 等
        target: 真实涡度
        scales: 邻域尺度列表（格点数）
        threshold: 涡度阈值
        temperature: 软阈值温度

    Returns:
        L_fss (标量)
    """
    target_bin = (target > threshold).float().detach()
    pred_soft = torch.sigmoid((pred - threshold) / temperature)

    # 确保输入是 B×C×H×W 格式用于 avg_pool2d
    if pred.dim() == 5:
        B, T, H, W, L = pred.shape
        pred_soft = pred_soft.permute(0, 1, 4, 2, 3).reshape(-1, 1, H, W)
        target_bin = target_bin.permute(0, 1, 4, 2, 3).reshape(-1, 1, H, W)
    elif pred.dim() == 4:
        pred_soft = pred_soft.reshape(-1, 1, *pred.shape[-2:])
        target_bin = target_bin.reshape(-1, 1, *target.shape[-2:])
    elif pred.dim() == 3:
        pred_soft = pred_soft.permute(2, 0, 1).unsqueeze(1)
        target_bin = target_bin.permute(2, 0, 1).unsqueeze(1)
    else:
        pred_soft = pred_soft.unsqueeze(0).unsqueeze(0)
        target_bin = target_bin.unsqueeze(0).unsqueeze(0)

    total_loss = 0.0
    for n in scales:
        if n == 1:
            F_n = pred_soft
            O_n = target_bin
        else:
            pool = nn.AvgPool2d(kernel_size=n, stride=1, padding=n // 2)
            F_n = pool(pred_soft)
            O_n = pool(target_bin)

        mse = ((O_n - F_n) ** 2).mean()
        mse_ref = (O_n ** 2).mean() + (F_n ** 2).mean()
        fss_n = 1.0 - mse / (mse_ref + 1e-8)
        total_loss += (1.0 - fss_n)

    return total_loss / len(scales)


def soft_auc_loss(pred: torch.Tensor, target: torch.Tensor,
                  threshold: float = 0.001,
                  temperature: float = 1.0,
                  max_pairs: int = 10000) -> torch.Tensor:
    """可微 Soft AUC 损失（加权均值版，完全无布尔索引，HIP 安全）。

    L_auc = σ( (mean(pred_neg) - mean(pred_pos)) / T )
    本质上是让正样本的平均预测值高于负样本的平均预测值。
    使用连续权重避免索引操作，与 ROCM/HIP 完全兼容。
    """
    target_bin = (target > threshold).float().detach()
    pred_flat = pred.reshape(-1)
    target_flat = target_bin.reshape(-1)

    # 正负样本的连续权重（避免硬索引）
    pos_weight = target_flat
    neg_weight = 1.0 - target_flat

    sum_pos = pos_weight.sum()
    sum_neg = neg_weight.sum()

    if sum_pos == 0 or sum_neg == 0:
        return torch.tensor(0.0, device=pred.device)

    # 加权均值（完全无索引操作）
    mean_pos = (pred_flat * pos_weight).sum() / sum_pos
    mean_neg = (pred_flat * neg_weight).sum() / sum_neg

    # 我们希望 mean_pos > mean_neg，所以当 mean_neg > mean_pos 时损失较大
    loss = torch.sigmoid((mean_neg - mean_pos) / temperature)
    return loss


class LPIPSLoss(nn.Module):
    """可微 LPIPS (Learned Perceptual Image Patch Similarity) 损失。

    使用预训练 AlexNet 计算预测与真实涡度场之间的感知相似度。
    对 3D 涡度场各高度层切片分别计算后取平均。

    LPIPS 网络权重被冻结（eval 模式，不参与梯度更新），
    但 LPIPS 本身的输出对输入是可微的。

    使用方式:
        lpips_loss = LPIPSLoss(device="cuda")
        L = lpips_loss(pred_zeta, true_zeta)  # (B, T, H, W, L)
    """

    def __init__(self, net: str = "alex", device: str = "cuda",
                 batch_interval: int = 5):
        super().__init__()
        self.batch_interval = batch_interval
        self._batch_count = 0
        self._device = device

        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net).to(device)
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
            self.lpips_fn.eval()
            self._available = True
        except ImportError:
            self._available = False
            self.lpips_fn = None

    def should_compute(self) -> bool:
        """按 batch_interval 间隔决定是否计算 LPIPS。"""
        self._batch_count += 1
        return self._available and (self._batch_count % self.batch_interval == 1)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """计算 LPIPS 损失（均值 over T×L 个切片）。"""
        if not self._available or self.lpips_fn is None:
            return torch.tensor(0.0, device=pred.device)

        B, T, H, W, L = pred.shape
        total = 0.0
        count = 0

        for b in range(B):
            p_min = pred[b].min()
            p_max = pred[b].max()
            t_min = target[b].min()
            t_max = target[b].max()

            p_norm = (pred[b] - p_min) / max(p_max - p_min, 1e-8)
            t_norm = (target[b] - t_min) / max(t_max - t_min, 1e-8)

            for t in range(T):
                for l in range(L):
                    p_slice = p_norm[t, :, :, l].unsqueeze(0).unsqueeze(0)
                    t_slice = t_norm[t, :, :, l].unsqueeze(0).unsqueeze(0)
                    p_3ch = p_slice.repeat(1, 3, 1, 1).clamp(0, 1)
                    t_3ch = t_slice.repeat(1, 3, 1, 1).clamp(0, 1)

                    total += self.lpips_fn(p_3ch, t_3ch).squeeze()
                    count += 1

        return total / max(count, 1)


# ═══════════════════════════════════════════════════════════════
# 推理时精确指标（numpy 实现）
# ═══════════════════════════════════════════════════════════════


def compute_binary_confusion(pred: np.ndarray, target: np.ndarray,
                             threshold: float = 0.001
                             ) -> tuple[int, int, int, int]:
    """计算 TP, FP, FN, TN。"""
    pred_bin = (pred > threshold).astype(np.int32)
    target_bin = (target > threshold).astype(np.int32)

    tp = int((pred_bin * target_bin).sum())
    fp = int((pred_bin * (1 - target_bin)).sum())
    fn = int(((1 - pred_bin) * target_bin).sum())
    tn = int(((1 - pred_bin) * (1 - target_bin)).sum())

    return tp, fp, fn, tn


def compute_csi(pred: np.ndarray, target: np.ndarray,
                threshold: float = 0.001) -> float:
    """计算 CSI (Critical Success Index) = TP / (TP + FP + FN)。"""
    tp, fp, fn, _ = compute_binary_confusion(pred, target, threshold)
    denom = tp + fp + fn
    if denom == 0:
        return float("nan")
    return tp / denom


def compute_fss(pred: np.ndarray, target: np.ndarray,
                scales: list[int] = [1, 3, 5, 11, 21],
                threshold: float = 0.001) -> dict[int, float]:
    """计算多尺度 FSS (Fractions Skill Score)。"""
    from scipy.ndimage import uniform_filter

    pred_bin = (pred > threshold).astype(np.float64)
    target_bin = (target > threshold).astype(np.float64)

    fss_values = {}
    for n in scales:
        if n == 1:
            O_n = target_bin
            F_n = pred_bin
        else:
            if pred_bin.ndim == 2:
                O_n = uniform_filter(target_bin, size=n, mode="reflect")
                F_n = uniform_filter(pred_bin, size=n, mode="reflect")
            elif pred_bin.ndim >= 3:
                O_n = uniform_filter(target_bin, size=(1, n, n), mode="reflect")
                F_n = uniform_filter(pred_bin, size=(1, n, n), mode="reflect")

        mse = ((O_n - F_n) ** 2).mean()
        mse_ref = (O_n ** 2).mean() + (F_n ** 2).mean()

        if mse_ref > 0:
            fss_n = 1.0 - mse / mse_ref
        else:
            fss_n = 1.0

        fss_values[n] = float(fss_n)

    fss_values["mean"] = float(np.mean(list(fss_values.values())))
    return fss_values


def compute_lpips_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """使用 LPIPS 计算感知相似度（numpy→torch 桥接）。"""
    try:
        import lpips
    except ImportError:
        return float("nan")

    lpips_fn = lpips.LPIPS(net="alex")
    for p in lpips_fn.parameters():
        p.requires_grad = False
    lpips_fn.eval()

    T, H, W, L_z = pred.shape
    total = 0.0
    count = 0

    for t in range(T):
        p_min, p_max = pred[t].min(), pred[t].max()
        t_min, t_max = target[t].min(), target[t].max()

        p_norm = (pred[t] - p_min) / max(p_max - p_min, 1e-8)
        t_norm = (target[t] - t_min) / max(t_max - t_min, 1e-8)

        for l in range(L_z):
            p_slice = torch.from_numpy(p_norm[:, :, l]).float().unsqueeze(0).unsqueeze(0)
            t_slice = torch.from_numpy(t_norm[:, :, l]).float().unsqueeze(0).unsqueeze(0)
            p_3ch = p_slice.repeat(1, 3, 1, 1).clamp(0, 1)
            t_3ch = t_slice.repeat(1, 3, 1, 1).clamp(0, 1)

            with torch.no_grad():
                total += lpips_fn(p_3ch, t_3ch).item()
            count += 1

    return total / max(count, 1)


def compute_auc(pred: np.ndarray, target: np.ndarray,
                threshold: float = 0.001) -> float:
    """计算 AUC-ROC (Area Under ROC Curve)。"""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")

    target_bin = (target > threshold).astype(np.int32).flatten()
    pred_flat = pred.astype(np.float64).flatten()

    unique_targets = np.unique(target_bin)
    if len(unique_targets) < 2:
        return float("nan")

    return float(roc_auc_score(target_bin, pred_flat))


def compute_all_metrics(pred: np.ndarray, target: np.ndarray,
                        threshold: float = 0.001,
                        fss_scales: list[int] = [1, 3, 5, 11, 21]
                        ) -> dict[str, float]:
    """一次性计算全部推理指标。"""
    tp, fp, fn, tn = compute_binary_confusion(pred, target, threshold)
    total = tp + fp + fn + tn

    csi_denom = tp + fp + fn
    csi = tp / csi_denom if csi_denom > 0 else float("nan")

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    iou_denom = tp + fp + fn
    iou = tp / iou_denom if iou_denom > 0 else float("nan")

    fss_dict = compute_fss(pred, target, scales=fss_scales, threshold=threshold)

    lpips_val = compute_lpips_numpy(pred, target)

    auc_val = compute_auc(pred, target, threshold)

    mae = float(np.abs(pred - target).mean())

    pos_ratio = (tp + fn) / total if total > 0 else 0.0

    metrics = {
        "mae": mae,
        "csi": csi,
        "fcsi_mean": fss_dict.get("mean", float("nan")),
        "f1": f1,
        "iou": iou,
        "lpips": lpips_val,
        "auc": auc_val,
        "pos_ratio": pos_ratio,
    }
    for s in fss_scales:
        metrics[f"fcsi_{s}"] = fss_dict.get(s, float("nan"))

    return metrics

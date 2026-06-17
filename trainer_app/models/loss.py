"""损失函数模块。

L = L_mae + λ1·L_grad (+ λ2·L_continuity)
  + λ3·L_csi (soft IoU) + λ4·L_fcsi (multi-scale FSS)
  + λ5·L_lpips (perceptual, every N batches) + λ6·L_auc (soft ranking)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import (
    soft_iou_loss,
    soft_fss_loss,
    soft_auc_loss,
    LPIPSLoss,
)


class PhysicsLoss(nn.Module):
    r"""物理感知组合损失函数。

    L = L_mae + λ1 * L_grad (+ λ2 * L_continuity)
      + λ3 * L_csi (+ λ4 * L_fcsi)
      + λ5 * L_lpips (+ λ6 * L_auc)

    其中:
    - L_mae: 预测与真实涡度的 L1 损失
    - L_grad: 涡度梯度幅值的 MSE 损失
    - L_continuity: 速度散度惩罚（∇·v → 0）
    - L_csi: 可微 Soft IoU (= CSI) 损失
    - L_fcsi: 多尺度可微 FSS 损失
    - L_lpips: 感知相似度损失（每 5 batch 采样）
    - L_auc: 可微 Soft AUC 损失（配对排序代理）

    可选 focal 加权：λ_focal > 0 时，按 |target| 对 MAE 施加逐元素权重。
    """

    def __init__(self, lambda_grad: float = 0.1,
                 lambda_continuity: float = 0.0,
                 lambda_focal: float = 0.0,
                 lambda_csi: float = 1.0,
                 lambda_fcsi: float = 0.5,
                 lambda_lpips: float = 0.1,
                 lambda_auc: float = 0.05,
                 lpips_device: str = "cuda",
                 lpips_interval: int = 5,
                 max_auc_pairs: int = 10000):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_continuity = lambda_continuity
        self.lambda_focal = lambda_focal
        self.lambda_csi = lambda_csi
        self.lambda_fcsi = lambda_fcsi
        self.lambda_lpips = lambda_lpips
        self.lambda_auc = lambda_auc

        self.mae = nn.L1Loss(reduction='none')
        self.mse = nn.MSELoss()

        # 新指标损失
        self.lpips_loss = LPIPSLoss(device=lpips_device,
                                     batch_interval=lpips_interval)
        self.max_auc_pairs = max_auc_pairs

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """计算场的梯度幅值 |∇x|，使用中心差分（零边界）。"""
        p = (1, 1, 1, 1, 1, 1)
        x_padded = F.pad(x, p, mode='replicate')

        x_plus_x = x_padded[..., 1:-1, 2:, 1:-1]
        x_minus_x = x_padded[..., 1:-1, :-2, 1:-1]
        x_plus_y = x_padded[..., 2:, 1:-1, 1:-1]
        x_minus_y = x_padded[..., :-2, 1:-1, 1:-1]
        x_plus_z = x_padded[..., 1:-1, 1:-1, 2:]
        x_minus_z = x_padded[..., 1:-1, 1:-1, :-2]

        dx = (x_plus_x - x_minus_x) * 0.5
        dy = (x_plus_y - x_minus_y) * 0.5
        dz = (x_plus_z - x_minus_z) * 0.5
        return torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)

    def _divergence(self, velocity: torch.Tensor) -> torch.Tensor:
        r"""计算速度场散度 ∇·v = ∂u/∂x + ∂v/∂y + ∂w/∂z。"""
        u = velocity[:, 0:1, :, :, :]
        v = velocity[:, 1:2, :, :, :]
        w = velocity[:, 2:3, :, :, :]

        p = (1, 1, 1, 1, 1, 1)
        du_dx = (F.pad(u, p, mode='replicate')[..., 1:-1, 2:, 1:-1]
                 - F.pad(u, p, mode='replicate')[..., 1:-1, :-2, 1:-1]) * 0.5
        dv_dy = (F.pad(v, p, mode='replicate')[..., 2:, 1:-1, 1:-1]
                 - F.pad(v, p, mode='replicate')[..., :-2, 1:-1, 1:-1]) * 0.5
        dw_dz = (F.pad(w, p, mode='replicate')[..., 1:-1, 1:-1, 2:]
                 - F.pad(w, p, mode='replicate')[..., 1:-1, 1:-1, :-2]) * 0.5

        return (du_dx + dv_dy + dw_dz).squeeze(1)

    def forward(
        self,
        zeta_pred: torch.Tensor,
        zeta_true: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """计算组合损失。

        Args:
            zeta_pred: 预测涡度 (B, T_out, H, W, L)
            zeta_true: 真实涡度 (B, T_out, H, W, L)
            velocity: 速度场 (B, T_out, 3, H, W, L)

        Returns:
            (total_loss, components_dict)
        """
        # --- L_mae (with optional focal) ---
        mae_per_element = self.mae(zeta_pred, zeta_true)
        if self.lambda_focal > 0:
            weight = 1.0 + self.lambda_focal * zeta_true.abs() / (
                zeta_true.abs().mean() + 1e-8)
            loss_mae = (mae_per_element * weight).mean()
        else:
            loss_mae = mae_per_element.mean()

        # --- L_grad ---
        loss_grad = self.mse(
            self._gradient_magnitude(zeta_pred),
            self._gradient_magnitude(zeta_true),
        )

        # --- L_csi (Soft IoU) ---
        loss_csi = soft_iou_loss(zeta_pred, zeta_true)

        # --- L_fcsi (Multi-scale FSS) ---
        loss_fcsi = soft_fss_loss(zeta_pred, zeta_true)

        # --- L_lpips (every N batches) ---
        if self.lpips_loss.should_compute():
            loss_lpips = self.lpips_loss(zeta_pred, zeta_true)
        else:
            loss_lpips = torch.tensor(0.0, device=zeta_pred.device)

        # --- L_auc (Soft AUC) ---
        loss_auc = soft_auc_loss(zeta_pred, zeta_true,
                                  max_pairs=self.max_auc_pairs)

        # --- Assembly ---
        total = (loss_mae
                 + self.lambda_grad * loss_grad
                 + self.lambda_csi * loss_csi
                 + self.lambda_fcsi * loss_fcsi
                 + self.lambda_lpips * loss_lpips
                 + self.lambda_auc * loss_auc)

        components = {
            "mae": loss_mae.item(),
            "grad": loss_grad.item(),
            "csi": loss_csi.item(),
            "fcsi": loss_fcsi.item(),
            "lpips": loss_lpips.item(),
            "auc": loss_auc.item(),
            "continuity": 0.0,
        }

        # --- L_continuity ---
        if velocity is not None and self.lambda_continuity > 0:
            B, T = velocity.shape[0], velocity.shape[1]
            div_losses = []
            for t in range(T):
                div = self._divergence(velocity[:, t, :, :, :, :])
                div_losses.append(self.mse(div, torch.zeros_like(div)))
            loss_continuity = torch.stack(div_losses).mean()
            total += self.lambda_continuity * loss_continuity
            components["continuity"] = loss_continuity.item()

        return total, components

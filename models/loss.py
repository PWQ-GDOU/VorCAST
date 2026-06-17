"""损失函数模块。

L = L_mae + λ1·L_grad (+ λ2·L_continuity)
  + λ3·L_csi (soft IoU) + λ4·L_fcsi (multi-scale FSS, optional)
  + λ5·L_lpips (perceptual, every N batches, optional) + λ6·L_auc (soft ranking)
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

        self.lpips_loss = LPIPSLoss(device=lpips_device,
                                     batch_interval=lpips_interval)
        self.max_auc_pairs = max_auc_pairs

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
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
        u = velocity[:, 0:1, :, :, :]
        v = velocity[:, 1:2, :, :, :]
        w = velocity[:, 2:3, :, :, :]
        p = (1, 1, 1, 1, 1, 1)
        du_dx = (F.pad(u, p, mode='replicate')[..., 1:-1, 2:, 1:-1] -
                 F.pad(u, p, mode='replicate')[..., 1:-1, :-2, 1:-1]) * 0.5
        dv_dy = (F.pad(v, p, mode='replicate')[..., 2:, 1:-1, 1:-1] -
                 F.pad(v, p, mode='replicate')[..., :-2, 1:-1, 1:-1]) * 0.5
        dw_dz = (F.pad(w, p, mode='replicate')[..., 1:-1, 1:-1, 2:] -
                 F.pad(w, p, mode='replicate')[..., 1:-1, 1:-1, :-2]) * 0.5
        return (du_dx + dv_dy + dw_dz).squeeze(1)

    def forward(self, zeta_pred, zeta_true, velocity=None):
        # 1. MAE + Focal
        mae_per_element = self.mae(zeta_pred, zeta_true)
        if self.lambda_focal > 0:
            weight = 1.0 + self.lambda_focal * zeta_true.abs() / (zeta_true.abs().mean() + 1e-8)
            loss_mae = (mae_per_element * weight).mean()
        else:
            loss_mae = mae_per_element.mean()

        # 2. Gradient
        loss_grad = self.mse(self._gradient_magnitude(zeta_pred),
                             self._gradient_magnitude(zeta_true))
        total = loss_mae + self.lambda_grad * loss_grad

        # 3. CSI (Soft IoU)
        loss_csi = soft_iou_loss(zeta_pred, zeta_true)

        # 4. FSS (Multi-scale) — only if lambda_fcsi > 0
        if self.lambda_fcsi > 0:
            loss_fcsi = soft_fss_loss(zeta_pred, zeta_true)
        else:
            loss_fcsi = torch.tensor(0.0, device=zeta_pred.device)

        # 5. LPIPS — only if lambda_lpips > 0 and should compute
        if self.lambda_lpips > 0 and self.lpips_loss.should_compute():
            loss_lpips = self.lpips_loss(zeta_pred, zeta_true)
        else:
            loss_lpips = torch.tensor(0.0, device=zeta_pred.device)

        # 6. AUC — always safe
        loss_auc = soft_auc_loss(zeta_pred, zeta_true, max_pairs=self.max_auc_pairs)

        total += (self.lambda_csi * loss_csi +
                  self.lambda_fcsi * loss_fcsi +
                  self.lambda_lpips * loss_lpips +
                  self.lambda_auc * loss_auc)

        components = {
            "mae": loss_mae.item(),
            "grad": loss_grad.item(),
            "csi": loss_csi.item(),
            "fcsi": loss_fcsi.item(),
            "lpips": loss_lpips.item(),
            "auc": loss_auc.item(),
            "continuity": 0.0,
        }

        # Continuity (optional)
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

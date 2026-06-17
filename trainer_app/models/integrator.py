"""RK2 时间积分器。

按照 method.md 中描述的二阶 Runge-Kutta（中点法）：
  预测步: ζ* = ζ^n + (Δt/2) · F(ζ^n, v^n)
  校正步: ζ^{n+1} = ζ^n + Δt · F(ζ*, v^{n+1/2})

其中 F = ∂ζ/∂t = PhysicsModule.forward()
"""
import torch
import torch.nn as nn

from .physics import PhysicsModule


class RK2Integrator(nn.Module):
    """二阶 Runge-Kutta 可微时间积分器。

    用于从初始涡度场和物理模块推进预测未来时刻的涡度场。
    """

    def __init__(self, physics_module: PhysicsModule, dt: float = 300.0):
        """初始化 RK2 积分器。

        Args:
            physics_module: 物理模块，计算 ∂ζ/∂t
            dt: 时间步长（秒），默认 300s = 5min
        """
        super().__init__()
        self.physics = physics_module
        self.dt = dt

    def _step(
        self,
        zeta: torch.Tensor,
        velocity: torch.Tensor,
        dt: float,
        omega_x: torch.Tensor | None = None,
        omega_y: torch.Tensor | None = None,
        vertical_velocity: torch.Tensor | None = None,
        rho: torch.Tensor | None = None,
        pressure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """单步推进。

        dt * physics_output easily overflows float16 over many steps.
        Explicitly cast to float32 — autocast(enabled=False) only stops
        further autocasting but does NOT upcast existing float16 tensors.
        """
        # Force float32: AMP autocast makes encoder/projection outputs float16,
        # and half-precision can't represent dt*physics_output over 36 steps.
        zeta_f = zeta.float()
        vel_f = velocity.float()
        ox_f = omega_x.float() if omega_x is not None else None
        oy_f = omega_y.float() if omega_y is not None else None
        vv_f = vertical_velocity.float() if vertical_velocity is not None else None
        rho_f = rho.float() if rho is not None else None
        p_f = pressure.float() if pressure is not None else None

        # F_n = F(ζ^n, v^n)
        f_n = self.physics(zeta_f, vel_f, ox_f, oy_f, vv_f, rho_f, p_f)

        # 预测步: ζ* = ζ^n + (dt/2) * F_n
        zeta_star = zeta_f + 0.5 * dt * f_n

        # F_* = F(ζ*, v^{n+1/2})
        f_star = self.physics(zeta_star, vel_f, ox_f, oy_f, vv_f, rho_f, p_f)

        # 校正步: ζ^{n+1} = ζ^n + dt * F_*
        zeta_next = zeta_f + dt * f_star

        # Safety clamp — with correct dx the physics stays in ~[-0.1, 0.1];
        # ±1.0 gives 30× headroom over the target range [-0.03, 0.03]
        return torch.clamp(zeta_next, -1.0, 1.0)

    def forward(
        self,
        zeta_init: torch.Tensor,
        velocities: torch.Tensor,
        omega_x: torch.Tensor | None = None,
        omega_y: torch.Tensor | None = None,
        vertical_velocities: torch.Tensor | None = None,
        rho: torch.Tensor | None = None,
        pressure: torch.Tensor | None = None,
        step_callback=None,
    ) -> torch.Tensor:
        """积分整个未来序列。

        Args:
            zeta_init: 初始涡度场 (B, H, W, L)
            velocities: 未来各时间步的速度场 (B, T_out, 3, H, W, L)
            omega_x: 水平涡度 x (B, T_out, H, W, L)，可选
            omega_y: 水平涡度 y (B, T_out, H, W, L)，可选
            vertical_velocities: 垂直速度 (B, T_out, H, W, L)，可选
            rho: 密度 (B, T_out, H, W, L)，可选
            pressure: 气压 (B, T_out, H, W, L)，可选
            step_callback: 可选回调(step_idx, zeta_tensor) → None

        Returns:
            所有未来时刻的涡度预测 (B, T_out, H, W, L)
        """
        B, T_out = velocities.shape[0], velocities.shape[1]
        H, W, L_z = velocities.shape[3], velocities.shape[4], velocities.shape[5]

        zeta = zeta_init
        predictions = []

        for t in range(T_out):
            v_t = velocities[:, t, :, :, :, :]  # (B, 3, H, W, L)
            ox_t = omega_x[:, t, :, :, :] if omega_x is not None else None
            oy_t = omega_y[:, t, :, :, :] if omega_y is not None else None
            vv_t = vertical_velocities[:, t, :, :, :] if vertical_velocities is not None else None
            rho_t = rho[:, t, :, :, :] if rho is not None else None
            p_t = pressure[:, t, :, :, :] if pressure is not None else None

            zeta = self._step(zeta, v_t, self.dt, ox_t, oy_t, vv_t, rho_t, p_t)
            predictions.append(zeta)
            if step_callback is not None:
                step_callback(t, zeta)

        return torch.stack(predictions, dim=1)  # (B, T_out, H, W, L)

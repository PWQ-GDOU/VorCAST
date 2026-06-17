"""物理模块：实现垂直涡度演化方程的可微分离散化算子。

三算子结构（四个数学项）：
1. 平流项: -v · ∇ζ
2. 倾斜+拉伸项: ω · ∇w  (源汇算子)
3. 扩散项: ν_t ∇²ζ
4. 斜压项: (1/ρ)(∂ρ/∂x ∂p/∂y - ∂ρ/∂y ∂p/∂x)  (源汇补充)

所有算子使用 PyTorch 实现，保持可微分性以支持端到端训练。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhysicsModule(nn.Module):
    r"""垂直涡度演化物理模块。

    计算 ∂ζ/∂t = -v·∇ζ + ω·∇w + ν_t∇²ζ + (1/ρ)(∂ρ/∂x·∂p/∂y - ∂ρ/∂y·∂p/∂x)
    """

    def __init__(self, config: dict):
        super().__init__()
        phys_cfg = config.get("model", {}).get("physics", {})
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})          # 必须存在，否则 encoder_downsample 读取失败
        encoder_depth = model_cfg.get("depth", 2)    # 必须存在，否则 downsample_factor 计算失败

        self.diffusion_coef = phys_cfg.get("diffusion_coef", 0.1)
        self.use_baroclinic = phys_cfg.get("use_baroclinic", False)

        # 计算真实网格间距（米）
        # 空间窗口 2.56° / 128 格点 × 111,320 m/度 ≈ 2,226 m（原生分辨率）
        spatial_degree = data_cfg.get("spatial_degree", 2.56)
        grid_size = data_cfg.get("grid_size", 128)
        deg_per_cell = spatial_degree / grid_size
        meters_per_deg_lat = 111_320.0
        dx_native = deg_per_cell * meters_per_deg_lat

        # 编码器下采样因子：根据 encoder_downsample 配置
        encoder_downsample = model_cfg.get("encoder_downsample", True)
        downsample_factor = 2 ** encoder_depth if encoder_downsample else 1

        # 物理模块运行在编码器输出分辨率，dx 需按实际格点间距调整
        self.dx = phys_cfg.get("dx_override", dx_native * downsample_factor)
        self.dy = phys_cfg.get("dy_override", self.dx)
        # 垂直间距：GridRad 约 0-20km / 29 层 ≈ 690m（垂直方向不下采样）
        vert_layers = data_cfg.get("vertical_layers", None) or 29
        self.dz = phys_cfg.get("dz_override", 20_000.0 / vert_layers)

        # 可学习的扩散系数（每个空间维度可不同，softplus 保证正值）
        inv_sp = math.log(math.exp(self.diffusion_coef) - 1) if self.diffusion_coef > 0 else 0.0
        self.raw_nu_x = nn.Parameter(torch.tensor(inv_sp))
        self.raw_nu_y = nn.Parameter(torch.tensor(inv_sp))
        self.raw_nu_z = nn.Parameter(torch.tensor(inv_sp))

    def _gradient_3d(self, f: torch.Tensor, dx: float | None = None,
                      dy: float | None = None, dz: float | None = None) -> tuple[torch.Tensor, ...]:
        """计算三维梯度 ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)。

        使用中心差分格式（replicate 填充避免边界环绕伪影）：
        ∂f/∂x[i] ≈ (f[i+1] - f[i-1]) / (2*dx)
        """
        if dx is None:
            dx = self.dx
        if dy is None:
            dy = self.dy
        if dz is None:
            dz = self.dz
        p = (1, 1, 1, 1, 1, 1)
        f_padded = F.pad(f, p, mode='replicate')
        f_plus_x = f_padded[..., 1:-1, 2:, 1:-1]
        f_minus_x = f_padded[..., 1:-1, :-2, 1:-1]
        f_plus_y = f_padded[..., 2:, 1:-1, 1:-1]
        f_minus_y = f_padded[..., :-2, 1:-1, 1:-1]
        f_plus_z = f_padded[..., 1:-1, 1:-1, 2:]
        f_minus_z = f_padded[..., 1:-1, 1:-1, :-2]

        df_dx = (f_plus_x - f_minus_x) / (2.0 * dx)
        df_dy = (f_plus_y - f_minus_y) / (2.0 * dy)
        df_dz = (f_plus_z - f_minus_z) / (2.0 * dz)
        return df_dx, df_dy, df_dz

    def _laplacian_3d(self, f: torch.Tensor, dx: float | None = None,
                       dy: float | None = None, dz: float | None = None) -> torch.Tensor:
        r"""计算三维拉普拉斯 ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²。

        使用中心差分（replicate 填充避免边界环绕伪影）：
        ∂²f/∂x²[i] ≈ (f[i+1] - 2*f[i] + f[i-1]) / dx²
        """
        if dx is None:
            dx = self.dx
        if dy is None:
            dy = self.dy
        if dz is None:
            dz = self.dz
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_dz2 = 1.0 / (dz * dz)
        p = (1, 1, 1, 1, 1, 1)
        f_padded = F.pad(f, p, mode='replicate')

        f_plus_x = f_padded[..., 2:, 1:-1, 1:-1]
        f_minus_x = f_padded[..., :-2, 1:-1, 1:-1]
        f_plus_y = f_padded[..., 1:-1, 2:, 1:-1]
        f_minus_y = f_padded[..., 1:-1, :-2, 1:-1]
        f_plus_z = f_padded[..., 1:-1, 1:-1, 2:]
        f_minus_z = f_padded[..., 1:-1, 1:-1, :-2]

        lap_x = (f_plus_x - 2 * f + f_minus_x) * inv_dx2
        lap_y = (f_plus_y - 2 * f + f_minus_y) * inv_dy2
        lap_z = (f_plus_z - 2 * f + f_minus_z) * inv_dz2
        return lap_x + lap_y + lap_z

    def advection(self, zeta: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        r"""平流项: -v · ∇ζ

        Args:
            zeta: 垂直涡度 (B, H, W, L)
            velocity: 三维速度场 (B, 3, H, W, L)，channels = (u, v, w)

        Returns:
            平流项贡献 (B, H, W, L)
        """
        dz_dx, dz_dy, dz_dz = self._gradient_3d(zeta)

        u = velocity[:, 0:1, :, :, :].squeeze(1)
        v = velocity[:, 1:2, :, :, :].squeeze(1)
        w = velocity[:, 2:3, :, :, :].squeeze(1)

        v_dot_grad = u * dz_dx + v * dz_dy + w * dz_dz
        return -v_dot_grad

    def tilt_and_stretch(self, zeta: torch.Tensor,
                         omega_x: torch.Tensor, omega_y: torch.Tensor,
                         vertical_velocity: torch.Tensor) -> torch.Tensor:
        r"""倾斜与拉伸项: ω · ∇w = ω_x·∂w/∂x + ω_y·∂w/∂y + ζ·∂w/∂z

        Args:
            zeta: 垂直涡度 (B, H, W, L)
            omega_x: 水平涡度 x 分量 (B, H, W, L)
            omega_y: 水平涡度 y 分量 (B, H, W, L)
            vertical_velocity: 垂直速度 w (B, H, W, L)

        Returns:
            倾斜与拉伸项贡献 (B, H, W, L)
        """
        dw_dx, dw_dy, dw_dz = self._gradient_3d(vertical_velocity)

        tilt_x = omega_x * dw_dx
        tilt_y = omega_y * dw_dy
        stretch = zeta * dw_dz

        return tilt_x + tilt_y + stretch

    def diffusion(self, zeta: torch.Tensor, dx: float | None = None,
                   dy: float | None = None, dz: float | None = None) -> torch.Tensor:
        r"""三维湍流扩散项: ν_t ∇²ζ

        Args:
            zeta: 垂直涡度 (B, H, W, L)
            dx, dy, dz: 网格间距（默认使用 self.dx, self.dy, self.dz）

        Returns:
            扩散项贡献 (B, H, W, L)
        """
        if dx is None:
            dx = self.dx
        if dy is None:
            dy = self.dy
        if dz is None:
            dz = self.dz
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_dz2 = 1.0 / (dz * dz)
        p = (1, 1, 1, 1, 1, 1)
        z_padded = F.pad(zeta, p, mode='replicate')

        z_plus_x = z_padded[..., 1:-1, 2:, 1:-1]
        z_minus_x = z_padded[..., 1:-1, :-2, 1:-1]
        z_plus_y = z_padded[..., 2:, 1:-1, 1:-1]
        z_minus_y = z_padded[..., :-2, 1:-1, 1:-1]
        z_plus_z = z_padded[..., 1:-1, 1:-1, 2:]
        z_minus_z = z_padded[..., 1:-1, 1:-1, :-2]

        # 各向异性扩散（每个维度可学习不同系数）
        lap_x = (z_plus_x - 2 * zeta + z_minus_x) * inv_dx2
        lap_y = (z_plus_y - 2 * zeta + z_minus_y) * inv_dy2
        lap_z = (z_plus_z - 2 * zeta + z_minus_z) * inv_dz2

        nu_x = F.softplus(self.raw_nu_x)
        nu_y = F.softplus(self.raw_nu_y)
        nu_z = F.softplus(self.raw_nu_z)
        result = nu_x * lap_x + nu_y * lap_y + nu_z * lap_z
        return result

    def baroclinic(self, rho: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        r"""斜压项: (1/ρ)(∂ρ/∂x·∂p/∂y - ∂ρ/∂y·∂p/∂x)

        代表力管效应——冷暖空气交汇时密度梯度和气压梯度不共线产生的力矩。

        Args:
            rho: 大气密度 (B, H, W, L)
            pressure: 气压 (B, H, W, L)

        Returns:
            斜压项贡献 (B, H, W, L)
        """
        drho_dx, drho_dy, _ = self._gradient_3d(rho)
        dp_dx, dp_dy, _ = self._gradient_3d(pressure)

        baroclinic_term = (drho_dx * dp_dy - drho_dy * dp_dx) / (rho + 1e-8)
        return baroclinic_term

    def forward(
        self,
        zeta: torch.Tensor,
        velocity: torch.Tensor,
        omega_x: torch.Tensor | None = None,
        omega_y: torch.Tensor | None = None,
        vertical_velocity: torch.Tensor | None = None,
        rho: torch.Tensor | None = None,
        pressure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""前向传播：计算 ∂ζ/∂t。

        Args:
            zeta: 垂直涡度 (B, H, W, L)
            velocity: 三维速度 (B, 3, H, W, L)，channels=(u, v, w)
            omega_x: 水平涡度 x 分量 (B, H, W, L)，可选
            omega_y: 水平涡度 y 分量 (B, H, W, L)，可选
            vertical_velocity: 垂直速度 w (B, H, W, L)，可选
            rho: 密度 (B, H, W, L)，可选（斜压项需要）
            pressure: 气压 (B, H, W, L)，可选（斜压项需要）

        Returns:
            ∂ζ/∂t (B, H, W, L)
        """
        # 1. 平流项
        d_zeta_dt = self.advection(zeta, velocity)

        # 2. 倾斜与拉伸项（源汇算子）
        if omega_x is not None and omega_y is not None and vertical_velocity is not None:
            d_zeta_dt += self.tilt_and_stretch(zeta, omega_x, omega_y, vertical_velocity)
        elif vertical_velocity is not None:
            # 简化：如果没有 omega_x/y，仅用拉伸项 ζ·∂w/∂z
            _, _, dw_dz = self._gradient_3d(vertical_velocity)
            d_zeta_dt += zeta * dw_dz

        # 3. 扩散项
        d_zeta_dt += self.diffusion(zeta)

        # 4. 斜压项（源汇补充）
        if self.use_baroclinic and rho is not None and pressure is not None:
            d_zeta_dt += self.baroclinic(rho, pressure)

        return d_zeta_dt

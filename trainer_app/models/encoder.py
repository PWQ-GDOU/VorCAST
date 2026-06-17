"""ResUNet 3D 编码器-解码器。

按照方法2设计：
- 编码器: 输入 (B,T_in,H,W,L,C) → 合并 T_in*C → Conv3D 1×1×1 → ResUNet 编码器
- 解码器: 物理模块输出 + skip connections → ConvTranspose3d → 全分辨率涡度
- 残差连接缓解梯度消失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class ResidualBlock3D(nn.Module):
    """3D 残差块：Conv3D → BN → ReLU → Conv3D → BN → 残差连接 → ReLU"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class EncoderStage(nn.Module):
    """单个编码器阶段：残差块 + 可选下采样（仅 H,W；L 保持不变）"""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2,
                 downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList()
        # stride=(2,2,1) when downsampling, else (1,1,1)
        stride = (2, 2, 1) if downsample else 1
        self.blocks.append(ResidualBlock3D(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            self.blocks.append(ResidualBlock3D(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class DecoderStage(nn.Module):
    """单个解码器阶段：上采样 + skip连接 + 残差块

    使用 interpolate 上采样替代 ConvTranspose3d，避免大数据集上显存溢出。
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # interpolate 2× 后接卷积融合（仅 H, W；L 保持不变）
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        # 上采样后 + skip 连接的融合块
        self.fusion = ResidualBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(2, 2, 1), mode='trilinear',
                          align_corners=False)
        x = F.relu(self.bn(self.conv(x)))
        # Pad/crop 到 skip 尺寸（处理奇数尺寸）
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fusion(x)


class ResUNet3DEncoder(nn.Module):
    """3D ResUNet 编码器（返回 skip connections 供解码器使用）。

    输入:  (B, T_in, H, W, L, C)
    输出:  (bottleneck_features, [skip_0, skip_1, ...])
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        t_in = data_cfg.get("history_steps", 12)
        self.c_radar = data_cfg.get("in_channels", 4)
        self.in_channels_merged = t_in * self.c_radar

        base_ch = model_cfg.get("base_channels", 64)
        self.depth = model_cfg.get("depth", 2)
        self.downsample = model_cfg.get("encoder_downsample", True)
        dropout_rate = model_cfg.get("dropout", 0.2)

        # 1×1×1 卷积：将合并通道映射到基础通道
        self.input_proj = nn.Conv3d(self.in_channels_merged, base_ch,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self.input_bn = nn.BatchNorm3d(base_ch)

        # 编码器各阶段
        self.stages = nn.ModuleList()
        ch = base_ch
        for i in range(self.depth):
            out_ch = min(ch * 2, 1024)
            self.stages.append(EncoderStage(ch, out_ch, downsample=self.downsample))
            ch = out_ch

        self.output_channels = ch
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (B, T_in, H, W, L, C) 原始输入

        Returns:
            (bottleneck, skips) 其中 skips = [skip_0, skip_1, ...]
            skip_0 是输入投影后的特征，skip_1 是第一个阶段后的特征，以此类推
        """
        B, T_in, H, W, L, C = x.shape

        # 合并时间与通道: (B, T_in, H, W, L, C) → (B, T_in*C, H, W, L)
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()
        x = x.view(B, T_in * C, H, W, L)

        # 1×1×1 投影
        x = F.relu(self.input_bn(self.input_proj(x)))
        skips = [x]  # skip_0: 全分辨率

        # 编码器各阶段，保存每个阶段的输出作为 skip
        # 使用 gradient checkpointing 避免存储中间激活值：
        #   B=1, 128×128×29 下 Stage1 残差块内部激活 ≈930MB，Stage2 ≈460MB
        #   checkpoint 后在反向传播时重算，节省 ~1.4GB 显存
        for stage in self.stages:
            x = torch.utils.checkpoint.checkpoint(stage, x, use_reentrant=False)
            skips.append(x)

        # 最后一个 skip 是瓶颈特征本身，不传给解码器
        x = self.dropout(x)
        skips = skips[:-1]  # 去掉瓶颈，只保留给解码器用的跳连

        return x, skips


class ResUNet3DDecoder(nn.Module):
    """3D ResUNet 解码器：将物理模块的低分辨率涡度预测上采样到全分辨率。

    使用 skip connections 从编码器恢复空间细节。
    按时间步分块处理（由 model.decoder_chunk_size 控制），
    避免 B×T_out 过大导致显存溢出。
    """

    def __init__(self, config: dict, bottleneck_channels: int,
                 skip_channels_list: list[int], max_timesteps_per_chunk: int | None = None):
        super().__init__()
        model_cfg = config.get("model", {})

        self.depth = model_cfg.get("depth", 2)
        base_ch = model_cfg.get("base_channels", 64)
        dropout_rate = model_cfg.get("dropout", 0.2)

        if max_timesteps_per_chunk is not None:
            self.max_chunk = max_timesteps_per_chunk
        else:
            chunk_size = model_cfg.get("decoder_chunk_size", 1)
            self.max_chunk = chunk_size if chunk_size > 0 else 999

        # 初始投影：将单通道涡度（或低通道数输入）映射到特征空间
        self.input_proj = nn.Sequential(
            nn.Conv3d(1, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(),
        )

        # 解码器各阶段（逆序：从最深到最浅）
        # skip_channels_list[0] 是全分辨率跳连，最后一个是最深跳连
        self.stages = nn.ModuleList()
        reversed_skips = list(reversed(skip_channels_list))
        in_ch = base_ch
        for i, skip_ch in enumerate(reversed_skips):
            out_ch = skip_ch  # 输出通道数与对应跳连匹配
            self.stages.append(DecoderStage(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # 最终输出投影：Tanh 约束输出到 [-output_scale, +output_scale]
        # 避免随机初始化时 decoder 输出数百倍于目标值造成梯度爆炸
        zeta_range = 0.03  # ζ_max ≈ 2 × AzShear_max ≈ 0.03 s⁻¹
        self.register_buffer("output_scale", torch.tensor(zeta_range))
        self.output_proj = nn.Sequential(
            nn.Conv3d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_ch // 2),
            nn.ReLU(),
            nn.Conv3d(in_ch // 2, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout3d(dropout_rate)

    def _decode_chunk(self, x_chunk: torch.Tensor,
                      skips_chunk: list[torch.Tensor]) -> torch.Tensor:
        """对单个时间块执行解码（共享权重）。"""
        x = self.input_proj(x_chunk)
        for stage, skip in zip(self.stages, reversed(skips_chunk)):
            x = stage(x, skip)
        x = self.dropout(x)
        return self.output_proj(x)

    def _decode_chunk_ckpt(self, x_chunk: torch.Tensor,
                           *skips: torch.Tensor) -> torch.Tensor:
        """checkpoint-compatible wrapper: accepts *skips instead of list."""
        return self._decode_chunk(x_chunk, list(skips))

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor],
                T_out: int) -> torch.Tensor:
        """
        Args:
            x: 物理模块预测 (B, T_out, H_bn, W_bn, L)  — 低分辨率涡度
            skips: 编码器跳连特征列表，每个 (B, C, H_i, W_i, L)
            T_out: 未来时间步数

        Returns:
            (B, T_out, H_full, W_full, L) 全分辨率涡度
        """
        B = x.shape[0]
        results = []

        for b in range(B):
            batch_results = []
            x_b = x[b]  # (T_out, H_bn, W_bn, L)
            # 每个 batch 的 skip（未扩展）
            skips_b = [s[b] for s in skips]  # list of (C, H_i, W_i, L)

            for t_start in range(0, T_out, self.max_chunk):
                t_end = min(t_start + self.max_chunk, T_out)
                chunk_size = t_end - t_start

                x_chunk = x_b[t_start:t_end]  # (chunk, H_bn, W_bn, L)
                x_chunk = x_chunk.unsqueeze(1)  # (chunk, 1, H_bn, W_bn, L)

                s_chunk = [s.unsqueeze(0).expand(chunk_size, -1, -1, -1, -1)
                           for s in skips_b]

                # gradient checkpointing: 避免存储 36 个时间步 × 全分辨率解码器激活
                out_chunk = torch.utils.checkpoint.checkpoint(
                    self._decode_chunk_ckpt, x_chunk, *s_chunk,
                    use_reentrant=False,
                )  # (chunk, 1, H, W, L)
                batch_results.append(out_chunk.squeeze(1))  # (chunk, H, W, L)

            results.append(torch.cat(batch_results, dim=0))  # (T_out, H, W, L)

        return torch.stack(results, dim=0) * self.output_scale  # (B, T_out, H, W, L)

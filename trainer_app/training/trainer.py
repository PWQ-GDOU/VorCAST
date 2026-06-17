"""核心训练引擎。

支持：
- 标准训练循环（含验证）
- GPU 混合精度训练
- 暂停/恢复/终止训练
- 后台运行模式
- 实时指标回调
"""
import time
import signal
import threading
from pathlib import Path
from typing import Any, Callable
from collections.abc import Callable as CallableType

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..utils.device import get_device
from ..utils.exceptions import TrainingError, TrainingInterrupted
from ..utils.visualization import generate_all_plots
from ..models.encoder import ResUNet3DEncoder, ResUNet3DDecoder
from ..models.physics import PhysicsModule
from ..models.integrator import RK2Integrator
from ..models.loss import PhysicsLoss
from ..log.logger import setup_logger
from ..log.manager import LogManager
from ..history.storage import HistoryStorage
from .monitor import TrainingMonitor, compute_binary_metrics
from .checkpoint import CheckpointManager


class Trainer:
    """龙卷风涡度预测模型训练器。

    整合 Encoder + PhysicsModule + RK2Integrator + Loss，
    提供完整的训练/验证/测试流程。
    """

    def __init__(self, config: dict):
        self.config = config
        training_cfg = config.get("training", {})
        device_cfg = config.get("device", {})
        model_cfg = config.get("model", {})
        loss_cfg = model_cfg.get("loss", {})

        # 设备
        self.device = get_device(
            device_cfg.get("gpu_enabled", True),
            device_cfg.get("gpu_id", 0),
        )
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")  # Enable TF32 on Ampere+

        # 超参数
        self.epochs = training_cfg.get("epochs", 100)
        self.batch_size = training_cfg.get("batch_size", 2)
        self.learning_rate = training_cfg.get("learning_rate", 0.001)
        self.weight_decay = training_cfg.get("weight_decay", 0.0001)
        self.use_amp = training_cfg.get("use_amp", True)
        self.early_stop_patience = training_cfg.get("early_stop_patience", 15)

        # 模型组件
        self.encoder = ResUNet3DEncoder(config).to(self.device)
        self.physics = PhysicsModule(config).to(self.device)
        self.integrator = RK2Integrator(self.physics, dt=300.0).to(self.device)

        # 解码器：将物理模块低分辨率涡度预测 + skip connections → 全分辨率
        model_cfg = config.get("model", {})
        base_ch = model_cfg.get("base_channels", 64)
        depth = model_cfg.get("depth", 2)
        skip_channels = [base_ch]  # skip_0: 输入投影输出
        ch = base_ch
        for i in range(depth - 1):  # 除去瓶颈层，共 depth-1 个跳连
            ch = min(ch * 2, 1024)
            skip_channels.append(ch)
        self.decoder = ResUNet3DDecoder(
            config, self.encoder.output_channels, skip_channels,
        ).to(self.device)
        self.criterion = PhysicsLoss(
            lambda_grad=loss_cfg.get("lambda_grad", 0.1),
            lambda_continuity=loss_cfg.get("lambda_continuity", 0.0),
            lambda_focal=loss_cfg.get("lambda_focal", 0.0),
            lambda_csi=loss_cfg.get("lambda_csi", 1.0),
            lambda_fcsi=loss_cfg.get("lambda_fcsi", 0.5),
            lambda_lpips=loss_cfg.get("lambda_lpips", 0.1),
            lambda_auc=loss_cfg.get("lambda_auc", 0.05),
            lpips_device=str(self.device),
            lpips_interval=loss_cfg.get("lpips_interval", 5),
            max_auc_pairs=loss_cfg.get("max_auc_pairs", 10000),
        )

        # 投影头：编码器输出 → 物理模块输入（涡度初值 + 速度场等）
        self._build_projection_heads()

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )

        # 学习率调度器
        scheduler_type = training_cfg.get("lr_scheduler", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs,
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=5,
            )

        self.scaler = GradScaler() if self.use_amp and self.device.type == "cuda" else None

        # 管理与监控
        self.monitor = TrainingMonitor()
        self.checkpoint_mgr = CheckpointManager(
            training_cfg.get("checkpoint_dir", "./checkpoints"),
            training_cfg.get("save_best_only", True),
        )
        log_cfg = config.get("logging", {})
        self.log_manager = LogManager(log_cfg.get("log_dir", "./logs"))
        self.history_storage = HistoryStorage()

        self.logger = None
        self._task_dir = None

        # 状态控制
        self._running = False
        self._paused = False
        self._stop_requested = False
        self._interrupted = False

        # 回调
        self._callbacks: list[CallableType] = []

    def _build_projection_heads(self):
        """从编码器输出投影得到物理模块所需的各输入场。

        速度场和涡度分量输出 T_out 个通道，每个未来时间步独立预测，
        而非将单帧复制 36 次。
        """
        enc_out_ch = self.encoder.output_channels
        data_cfg = self.config.get("data", {})
        self.T_out = data_cfg.get("future_steps", 36)
        self.grid_size = data_cfg.get("grid_size", 128)
        vert_layers = data_cfg.get("vertical_layers", None) or 20

        # 投影头：编码特征 → 涡度初值 (1, H', W', L)
        # ζ = 2 × AzShear, range ~[-0.03, 0.03]
        zeta_scale = 0.03
        self.zeta_proj = nn.Sequential(
            nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            nn.BatchNorm3d(enc_out_ch // 2),
            nn.ReLU(),
            nn.Conv3d(enc_out_ch // 2, 1, 3, padding=1),
            nn.Tanh(),
        ).to(self.device)
        self.zeta_scale = zeta_scale

        # 投影头：编码特征 → 所有时间步的速度场 (T_out * 3, H', W', L)
        self.velocity_proj = nn.Sequential(
            nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            nn.BatchNorm3d(enc_out_ch // 2),
            nn.ReLU(),
            nn.Conv3d(enc_out_ch // 2, self.T_out * 3, 3, padding=1),
            nn.Tanh(),
        ).to(self.device)

        # 投影头：编码特征 → 所有时间步的水平涡度 (T_out * 2, H', W', L)
        self.omega_proj = nn.Sequential(
            nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            nn.BatchNorm3d(enc_out_ch // 2),
            nn.ReLU(),
            nn.Conv3d(enc_out_ch // 2, self.T_out * 2, 3, padding=1),
            nn.Tanh(),
        ).to(self.device)

    def parameters(self):
        """获取所有可训练参数（去重：integrator 已包含 physics 参数）。"""
        seen = set()
        for m in [self.encoder, self.decoder, self.integrator,
                   self.zeta_proj, self.velocity_proj, self.omega_proj]:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def load_pretrained(self, ckpt_path: str, load_optimizer: bool = False,
                         strict: bool = True) -> dict:
        """从 .pth 文件加载预训练权重，在此基础上继续训练。

        支持两种格式：
        1. 本训练器保存的 checkpoint（含 model_state/opt_state/epoch）
        2. 原始 state_dict .pth 文件（仅模型权重）

        Args:
            ckpt_path: .pth 文件路径
            load_optimizer: 是否同时恢复优化器状态（继续训练）
            strict: 权重加载是否严格匹配所有参数名（False 允许部分迁移学习）

        Returns:
            加载信息 dict
        """
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # 格式1: 完整 checkpoint
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
            self._load_model_state(model_state, strict)

            if load_optimizer and "opt_state" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["opt_state"])

            info = {
                "source": Path(ckpt_path).name,
                "epoch": checkpoint.get("epoch"),
                "metrics": checkpoint.get("metrics"),
                "loaded_optimizer": load_optimizer,
            }
        else:
            # 格式2: 原始 state_dict
            self._load_model_state(checkpoint, strict)
            info = {
                "source": Path(ckpt_path).name,
                "epoch": None,
                "metrics": None,
                "loaded_optimizer": False,
            }

        return info

    def _load_model_state(self, state_dict: dict, strict: bool):
        """将 state_dict 中的权重加载到各子模块，返回加载结果。"""
        key_map = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "physics": self.physics,
            "zeta_proj": self.zeta_proj,
            "velocity_proj": self.velocity_proj,
            "omega_proj": self.omega_proj,
            "integrator": self.integrator,
        }

        loaded = []
        missing = []
        for key, module in key_map.items():
            if key in state_dict:
                result = module.load_state_dict(state_dict[key], strict=strict)
                loaded.append(key)
                if hasattr(result, "missing_keys"):
                    missing.extend(result.missing_keys)
            elif not strict:
                loaded.append(f"{key} (跳过)")

        return {"loaded": loaded, "missing": missing}

    def on_metric_update(self, callback: CallableType):
        """注册指标更新回调（供 TUI 使用）。"""
        self._callbacks.append(callback)

    def _notify_callbacks(self, metrics: dict):
        for cb in self._callbacks:
            try:
                cb(metrics)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Callback failed: {e}")

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """前向传播：Encoder → Projection → RK2 → Decoder。

        Args:
            batch: {"input": (B, T_in, H, W, L, C), "target": (B, T_out, H, W, L)}

        Returns:
            (predictions, targets, projected_fields)
        """
        x = batch["input"].to(self.device)     # (B, T_in, H, W, L, C)
        target = batch["target"].to(self.device)  # (B, T_out, H, W, L)
        B, T_out = x.shape[0], target.shape[1]

        # 编码 → 瓶颈特征 + skip connections
        features, skips = self.encoder(x)  # features: (B, C_enc, H', W', L'), skips: list

        # 投影得到各物理场
        zeta_init = self.zeta_proj(features).squeeze(1) * self.zeta_scale  # (B, H', W', L')

        # 时间依赖的速度场：投影输出 T_out×3 通道 → (B, T_out, 3, H', W', L')
        # 水平分量 (u,v) ×20 m/s，垂直分量 (w) ×5 m/s（风暴中垂直速度远小于水平）
        vel_raw = self.velocity_proj(features)  # (B, T_out*3, H', W', L')
        vel_raw = vel_raw.reshape(B, T_out, 3, *vel_raw.shape[-3:])
        vel_u = vel_raw[:, :, 0:1, :, :, :] * 20.0
        vel_v = vel_raw[:, :, 1:2, :, :, :] * 20.0
        vel_w = vel_raw[:, :, 2:3, :, :, :] * 5.0
        velocities_seq = torch.cat([vel_u, vel_v, vel_w], dim=2)  # (B, T_out, 3, H', W', L')

        # 时间依赖的水平涡度：投影输出 T_out×2 通道 → (B, T_out, 2, H', W', L')
        omega_raw = self.omega_proj(features) * 0.1  # (B, T_out*2, H', W', L')
        omega_raw = omega_raw.reshape(B, T_out, 2, *omega_raw.shape[-3:])

        omega_x_seq = omega_raw[:, :, 0, :, :, :]  # (B, T_out, H', W', L')
        omega_y_seq = omega_raw[:, :, 1, :, :, :]  # (B, T_out, H', W', L')
        vv_seq = velocities_seq[:, :, 2, :, :, :]   # (B, T_out, H', W', L')

        # RK2 积分预测未来涡度（瓶颈分辨率）
        zeta_pred_lowres = self.integrator(
            zeta_init, velocities_seq,
            omega_x=omega_x_seq, omega_y=omega_y_seq,
            vertical_velocities=vv_seq,
        )  # (B, T_out, H', W', L')

        # 解码器：低分辨率涡度 + skip connections → 全分辨率
        predictions = self.decoder(zeta_pred_lowres, skips, T_out)  # (B, T_out, H, W, L)

        projected = {
            "velocity": velocities_seq,
            "zeta_init": zeta_init,
            "omega_x": omega_x_seq[:, 0, :, :, :],
            "omega_y": omega_y_seq[:, 0, :, :, :],
        }
        return predictions, target, projected

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """训练一个 epoch。"""
        self.encoder.train()
        self.decoder.train()
        self.physics.train()
        self.zeta_proj.train()
        self.velocity_proj.train()
        self.omega_proj.train()

        epoch_loss = 0.0
        epoch_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": [],
                         "f1": [], "dice": [], "iou": [], "pos_ratio": [],
                         "zero_baseline_acc": [], "pred_std": [], "target_std": [],
                         "csi_loss": [], "fcsi_loss": [], "lpips_loss": [], "auc_loss": []}

        for batch_idx, batch in enumerate(train_loader):
            if self._stop_requested:
                break
            while self._paused:
                time.sleep(0.1)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    pred, target, proj = self.forward(batch)
                    loss, components = self.criterion(pred, target, proj.get("velocity"))
                # Skip NaN batches (numerical instability recovery)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred, target, proj = self.forward(batch)
                loss, components = self.criterion(pred, target, proj.get("velocity"))
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 二分类指标
            bin_metrics = compute_binary_metrics(pred, target)

            # 预测值统计（用于诊断模型是否预测近零值）
            with torch.no_grad():
                pred_std = pred.std().item()
                target_std = target.std().item()

            metrics = {
                "loss": components["mae"],
                "total_loss": loss.item(),
                "accuracy": bin_metrics["accuracy"],
                "precision": bin_metrics["precision"],
                "recall": bin_metrics["recall"],
                "f1": bin_metrics["f1"],
                "dice": bin_metrics["dice"],
                "iou": bin_metrics["iou"],
                "pos_ratio": bin_metrics["pos_ratio"],
                "zero_baseline_acc": bin_metrics["zero_baseline_acc"],
                "grad_loss": components["grad"],
                "csi_loss": components.get("csi", 0.0),
                "fcsi_loss": components.get("fcsi", 0.0),
                "lpips_loss": components.get("lpips", 0.0),
                "auc_loss": components.get("auc", 0.0),
                "continuity_loss": components.get("continuity", 0.0),
                "lr": self.optimizer.param_groups[0]["lr"],
                "pred_std": pred_std,
                "target_std": target_std,
            }

            self.monitor.update("train", metrics, epoch, batch_idx)
            epoch_loss += loss.item()
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k].append(metrics[k])

            if batch_idx % 10 == 0 or batch_idx == 0:
                self._notify_callbacks({
                    "type": "batch",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_batches": len(train_loader),
                    **metrics,
                })
                if self.logger:
                    self.logger.info(
                        f"Epoch {epoch}  Batch {batch_idx}  "
                        f"Loss: {metrics['loss']:.4f}  "
                        f"Dice: {metrics['dice']:.3f}  IoU/CSI: {metrics['iou']:.3f}  "
                        f"Acc: {metrics['accuracy']:.3f}(BL:{metrics['zero_baseline_acc']:.2f})  "
                        f"L_csi={metrics.get('csi_loss', 0):.4f} L_fcsi={metrics.get('fcsi_loss', 0):.4f} L_auc={metrics.get('auc_loss', 0):.4f}"
                    )

            # Always update TUI on every batch (lightweight) for progress visibility
            if batch_idx > 0 and batch_idx % 10 != 0:
                self._notify_callbacks({
                    "type": "batch",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_batches": len(train_loader),
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "pos_ratio": metrics["pos_ratio"],
                    "zero_baseline_acc": metrics["zero_baseline_acc"],
                    "grad_loss": metrics["grad_loss"],
                    "total_loss": metrics["total_loss"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "pred_std": metrics["pred_std"],
                    "target_std": metrics["target_std"],
                    "lr": metrics["lr"],
                })

        return {k: sum(v) / len(v) if v else 0 for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int,
                 return_sample: bool = False) -> dict | tuple[dict, np.ndarray | None, np.ndarray | None]:
        """验证一个 epoch。

        Args:
            val_loader: 验证 DataLoader
            epoch: 当前 epoch
            return_sample: 是否返回一组预测样本（用于可视化）

        Returns:
            仅指标时返回 dict
            需要样本时返回 (summary, pred_sample, target_sample)
        """
        self.encoder.eval()
        self.decoder.eval()
        self.physics.eval()
        self.zeta_proj.eval()
        self.velocity_proj.eval()
        self.omega_proj.eval()

        epoch_metrics_val = {"loss": [], "accuracy": [], "precision": [], "recall": [],
                             "f1": [], "dice": [], "iou": [], "pos_ratio": [],
                             "zero_baseline_acc": [], "pred_std": [], "target_std": [],
                             "csi_loss": [], "fcsi_loss": [], "lpips_loss": [], "auc_loss": []}
        pred_sample = None
        target_sample = None

        for batch_idx, batch in enumerate(val_loader):
            pred, target, proj = self.forward(batch)
            loss, components = self.criterion(pred, target, proj.get("velocity"))
            bin_metrics = compute_binary_metrics(pred, target)

            if return_sample and pred_sample is None and batch_idx == 0:
                pred_sample = pred[0].cpu().numpy()
                target_sample = target[0].cpu().numpy()

            pred_std_v = pred.std().item()
            target_std_v = target.std().item()

            metrics = {
                "loss": components["mae"],
                "total_loss": loss.item(),
                "accuracy": bin_metrics["accuracy"],
                "precision": bin_metrics["precision"],
                "recall": bin_metrics["recall"],
                "f1": bin_metrics["f1"],
                "dice": bin_metrics["dice"],
                "iou": bin_metrics["iou"],
                "pos_ratio": bin_metrics["pos_ratio"],
                "zero_baseline_acc": bin_metrics["zero_baseline_acc"],
                "grad_loss": components["grad"],
                "csi_loss": components.get("csi", 0.0),
                "fcsi_loss": components.get("fcsi", 0.0),
                "lpips_loss": components.get("lpips", 0.0),
                "auc_loss": components.get("auc", 0.0),
                "continuity_loss": components.get("continuity", 0.0),
                "pred_std": pred_std_v,
                "target_std": target_std_v,
            }
            self.monitor.update("val", metrics, epoch, batch_idx)
            for k in epoch_metrics_val:
                if k in metrics:
                    epoch_metrics_val[k].append(metrics[k])

        summary = {k: sum(v) / len(v) if v else 0 for k, v in epoch_metrics_val.items()}
        if self.logger:
            self.logger.info(
                f"Epoch {epoch}  Valid  "
                f"Loss: {summary['loss']:.4f}  "
                f"Dice: {summary.get('dice', 0):.3f}  "
                f"IoU/CSI: {summary.get('iou', 0):.3f}  "
                f"F1: {summary.get('f1', 0):.3f}  "
                f"| L_csi={summary.get('csi_loss', 0):.4f}  "
                f"L_fcsi={summary.get('fcsi_loss', 0):.4f}  "
                f"L_lpips={summary.get('lpips_loss', 0):.4f}  "
                f"L_auc={summary.get('auc_loss', 0):.4f}"
            )

        if return_sample:
            return summary, pred_sample, target_sample
        return summary

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None,
              task_name: str = "training") -> dict:
        """完整训练流程。

        Returns:
            最终训练结果字典
        """
        log_cfg = self.config.get("logging", {})
        self._task_dir = self.log_manager.create_task_dir(task_name)
        self.logger = setup_logger(
            "trainer", str(self.log_manager._log_dir), task_name,
            level=log_cfg.get("log_level", "INFO"),
        )
        self.log_manager.save_config(self._task_dir, self.config)

        self.logger.info(f"开始训练: {task_name}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"Epochs: {self.epochs}, Batch Size: {self.batch_size}, LR: {self.learning_rate}")

        # 加载预训练权重
        training_cfg = self.config.get("training", {})
        pretrained_path = training_cfg.get("pretrained_path")
        if pretrained_path and Path(pretrained_path).exists():
            load_opt = training_cfg.get("load_optimizer_state", False)
            strict_load = self.config.get("model", {}).get("strict_load", True)
            self.logger.info(f"加载预训练权重: {pretrained_path}")
            info = self.load_pretrained(pretrained_path, load_optimizer=load_opt, strict=strict_load)
            self.logger.info(f"已加载模块: {info['loaded']}")
            if info.get("missing"):
                self.logger.warning(f"未匹配参数: {info['missing']}")
        elif pretrained_path:
            self.logger.warning(f"预训练权重文件不存在，将从头训练: {pretrained_path}")

        # 注册数据库记录
        data_cfg = self.config.get("data", {})
        run_id = self.history_storage.create_run(
            task_name=task_name,
            config=self.config,
            device=str(self.device),
            dataset1=data_cfg.get("dataset1_path", ""),
            dataset2=data_cfg.get("dataset2_path", ""),
            log_dir=str(self._task_dir),
        )

        self._running = True
        self.monitor.reset()
        best_val_loss = float("inf")
        patience_counter = 0
        final_metrics = {}
        last_pred_sample = None
        last_target_sample = None

        # Notify TUI that training loop is about to start
        self._notify_callbacks({
            "type": "status",
            "message": f"Training started on {self.device} — first batch may take 1-3 min (CUDA JIT)",
            "total_batches": len(train_loader) if train_loader else 0,
            "total_epochs": self.epochs,
        })

        # 设置信号处理（仅主线程可用）
        original_sigint = None

        def handle_interrupt(signum, frame):
            self._stop_requested = True
            self._interrupted = True
            if self.logger:
                self.logger.warning("收到中断信号，正在优雅退出...")

        try:
            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handle_interrupt)
        except (ValueError, OSError):
            pass  # 不在主线程，无法设置信号处理器

        try:
            for epoch in range(1, self.epochs + 1):
                if self._stop_requested:
                    break

                # 训练
                train_metrics = self.train_epoch(train_loader, epoch)
                self.logger.info(
                    f"Epoch {epoch}  Train Avg  "
                    f"Loss: {train_metrics['loss']:.4f}  "
                    f"Dice: {train_metrics.get('dice', 0):.3f}  "
                    f"IoU/CSI: {train_metrics.get('iou', 0):.3f}  "
                    f"F1: {train_metrics.get('f1', 0):.3f}  "
                    f"| L_csi={train_metrics.get('csi_loss', 0):.4f}  "
                    f"L_fcsi={train_metrics.get('fcsi_loss', 0):.4f}  "
                    f"L_lpips={train_metrics.get('lpips_loss', 0):.4f}  "
                    f"L_auc={train_metrics.get('auc_loss', 0):.4f}"
                )

                # 通知 TUI
                self._notify_callbacks({
                    "type": "epoch",
                    "epoch": epoch,
                    "total_epochs": self.epochs,
                    "phase": "train",
                    **train_metrics,
                })

                # 保存到数据库
                self.history_storage.save_metric(
                    run_id, epoch, "train", **train_metrics,
                )
                self.history_storage.update_run_status(
                    run_id, "running", current_epoch=epoch,
                )

                # 验证
                if val_loader is not None and epoch % self.config["training"].get("val_every", 1) == 0:
                    val_metrics, pred_sample, target_sample = self.validate(
                        val_loader, epoch, return_sample=True,
                    )
                    if pred_sample is not None:
                        last_pred_sample = pred_sample
                        last_target_sample = target_sample

                    self.history_storage.save_metric(
                        run_id, epoch, "val", **val_metrics,
                    )

                    self._notify_callbacks({
                        "type": "epoch",
                        "epoch": epoch,
                        "total_epochs": self.epochs,
                        "phase": "val",
                        **val_metrics,
                    })

                    # Per-epoch comparison plot (pred vs target vorticity)
                    if last_pred_sample is not None and last_target_sample is not None:
                        try:
                            self._save_epoch_plot(
                                epoch, last_pred_sample, last_target_sample)
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"Epoch plot failed: {e}")

                    # 早停检查
                    val_loss = val_metrics["loss"]
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.checkpoint_mgr.save_best(
                            {"val_loss": val_loss, "epoch": epoch},
                            model_state={
                                "encoder": self.encoder.state_dict(),
                                "decoder": self.decoder.state_dict(),
                                "physics": self.physics.state_dict(),
                                "zeta_proj": self.zeta_proj.state_dict(),
                                "velocity_proj": self.velocity_proj.state_dict(),
                                "omega_proj": self.omega_proj.state_dict(),
                            },
                            opt_state=self.optimizer.state_dict(),
                            epoch=epoch,
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stop_patience:
                            self.logger.info(f"早停触发: epoch {epoch}")
                            break

                # 学习率调度
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"] if val_loader else train_metrics["loss"])
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step()

            final_metrics = {
                "best_val_loss": best_val_loss if best_val_loss != float("inf") else train_metrics["loss"],
            }
            status = "completed" if not self._interrupted else "interrupted"

        except Exception as e:
            self.logger.error(f"训练异常: {e}", exc_info=True)
            status = "failed"
            final_metrics["error"] = str(e)

        finally:
            if original_sigint is not None:
                try:
                    signal.signal(signal.SIGINT, original_sigint)
                except (ValueError, OSError):
                    pass
            self._running = False

            current_epoch = epoch if "epoch" in dir() else 0

            # 保存最终 checkpoint（OOM-safe：先清显存，state_dict 移到 CPU）
            try:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if not self._task_dir:
                    self._task_dir = self.log_manager.create_task_dir(task_name)

                def _to_cpu(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu()
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_to_cpu(v) for v in obj]
                    return obj

                self.checkpoint_mgr.save(
                    "final_checkpoint.pt",
                    model_state=_to_cpu({
                        "encoder": self.encoder.state_dict(),
                        "decoder": self.decoder.state_dict(),
                        "physics": self.physics.state_dict(),
                    }),
                    opt_state=_to_cpu(self.optimizer.state_dict()),
                    epoch=current_epoch,
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Final checkpoint save failed (ignored): {e}")

            # 保存指标 CSV
            try:
                all_train = self.monitor._history.get("train", [])
                all_val = self.monitor._history.get("val", [])
                self.log_manager.save_metrics_csv(
                    self._task_dir,
                    {
                        "epoch": [r["epoch"] for r in all_train],
                        "phase": [r["phase"] for r in all_train],
                        "loss": [r.get("loss", 0) for r in all_train],
                        "total_loss": [r.get("total_loss", 0) for r in all_train],
                        "accuracy": [r.get("accuracy", 0) for r in all_train],
                        "precision": [r.get("precision", 0) for r in all_train],
                        "recall": [r.get("recall", 0) for r in all_train],
                        "f1": [r.get("f1", 0) for r in all_train],
                        "dice": [r.get("dice", 0) for r in all_train],
                        "iou": [r.get("iou", 0) for r in all_train],
                        "csi": [r.get("csi", 0) for r in all_train],
                        "grad_loss": [r.get("grad_loss", 0) for r in all_train],
                        "csi_loss": [r.get("csi_loss", 0) for r in all_train],
                        "fcsi_loss": [r.get("fcsi_loss", 0) for r in all_train],
                        "lpips_loss": [r.get("lpips_loss", 0) for r in all_train],
                        "auc_loss": [r.get("auc_loss", 0) for r in all_train],
                        "continuity_loss": [r.get("continuity_loss", 0) for r in all_train],
                        "pred_std": [r.get("pred_std", 0) for r in all_train],
                        "target_std": [r.get("target_std", 0) for r in all_train],
                        "pos_ratio": [r.get("pos_ratio", 0) for r in all_train],
                        "lr": [r.get("lr", 0) for r in all_train],
                        # Validation metrics
                        "val_loss": [r.get("loss", 0) for r in all_val],
                        "val_accuracy": [r.get("accuracy", 0) for r in all_val],
                        "val_f1": [r.get("f1", 0) for r in all_val],
                        "val_iou": [r.get("iou", 0) for r in all_val],
                        "val_csi": [r.get("csi", 0) for r in all_val],
                    },
                )
                self.log_manager.link_checkpoint(
                    self._task_dir, str(self.checkpoint_mgr._dir),
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Metrics save failed (ignored): {e}")

            # 生成训练效果图
            try:
                monitor_data = self.monitor.to_dict()
                plot_results = generate_all_plots(
                    monitor_data=monitor_data,
                    output_dir=self._task_dir,
                    predictions=last_pred_sample,
                    targets=last_target_sample,
                    best_loss=best_val_loss if best_val_loss != float("inf") else 0,
                    best_epoch=current_epoch,
                    total_epochs=self.epochs,
                    elapsed=self.monitor.get_elapsed_time(),
                    device=str(self.device),
                )
                if self.logger:
                    self.logger.info(f"效果图已生成: {len(plot_results)} 个文件 → {self._task_dir}/plots/")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"效果图生成失败: {e}")

            # 更新数据库
            try:
                self.history_storage.update_run_status(
                    run_id, status,
                    end_time=None,
                    total_epochs=current_epoch,
                    best_loss=best_val_loss if best_val_loss != float("inf") else None,
                    best_accuracy=final_metrics.get("best_accuracy"),
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"History update failed (ignored): {e}")

            self.logger.info(f"训练结束: status={status}")

        return {**final_metrics, "status": status, "run_id": run_id,
                "log_dir": str(self._task_dir)}

    def pause(self):
        """暂停训练。"""
        self._paused = True
        if self.logger:
            self.logger.info("训练已暂停")

    def resume(self):
        """恢复训练。"""
        self._paused = False
        if self.logger:
            self.logger.info("训练已恢复")

    def stop(self):
        """请求终止训练。"""
        self._stop_requested = True
        if self.logger:
            self.logger.info("训练终止请求已发出")

    def _save_epoch_plot(self, epoch: int, pred: np.ndarray, target: np.ndarray):
        """Save vorticity comparison plot for a single epoch to the plots dir."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = Path(self._task_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        T = min(pred.shape[0], target.shape[0])
        mid_l = pred.shape[3] // 2
        t_idx = [max(0, T // 6), T // 3, T // 2, T - 1]
        t_lbl = [f"t+{(ti+1)*5}min" for ti in t_idx]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Epoch {epoch} — Pred vs True Vorticity", fontsize=13, fontweight="bold")

        for i, (ti, tlbl) in enumerate(zip(t_idx, t_lbl)):
            im = axes[0, i].imshow(target[ti, :, :, mid_l], cmap="RdBu_r",
                                   origin="lower", aspect="auto")
            axes[0, i].set_title(f"TRUE  {tlbl}")
            plt.colorbar(im, ax=axes[0, i], fraction=0.046)

        for i, (ti, tlbl) in enumerate(zip(t_idx, t_lbl)):
            im = axes[1, i].imshow(pred[ti, :, :, mid_l], cmap="RdBu_r",
                                   origin="lower", aspect="auto")
            axes[1, i].set_title(f"PRED  {tlbl}")
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)

        fig.tight_layout()
        path = str(plots_dir / f"epoch_{epoch:03d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused


def run_training_cli(config: dict, verbose: bool = False):
    """纯 CLI 模式运行训练（不使用 TUI），输出详细诊断日志。

    输出分为四个层次:
      1. 启动诊断 — 配置 / 数据统计 / 模型参数 / 设备信息
      2. 每 batch — 进度条 + 关键指标; verbose 模式额外输出完整指标行
      3. 每 epoch — train + val 完整对比, 含趋势预警
      4. 训练结束 — 结构化诊断报告 (可供 AI 分析)

    Args:
        config: 训练配置字典
        verbose: True 时每 50 batch 输出完整指标行
    """
    import os
    import shutil
    import time as time_module
    from ..data.dataset import scan_processed_files, create_dataloaders
    from ..data.split import merge_datasets

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    loss_cfg = model_cfg.get("loss", {})
    physics_cfg = model_cfg.get("physics", {})

    processed_dir = data_cfg.get("processed_dir", "./processed")

    # ── 启动: 扫描数据 ────────────────────────────────────────
    files = scan_processed_files(processed_dir)
    if not files:
        print("=" * 72)
        print("ERROR: No preprocessed .npz files found")
        print(f"  Expected: {os.path.abspath(processed_dir)}")
        print("  Run preprocessing in TUI first, or use --dataset1/--dataset2")
        print("=" * 72)
        return None

    train_files, val_files, test_files = merge_datasets(files, [])
    train_loader, val_loader, _ = create_dataloaders(
        train_files, val_files, test_files, config,
    )

    # ── 启动: 数据统计 ────────────────────────────────────────
    print()
    print("=" * 72)
    print("  TRAINING STARTUP DIAGNOSTICS")
    print("=" * 72)

    # 分析一个样本的形状
    total_batches = len(train_loader)
    total_epochs = training_cfg.get("epochs", 100)
    batch_size = training_cfg.get("batch_size", 2)

    sample_shape = "N/A"
    sample_target_range = "N/A"
    try:
        sample_batch = next(iter(train_loader))
        inp = sample_batch["input"]
        tgt = sample_batch["target"]
        sample_shape = f"input={list(inp.shape)} target={list(tgt.shape)}"
        sample_target_range = (f"min={tgt.min().item():.6f} max={tgt.max().item():.6f} "
                               f"mean={tgt.mean().item():.6f} std={tgt.std().item():.6f}")
    except Exception:
        pass

    print(f"  DATA:")
    print(f"    samples: total={len(files)} train={len(train_files)} "
          f"val={len(val_files)} test={len(test_files)}")
    print(f"    shape:   {sample_shape}")
    print(f"    target:  {sample_target_range}")
    print(f"    batches/epoch: {total_batches}  (batch_size={batch_size})")

    # ── 启动: 模型架构 ────────────────────────────────────────
    trainer = Trainer(config)
    device_str = str(trainer.device)

    # 统计参数量
    def _count_params(module, name):
        n = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return n, trainable

    total_params = 0
    total_trainable = 0
    components = [
        ("Encoder", trainer.encoder),
        ("Decoder", trainer.decoder),
        ("PhysicsModule", trainer.physics),
        ("RK2Integrator", trainer.integrator),
        ("ZetaProj", trainer.zeta_proj),
        ("VelocityProj", trainer.velocity_proj),
        ("OmegaProj", trainer.omega_proj),
    ]
    print(f"  MODEL:")
    for name, mod in components:
        n, t = _count_params(mod, name)
        total_params += n
        total_trainable += t
        print(f"    {name:16s} params={n:>10,}  trainable={t:>10,}")
    print(f"    {'TOTAL':16s} params={total_params:>10,}  trainable={total_trainable:>10,}")

    # ── 启动: 超参数 ──────────────────────────────────────────
    print(f"  HYPERPARAMS:")
    print(f"    epochs={total_epochs}  batch_size={batch_size}  "
          f"lr={training_cfg.get('learning_rate', 0.001)}")
    print(f"    weight_decay={training_cfg.get('weight_decay', 0.0001)}  "
          f"lr_scheduler={training_cfg.get('lr_scheduler', 'cosine')}")
    print(f"    AMP={training_cfg.get('use_amp', True)}  "
          f"early_stop_patience={training_cfg.get('early_stop_patience', 15)}")
    print(f"    num_workers(DataLoader)={training_cfg.get('num_workers', 0)}  "
          f"val_every={training_cfg.get('val_every', 1)}")
    print(f"  MODEL CONFIG:")
    print(f"    depth={model_cfg.get('depth', 2)}  "
          f"base_channels={model_cfg.get('base_channels', 64)}  "
          f"dropout={model_cfg.get('dropout', 0.3)}")
    print(f"    decoder_chunk_size={model_cfg.get('decoder_chunk_size', 4)}")
    print(f"    physics: diffusion={physics_cfg.get('diffusion_coef', 0.1)}  "
          f"baroclinic={physics_cfg.get('use_baroclinic', False)}")
    print(f"  LOSS:")
    print(f"    lambda_grad={loss_cfg.get('lambda_grad', 0.1)}  "
          f"lambda_continuity={loss_cfg.get('lambda_continuity', 0.0)}  "
          f"lambda_focal={loss_cfg.get('lambda_focal', 5.0)}")

    # ── 启动: 设备 ────────────────────────────────────────────
    print(f"  DEVICE: {device_str}", end="")
    if trainer.device.type == "cuda":
        props = torch.cuda.get_device_properties(trainer.device)
        vram_gb = props.total_memory / (1024**3)
        print(f"  ({props.name}, {vram_gb:.1f} GB VRAM)")
    else:
        print("  (CPU — training will be slow)")

    print("=" * 72)
    print()

    # ── 运行时状态追踪 ────────────────────────────────────────
    term_w = max(80, shutil.get_terminal_size().columns)
    bar_w = max(20, min(30, term_w - 70))
    verbose_every = 50  # verbose 模式下每 N batch 输出详细行

    # 追踪历史（用于最终诊断报告）
    train_history: list[dict] = []
    val_history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics: dict = {}
    epoch_start = time_module.time()
    train_start = epoch_start
    nan_skip_total = 0
    prev_train_loss = None

    def cli_callback(metrics: dict):
        nonlocal best_val_loss, best_epoch, best_val_metrics, epoch_start
        nonlocal nan_skip_total, prev_train_loss

        typ = metrics.get("type", "")

        if typ == "status":
            msg = metrics.get("message", "")
            if msg:
                total_b = metrics.get("total_batches", 0)
                total_e = metrics.get("total_epochs", 0)
                extra = f" ({total_e} epochs, {total_b} batches/epoch)" if total_b else ""
                print(f"[INFO] {msg}{extra}")
            return

        if typ == "batch":
            epoch = metrics.get("epoch", 0)
            batch = metrics.get("batch", 0)
            total_b = metrics.get("total_batches", total_batches)
            loss = metrics.get("loss", 0)
            acc = metrics.get("accuracy", 0)
            total_loss = metrics.get("total_loss", 0)
            grad_loss = metrics.get("grad_loss", 0)
            current_lr = metrics.get("lr", 0)

            elapsed = time_module.time() - epoch_start
            pct = (batch / total_b * 100) if total_b else 0

            filled = int(bar_w * batch / total_b) if total_b else 0
            bar = "█" * filled + "░" * (bar_w - filled)

            # 原地刷新进度条
            line = (f"\rEpoch {epoch:3d}/{total_epochs} |{bar}| "
                    f"{batch:4d}/{total_b:4d} ({pct:5.1f}%) | "
                    f"Loss={loss:.4f} Acc={acc:.4f} "
                    f"| Total={total_loss:.3f} Grad={grad_loss:.4f} "
                    f"LR={current_lr:.1e} | {elapsed:.0f}s ")
            if len(line) > term_w:
                line = line[:term_w - 1]
            print(line, end="", flush=True)

            # verbose: 详细行
            if verbose and (batch % verbose_every == 0 or batch == 1):
                print()  # 结束进度行
                _print_verbose_batch(metrics, term_w)
                # 重新开始进度行 (下一 batch 会刷新掉)

        elif typ == "epoch":
            epoch = metrics.get("epoch", 0)
            phase = metrics.get("phase", "train")
            loss_val = metrics.get("loss", 0)
            acc = metrics.get("accuracy", 0)

            elapsed = time_module.time() - epoch_start
            epoch_start = time_module.time()

            print()  # 结束进度行

            if phase == "train":
                train_history.append(metrics)
                prev_train_loss = loss_val

                pred_std = metrics.get("pred_std", 0)
                target_std = metrics.get("target_std", 0)
                pos_ratio = metrics.get("pos_ratio", 0)
                zbl_acc = metrics.get("zero_baseline_acc", 0)

                _print_epoch_row("TRAIN", epoch, total_epochs, metrics, elapsed, "")

                # 原始数据分布 (不附加诊断判断)
                print(f"        pred_std={pred_std:.6f} target_std={target_std:.6f} "
                      f"pos_ratio={pos_ratio:.4f} zero_baseline_acc={zbl_acc:.4f}")

            elif phase == "val":
                val_history.append(metrics)

                is_best = False
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_epoch = epoch
                    best_val_metrics = dict(metrics)
                    is_best = True

                _print_epoch_row("VAL", epoch, total_epochs, metrics, elapsed,
                                 " ★ NEW BEST" if is_best else "")

                # Train/Val gap (仅输出差值，不做判断)
                if train_history:
                    last_train = train_history[-1]
                    train_loss = last_train.get("loss", 0)
                    gap = loss_val - train_loss
                    print(f"        Train→Val gap: {gap:+.4f}")

    trainer.on_metric_update(cli_callback)
    result = trainer.train(train_loader, val_loader)

    # ── 训练结束: 诊断报告 ────────────────────────────────────
    _print_diagnostic_report(
        result=result,
        config=config,
        train_history=train_history,
        val_history=val_history,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        total_params=total_params,
        total_trainable=total_trainable,
        train_start=train_start,
        total_epochs=total_epochs,
    )

    return result


# ── CLI 辅助输出函数 ────────────────────────────────────────────

def _print_epoch_row(tag: str, epoch: int, total: int, m: dict, elapsed: float,
                     suffix: str = ""):
    """打印 epoch 汇总行。"""
    f1 = m.get("f1")
    dice = m.get("dice")
    iou = m.get("iou")
    prec = m.get("precision")
    rec = m.get("recall")
    extra = ""
    if f1 is not None:
        extra += f" F1={f1:.3f}"
    if dice is not None:
        extra += f" Dice={dice:.3f}"
    if iou is not None:
        extra += f" IoU={iou:.3f}"
    if prec is not None:
        extra += f" Prec={prec:.3f}"
    if rec is not None:
        extra += f" Rec={rec:.3f}"

    print(f"  [{tag:5s}] Epoch {epoch:3d}/{total:3d} | "
          f"Loss={m.get('loss', 0):.4f} Acc={m.get('accuracy', 0):.4f}{extra} | "
          f"{elapsed:.1f}s{suffix}")


def _print_verbose_batch(m: dict, term_w: int):
    """verbose 模式: 每 N batch 输出完整指标行。"""
    loss = m.get("loss", 0)
    acc = m.get("accuracy", 0)
    f1 = m.get("f1", 0)
    dice = m.get("dice", 0)
    iou = m.get("iou", 0)
    pos = m.get("pos_ratio", 0)
    zbl = m.get("zero_baseline_acc", 0)
    pred_std = m.get("pred_std", 0)
    target_std = m.get("target_std", 0)
    grad = m.get("grad_loss", 0)
    total_l = m.get("total_loss", 0)
    lr = m.get("lr", 0)
    epoch = m.get("epoch", 0)
    batch = m.get("batch", 0)
    total_b = m.get("total_batches", 1)
    csi_l = m.get("csi_loss", 0)
    fcsi_l = m.get("fcsi_loss", 0)
    lpips_l = m.get("lpips_loss", 0)
    auc_l = m.get("auc_loss", 0)

    std_ratio = pred_std / (target_std + 1e-8)

    line = (f"    [DETAIL] E{epoch:3d} B{batch:4d}/{total_b:4d} | "
            f"Loss={loss:.4f} Total={total_l:.3f} Grad={grad:.4f} LR={lr:.1e} | "
            f"Acc={acc:.4f} zbl={zbl:.4f} "
            f"F1={f1:.3f} Dice={dice:.3f} IoU={iou:.3f} | "
            f"CSI_L={csi_l:.4f} FSS_L={fcsi_l:.4f} LPIPS_L={lpips_l:.4f} AUC_L={auc_l:.4f} | "
            f"Pos%={pos:.4f} pred_std={pred_std:.4f} tgt_std={target_std:.4f} "
            f"ratio={std_ratio:.3f}")
    if len(line) > term_w - 1:
        line = line[:term_w - 1]
    print(line)


def _print_diagnostic_report(
    result: dict, config: dict,
    train_history: list[dict], val_history: list[dict],
    best_val_loss: float, best_epoch: int, best_val_metrics: dict,
    total_params: int, total_trainable: int,
    train_start: float, total_epochs: int,
):
    """打印结构化诊断报告，适合 AI 分析。"""
    total_elapsed = time.time() - train_start
    status = result.get("status", "unknown")

    print()
    print("=" * 72)
    print("  TRAINING DIAGNOSTIC REPORT")
    print("=" * 72)

    # ── 基本信息 ──
    log_dir = result.get("log_dir", "N/A")
    h, rem = divmod(total_elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"  STATUS:           {status}")
    print(f"  DURATION:         {total_elapsed:.1f}s ({int(h)}h {int(m)}m {int(s)}s)")
    print(f"  LOG_DIR:          {log_dir}")
    print(f"  TOTAL_PARAMS:     {total_params:,} (trainable: {total_trainable:,})")

    # ── 最佳模型 ──
    print(f"  ---")
    print(f"  BEST_MODEL:")
    print(f"    epoch:           {best_epoch}")
    if best_val_loss != float("inf"):
        print(f"    val_loss:        {best_val_loss:.6f}")
    if best_val_metrics:
        for k in ("accuracy", "f1", "dice", "iou", "precision", "recall",
                   "pred_std", "target_std", "pos_ratio", "zero_baseline_acc"):
            v = best_val_metrics.get(k)
            if v is not None:
                print(f"    {k:18s} {v:.6f}")

    # ── 训练曲线摘要 ──
    print(f"  ---")
    print(f"  TRAINING_CURVE:")
    if train_history:
        first = train_history[0]
        last = train_history[-1]
        print(f"    train_loss:      {first.get('loss', 0):.6f} → {last.get('loss', 0):.6f}")
        print(f"    train_acc:       {first.get('accuracy', 0):.4f} → {last.get('accuracy', 0):.4f}")
        print(f"    train_f1:        {first.get('f1', 0):.4f} → {last.get('f1', 0):.4f}")
    else:
        print(f"    (no training history)")

    if val_history:
        first_v = val_history[0]
        last_v = val_history[-1]
        print(f"    val_loss:        {first_v.get('loss', 0):.6f} → {last_v.get('loss', 0):.6f}")
        print(f"    val_acc:         {first_v.get('accuracy', 0):.4f} → {last_v.get('accuracy', 0):.4f}")
        print(f"    val_f1:          {first_v.get('f1', 0):.4f} → {last_v.get('f1', 0):.4f}")
    else:
        print(f"    (no validation history)")

    # ── 训练曲线统计 (仅原始数据) ──
    print(f"  ---")
    print(f"  CURVE_STATS:")
    if train_history:
        losses = [h.get("loss", 0) for h in train_history]
        accs = [h.get("accuracy", 0) for h in train_history]
        print(f"    train_loss_range:    [{min(losses):.6f}, {max(losses):.6f}]")
        print(f"    train_acc_range:     [{min(accs):.4f}, {max(accs):.4f}]")
        if len(losses) >= 3:
            print(f"    train_loss_last3trend: {losses[-1] - losses[-3]:+.6f}")
    if val_history:
        vlosses = [h.get("loss", 0) for h in val_history]
        vaccs = [h.get("accuracy", 0) for h in val_history]
        print(f"    val_loss_range:      [{min(vlosses):.6f}, {max(vlosses):.6f}]")
        print(f"    val_acc_range:       [{min(vaccs):.4f}, {max(vaccs):.4f}]")
        if len(vlosses) >= 3:
            print(f"    val_loss_last3trend:   {vlosses[-1] - vlosses[-3]:+.6f}")
    if train_history and val_history:
        gaps = [
            vh.get("loss", 0) - th.get("loss", 0)
            for th, vh in zip(train_history, val_history)
        ]
        if gaps:
            print(f"    train_val_gap_range:  [{min(gaps):+.6f}, {max(gaps):+.6f}]")

    if best_val_metrics:
        pred_s = best_val_metrics.get("pred_std")
        tgt_s = best_val_metrics.get("target_std")
        if pred_s is not None and tgt_s is not None and tgt_s > 0:
            print(f"    best_pred_std/target_std: {pred_s / tgt_s:.5f}")

    if train_history and val_history:
        avg_grad = sum(h.get("grad_loss", 0) for h in train_history) / len(train_history)
        print(f"    avg_grad_loss:       {avg_grad:.6f}")

    if best_epoch < len(train_history):
        patience = config.get("training", {}).get("early_stop_patience", 15)
        print(f"    best_epoch:          {best_epoch}")
        print(f"    total_epochs_run:    {len(train_history)}")
        print(f"    early_stop_patience:  {patience}")

    # ── 配置存根 (供 AI 复现) ──
    print(f"  ---")
    print(f"  CONFIG_STUB_FOR_REPRODUCTION:")
    tc = config.get("training", {})
    mc = config.get("model", {})
    dc = config.get("data", {})
    lc = mc.get("loss", {})
    print(f"    batch_size={tc.get('batch_size')} epochs={tc.get('epochs')} "
          f"lr={tc.get('learning_rate')} wd={tc.get('weight_decay')} "
          f"lr_sched={tc.get('lr_scheduler')} amp={tc.get('use_amp')}")
    print(f"    depth={mc.get('depth')} base_ch={mc.get('base_channels')} "
          f"dropout={mc.get('dropout')} dec_chunk={mc.get('decoder_chunk_size')}")
    print(f"    λ_grad={lc.get('lambda_grad')} λ_cont={lc.get('lambda_continuity')} "
          f"λ_focal={lc.get('lambda_focal')}")
    print(f"    hist_steps={dc.get('history_steps')} future_steps={dc.get('future_steps')} "
          f"grid={dc.get('grid_size')}")
    print("=" * 72)

"""核心训练引擎 — 支持 DDP 多卡并行。

包含：
- TornadoModel: 将所有子模块打包为一个 Module，供 DDP 包装
- Trainer: 完整的训练/验证/早停/检查点管理
- run_training_cli: 命令行启动入口，自动处理 DistributedSampler
"""
import os
import time
import signal
from pathlib import Path
from typing import Any, Callable
from collections.abc import Callable as CallableType

import torch
import torch.distributed as dist
import numpy as np
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from ..utils.device import get_device      # 保留但不再用于分布式，仅用于非 DDP 场景
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


class TornadoModel(nn.Module):
    """将所有子模块打包为一个 Module，便于 DDP 包装。"""

    def __init__(self, encoder, decoder, physics, integrator,
                 zeta_proj, velocity_proj, omega_proj,
                 T_out, zeta_scale):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.physics = physics
        self.integrator = integrator
        self.zeta_proj = zeta_proj
        self.velocity_proj = velocity_proj
        self.omega_proj = omega_proj
        self.T_out = T_out
        self.zeta_scale = zeta_scale

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = batch["input"]                      # (B, T_in, H, W, L, C)
        target = batch["target"]                # (B, T_out, H, W, L)
        B, T_out = x.shape[0], target.shape[1]

        # 编码 → 瓶颈特征 + skip connections
        features, skips = self.encoder(x)

        # 投影得到各物理场
        zeta_init = self.zeta_proj(features).squeeze(1) * self.zeta_scale

        # 时间依赖的速度场
        vel_raw = self.velocity_proj(features)
        vel_raw = vel_raw.reshape(B, T_out, 3, *vel_raw.shape[-3:])
        vel_u = vel_raw[:, :, 0:1, :, :, :] * 20.0
        vel_v = vel_raw[:, :, 1:2, :, :, :] * 20.0
        vel_w = vel_raw[:, :, 2:3, :, :, :] * 5.0
        velocities_seq = torch.cat([vel_u, vel_v, vel_w], dim=2)

        # 水平涡度分量
        omega_raw = self.omega_proj(features) * 0.1
        omega_raw = omega_raw.reshape(B, T_out, 2, *omega_raw.shape[-3:])
        omega_x_seq = omega_raw[:, :, 0, :, :, :]
        omega_y_seq = omega_raw[:, :, 1, :, :, :]
        vv_seq = velocities_seq[:, :, 2, :, :, :]   # 垂直速度

        # RK2 积分预测未来涡度
        zeta_pred_lowres = self.integrator(
            zeta_init, velocities_seq,
            omega_x=omega_x_seq, omega_y=omega_y_seq,
            vertical_velocities=vv_seq,
        )  # (B, T_out, H', W', L')

        # 解码器 → 全分辨率
        predictions = self.decoder(zeta_pred_lowres, skips, T_out)

        projected = {
            "velocity": velocities_seq,
            "zeta_init": zeta_init,
            "omega_x": omega_x_seq[:, 0, :, :, :],
            "omega_y": omega_y_seq[:, 0, :, :, :],
        }
        return predictions, target, projected


class Trainer:
    """龙卷风涡度预测模型训练器（DDP 版）。"""

    def __init__(self, config: dict):
        self.config = config
        training_cfg = config.get("training", {})
        device_cfg = config.get("device", {})
        model_cfg = config.get("model", {})
        loss_cfg = model_cfg.get("loss", {})

        # ── 分布式初始化 ──
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend="nccl")   # ROCm 支持 nccl
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        torch.backends.cudnn.benchmark = True

        # 超参数（学习率线性缩放）
        self.epochs = training_cfg.get("epochs", 100)
        self.batch_size = training_cfg.get("batch_size", 2)
        self.learning_rate = training_cfg.get("learning_rate", 0.001) * self.world_size
        self.weight_decay = training_cfg.get("weight_decay", 0.0001)
        self.use_amp = training_cfg.get("use_amp", True)
        self.early_stop_patience = training_cfg.get("early_stop_patience", 15)

        # ── 构建子模块（移到当前设备）──
        self.encoder = ResUNet3DEncoder(config).to(self.device)
        self.physics = PhysicsModule(config).to(self.device)
        self.integrator = RK2Integrator(self.physics, dt=300.0).to(self.device)

        # 解码器
        base_ch = model_cfg.get("base_channels", 64)
        depth = model_cfg.get("depth", 2)
        skip_channels = [base_ch]
        ch = base_ch
        for i in range(depth - 1):
            ch = min(ch * 2, 1024)
            skip_channels.append(ch)
        self.decoder = ResUNet3DDecoder(
            config, self.encoder.output_channels, skip_channels,
        ).to(self.device)

        # 损失函数（支持新参数）
        self.criterion = PhysicsLoss(
            lambda_grad=loss_cfg.get("lambda_grad", 0.1),
            lambda_continuity=loss_cfg.get("lambda_continuity", 0.0),
            lambda_focal=loss_cfg.get("lambda_focal", 5.0),
            lambda_csi=loss_cfg.get("lambda_csi", 1.0),
            lambda_fcsi=loss_cfg.get("lambda_fcsi", 0.5),
            lambda_lpips=loss_cfg.get("lambda_lpips", 0.1),
            lambda_auc=loss_cfg.get("lambda_auc", 0.05),
            lpips_device=str(self.device),
            lpips_interval=loss_cfg.get("lpips_interval", 5),
            max_auc_pairs=loss_cfg.get("max_auc_pairs", 10000),
        )

        # 投影头及参数
        self._build_projection_heads()

        # ── 包装为 TornadoModel 并用 DDP ──
        self.model = TornadoModel(
            self.encoder, self.decoder, self.physics, self.integrator,
            self.zeta_proj, self.velocity_proj, self.omega_proj,
            self.T_out, self.zeta_scale
        ).to(self.device)

        if self.device.type == "cuda":
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             find_unused_parameters=True)

        # 优化器（参数来自原始子模块，它们与 DDP 模型共享内存）
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

        # 管理与监控（只在 rank 0 创建实际对象）
        self.is_rank0 = (self.local_rank == 0)
        self.monitor = TrainingMonitor() if self.is_rank0 else None
        if self.is_rank0:
            self.checkpoint_mgr = CheckpointManager(
                training_cfg.get("checkpoint_dir", "./checkpoints"),
                training_cfg.get("save_best_only", True),
            )
            log_cfg = config.get("logging", {})
            self.log_manager = LogManager(log_cfg.get("log_dir", "./logs"))
            self.history_storage = HistoryStorage()
        else:
            self.checkpoint_mgr = None
            self.log_manager = None
            self.history_storage = None

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
        enc_out_ch = self.encoder.output_channels
        data_cfg = self.config.get("data", {})
        self.T_out = data_cfg.get("future_steps", 36)
        self.grid_size = data_cfg.get("grid_size", 128)
        self.vert_layers = data_cfg.get("vertical_layers", None) or 20

        self.zeta_scale = 0.03
        self.zeta_proj = nn.Sequential(
            nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            nn.BatchNorm3d(enc_out_ch // 2),
            nn.ReLU(),
            nn.Conv3d(enc_out_ch // 2, 1, 3, padding=1),
            nn.Tanh(),
        ).to(self.device)

        self.velocity_proj = nn.Sequential(
            nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            nn.BatchNorm3d(enc_out_ch // 2),
            nn.ReLU(),
            nn.Conv3d(enc_out_ch // 2, self.T_out * 3, 3, padding=1),
            nn.Tanh(),
        ).to(self.device)

        self.omega_proj = nn.Sequential(
            nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            nn.BatchNorm3d(enc_out_ch // 2),
            nn.ReLU(),
            nn.Conv3d(enc_out_ch // 2, self.T_out * 2, 3, padding=1),
            nn.Tanh(),
        ).to(self.device)

    def parameters(self):
        seen = set()
        for m in [self.encoder, self.decoder, self.integrator,
                   self.zeta_proj, self.velocity_proj, self.omega_proj]:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    # ── 加载预训练权重 ──
    def load_pretrained(self, ckpt_path: str, load_optimizer: bool = False,
                         strict: bool = True) -> dict:
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
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
            self._load_model_state(checkpoint, strict)
            info = {
                "source": Path(ckpt_path).name,
                "epoch": None,
                "metrics": None,
                "loaded_optimizer": False,
            }
        return info

    def _load_model_state(self, state_dict: dict, strict: bool):
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
        self._callbacks.append(callback)

    def _notify_callbacks(self, metrics: dict):
        if not self.is_rank0:
            return
        for cb in self._callbacks:
            try:
                cb(metrics)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Callback failed: {e}")

    # ── 训练/验证循环调用 self.model ──
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        self.model.train()

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
                    pred, target, proj = self.model(batch)
                    loss, components = self.criterion(pred, target, proj.get("velocity"))
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred, target, proj = self.model(batch)
                loss, components = self.criterion(pred, target, proj.get("velocity"))
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 指标计算（只在 rank 0 详细记录）
            bin_metrics = compute_binary_metrics(pred, target)
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

            if self.is_rank0:
                self.monitor.update("train", metrics, epoch, batch_idx)
            epoch_loss += loss.item()
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k].append(metrics[k])

            if self.is_rank0 and (batch_idx % 10 == 0 or batch_idx == 0):
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
                        f"Dice: {metrics['dice']:.3f}  IoU: {metrics['iou']:.3f}  "
                        f"Acc: {metrics['accuracy']:.3f}(BL:{metrics['zero_baseline_acc']:.2f})"
                    )

        return {k: sum(v) / len(v) if v else 0 for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int,
                 return_sample: bool = False) -> dict | tuple[dict, np.ndarray | None, np.ndarray | None]:
        self.model.eval()

        epoch_metrics_val = {"loss": [], "accuracy": [], "precision": [], "recall": [],
                             "f1": [], "dice": [], "iou": [], "pos_ratio": [],
                             "zero_baseline_acc": [], "pred_std": [], "target_std": [],
                             "csi_loss": [], "fcsi_loss": [], "lpips_loss": [], "auc_loss": []}
        pred_sample = None
        target_sample = None

        for batch_idx, batch in enumerate(val_loader):
            pred, target, proj = self.model(batch)
            loss, components = self.criterion(pred, target, proj.get("velocity"))
            bin_metrics = compute_binary_metrics(pred, target)

            if self.is_rank0 and return_sample and pred_sample is None and batch_idx == 0:
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
            if self.is_rank0:
                self.monitor.update("val", metrics, epoch, batch_idx)
            for k in epoch_metrics_val:
                if k in metrics:
                    epoch_metrics_val[k].append(metrics[k])

        summary = {k: sum(v) / len(v) if v else 0 for k, v in epoch_metrics_val.items()}
        if self.is_rank0 and self.logger:
            self.logger.info(
                f"Epoch {epoch}  Valid  "
                f"Loss: {summary['loss']:.4f}  "
                f"Dice: {summary.get('dice', 0):.3f}  "
                f"IoU: {summary.get('iou', 0):.3f}  "
                f"F1: {summary.get('f1', 0):.3f}"
            )

        if return_sample:
            return summary, pred_sample, target_sample
        return summary

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None,
              task_name: str = "training") -> dict:
        if self.is_rank0:
            log_cfg = self.config.get("logging", {})
            self._task_dir = self.log_manager.create_task_dir(task_name)
            self.logger = setup_logger(
                "trainer", str(self.log_manager._log_dir), task_name,
                level=log_cfg.get("log_level", "INFO"),
            )
            self.log_manager.save_config(self._task_dir, self.config)
            self.logger.info(f"开始训练: {task_name}")
            self.logger.info(f"设备: {self.device}, 世界大小: {self.world_size}")
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

            # 数据库记录
            data_cfg = self.config.get("data", {})
            run_id = self.history_storage.create_run(
                task_name=task_name,
                config=self.config,
                device=str(self.device),
                dataset1=data_cfg.get("dataset1_path", ""),
                dataset2=data_cfg.get("dataset2_path", ""),
                log_dir=str(self._task_dir),
            )
        else:
            run_id = -1  # 非 rank0 占位

        self._running = True
        if self.is_rank0:
            self.monitor.reset()
        best_val_loss = float("inf")
        patience_counter = 0
        final_metrics = {}
        last_pred_sample = None
        last_target_sample = None

        if self.is_rank0:
            self._notify_callbacks({
                "type": "status",
                "message": f"Training started on {self.device} — first batch may take 1-3 min (CUDA JIT)",
                "total_batches": len(train_loader),
                "total_epochs": self.epochs,
            })

        # 信号处理（仅主线程）
        original_sigint = None

        def handle_interrupt(signum, frame):
            self._stop_requested = True
            self._interrupted = True
            if self.is_rank0 and self.logger:
                self.logger.warning("收到中断信号，正在优雅退出...")

        try:
            original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handle_interrupt)
        except (ValueError, OSError):
            pass

        try:
            for epoch in range(1, self.epochs + 1):
                if self._stop_requested:
                    break

                # 设置 epoch（对 DistributedSampler）
                if self.world_size > 1 and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

                train_metrics = self.train_epoch(train_loader, epoch)

                if self.is_rank0:
                    self.logger.info(
                        f"Epoch {epoch}  Train Avg  "
                        f"Loss: {train_metrics['loss']:.4f}  "
                        f"Dice: {train_metrics.get('dice', 0):.3f}  "
                        f"IoU: {train_metrics.get('iou', 0):.3f}  "
                        f"F1: {train_metrics.get('f1', 0):.3f}"
                    )
                    self._notify_callbacks({
                        "type": "epoch",
                        "epoch": epoch,
                        "total_epochs": self.epochs,
                        "phase": "train",
                        **train_metrics,
                    })
                    self.history_storage.save_metric(run_id, epoch, "train", **train_metrics)
                    self.history_storage.update_run_status(run_id, "running", current_epoch=epoch)

                # 验证（只在主进程做，或者所有进程都做但只在 rank0 聚合）
                if val_loader is not None and epoch % self.config["training"].get("val_every", 1) == 0:
                    val_metrics, pred_sample, target_sample = self.validate(
                        val_loader, epoch, return_sample=True,
                    )
                    if self.is_rank0:
                        if pred_sample is not None:
                            last_pred_sample = pred_sample
                            last_target_sample = target_sample
                        self.history_storage.save_metric(run_id, epoch, "val", **val_metrics)
                        self._notify_callbacks({
                            "type": "epoch",
                            "epoch": epoch,
                            "total_epochs": self.epochs,
                            "phase": "val",
                            **val_metrics,
                        })
                        # 绘图
                        if last_pred_sample is not None and last_target_sample is not None:
                            try:
                                self._save_epoch_plot(epoch, last_pred_sample, last_target_sample)
                            except Exception as e:
                                if self.logger:
                                    self.logger.warning(f"Epoch plot failed: {e}")
                        # 早停 / 最佳模型保存
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
                    if val_loader is not None and self.is_rank0:
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step(train_metrics["loss"])
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step()

            final_metrics = {
                "best_val_loss": best_val_loss if best_val_loss != float("inf") else (train_metrics["loss"] if 'train_metrics' in dir() else 0),
            }
            status = "completed" if not self._interrupted else "interrupted"

        except Exception as e:
            if self.is_rank0 and self.logger:
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

            if self.is_rank0:
                current_epoch = epoch if 'epoch' in dir() else 0
                # 最终 checkpoint
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

                # 保存 CSV
                try:
                    self.log_manager.save_metrics_csv(
                        self._task_dir,
                        {
                            "epoch": [r["epoch"] for r in self.monitor._history.get("train", [])],
                            "phase": [r["phase"] for r in self.monitor._history.get("train", [])],
                            "loss": [r.get("loss", 0) for r in self.monitor._history.get("train", [])],
                        },
                    )
                    self.log_manager.link_checkpoint(self._task_dir, str(self.checkpoint_mgr._dir))
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Metrics save failed (ignored): {e}")

                # 生成效果图
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

                if self.logger:
                    self.logger.info(f"训练结束: status={status}")

            if self.world_size > 1:
                dist.destroy_process_group()

        return {**final_metrics, "status": status, "run_id": run_id if self.is_rank0 else -1,
                "log_dir": str(self._task_dir) if self.is_rank0 else ""}

    def pause(self):
        self._paused = True
        if self.is_rank0 and self.logger:
            self.logger.info("训练已暂停")

    def resume(self):
        self._paused = False
        if self.is_rank0 and self.logger:
            self.logger.info("训练已恢复")

    def stop(self):
        self._stop_requested = True
        if self.is_rank0 and self.logger:
            self.logger.info("训练终止请求已发出")

    def _save_epoch_plot(self, epoch: int, pred: np.ndarray, target: np.ndarray):
        if not self.is_rank0:
            return
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
            im = axes[0, i].imshow(target[ti, :, :, mid_l], cmap="RdBu_r", origin="lower", aspect="auto")
            axes[0, i].set_title(f"TRUE  {tlbl}")
            plt.colorbar(im, ax=axes[0, i], fraction=0.046)
        for i, (ti, tlbl) in enumerate(zip(t_idx, t_lbl)):
            im = axes[1, i].imshow(pred[ti, :, :, mid_l], cmap="RdBu_r", origin="lower", aspect="auto")
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
    """纯 CLI 模式运行训练（DDP 版），用于命令行直接启动。"""
    import os
    import shutil
    import time as time_module
    from ..data.dataset import scan_processed_files, create_dataloaders
    from ..data.split import merge_datasets

    # 分布式环境由 torchrun 设置，local_rank 已在环境变量中
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    processed_dir = data_cfg.get("processed_dir", "./processed")

    # 扫描数据
    files = scan_processed_files(processed_dir)
    if not files:
        if local_rank == 0:
            print("ERROR: No preprocessed .npz files found")
        return None

    train_files, val_files, test_files = merge_datasets(files, [])
    train_loader, val_loader, _ = create_dataloaders(
        train_files, val_files, test_files, config,
    )

    # 只在 rank 0 输出启动诊断
    if local_rank == 0:
        print("=" * 72)
        print("  TRAINING STARTUP DIAGNOSTICS")
        print("=" * 72)
        total_batches = len(train_loader)
        batch_size = training_cfg.get("batch_size", 2)
        sample_shape = "N/A"
        try:
            sample_batch = next(iter(train_loader))
            inp = sample_batch["input"]
            tgt = sample_batch["target"]
            sample_shape = f"input={list(inp.shape)} target={list(tgt.shape)}"
            sample_target_range = (f"min={tgt.min().item():.6f} max={tgt.max().item():.6f} "
                                   f"mean={tgt.mean().item():.6f} std={tgt.std().item():.6f}")
        except Exception:
            sample_target_range = "N/A"
        print(f"  DATA: samples total={len(files)} train={len(train_files)} val={len(val_files)}")
        print(f"  shape: {sample_shape}")
        print(f"  target: {sample_target_range}")
        print(f"  batches/epoch: {total_batches} (batch_size={batch_size})")

    trainer = Trainer(config)

    if local_rank == 0:
        total_params = sum(p.numel() for p in trainer.parameters())
        print(f"  MODEL total params: {total_params:,}")
        print(f"  DEVICE: {trainer.device} (world_size={trainer.world_size})")
        print("=" * 72)

    # 添加 DDP 数据加载器
    if trainer.world_size > 1:
        train_sampler = DistributedSampler(
            train_loader.dataset, num_replicas=trainer.world_size, rank=local_rank, shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset, batch_size=train_loader.batch_size,
            sampler=train_sampler,
            num_workers=train_loader.num_workers,
            pin_memory=True,
            collate_fn=train_loader.collate_fn,
        )
        if val_loader is not None:
            val_sampler = DistributedSampler(
                val_loader.dataset, num_replicas=trainer.world_size, rank=local_rank, shuffle=False
            )
            val_loader = DataLoader(
                val_loader.dataset, batch_size=val_loader.batch_size,
                sampler=val_sampler,
                num_workers=val_loader.num_workers,
                pin_memory=True,
                collate_fn=val_loader.collate_fn,
            )

    # 回调
    def cli_callback(metrics: dict):
        if local_rank != 0:
            return
        typ = metrics.get("type", "")
        if typ == "status":
            msg = metrics.get("message", "")
            if msg:
                print(f"[INFO] {msg}")
        elif typ == "batch":
            epoch = metrics.get("epoch", 0)
            batch = metrics.get("batch", 0)
            total_b = metrics.get("total_batches", 0)
            loss = metrics.get("loss", 0)
            acc = metrics.get("accuracy", 0)
            pct = (batch / total_b * 100) if total_b else 0
            print(f"\rEpoch {epoch:3d} Batch {batch:4d}/{total_b:4d} ({pct:.1f}%) Loss={loss:.4f} Acc={acc:.4f}", end="", flush=True)
        elif typ == "epoch":
            epoch = metrics.get("epoch", 0)
            phase = metrics.get("phase", "train")
            loss_val = metrics.get("loss", 0)
            acc = metrics.get("accuracy", 0)
            print()
            print(f"  [{phase.upper():5s}] Epoch {epoch} Loss={loss_val:.4f} Acc={acc:.4f}")

    trainer.on_metric_update(cli_callback)
    result = trainer.train(train_loader, val_loader)

    if local_rank == 0:
        print()
        print("=" * 72)
        print(f"  TRAINING FINISHED. Status: {result.get('status')}")
        print(f"  Best val loss: {result.get('best_val_loss', 'N/A')}")
        print(f"  Log dir: {result.get('log_dir', 'N/A')}")
        print("=" * 72)

    return result

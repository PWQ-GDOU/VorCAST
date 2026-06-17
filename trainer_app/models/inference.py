"""模型推理/预测引擎。

加载训练好的 checkpoint，对输入数据进行未来垂直涡度场预测。
使用完整的 Encoder → Projection → RK2 Integrator → Decoder 架构。
"""
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np

from ..utils.device import get_device
from ..models.encoder import ResUNet3DEncoder, ResUNet3DDecoder
from ..models.physics import PhysicsModule
from ..models.integrator import RK2Integrator
from ..models.metrics import compute_all_metrics


class InferenceEngine:
    """龙卷风涡度预测推理引擎。

    使用方式:
        engine = InferenceEngine(config, prediction_steps=18, dt=300)
        engine.load_checkpoint("best_model.pt")
        predictions = engine.predict(input_file="event_001.npz")
        engine.save_predictions(predictions, "output.npz")
    """

    def __init__(self, config: dict, prediction_steps: int | None = None,
                 time_step_seconds: float | None = None, gpu_enabled: bool | None = None,
                 lon_range: tuple[float, float] | None = None,
                 lat_range: tuple[float, float] | None = None,
                 vert_range: tuple[int, int] | None = None):
        self.config = config

        # Spatial window for input cropping (optional bounding box)
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.vert_range = vert_range

        # 推理参数：优先使用显式入参，其次 config 的 inference 节，最后 fallback 到 data 节
        infer_cfg = config.get("inference", {})
        data_cfg = config.get("data", {})
        device_cfg = config.get("device", {})

        if prediction_steps is not None:
            self.prediction_steps = prediction_steps
        else:
            self.prediction_steps = infer_cfg.get("prediction_steps") or data_cfg.get("future_steps", 36)

        if time_step_seconds is not None:
            self.dt = time_step_seconds
        else:
            self.dt = infer_cfg.get("time_step_seconds", 300.0)

        if gpu_enabled is not None:
            self._gpu_enabled = gpu_enabled
        else:
            self._gpu_enabled = infer_cfg.get("gpu_enabled",
                                device_cfg.get("gpu_enabled", True))

        self.device = get_device(self._gpu_enabled, device_cfg.get("gpu_id", 0))

        self.encoder = None
        self.decoder = None
        self.physics = None
        self.integrator = None
        self.zeta_proj = None
        self.velocity_proj = None
        self.omega_proj = None
        self._loaded = False

    def load_checkpoint(self, ckpt_path: str, strict: bool = True) -> dict:
        """加载训练好的模型权重。

        Args:
            ckpt_path: .pth checkpoint 文件路径
            strict: 是否严格匹配所有参数名

        Returns:
            加载信息 dict
        """
        self._build_model()

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
            epoch = checkpoint.get("epoch")
            metrics = checkpoint.get("metrics")
        else:
            model_state = checkpoint
            epoch = None
            metrics = None

        loaded = []
        missing = []

        key_map = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "physics": self.physics,
            "integrator": self.integrator,
            "zeta_proj": self.zeta_proj,
            "velocity_proj": self.velocity_proj,
            "omega_proj": self.omega_proj,
        }

        for key, module in key_map.items():
            if module is None:
                continue
            if key in model_state:
                result = module.load_state_dict(model_state[key], strict=strict)
                loaded.append(key)
                if hasattr(result, "missing_keys") and result.missing_keys:
                    missing.extend([f"{key}.{m}" for m in result.missing_keys])
            elif not strict:
                loaded.append(f"{key}(跳过)")

        self._loaded = True

        for m in key_map.values():
            if m is not None:
                m.eval()

        return {
            "source": Path(ckpt_path).name,
            "epoch": epoch,
            "metrics": metrics,
            "loaded": loaded,
            "missing": missing,
        }

    def _build_model(self):
        """构建完整 Encoder→Projection→RK2→Decoder 模型结构。

        投影头的输出通道数始终使用 data.future_steps（与训练时一致），
        确保 checkpoint 加载不出现尺寸不匹配。推理时通过切片限制步数。
        """
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        self.model_T_out = data_cfg.get("future_steps", 36)  # 训练时的 T_out

        self.encoder = ResUNet3DEncoder(self.config).to(self.device)
        self.physics = PhysicsModule(self.config).to(self.device)
        self.integrator = RK2Integrator(self.physics, dt=self.dt).to(self.device)

        # 解码器：与训练时的结构一致
        base_ch = model_cfg.get("base_channels", 64)
        depth = model_cfg.get("depth", 2)
        skip_channels = [base_ch]
        ch = base_ch
        for _ in range(depth - 1):
            ch = min(ch * 2, 1024)
            skip_channels.append(ch)
        self.decoder = ResUNet3DDecoder(
            self.config, self.encoder.output_channels, skip_channels,
        ).to(self.device)

        enc_out_ch = self.encoder.output_channels

        # 涡度初值投影（单通道，Tanh 限幅后由 predict 乘 zeta_scale=0.03）
        self.zeta_proj = torch.nn.Sequential(
            torch.nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            torch.nn.BatchNorm3d(enc_out_ch // 2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(enc_out_ch // 2, 1, 3, padding=1),
            torch.nn.Tanh(),
        ).to(self.device)

        # 时间依赖的速度场投影：model_T_out × 3 通道（与训练 checkpoint 一致）
        self.velocity_proj = torch.nn.Sequential(
            torch.nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            torch.nn.BatchNorm3d(enc_out_ch // 2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(enc_out_ch // 2, self.model_T_out * 3, 3, padding=1),
            torch.nn.Tanh(),
        ).to(self.device)

        # 时间依赖的水平涡度投影：model_T_out × 2 通道（与训练 checkpoint 一致）
        self.omega_proj = torch.nn.Sequential(
            torch.nn.Conv3d(enc_out_ch, enc_out_ch // 2, 3, padding=1),
            torch.nn.BatchNorm3d(enc_out_ch // 2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(enc_out_ch // 2, self.model_T_out * 2, 3, padding=1),
            torch.nn.Tanh(),
        ).to(self.device)

    @torch.no_grad()
    def predict(self, input_data: np.ndarray, num_steps: int | None = None,
                step_callback=None, target_data: np.ndarray | None = None,
                compute_metrics: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
        """使用模型预测未来垂直涡度场。

        Args:
            input_data: (T_in, H, W, L, C) 或 (1, T_in, H, W, L, C) 输入雷达数据
            num_steps: 预测步数，None 则使用 model_T_out（与训练一致）
            step_callback: 可选回调(step_idx, zeta_np_array) → None
            target_data: (T_out, H, W, L) 真实涡度，用于计算指标
            compute_metrics: 是否计算并返回 CSI/FSS/LPIPS/AUC 等指标

        Returns:
            如果 compute_metrics=False: (T_out, H, W, L) 预测的垂直涡度场序列
            如果 compute_metrics=True: ((T_out, H, W, L), metrics_dict)
        """
        if not self._loaded:
            raise RuntimeError("请先调用 load_checkpoint() 加载权重")
        if not hasattr(self, "model_T_out"):
            raise RuntimeError("模型未构建，请先调用 load_checkpoint()")

        T_req = num_steps if num_steps is not None else self.prediction_steps
        # 推理不能超过训练时的 T_out（投影头输出通道数受限）
        if T_req > self.model_T_out:
            raise ValueError(
                f"请求预测 {T_req} 步，但模型仅支持最多 {self.model_T_out} 步 "
                f"（训练时的 future_steps）。请减小 num_steps 或使用完整步数。"
            )

        if input_data.ndim == 5:
            input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(self.device)
        else:
            input_tensor = torch.from_numpy(input_data).to(self.device)

        B = input_tensor.shape[0]
        T_full = self.model_T_out  # 投影头输出通道数对应的步数

        # Encoder: 瓶颈特征 + skip connections
        features, skips = self.encoder(input_tensor)

        # 投影头 (始终输出完整 T_full 步)
        zeta_init = self.zeta_proj(features).squeeze(1) * 0.03  # (B, H', W', L')

        vel_raw = self.velocity_proj(features)  # (B, T_full*3, H', W', L')
        vel_raw = vel_raw.reshape(B, T_full, 3, *vel_raw.shape[-3:])
        # 水平 (u,v) ×20, 垂直 (w) ×5
        vu = vel_raw[:, :, 0:1, :, :, :] * 20.0
        vv = vel_raw[:, :, 1:2, :, :, :] * 20.0
        vw = vel_raw[:, :, 2:3, :, :, :] * 5.0
        vel_raw = torch.cat([vu, vv, vw], dim=2)

        omega_raw = self.omega_proj(features) * 0.1  # (B, T_full*2, H', W', L')
        omega_raw = omega_raw.reshape(B, T_full, 2, *omega_raw.shape[-3:])

        omega_x_seq = omega_raw[:, :, 0, :, :, :]
        omega_y_seq = omega_raw[:, :, 1, :, :, :]
        vv_seq = vel_raw[:, :, 2, :, :, :]

        # 只预测请求的步数（切片速度场以节省积分时间）
        vel_use = vel_raw[:, :T_req, :, :, :, :]
        ox_use = omega_x_seq[:, :T_req, :, :, :]
        oy_use = omega_y_seq[:, :T_req, :, :, :]
        vv_use = vv_seq[:, :T_req, :, :, :]

        # RK2 积分预测未来涡度（瓶颈分辨率）
        zeta_pred_lowres = self.integrator(
            zeta_init, vel_use,
            omega_x=ox_use, omega_y=oy_use,
            vertical_velocities=vv_use,
        )  # (B, T_req, H', W', L')

        # 解码器：低分辨率涡度 + skip connections → 全分辨率
        predictions = self.decoder(zeta_pred_lowres, skips, T_req)  # (B, T_req, H, W, L)

        result = predictions.squeeze(0).cpu().numpy()  # (T_req, H, W, L)

        if compute_metrics and target_data is not None:
            metrics = compute_all_metrics(result, target_data[:T_req])
            return result, metrics
        return result

    @torch.no_grad()
    def predict_from_file(self, input_path: str, output_path: str | None = None,
                          num_steps: int | None = None,
                          step_callback=None,
                          compute_metrics: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
        """从预处理后的 .npz 文件读取输入并预测。

        Args:
            input_path: 预处理后的 .npz 文件路径（需包含 'input' 数组）
            output_path: 可选的输出保存路径
            num_steps: 预测步数
            step_callback: 可选回调(step_idx, zeta_np_array) → None
            compute_metrics: 是否计算 CSI/FSS/LPIPS/AUC 等指标（需 npz 包含 'target'）

        Returns:
            (T_out, H, W, L) 预测涡度场，或 (predictions, metrics)
        """
        data = np.load(input_path, allow_pickle=True)
        input_arr = data["input"]
        target_arr = data.get("target", None) if compute_metrics else None

        if compute_metrics and target_arr is not None:
            predictions, metrics = self.predict(
                input_arr, num_steps=num_steps,
                step_callback=step_callback,
                target_data=target_arr, compute_metrics=True)
        else:
            predictions = self.predict(input_arr, num_steps=num_steps,
                                       step_callback=step_callback)

        if output_path:
            self.save_predictions(predictions, output_path, metadata={
                "input_source": Path(input_path).name,
                "prediction_steps": num_steps or self.prediction_steps,
                "dt_seconds": self.dt,
                "device": str(self.device),
            }, metrics=metrics if compute_metrics and target_arr is not None else None)

        if compute_metrics and target_arr is not None:
            return predictions, metrics
        return predictions

    @torch.no_grad()
    def predict_batch(self, input_dir: str, output_dir: str,
                      num_steps: int | None = None,
                      progress_callback=None) -> list[str]:
        """批量预测一个目录下的所有 .npz 文件。

        Args:
            input_dir: 输入文件目录
            output_dir: 输出目录
            num_steps: 预测步数
            progress_callback: 进度回调 (current, total, filename)

        Returns:
            输出文件路径列表
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = sorted(input_path.glob("*.npz"))
        total = len(files)
        outputs = []

        for idx, npz_file in enumerate(files):
            if progress_callback:
                progress_callback(idx + 1, total, npz_file.name)

            try:
                out_file = output_path / f"{npz_file.stem}_pred.npz"
                self.predict_from_file(str(npz_file), str(out_file), num_steps=num_steps)
                outputs.append(str(out_file))
            except Exception as e:
                if progress_callback:
                    progress_callback(idx + 1, total, f"错误: {npz_file.name} - {e}")

        return outputs

    @staticmethod
    def save_predictions(predictions: np.ndarray, output_path: str,
                         metadata: dict | None = None,
                         metrics: dict | None = None):
        """保存预测结果到 .npz 文件。"""
        save_dict = {"predictions": predictions.astype(np.float32)}
        if metadata:
            save_dict["metadata"] = metadata
        if metrics:
            save_dict["metrics"] = metrics
        np.savez_compressed(output_path, **save_dict)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def inference_info(self) -> dict:
        """返回当前推理配置摘要。"""
        info = {
            "prediction_steps": self.prediction_steps,
            "dt_seconds": self.dt,
            "prediction_minutes": self.prediction_steps * self.dt / 60.0,
            "device": str(self.device),
            "loaded": self._loaded,
        }
        if self.lon_range:
            info["lon_range"] = self.lon_range
        if self.lat_range:
            info["lat_range"] = self.lat_range
        if self.vert_range:
            info["vert_range"] = self.vert_range
        return info

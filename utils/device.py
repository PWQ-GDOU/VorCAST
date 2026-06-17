import torch
from .exceptions import GPUNotAvailableError


def get_device(gpu_enabled: bool = True, gpu_id: int = 0) -> torch.device:
    """获取计算设备。

    如果启用 GPU 且 CUDA 可用则返回 cuda 设备，否则返回 CPU。
    """
    if gpu_enabled and torch.cuda.is_available():
        device_id = min(gpu_id, torch.cuda.device_count() - 1)
        return torch.device(f"cuda:{device_id}")
    if gpu_enabled and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_gpu(gpu_enabled: bool = True) -> dict:
    """检查 GPU 状态并返回信息字典。"""
    info = {
        "gpu_enabled": gpu_enabled,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "device": "cpu",
    }
    if gpu_enabled and torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(device_id)
        info["device"] = f"cuda:{device_id}"
        info["memory_total"] = torch.cuda.get_device_properties(device_id).total_memory
    return info


def enable_gpu(gpu_id: int = 0) -> torch.device:
    """强制启用 GPU，不可用时抛出异常。"""
    if not torch.cuda.is_available():
        raise GPUNotAvailableError("CUDA GPU 不可用")
    return torch.device(f"cuda:{gpu_id}")


def disable_gpu() -> torch.device:
    """返回 CPU 设备。"""
    return torch.device("cpu")

"""PyTorch Dataset 与 DataLoader 工厂。"""
import numpy as np
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader


class StormEventDataset(Dataset):
    """加载预处理后的风暴事件 .npz 文件。"""

    def __init__(self, file_paths: list[str], config: dict | None = None):
        self.file_paths = [Path(p) for p in file_paths if Path(p).exists()]
        if not self.file_paths:
            raise ValueError("没有找到有效的预处理数据文件")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.file_paths[idx], allow_pickle=True)

        input_data = torch.from_numpy(data["input"])    # [T_in, H, W, L, C]
        target = torch.from_numpy(data["target"])       # [T_out, H, W, L]

        result: dict[str, Any] = {
            "input": input_data,
            "target": target,
            "event_path": str(self.file_paths[idx]),
        }
        if "storm_uv" in data:
            result["storm_uv"] = torch.from_numpy(data["storm_uv"])      # [T_total, 2]
        if "ef_label" in data:
            result["ef_label"] = int(data["ef_label"])
        return result


class ConcatDataset(Dataset):
    """合并两个数据集。"""

    def __init__(self, dataset1_files: list[str], dataset2_files: list[str],
                 config: dict | None = None):
        files = list(dataset1_files) + list(dataset2_files)
        self._inner = StormEventDataset(files, config)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


def create_dataloaders(
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试 DataLoader。

    Args:
        train_files: 训练集文件路径列表
        val_files: 验证集文件路径列表
        test_files: 测试集文件路径列表
        config: 完整配置字典

    Returns:
        (train_loader, val_loader, test_loader)
    """
    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 2)
    num_workers = training_cfg.get("num_workers", 0)

    train_ds = StormEventDataset(train_files, config)
    val_ds = StormEventDataset(val_files, config) if val_files else None
    test_ds = StormEventDataset(test_files, config) if test_files else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    ) if test_ds else None

    return train_loader, val_loader, test_loader


def scan_processed_files(processed_dir: str) -> list[str]:
    """扫描已处理的 .npz 文件，返回路径列表。"""
    p = Path(processed_dir)
    if not p.exists():
        return []
    return sorted([str(f) for f in p.glob("*.npz")])

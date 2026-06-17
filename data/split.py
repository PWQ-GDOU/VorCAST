"""按事件分割训练/验证/测试集。

按照方法2要求：
- 按事件分割（避免同一风暴的不同时间片段出现在训练集和验证集中，防止数据泄露）
- 比例：70% 训练，15% 验证，15% 测试
- 每个 .npz 文件 = 一个独立事件（storm_{event_id}.npz），文件级分割即事件级分割
"""
import re
import numpy as np
from pathlib import Path


def _extract_event_id(filepath: str) -> str:
    """从 .npz 文件路径中提取事件 ID。

    文件名格式:
      storm_{storm_num}_{date}_w{idx}.npz  (多窗口)
      storm_{storm_num}_{date}.npz          (单窗口)
      event_{id}.npz
    """
    name = Path(filepath).stem
    # storm_123_20230105_w0 或 storm_123_20230105 格式
    m = re.match(r"storm_(\d+_\d+)", name)
    if m:
        return m.group(1)
    # event_001 格式
    m = re.match(r"event_(\d+)", name)
    if m:
        return m.group(1)
    # 退化：用文件名本身作为事件 ID
    return name


def split_by_event(
    file_paths: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """按事件将文件列表分割为 train/val/test。

    保证同一事件的所有文件（含滑动窗口产生的多个样本）只出现
    在一个子集中，防止数据泄露。

    Args:
        file_paths: 所有已处理 .npz 文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        f"比例之和必须为 1，当前: {train_ratio + val_ratio + test_ratio:.3f}"

    files = sorted(file_paths)
    if not files:
        raise ValueError("文件列表为空")

    # 按事件 ID 分组（滑动窗口产生同一事件的多个 .npz）
    event_groups: dict[str, list[str]] = {}
    for f in files:
        eid = _extract_event_id(f)
        event_groups.setdefault(eid, []).append(f)

    event_ids = sorted(event_groups.keys())
    n_events = len(event_ids)

    rng = np.random.RandomState(seed)
    shuffled_events = event_ids.copy()
    rng.shuffle(shuffled_events)

    train_end = max(1, int(n_events * train_ratio))
    val_end = train_end + max(1, int(n_events * val_ratio))

    train_events = shuffled_events[:train_end]
    val_events = shuffled_events[train_end:val_end]
    test_events = shuffled_events[val_end:]

    train_files = sorted(f for eid in train_events for f in event_groups[eid])
    val_files = sorted(f for eid in val_events for f in event_groups[eid])
    test_files = sorted(f for eid in test_events for f in event_groups[eid])

    # 验证无事件 ID 跨集泄露
    train_ids = set(train_events)
    val_ids = set(val_events)
    test_ids = set(test_events)
    assert train_ids.isdisjoint(val_ids), "事件 ID 在 train/val 之间泄露!"
    assert train_ids.isdisjoint(test_ids), "事件 ID 在 train/test 之间泄露!"
    assert val_ids.isdisjoint(test_ids), "事件 ID 在 val/test 之间泄露!"

    return train_files, val_files, test_files


def merge_datasets(dataset1_files: list[str], dataset2_files: list[str],
                   train_ratio: float = 0.70, val_ratio: float = 0.15,
                   seed: int = 42):
    """合并两个数据集后按事件分割。

    两个训练集（如 NEXRAD 雷达数据 + 风暴轨迹数据衍生的不同预处理批次）
    的样本合并后统一做 train/val/test 分割。
    """
    all_files = sorted(set(list(dataset1_files) + list(dataset2_files)))
    if not all_files:
        raise ValueError("合并后文件为空")
    return split_by_event(all_files, train_ratio, val_ratio,
                          1.0 - train_ratio - val_ratio, seed)

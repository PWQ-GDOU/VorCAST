import os
import yaml
from pathlib import Path
from typing import Any

from .exceptions import ConfigError

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config_default.yaml"


def load_default_config() -> dict:
    """加载默认配置文件。"""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 覆盖 base。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict):
    """校验配置完整性和合法性。"""
    required_sections = ["data", "model", "training", "device", "logging"]
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"缺少配置节: '{section}'")

    # 校验训练参数
    training = config.get("training", {})
    if training.get("batch_size", 0) < 1:
        raise ConfigError("batch_size 必须 ≥ 1")
    if training.get("epochs", 0) < 1:
        raise ConfigError("epochs 必须 ≥ 1")
    if training.get("learning_rate", 0) <= 0:
        raise ConfigError("learning_rate 必须 > 0")

    # 校验数据参数
    data = config.get("data", {})
    if data.get("history_steps", 0) < 1:
        raise ConfigError("history_steps 必须 ≥ 1")
    if data.get("future_steps", 0) < 1:
        raise ConfigError("future_steps 必须 ≥ 1")


def load_config(user_config_path: str | None = None) -> dict:
    """加载配置：默认配置 + 可选用户配置合并。"""
    config = load_default_config()
    if user_config_path:
        path = Path(user_config_path)
        if not path.exists():
            raise ConfigError(f"配置文件不存在: {user_config_path}")
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config = deep_merge(config, user_config)
    validate_config(config)
    return config


def save_config(config: dict, path: str):
    """保存配置到文件。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

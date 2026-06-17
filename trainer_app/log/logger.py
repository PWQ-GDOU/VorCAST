import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "trainer",
    log_dir: str = "./logs",
    task_name: str | None = None,
    level: str = "INFO",
    log_format: str | None = None,
) -> logging.Logger:
    """创建配置好的 logger 实例。

    Args:
        name: logger 名称
        log_dir: 日志根目录
        task_name: 任务名称，为空则自动生成时间戳名称
        level: 日志级别
        log_format: 日志格式字符串

    Returns:
        配置完成的 Logger
    """
    if task_name is None:
        task_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    task_log_dir = Path(log_dir) / task_name
    task_log_dir.mkdir(parents=True, exist_ok=True)

    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    # 文件 handler
    file_handler = logging.FileHandler(
        task_log_dir / "training.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt="%H:%M:%S"))
    logger.addHandler(console_handler)

    return logger


def get_task_log_dir(log_dir: str, task_name: str) -> Path:
    """获取任务日志目录路径。"""
    return Path(log_dir) / task_name

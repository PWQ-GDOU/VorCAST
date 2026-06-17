class TrainerError(Exception):
    """训练器基础异常"""


class ConfigError(TrainerError):
    """配置错误"""


class DataError(TrainerError):
    """数据加载/预处理错误"""


class ModelError(TrainerError):
    """模型构建/推理错误"""


class TrainingError(TrainerError):
    """训练过程错误"""


class GPUNotAvailableError(TrainerError):
    """GPU 不可用错误"""


class TrainingInterrupted(TrainerError):
    """训练被用户中断"""

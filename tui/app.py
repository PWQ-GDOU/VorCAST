"""Textual TUI 主应用。"""
import torch
from textual.app import App, ComposeResult
from textual.widgets import Footer

from ..utils.config import load_config
from .screens.main_menu import MainMenuScreen
from .screens.dataset_select import DatasetSelectScreen
from .screens.config_screen import ConfigScreen
from .screens.training_run import TrainingRunScreen
from .screens.history_view import HistoryViewScreen
from .screens.log_view import LogViewScreen
from .screens.inference_screen import InferenceScreen


class TrainerTUI(App):
    """龙卷风垂直涡度预测训练器 — 终端用户界面。

    基于 Textual 框架，纯终端环境运行。
    支持键盘导航，完整中文显示。
    """

    TITLE = "Tornado Vorticity Prediction Trainer"
    SUB_TITLE = "Nowcast3D Gray-box Physics Model"

    CSS = """
    Header {
        background: $panel;
        color: $text;
        text-style: bold;
    }
    Footer {
        background: $panel;
        color: $text-muted;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        padding: 1 0;
        content-align: center middle;
    }
    .hint {
        color: $text-muted;
        text-style: italic;
    }
    Screen {
        align: center middle;
        overflow: auto;
    }
    #training-status {
        text-style: bold;
        padding: 1 0;
    }
    ScrollableContainer {
        overflow-y: auto;
        overflow-x: hidden;
    }
    Label {
        text-overflow: ellipsis;
    }
    """

    SCREENS = {
        "main_menu": MainMenuScreen,
        "dataset_select": DatasetSelectScreen,
        "config_screen": ConfigScreen,
        "training_run": TrainingRunScreen,
        "history_view": HistoryViewScreen,
        "log_view": LogViewScreen,
        "inference_screen": InferenceScreen,
    }

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def on_mount(self):
        """启动时进入主菜单。"""
        self.push_screen("main_menu")

    def compose(self) -> ComposeResult:
        yield Footer()

    def push_screen(self, name: str):
        """按名称压入页面（已安装则直接复用）。"""
        screen_cls = self.SCREENS.get(name)
        if screen_cls:
            if not self.is_screen_installed(name):
                self.install_screen(screen_cls(), name)
            super().push_screen(name)
        else:
            super().push_screen(name)


def run_app(config: dict):
    """启动 TUI 应用。"""
    app = TrainerTUI(config)
    app.run()

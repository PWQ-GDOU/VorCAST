"""训练参数配置页面。"""
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input, Button, Select, Switch
from textual.reactive import reactive


class ConfigScreen(Screen):
    """训练超参数配置页面。

    可配置项：
    - 窗口大小（空间度数、格点数）
    - 垂直层数
    - 归一化方式
    - 训练超参数（batch size, epochs, LR）
    - GPU 开关
    """

    BINDINGS = [
        ("escape", "dismiss", "返回"),
        ("ctrl+t", "start_training", "训练"),
        ("ctrl+s", "save", "保存"),
        ("ctrl+d", "reset", "重置"),
    ]

    CSS = """
    ConfigScreen {
        align: center middle;
    }
    #config-container {
        width: 70;
        height: auto;
        max-height: 90%;
        border: solid $primary;
        padding: 1 2;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    .config-row {
        height: 3;
        margin: 1 0;
    }
    .config-label {
        width: 30;
        padding: 0 1;
    }
    .hint {
        color: $text-muted;
        text-style: italic;
    }
    Input {
        width: 35;
    }
    Select {
        width: 35;
    }
    #btn-row {
        align: center middle;
        padding-top: 1;
    }
    Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="config-container"):
            yield Static("Training Configuration", classes="section-title")

            # Data parameters
            yield Static("Data Parameters", classes="section-title")
            with Horizontal(classes="config-row"):
                yield Static("Spatial Window (deg):", classes="config-label")
                yield Input(value="2.56", id="spatial_degree", type="number")
            with Horizontal(classes="config-row"):
                yield Static("Grid Size (HxW):", classes="config-label")
                yield Input(value="128", id="grid_size", type="integer")
            with Horizontal(classes="config-row"):
                yield Static("Vertical Layers (empty=all):", classes="config-label")
                yield Input(value="", id="vertical_layers", placeholder="Leave empty for all")
            with Horizontal(classes="config-row"):
                yield Static("Refl. Normalization:", classes="config-label")
                yield Select(
                    [("Min-Max (0~1)", "minmax"), ("Z-Score", "zscore")],
                    id="ref_norm", value="minmax",
                )

            # Model parameters
            yield Static("Model Parameters", classes="section-title")
            with Horizontal(classes="config-row"):
                yield Static("Encoder Depth:", classes="config-label")
                yield Input(value="4", id="depth", type="integer")
            with Horizontal(classes="config-row"):
                yield Static("Base Channels:", classes="config-label")
                yield Input(value="64", id="base_channels", type="integer")
            with Horizontal(classes="config-row"):
                yield Static("Diffusion Coef nu_t:", classes="config-label")
                yield Input(value="0.1", id="diffusion_coef", type="number")

            # Pretrained weights
            yield Static("Pretrained Weights (optional)", classes="section-title")
            yield Static("Select existing .pt / .pth weights to continue training", classes="hint")
            with Horizontal(classes="config-row"):
                yield Static("Weights Path:", classes="config-label")
                yield Input(value="", id="pretrained_path", placeholder="Leave empty to train from scratch")
                yield Button("Browse", id="btn-browse-pretrained")
            with Horizontal(classes="config-row"):
                yield Static("Load Optimizer State:", classes="config-label")
                yield Switch(value=False, id="load-opt-switch")
            with Horizontal(classes="config-row"):
                yield Static("Strict Param Match:", classes="config-label")
                yield Switch(value=True, id="strict-load-switch")

            # Training parameters
            yield Static("Training Parameters", classes="section-title")
            with Horizontal(classes="config-row"):
                yield Static("Run Name:", classes="config-label")
                yield Input(value="", id="run_name", placeholder="Leave empty for timestamp")
            with Horizontal(classes="config-row"):
                yield Static("Output Root Dir:", classes="config-label")
                yield Input(value="./output", id="output_dir", placeholder="./output")
            with Horizontal(classes="config-row"):
                yield Static("Batch Size:", classes="config-label")
                yield Input(value="16", id="batch_size", type="integer")
            with Horizontal(classes="config-row"):
                yield Static("Epochs:", classes="config-label")
                yield Input(value="100", id="epochs", type="integer")
            with Horizontal(classes="config-row"):
                yield Static("Learning Rate:", classes="config-label")
                yield Input(value="0.001", id="learning_rate", type="number")
            with Horizontal(classes="config-row"):
                yield Static("Early Stop Patience:", classes="config-label")
                yield Input(value="15", id="early_stop", type="integer")
            with Horizontal(classes="config-row"):
                yield Static("GPU Acceleration:", classes="config-label")
                yield Switch(value=True, id="gpu-switch")
            with Horizontal(classes="config-row"):
                yield Static("Mixed Precision (AMP):", classes="config-label")
                yield Switch(value=True, id="amp-switch")

            # 操作按钮
            with Horizontal(id="btn-row"):
                yield Button("开始训练 (Ctrl+T)", id="btn-start", variant="primary")
                yield Button("保存配置 (Ctrl+S)", id="btn-save")
                yield Button("重置默认 (Ctrl+D)", id="btn-reset")
                yield Button("返回 (ESC)", id="btn-back")
        yield Footer()

    def on_mount(self):
        """Populate fields from current app config."""
        self._populate_from_config()

    def _populate_from_config(self):
        """Read values from self.app.config into UI input fields."""
        cfg = self.app.config
        data = cfg.get("data", {})
        model = cfg.get("model", {})
        training = cfg.get("training", {})
        device = cfg.get("device", {})

        self.query_one("#spatial_degree", Input).value = str(data.get("spatial_degree", 2.56))
        self.query_one("#grid_size", Input).value = str(data.get("grid_size", 128))
        vl = data.get("vertical_layers", "")
        self.query_one("#vertical_layers", Input).value = str(vl) if vl else ""
        self.query_one("#ref_norm", Select).value = data.get("reflectivity_norm", "minmax")

        self.query_one("#depth", Input).value = str(model.get("depth", 4))
        self.query_one("#base_channels", Input).value = str(model.get("base_channels", 64))
        phys = model.get("physics", {})
        self.query_one("#diffusion_coef", Input).value = str(phys.get("diffusion_coef", 0.1))

        self.query_one("#pretrained_path", Input).value = training.get("pretrained_path", "")
        self.query_one("#load-opt-switch", Switch).value = training.get("load_optimizer_state", False)
        self.query_one("#strict-load-switch", Switch).value = model.get("strict_load", True)

        self.query_one("#run_name", Input).value = training.get("run_name", "")
        self.query_one("#output_dir", Input).value = training.get("output_dir", "./output")
        self.query_one("#batch_size", Input).value = str(training.get("batch_size", 16))
        self.query_one("#epochs", Input).value = str(training.get("epochs", 100))
        self.query_one("#learning_rate", Input).value = str(training.get("learning_rate", 0.001))
        self.query_one("#early_stop", Input).value = str(training.get("early_stop_patience", 15))

        self.query_one("#gpu-switch", Switch).value = device.get("gpu_enabled", True)
        self.query_one("#amp-switch", Switch).value = training.get("use_amp", True)

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id

        if btn_id == "btn-start":
            self.action_start_training()
        elif btn_id == "btn-save":
            self.action_save()
        elif btn_id == "btn-reset":
            self.action_reset()
        elif btn_id == "btn-back":
            self.dismiss()
        elif btn_id == "btn-browse-pretrained":
            self._browse_pretrained()

    def action_start_training(self):
        self._apply_config()
        self.dismiss()
        self.app.push_screen("training_run")

    def action_save(self):
        self._apply_config()
        self._save_config_to_file()

    def action_reset(self):
        self._reset_defaults()

    def _browse_pretrained(self):
        """Open file browser to select a pretrained checkpoint."""
        from pathlib import Path
        current = self.query_one("#pretrained_path", Input).value.strip()
        start = current if current and Path(current).exists() else "."
        self._browser_target = "pretrained_path"
        from .file_browser import FileBrowserScreen
        self.app.push_screen(
            FileBrowserScreen(start, ext_filter=[".pt", ".pth"])
        )

    def on_screen_resume(self):
        """Handle result from file browser."""
        result = getattr(self.app, '_browser_result', None)
        if result is not None and hasattr(self, '_browser_target'):
            input_id = self._browser_target
            self.query_one(f"#{input_id}", Input).value = result
            self.app._browser_result = None
            del self._browser_target

    def _apply_config(self):
        """将 UI 中的值应用到全局配置。"""
        cfg = self.app.config

        # 数据
        data = cfg.setdefault("data", {})
        data["spatial_degree"] = float(self.query_one("#spatial_degree", Input).value)
        data["grid_size"] = int(self.query_one("#grid_size", Input).value)
        vl = self.query_one("#vertical_layers", Input).value
        data["vertical_layers"] = int(vl) if vl.strip() else None
        data["reflectivity_norm"] = self.query_one("#ref_norm", Select).value

        # 模型
        model = cfg.setdefault("model", {})
        model["depth"] = int(self.query_one("#depth", Input).value)
        model["base_channels"] = int(self.query_one("#base_channels", Input).value)
        model.setdefault("physics", {})["diffusion_coef"] = float(
            self.query_one("#diffusion_coef", Input).value
        )

        # 训练
        training = cfg.setdefault("training", {})
        run_name = self.query_one("#run_name", Input).value.strip()
        training["run_name"] = run_name if run_name else None
        output_dir = self.query_one("#output_dir", Input).value.strip()
        training["output_dir"] = output_dir if output_dir else "./output"
        training["batch_size"] = int(self.query_one("#batch_size", Input).value)
        training["epochs"] = int(self.query_one("#epochs", Input).value)
        training["learning_rate"] = float(self.query_one("#learning_rate", Input).value)
        training["early_stop_patience"] = int(self.query_one("#early_stop", Input).value)

        # 预训练权重
        pretrained = self.query_one("#pretrained_path", Input).value.strip()
        training["pretrained_path"] = pretrained if pretrained else None
        training["load_optimizer_state"] = self.query_one("#load-opt-switch", Switch).value
        model["strict_load"] = self.query_one("#strict-load-switch", Switch).value

        # 设备
        cfg.setdefault("device", {})["gpu_enabled"] = self.query_one("#gpu-switch", Switch).value
        training["use_amp"] = self.query_one("#amp-switch", Switch).value

    def _save_config_to_file(self):
        """保存配置到文件。"""
        from ...utils.config import save_config
        path = "./user_config.yaml"
        save_config(self.app.config, path)
        self.notify(f"Config saved to {path}", severity="information")

    def _reset_defaults(self):
        """重置为默认配置。"""
        from ...utils.config import load_config
        self.app.config = load_config()
        # 更新 UI
        data = self.app.config["data"]
        self.query_one("#spatial_degree", Input).value = str(data.get("spatial_degree", 2.56))
        self.query_one("#grid_size", Input).value = str(data.get("grid_size", 128))
        self.query_one("#ref_norm", Select).value = data.get("reflectivity_norm", "minmax")
        training = self.app.config["training"]
        self.query_one("#epochs", Input).value = str(training.get("epochs", 100))
        self.query_one("#learning_rate", Input).value = str(training.get("learning_rate", 0.001))
        self.query_one("#pretrained_path", Input).value = ""
        self.query_one("#load-opt-switch", Switch).value = False
        self.notify("Reset to default config")

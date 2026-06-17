"""数据集选择页面。"""
import os
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input, Button


class DatasetSelectScreen(Screen):
    """数据集选择与预处理页面。

    数据集1: Severe Events Tracks — .csv 文件目录（风暴轨迹元信息）
    数据集2: NEXRAD GridRad — .nc (NetCDF) 文件目录（三维雷达体扫数据）

    支持：
    - 浏览并选择本地数据集目录
    - 启动前自动验证文件格式
    - 异步预处理
    """

    BINDINGS = [
        ("escape", "dismiss", "返回"),
        ("ctrl+v", "validate", "验证"),
        ("ctrl+p", "preprocess", "预处理"),
        ("ctrl+s", "skip", "跳过"),
    ]

    CSS = """
    DatasetSelectScreen {
        align: center middle;
    }
    #ds-container {
        width: 75;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    .hint {
        color: $text-muted;
        text-style: italic;
    }
    Input {
        width: 100%;
        margin-bottom: 1;
    }
    #ds-status {
        color: $text-muted;
        padding: 1 0;
        max-height: 10;
    }
    #validate-result {
        padding: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="ds-container"):
            yield Static("Dataset Selection & Preprocessing", classes="section-title")

            yield Static("Dataset 1 - NEXRAD 3D Radar (.nc)", classes="section-title")
            yield Static("NetCDF files every 5 min: Reflectivity/AzShear/Divergence etc.", classes="hint")
            yield Input(placeholder="Enter .nc file directory path...", id="ds1-input")
            yield Button("Browse", id="btn-browse1")

            yield Static("Dataset 2 - Storm Tracks (.csv)", classes="section-title")
            yield Static("Storm Number/Time/Lon/Lat/u/v-motion/EF tracks", classes="hint")
            yield Input(placeholder="Enter .csv file directory path...", id="ds2-input")
            yield Button("Browse", id="btn-browse2")

            yield Static("Preprocessing Output Dir:", classes="section-title")
            yield Input(placeholder="Default: ./processed", id="processed-input", value="./processed")

            yield Static("", id="validate-result")
            yield Static("", id="ds-status")
            with Horizontal():
                yield Button("Validate (Ctrl+V)", id="btn-validate", variant="default")
                yield Button("Preprocess (Ctrl+P)", id="btn-preprocess", variant="primary")
                yield Button("Skip (Ctrl+S)", id="btn-skip")
                yield Button("Back (ESC)", id="btn-back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id

        if btn_id in ("btn-browse1", "btn-browse2"):
            self._browse_directory(btn_id)
        elif btn_id == "btn-validate":
            self.action_validate()
        elif btn_id == "btn-preprocess":
            self.action_preprocess()
        elif btn_id == "btn-skip":
            self.action_skip()
        elif btn_id == "btn-back":
            self.dismiss()

    def action_validate(self):
        self._validate_datasets()

    def action_preprocess(self):
        self._start_preprocess()

    def action_skip(self):
        self._skip_to_config()

    def _get_ds_paths(self) -> tuple[str, str]:
        """获取两个数据集路径。"""
        ds1 = self.query_one("#ds1-input", Input).value.strip()
        ds2 = self.query_one("#ds2-input", Input).value.strip()
        return ds1, ds2

    def _browse_directory(self, btn_id: str):
        """Open interactive file browser for directory selection."""
        input_id = "ds1-input" if btn_id == "btn-browse1" else "ds2-input"
        ext_filter = [".nc"] if btn_id == "btn-browse1" else [".csv"]
        current = self.query_one(f"#{input_id}", Input).value.strip()
        start = current if current and Path(current).exists() else "."
        self._browser_target = input_id
        from .file_browser import FileBrowserScreen
        self.app.push_screen(FileBrowserScreen(start, ext_filter=ext_filter))

    def on_screen_resume(self):
        """Handle result from file browser."""
        result = getattr(self.app, '_browser_result', None)
        if result is not None and hasattr(self, '_browser_target'):
            input_id = self._browser_target
            dir_path = str(Path(result).parent if Path(result).is_file() else result)
            self.query_one(f"#{input_id}", Input).value = dir_path
            self.query_one("#ds-status", Static).update(f"Selected: {dir_path}")
            self.app._browser_result = None
            del self._browser_target

    def _validate_datasets(self):
        """验证所选数据集目录是否包含正确的文件格式。"""
        ds1, ds2 = self._get_ds_paths()

        if not ds1 or not ds2:
            self.query_one("#validate-result", Static).update(
                "[Error] Please fill in both dataset paths"
            )
            return

        from ...data.preprocess import DataPreprocessor
        preprocessor = DataPreprocessor(self.app.config)
        result = preprocessor.validate_datasets(ds1, ds2)

        if result["valid"]:
            msg = (
                f"[OK] Validation passed!\n"
                f"  CSV files: {result['csv_count']}\n"
                f"  NetCDF files: {result['nc_count']}"
            )
            self.query_one("#validate-result", Static).update(msg)
            self.notify("Validation passed", severity="information")
        else:
            msg = "[FAIL] Validation failed:\n" + "\n".join(f"  - {e}" for e in result["errors"])
            self.query_one("#validate-result", Static).update(msg)
            self.notify("Validation failed, check directory paths and file formats", severity="error")

    def _start_preprocess(self):
        """启动数据预处理。"""
        ds1, ds2 = self._get_ds_paths()
        processed = self.query_one("#processed-input", Input).value.strip()

        if not ds1 or not ds2:
            self.query_one("#ds-status", Static).update("Error: Please fill in both dataset paths")
            return

        from ...data.preprocess import DataPreprocessor
        preprocessor = DataPreprocessor(self.app.config)
        result = preprocessor.validate_datasets(ds1, ds2)
        if not result["valid"]:
            self.query_one("#ds-status", Static).update(
                "Validation failed, click [Validate] for details"
            )
            return

        self.query_one("#ds-status", Static).update("Preprocessing data, please wait...")

        # 更新全局配置
        self.app.config["data"]["dataset1_path"] = ds1
        self.app.config["data"]["dataset2_path"] = ds2
        self.app.config["data"]["processed_dir"] = processed or "./processed"

        # 在 worker 中异步运行预处理
        self.run_worker(self._run_preprocess(ds1, ds2, processed), exclusive=True)

    async def _run_preprocess(self, ds1: str, ds2: str, processed: str):
        """异步执行预处理（在后台线程中运行以不阻塞 UI）。"""
        from ...data.preprocess import DataPreprocessor

        preprocessor = DataPreprocessor(self.app.config)

        def progress_callback(current, total, msg):
            self.query_one("#ds-status", Static).update(
                f"Progress: {current}/{total}\n{msg}"
            )

        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: preprocessor.process_events(ds1, ds2, progress_callback),
        )
        self.query_one("#ds-status", Static).update(
            "Preprocessing complete! Click [Skip] to enter training config."
        )

    def _skip_to_config(self):
        """跳过预处理，直接进入训练参数配置。"""
        ds1, ds2 = self._get_ds_paths()
        processed = self.query_one("#processed-input", Input).value.strip()

        if ds1:
            self.app.config["data"]["dataset1_path"] = ds1
        if ds2:
            self.app.config["data"]["dataset2_path"] = ds2
        if processed:
            self.app.config["data"]["processed_dir"] = processed

        self.dismiss()
        self.app.push_screen("config_screen")

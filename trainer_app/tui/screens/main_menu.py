"""主菜单页面。"""
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button


class MainMenuScreen(Screen):
    """训练器主菜单。

    提供导航入口：训练配置、历史记录、日志查看、退出。
    只允许一个后台训练任务运行。
    """

    BINDINGS = [
        ("escape", "quit_app", "退出"),
        ("ctrl+t", "nav_train", "训练"),
        ("ctrl+i", "nav_infer", "推理"),
        ("ctrl+d", "nav_dataset", "数据集"),
        ("ctrl+g", "nav_config", "配置"),
        ("ctrl+h", "nav_history", "历史"),
        ("ctrl+l", "nav_logs", "日志"),
        ("ctrl+q", "quit_app", "退出"),
    ]

    CSS = """
    MainMenuScreen {
        align: center middle;
    }
    #menu-container {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }
    #title {
        content-align: center middle;
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    #subtitle {
        content-align: center middle;
        color: $text-muted;
        padding-bottom: 1;
    }
    #shortcuts {
        content-align: center middle;
        color: $text-muted;
        padding-bottom: 1;
    }
    Button {
        width: 100%;
        margin: 1 0;
    }
    #gpu-indicator {
        content-align: center middle;
        color: $success;
        padding-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="menu-container"):
            yield Static("Tornado Vertical Vorticity Prediction Trainer", id="title")
            yield Static("Based on Nowcast3D Gray-box Physics Model", id="subtitle")
            yield Static("Ctrl+T=Train  Ctrl+I=Infer  Ctrl+G=Config  Ctrl+H=History  Ctrl+L=Logs  Ctrl+Q=Quit", id="shortcuts")
            yield Button("Start Training (Ctrl+T)", id="btn-train", variant="primary")
            yield Button("Model Inference (Ctrl+I)", id="btn-infer", variant="primary")
            yield Button("Dataset Selection & Preprocessing (Ctrl+D)", id="btn-dataset", variant="default")
            yield Button("Training Config (Ctrl+G)", id="btn-config", variant="default")
            yield Button("Resume Training (none running)", id="btn-resume", variant="default", disabled=True)
            yield Button("Kill All Workers", id="btn-kill-workers", variant="warning", disabled=True)
            yield Button("Training History (Ctrl+H)", id="btn-history", variant="default")
            yield Button("View Logs (Ctrl+L)", id="btn-logs", variant="default")
            yield Button("Quit (Ctrl+Q)", id="btn-quit", variant="error")
            yield Static("GPU: Detecting...", id="gpu-indicator")
        yield Footer()

    def on_mount(self):
        """初始化时检测 GPU 状态和后台训练。"""
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            self.query_one("#gpu-indicator", Static).update(
                f"GPU: {name} [Available]"
            )
        else:
            self.query_one("#gpu-indicator", Static).update(
                "GPU: Unavailable [Using CPU]"
            )
        self._update_resume_button()
        # 定时刷新，替代不存在的 on_screen_resume
        self.set_interval(3.0, self._update_resume_button)

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """检查进程是否存活（跨平台）。"""
        import sys
        if sys.platform != "win32":
            import os
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, pid)
            if not handle:
                return False
            WAIT_TIMEOUT = 0x00000102
            result = kernel32.WaitForSingleObject(handle, 0)
            kernel32.CloseHandle(handle)
            return result == WAIT_TIMEOUT
        except Exception:
            return False

    def _update_resume_button(self):
        """扫描输出目录，检测后台 Worker 并更新按钮状态。"""
        import json
        import os

        try:
            btn_train = self.query_one("#btn-train", Button)
            btn_resume = self.query_one("#btn-resume", Button)
            btn_kill = self.query_one("#btn-kill-workers", Button)
        except Exception:
            return

        output_base = self.app.config.get("training", {}).get("output_dir", "./output")
        if not os.path.isdir(output_base):
            btn_resume.label = "Resume Training (none running)"
            btn_resume.disabled = True
            btn_resume.variant = "default"
            btn_kill.disabled = True
            btn_kill.label = "Kill All Workers"
            btn_train.disabled = False
            self.app._active_run_dir = None
            return

        running_run = None
        running_ts = 0
        dead_pid_count = 0

        try:
            for entry in os.scandir(output_base):
                if not entry.is_dir():
                    continue
                status_path = os.path.join(entry.path, "status.json")
                pid_path = os.path.join(entry.path, "worker.pid")

                # Read PID for liveness check
                pid = None
                if os.path.exists(pid_path):
                    try:
                        with open(pid_path, "r", encoding="utf-8") as f:
                            pid = int(f.read().strip())
                    except Exception:
                        pass

                # Read status
                s = None
                if os.path.exists(status_path):
                    try:
                        with open(status_path, "r", encoding="utf-8") as f:
                            s = json.load(f)
                    except Exception:
                        pass

                if s and s.get("status") in ("running", "starting", "paused"):
                    if pid is not None and self._is_pid_alive(pid):
                        ts = s.get("timestamp", 0)
                        if ts > running_ts:
                            running_ts = ts
                            running_run = (entry.path, s, pid)
                    else:
                        # Status says running but PID is dead or missing — orphaned
                        dead_pid_count += 1
                elif pid is not None:
                    # Has PID file but status is not running
                    if not self._is_pid_alive(pid):
                        dead_pid_count += 1
        except Exception:
            pass

        # Update resume button
        if running_run is not None:
            run_dir, s, pid = running_run
            st = s.get("status", "running")
            ep = s.get("epoch", 0)
            total = s.get("total_epochs", "?")
            btn_resume.label = f"Resume Training (E{ep}/{total}, {st})"
            btn_resume.disabled = False
            btn_resume.variant = "success"
            self.app._active_run_dir = run_dir
            # 有任务在跑，禁止新训练
            btn_train.disabled = True
        else:
            btn_resume.label = "Resume Training (none running)"
            btn_resume.disabled = True
            btn_resume.variant = "default"
            btn_train.disabled = False
            self.app._active_run_dir = None

        # Update kill button — show if any dead workers to clean up
        if dead_pid_count > 0:
            btn_kill.label = f"Kill All Workers ({dead_pid_count} dead)"
            btn_kill.disabled = False
            btn_kill.variant = "warning"
        else:
            btn_kill.label = "Kill All Workers"
            btn_kill.disabled = True
            btn_kill.variant = "default"

    def _kill_orphan_workers(self):
        """终止所有有 PID 文件的 Worker 进程并清理。"""
        import os
        import sys

        output_base = self.app.config.get("training", {}).get("output_dir", "./output")
        if not os.path.isdir(output_base):
            self.notify("No output directory found", severity="warning")
            return

        killed = 0
        for entry in os.scandir(output_base):
            if not entry.is_dir():
                continue
            pid_path = os.path.join(entry.path, "worker.pid")
            if not os.path.exists(pid_path):
                continue
            try:
                with open(pid_path, encoding="utf-8") as f:
                    pid = int(f.read().strip())
            except Exception:
                continue
            try:
                if sys.platform == "win32":
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.OpenProcess(0x0001, False, pid)
                    if handle:
                        kernel32.TerminateProcess(handle, 1)
                        kernel32.CloseHandle(handle)
                        killed += 1
                else:
                    import signal
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                os.remove(pid_path)
            except Exception:
                pass

        self.notify(f"Terminated {killed} worker(s)", severity="information")
        self._update_resume_button()

    def on_button_pressed(self, event: Button.Pressed):
        """Navigate to corresponding screen."""
        btn_id = event.button.id
        app = self.app

        if btn_id in ("btn-train", "btn-dataset"):
            # 检查是否有后台任务在跑
            if hasattr(app, "_active_run_dir") and app._active_run_dir:
                self.notify(
                    "A training task is already running. Resume or stop it first.",
                    severity="warning",
                )
                return
            app.push_screen("dataset_select")
        elif btn_id == "btn-infer":
            app.push_screen("inference_screen")
        elif btn_id == "btn-resume":
            app.push_screen("training_run")
        elif btn_id == "btn-config":
            app.push_screen("config_screen")
        elif btn_id == "btn-history":
            app.push_screen("history_view")
        elif btn_id == "btn-logs":
            app.push_screen("log_view")
        elif btn_id == "btn-kill-workers":
            self._kill_orphan_workers()
        elif btn_id == "btn-quit":
            app.exit()

    def action_nav_train(self):
        if hasattr(self.app, "_active_run_dir") and self.app._active_run_dir:
            self.notify(
                "A training task is already running. Resume or stop it first.",
                severity="warning",
            )
            return
        self.app.push_screen("dataset_select")

    def action_nav_infer(self):
        self.app.push_screen("inference_screen")

    def action_nav_dataset(self):
        if hasattr(self.app, "_active_run_dir") and self.app._active_run_dir:
            self.notify(
                "A training task is already running. Resume or stop it first.",
                severity="warning",
            )
            return
        self.app.push_screen("dataset_select")

    def action_nav_config(self):
        self.app.push_screen("config_screen")

    def action_nav_history(self):
        self.app.push_screen("history_view")

    def action_nav_logs(self):
        self.app.push_screen("log_view")

    def action_quit_app(self):
        self.app.exit()

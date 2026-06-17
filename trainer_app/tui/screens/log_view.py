"""日志查看页面。"""
import os
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, ListView, ListItem, Label


class LogViewScreen(Screen):
    """日志文件浏览页面。

    功能：
    - 列出所有任务日志文件夹
    - 点击查看日志内容
    - 显示日志文件夹信息
    """

    BINDINGS = [
        ("escape", "dismiss", "返回"),
        ("ctrl+r", "refresh", "刷新"),
        ("ctrl+v", "view", "查看"),
        ("ctrl+c", "clean", "清理"),
    ]

    CSS = """
    LogViewScreen {
        align: center middle;
    }
    #log-container {
        width: 80%;
        height: 90%;
        border: solid $primary;
        padding: 1 2;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    ListView {
        height: 1fr;
        border: solid $surface;
    }
    #log-content {
        height: 1fr;
        border: solid $surface;
        overflow-y: auto;
    }
    #log-info {
        color: $text-muted;
        padding: 1 0;
    }
    Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="log-container"):
            yield Static("Log File Manager", classes="section-title")
            yield Static("Task Log Folders:", classes="section-title")
            yield ListView(id="log-list")
            yield Static("", id="log-info")
            with Horizontal():
                yield Button("Refresh (Ctrl+R)", id="btn-refresh")
                yield Button("View (Ctrl+V)", id="btn-view")
                yield Button("Clean (Ctrl+C)", id="btn-clean")
                yield Button("Back (ESC)", id="btn-back")
        yield Footer()

    def on_mount(self):
        self._load_log_dirs()

    def _load_log_dirs(self):
        """加载日志文件夹列表。"""
        from ...log.manager import LogManager

        log_dir = self.app.config.get("logging", {}).get("log_dir", "./logs")
        mgr = LogManager(log_dir)
        dirs = mgr.list_task_dirs()

        list_view = self.query_one("#log-list", ListView)
        list_view.clear()

        if not dirs:
            list_view.append(ListItem(Label("(no log folders)")))
        else:
            for d in dirs:
                files = list(d.glob("*"))
                nc_files = [f for f in files if f.is_file()]
                info = f"{d.name}  ({len(nc_files)} files)"
                list_view.append(ListItem(Label(info)))

        self.query_one("#log-info", Static).update(
            f"Log root: {log_dir} | {len(dirs)} task folders"
        )

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id

        if btn_id == "btn-refresh":
            self._load_log_dirs()
        elif btn_id == "btn-view":
            self._view_log_content()
        elif btn_id == "btn-clean":
            self._clean_old_logs()
        elif btn_id == "btn-back":
            self.dismiss()

    def action_refresh(self):
        self._load_log_dirs()

    def action_view(self):
        self._view_log_content()

    def action_clean(self):
        self._clean_old_logs()

    def _get_selected_dir(self) -> Path | None:
        """获取当前选中的日志文件夹。"""
        list_view = self.query_one("#log-list", ListView)
        if list_view.index is None:
            return None

        from ...log.manager import LogManager
        log_dir = self.app.config.get("logging", {}).get("log_dir", "./logs")
        mgr = LogManager(log_dir)
        dirs = mgr.list_task_dirs()

        if list_view.index < len(dirs):
            return dirs[list_view.index]
        return None

    def _view_log_content(self):
        """查看选中日志文件夹内的 training.log。"""
        folder = self._get_selected_dir()
        if folder is None:
            self.notify("Please select a log folder first", severity="warning")
            return

        log_file = folder / "training.log"
        if not log_file.exists():
            self.notify(f"training.log not found: {log_file}")
            return

        try:
            content = log_file.read_text(encoding="utf-8")
            # 显示最后 100 行
            lines = content.strip().split("\n")
            preview = "\n".join(lines[-100:])
            self.notify(preview[:1000], title=f"日志预览: {folder.name}")
        except Exception as e:
            self.notify(f"Read failed: {e}", severity="error")

    def _clean_old_logs(self):
        """清理30天前的旧日志。"""
        from ...log.manager import LogManager
        log_dir = self.app.config.get("logging", {}).get("log_dir", "./logs")
        mgr = LogManager(log_dir)
        mgr.clean_old_logs(keep_days=30)
        self.notify("已清理30天前的旧日志", severity="information")
        self._load_log_dirs()

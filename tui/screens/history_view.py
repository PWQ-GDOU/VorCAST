"""历史训练记录浏览页面。"""
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, Input, Select, DataTable


class HistoryViewScreen(Screen):
    """训练历史记录浏览与查询。

    功能：
    - 表格列出所有历史训练记录
    - 按条件筛选（时间、数据集名、模型类型）
    - 点击查看详情
    - 导出 CSV
    """

    BINDINGS = [
        ("escape", "dismiss", "返回"),
        ("ctrl+f", "search", "搜索"),
        ("ctrl+r", "refresh", "刷新"),
        ("ctrl+d", "detail", "详情"),
        ("ctrl+e", "export_csv", "导出"),
        ("ctrl+p", "plots", "效果图"),
        ("delete", "delete_record", "删除"),
    ]

    CSS = """
    HistoryViewScreen {
        align: center middle;
    }
    #history-container {
        width: 90%;
        height: 90%;
        border: solid $primary;
        padding: 1 2;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    #filter-row {
        height: 3;
        margin-bottom: 1;
    }
    #filter-row Input {
        width: 30;
    }
    #filter-row Select {
        width: 20;
    }
    DataTable {
        height: 1fr;
        border: solid $surface;
    }
    #stat-bar {
        color: $text-muted;
        padding: 1 0;
    }
    Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="history-container"):
            yield Static("Training History", classes="section-title")

            with Horizontal(id="filter-row"):
                yield Input(placeholder="Filter by dataset name...", id="filter-dataset")
                yield Input(placeholder="Date from (YYYY-MM-DD)...", id="filter-date-from")
                yield Input(placeholder="Date to (YYYY-MM-DD)...", id="filter-date-to")
                yield Select(
                    [("All", "all"), ("Running", "running"),
                     ("Completed", "completed"), ("Interrupted", "interrupted"),
                     ("Failed", "failed")],
                    id="filter-status", value="all",
                )
                yield Button("Search", id="btn-search")

            yield DataTable(id="history-table", cursor_type="row")
            yield Static("0 records total", id="stat-bar")

            with Horizontal():
                yield Button("Refresh (Ctrl+R)", id="btn-refresh")
                yield Button("Detail (Ctrl+D)", id="btn-detail")
                yield Button("Export (Ctrl+E)", id="btn-export")
                yield Button("Plots (Ctrl+P)", id="btn-gen-plots")
                yield Button("Delete (DEL)", id="btn-delete")
                yield Button("Back (ESC)", id="btn-back")
        yield Footer()

    def on_mount(self):
        """初始化表格列和加载数据。"""
        table = self.query_one("#history-table", DataTable)
        table.add_columns("ID", "Task Name", "Start Time", "End Time", "Status",
                          "Dataset", "Epoch", "Best Loss", "Best Acc")
        self._load_data()

    def _load_data(self):
        """从数据库加载历史记录到表格。"""
        from ...history.storage import HistoryStorage
        from ...history.query import HistoryQuery

        storage = HistoryStorage()
        query = HistoryQuery(storage)

        # 获取筛选条件
        dataset = self.query_one("#filter-dataset", Input).value
        date_from = self.query_one("#filter-date-from", Input).value
        date_to = self.query_one("#filter-date-to", Input).value
        status = self.query_one("#filter-status", Select).value

        if any([dataset, date_from, date_to]) or (status and status != "all"):
            runs = query.filter_runs(
                dataset_name=dataset or None,
                date_from=date_from or None,
                date_to=date_to or None,
                status=status if status != "all" else None,
            )
        else:
            runs = query.list_runs()

        table = self.query_one("#history-table", DataTable)
        table.clear()

        for run in runs:
            table.add_row(
                str(run.get("id", "")),
                str(run.get("task_name", "")),
                str(run.get("start_time", ""))[:19],
                str(run.get("end_time", "") or "--")[:19],
                str(run.get("status", "")),
                str(run.get("dataset_name", "")),
                f"{run.get('current_epoch', 0)}/{run.get('total_epochs', 0)}",
                f"{run.get('best_loss', '--'):.4f}" if run.get("best_loss") else "--",
                f"{run.get('best_accuracy', '--'):.3f}" if run.get("best_accuracy") else "--",
            )

        self._stats = query.get_statistics()
        self.query_one("#stat-bar", Static).update(
            f"{len(runs)} records | "
            f"Total runs: {self._stats['total_runs']} | "
            f"Completed: {self._stats['completed_runs']} | "
            f"Global best Loss: {self._stats.get('best_loss_overall', '--')}"
        )

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id

        if btn_id == "btn-search":
            self._load_data()
        elif btn_id == "btn-refresh":
            self._load_data()
        elif btn_id == "btn-detail":
            self._show_detail()
        elif btn_id == "btn-export":
            self._export_csv()
        elif btn_id == "btn-gen-plots":
            self._generate_plots()
        elif btn_id == "btn-delete":
            self._delete_selected()
        elif btn_id == "btn-back":
            self.dismiss()

    def action_search(self):
        self._load_data()

    def action_refresh(self):
        self._load_data()

    def action_detail(self):
        self._show_detail()

    def action_export_csv(self):
        self._export_csv()

    def action_plots(self):
        self._generate_plots()

    def action_delete_record(self):
        self._delete_selected()

    def _get_selected_run_id(self) -> int | None:
        """获取当前选中行的 run_id。"""
        table = self.query_one("#history-table", DataTable)
        row = table.cursor_row
        if row is not None and len(table.rows) > 0:
            cell = table.get_cell_at((row, 0))
            return cell
        return None

    def _show_detail(self):
        """显示选中记录的详情。"""
        row = self._get_selected_run_id()
        if row is None:
            self.notify("Please select a record first", severity="warning")
            return

        from ...history.storage import HistoryStorage
        from ...history.query import HistoryQuery

        storage = HistoryStorage()
        query = HistoryQuery(storage)

        run = storage.get_run(int(row))
        if run is None:
            self.notify("Record not found")
            return

        # Epoch 级指标摘要
        summary = query.get_metrics_summary(int(row))
        def _fmt_metric(val, precision=4):
            if val is None:
                return "--"
            return f"{val:.{precision}f}"

        summary_str = "\n".join(
            f"  Epoch {s['epoch']} [{s['phase']}]: "
            f"Loss={_fmt_metric(s['avg_loss'])} Acc={_fmt_metric(s['avg_accuracy'], 3)}"
            for s in summary[:20]
        )

        detail = (
            f"Task: {run.get('task_name')}\n"
            f"Status: {run.get('status')}\n"
            f"Dataset: {run.get('dataset_name')}\n"
            f"Model: {run.get('model_type')}\n"
            f"Time: {run.get('start_time', '')} ~ {run.get('end_time', '')}\n"
            f"Device: {run.get('device')}\n"
            f"Log: {run.get('log_dir')}\n"
            f"Best Loss: {run.get('best_loss')}\n"
            f"Metrics Summary:\n{summary_str}"
        )
        self.notify(detail[:500], title=f"Run #{row} Details")

    def _export_csv(self):
        """导出选中记录为 CSV。"""
        row = self._get_selected_run_id()
        if row is None:
            self.notify("Please select a record first", severity="warning")
            return

        from ...history.storage import HistoryStorage
        from ...history.query import HistoryQuery

        storage = HistoryStorage()
        query = HistoryQuery(storage)
        path = f"./export_run_{row}.csv"
        query.export_metrics_csv(int(row), path)
        self.notify(f"Exported to {path}")

    def _generate_plots(self):
        """根据选中的历史记录生成效果图。"""
        row = self._get_selected_run_id()
        if row is None:
            self.notify("Please select a record first", severity="warning")
            return

        from ...history.storage import HistoryStorage
        from ...utils.visualization import generate_all_plots

        storage = HistoryStorage()
        run = storage.get_run(int(row))
        if run is None:
            self.notify("Record not found")
            return

        metrics = storage.get_run_metrics(int(row))
        if not metrics:
            self.notify("No metrics data for this record")
            return

        # 转换为 monitor 兼容格式
        history = {"train": [], "val": []}
        for m in metrics:
            phase = m.get("phase", "train")
            history.setdefault(phase, []).append({
                "epoch": m["epoch"],
                "batch": m.get("batch"),
                "loss": m.get("loss"),
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "lr": m.get("lr"),
            })
        monitor_data = {"history": history, "elapsed": 0}

        log_dir = run.get("log_dir") or "./output_plots"
        total_epochs = run.get("total_epochs", 0)

        results = generate_all_plots(
            monitor_data=monitor_data,
            output_dir=log_dir,
            best_loss=run.get("best_loss") or 0,
            best_epoch=run.get("current_epoch", 0),
            total_epochs=total_epochs,
            elapsed=0,
            device=run.get("device", ""),
        )
        self.notify(f"已生成 {len(results)} 个效果图 → {log_dir}/plots/")

    def _delete_selected(self):
        """删除选中的训练记录。"""
        row = self._get_selected_run_id()
        if row is None:
            self.notify("Please select a record first", severity="warning")
            return

        from ...history.storage import HistoryStorage
        storage = HistoryStorage()
        storage.delete_run(int(row))
        self.notify(f"已删除 Run #{row}")
        self._load_data()

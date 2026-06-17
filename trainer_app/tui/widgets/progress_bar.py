"""训练进度条组件。"""
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, ProgressBar
from textual.reactive import reactive


class TrainingProgress(Vertical):
    """训练进度显示组件。

    显示当前 epoch/总 epoch、完成百分比、已用时间。
    """

    epoch = reactive(0)
    total_epochs = reactive(100)
    batch = reactive(0)
    total_batches = reactive(0)
    elapsed = reactive("00:00:00")

    def compose(self) -> ComposeResult:
        yield Static("Training Progress", classes="section-title")
        yield Static("Epoch: 0/100", id="epoch-label")
        yield ProgressBar(total=100, show_eta=False, id="epoch-progress")
        yield Static("Batch: 0/0", id="batch-label")
        yield ProgressBar(total=100, show_eta=False, id="batch-progress")
        yield Static("Elapsed: 00:00:00", id="elapsed-label")

    def watch_epoch(self, epoch: int):
        if not self.is_mounted:
            return
        self.query_one("#epoch-label", Static).update(
            f"Epoch: {epoch}/{self.total_epochs}"
        )
        if self.total_epochs > 0:
            self.query_one("#epoch-progress", ProgressBar).update(
                progress=min(epoch / self.total_epochs * 100, 100)
            )

    def watch_total_epochs(self, total: int):
        if not self.is_mounted:
            return
        self.query_one("#epoch-label", Static).update(
            f"Epoch: {self.epoch}/{total}"
        )
        self.query_one("#epoch-progress", ProgressBar).update(total=total)

    def watch_batch(self, batch: int):
        if not self.is_mounted:
            return
        self.query_one("#batch-label", Static).update(
            f"Batch: {batch}/{self.total_batches}"
        )
        if self.total_batches > 0:
            self.query_one("#batch-progress", ProgressBar).update(
                progress=min(batch / self.total_batches * 100, 100)
            )

    def watch_total_batches(self, total: int):
        if not self.is_mounted:
            return
        self.query_one("#batch-label", Static).update(
            f"Batch: {self.batch}/{total}"
        )
        self.query_one("#batch-progress", ProgressBar).update(total=total)

    def watch_elapsed(self, elapsed: str):
        if not self.is_mounted:
            return
        self.query_one("#elapsed-label", Static).update(f"Elapsed: {elapsed}")

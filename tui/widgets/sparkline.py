"""ASCII sparkline widget for terminal-based metric visualization."""
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive


class Sparkline(Vertical):
    """Draw a mini line chart using Unicode block characters (▁▂▃▄▅▆▇█).

    Shows a running history of a metric with label and current value.
    """

    values = reactive(list)
    label = reactive("metric")
    max_points = 60

    BLOCKS = " ▁▂▃▄▅▆▇█"

    def __init__(self, label: str = "metric", **kwargs):
        super().__init__(**kwargs)
        self.label = label
        safe = "".join(c for c in label if c.isascii() and (c.isalnum() or c in "-_"))
        self._safe_id = "spark-" + safe[:20] if safe else "spark-metric"

    def compose(self) -> ComposeResult:
        yield Static("", id=self._safe_id)

    def on_mount(self):
        self.refresh_display()

    def watch_values(self, vals: list):
        self.refresh_display()

    def watch_label(self, val: str):
        self.refresh_display()

    def push(self, value: float):
        """Add a value and redraw."""
        v = list(self.values)
        v.append(value)
        if len(v) > self.max_points:
            v = v[-self.max_points:]
        self.values = v

    def refresh_display(self):
        if not self.is_mounted:
            return
        vals = list(self.values)
        if not vals:
            return

        mn = min(vals)
        mx = max(vals)
        rng = mx - mn if mx > mn else 1.0

        # Normalize to 0-8 block index
        bars = []
        for v in vals:
            idx = min(8, max(0, int((v - mn) / rng * 8)))
            bars.append(self.BLOCKS[idx])

        line = "".join(bars)
        cur = vals[-1]
        mn_str = f"{mn:.4f}"
        mx_str = f"{mx:.4f}"
        cur_str = f"{cur:.4f}"

        text = f"{self.label}: [{mn_str} {line} {mx_str}] cur={cur_str}"
        self.query_one(Static).update(text)


class HistoryTable(Vertical):
    """Compact table showing epoch-by-epoch training vs validation metrics."""

    rows = reactive(list)  # list of dicts

    def compose(self) -> ComposeResult:
        yield Static("", id="history-table-content")

    def watch_rows(self, rows: list):
        lines = []
        lines.append("Epoch │ Train Loss │ Train Acc │ Val Loss │ Val Acc │   LR")
        lines.append("──────┼────────────┼───────────┼──────────┼─────────┼──────")
        for r in rows[-20:]:
            ep = r.get("epoch", "?")
            tl = r.get("train_loss", 0)
            ta = r.get("train_acc", 0)
            vl = r.get("val_loss", 0) if r.get("val_loss") else "--"
            va = r.get("val_acc", 0) if r.get("val_acc") else "--"
            lr = r.get("lr", 0)
            lines.append(
                f"  {ep:3d} │ {tl:8.4f} │ {ta:7.3f} │ {vl:>8} │ {va:>7} │ {lr:.2e}"
            )
        self.query_one(Static).update("\n".join(lines))

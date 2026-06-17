"""实时指标面板组件。"""
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static
from textual.reactive import reactive


class MetricCard(Static):
    """单个指标卡片。"""
    value = reactive("--")

    def __init__(self, label: str, metric_id: str):
        super().__init__()
        self._label = label
        self._metric_id = metric_id

    def render(self) -> str:
        return f"{self._label}\n{self.value}"

    def watch_value(self, value: str):
        self.refresh()


class MetricsPanel(Vertical):
    """实时训练指标面板。

    展示四行指标：
    Row1: Loss, Accuracy, IoU/CSI
    Row2: F1, Dice, Grad Loss
    Row3: CSI Loss, FSS Loss, LPIPS Loss
    Row4: AUC Loss, LR, Phase
    """

    loss = reactive("--")
    accuracy = reactive("--")
    iou = reactive("--")
    f1 = reactive("--")
    dice = reactive("--")
    grad_loss = reactive("--")
    csi_loss = reactive("--")
    fcsi_loss = reactive("--")
    lpips_loss = reactive("--")
    auc_loss = reactive("--")
    lr = reactive("--")
    phase = reactive("train")
    total_loss = reactive("--")

    def compose(self) -> ComposeResult:
        yield Static("Training Metrics", classes="section-title")
        yield Static(f"Phase: {self.phase}", id="phase-label")
        with Horizontal(id="metrics-row1"):
            yield MetricCard("Loss", "loss")
            yield MetricCard("Accuracy", "accuracy")
            yield MetricCard("IoU/CSI", "iou")
        with Horizontal(id="metrics-row2"):
            yield MetricCard("F1", "f1")
            yield MetricCard("Dice", "dice")
            yield MetricCard("Grad Loss", "grad_loss")
        with Horizontal(id="metrics-row3"):
            yield MetricCard("CSI Loss", "csi_loss")
            yield MetricCard("FSS Loss", "fcsi_loss")
            yield MetricCard("LPIPS", "lpips_loss")
        with Horizontal(id="metrics-row4"):
            yield MetricCard("AUC Loss", "auc_loss")
            yield MetricCard("LR", "lr")
            yield MetricCard("Total Loss", "total_loss")

    def watch_loss(self, val: str):
        self._update_card("loss", val)

    def watch_accuracy(self, val: str):
        self._update_card("accuracy", val)

    def watch_iou(self, val: str):
        self._update_card("iou", val)

    def watch_f1(self, val: str):
        self._update_card("f1", val)

    def watch_dice(self, val: str):
        self._update_card("dice", val)

    def watch_grad_loss(self, val: str):
        self._update_card("grad_loss", val)

    def watch_csi_loss(self, val: str):
        self._update_card("csi_loss", val)

    def watch_fcsi_loss(self, val: str):
        self._update_card("fcsi_loss", val)

    def watch_lpips_loss(self, val: str):
        self._update_card("lpips_loss", val)

    def watch_auc_loss(self, val: str):
        self._update_card("auc_loss", val)

    def watch_lr(self, val: str):
        self._update_card("lr", val)

    def watch_total_loss(self, val: str):
        self._update_card("total_loss", val)

    def watch_phase(self, phase: str):
        if self.is_mounted:
            self.query_one("#phase-label", Static).update(f"Phase: {phase}")

    def _update_card(self, metric_id: str, value: str):
        for card in self.query(MetricCard):
            if card._metric_id == metric_id:
                card.value = value

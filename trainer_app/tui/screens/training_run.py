"""Detailed training monitor screen — full-screen dedicated view."""
import json
import os
import sys
import subprocess
import time
import threading
from datetime import datetime
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, Log
from textual.reactive import reactive

from ..widgets.progress_bar import TrainingProgress
from ..widgets.gpu_monitor import GPUMonitor
from ..widgets.sparkline import Sparkline, HistoryTable


class TrainingRunScreen(Screen):
    """Full-screen training monitor with detailed progress display.

    Layout:
      Top:    Status bar (run name, status, epoch, elapsed, ETA)
      Row 1:  Progress bars (epoch + batch) | Loss sparkline
      Row 2:  Best metrics so far | Current metrics + training/validation comparison
      Row 3:  Epoch history table (scrollable last 20 epochs)
      Bottom: GPU status | Log stream
      Dock:   Control buttons + shortcut bar
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+s", "start", "Start"),
        ("ctrl+p", "pause", "Pause"),
        ("ctrl+r", "resume", "Resume"),
        ("ctrl+x", "stop", "Stop"),
        ("ctrl+b", "background", "Background"),
    ]

    CSS = """
    TrainingRunScreen {
        layout: vertical;
    }

    #status-bar {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary;
        dock: top;
    }
    #status-run-name {
        text-style: bold;
        color: $accent;
    }
    #status-state {
        text-style: bold;
    }
    #status-eta {
        color: $warning;
    }

    #main-grid {
        height: 1fr;
        layout: grid;
        grid-size: 2 3;
        grid-columns: 1fr 1fr;
        grid-rows: auto auto 1fr;
        padding: 0 1;
    }

    #progress-cell {
        border-right: dashed $surface;
        padding: 0 1 0 0;
    }
    #spark-cell {
        padding: 0 0 0 1;
    }
    #best-cell {
        border-right: dashed $surface;
        padding: 0 1 0 0;
    }
    #metrics-cell {
        padding: 0 0 0 1;
    }
    #history-cell {
        column-span: 2;
        padding-top: 1;
        border-top: dashed $surface;
    }

    #bottom-bar {
        dock: bottom;
        height: auto;
        min-height: 8;
        max-height: 15;
        border-top: solid $primary;
        padding: 0 1;
    }
    #log-stream {
        height: 1fr;
        min-height: 4;
    }

    #control-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        align: center middle;
    }
    #shortcut-bar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        align: center middle;
        color: $text-muted;
    }

    Button {
        margin: 0 1;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    """

    status_text = reactive("Ready...")
    is_training = reactive(False)
    is_paused = reactive(False)

    def compose(self) -> ComposeResult:
        yield Header()

        # Status bar
        with Horizontal(id="status-bar"):
            yield Static("Run: --", id="status-run-name")
            yield Static("", id="status-state")
            yield Static("", id="status-epoch-progress")
            yield Static("", id="status-elapsed")
            yield Static("", id="status-eta")

        # Main grid
        with Container(id="main-grid"):
            with Vertical(id="progress-cell"):
                yield Static("Training Progress", classes="section-title")
                yield TrainingProgress(id="progress")
                yield GPUMonitor(id="gpu")

            with Vertical(id="spark-cell"):
                yield Static("Loss Trend (sparkline)", classes="section-title")
                yield Sparkline(label="Loss", id="spark-loss")
                yield Sparkline(label="Acc", id="spark-acc")

            with Vertical(id="best-cell"):
                yield Static("Best Metrics So Far", classes="section-title")
                yield Static("No data yet", id="best-display")

            with Vertical(id="metrics-cell"):
                yield Static("Current Metrics", classes="section-title")
                yield Static("No data yet", id="cur-display")

            with Vertical(id="history-cell"):
                yield Static("Epoch History (recent 20)", classes="section-title")
                yield HistoryTable(id="history-table")

        # Bottom: log
        with Vertical(id="bottom-bar"):
            yield Log(id="log-stream", auto_scroll=True, max_lines=300)
            yield GPUMonitor(id="gpu-bottom")

        # Controls
        with Horizontal(id="control-bar"):
            yield Button("Start (Ctrl+S)", id="btn-start-training", variant="primary")
            yield Button("Pause (Ctrl+P)", id="btn-pause", disabled=True)
            yield Button("Resume (Ctrl+R)", id="btn-resume", disabled=True)
            yield Button("Stop (Ctrl+X)", id="btn-stop", disabled=True)
            yield Button("Background (Ctrl+B)", id="btn-background", disabled=True)
            yield Button("Export", id="btn-export", disabled=True)
            yield Button("Back (ESC)", id="btn-back")
        yield Static(
            "Ctrl+S=Start  Ctrl+P=Pause  Ctrl+R=Resume  Ctrl+X=Stop  Ctrl+B=Background  ESC=Back",
            id="shortcut-bar",
        )
        yield Footer()

    # ── lifecycle ──────────────────────────────────────────────

    def on_mount(self):
        self.query_one("#gpu", GPUMonitor).refresh_status()

        # Reconnect to running worker — only if it's actually alive
        if hasattr(self.app, "_active_run_dir") and self.app._active_run_dir:
            run_dir = self.app._active_run_dir
            status_path = os.path.join(run_dir, "status.json")
            if not os.path.exists(status_path):
                self.app._active_run_dir = None
                return

            try:
                with open(status_path, encoding="utf-8") as f:
                    s = json.load(f)
            except Exception:
                self.app._active_run_dir = None
                return

            worker_status = s.get("status", "")
            # Only reconnect to actively running/paused workers
            if worker_status not in ("running", "starting", "paused"):
                self.app._active_run_dir = None
                return

            # Read PID file
            pid_path = os.path.join(run_dir, "worker.pid")
            if not os.path.exists(pid_path):
                self.app._active_run_dir = None
                return
            try:
                with open(pid_path, encoding="utf-8") as f:
                    pid = int(f.read().strip())
            except Exception:
                self.app._active_run_dir = None
                return

            # Check PID is actually alive
            if not self._is_pid_alive(pid):
                self.app._active_run_dir = None
                return

            self._worker_pid = pid
            self._run_dir = run_dir
            self.is_training = True

            if worker_status == "paused":
                self.is_paused = True
                self.status_text = "Training PAUSED (reconnected)"
                self._set_button_states(started=True, paused=True)
            else:
                self.status_text = "Training RUNNING (reconnected)"
                self._set_button_states(started=True, paused=False)

            self._log(f"Reconnected to worker PID {self._worker_pid}")
            self._log(f"Status: {worker_status}, Epoch: {s.get('epoch')}/{s.get('total_epochs')}")

            self._epoch_history = []
            self._loss_vals = []
            self._acc_vals = []
            self._best_loss = float("inf")
            self._best_csi = 0.0
            self._best_acc = 0.0
            self._best_epoch_loss = 0
            self._best_epoch_csi = 0
            self._best_epoch_acc = 0
            self._last_seen_epoch = 0
            self._last_seen_batch = 0
            self._load_epoch_history()
            self.set_interval(0.5, self._refresh_ui)

    # ── button dispatch ────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-start-training":
            self.action_start()
        elif bid == "btn-pause":
            self.action_pause()
        elif bid == "btn-resume":
            self.action_resume()
        elif bid == "btn-stop":
            self.action_stop()
        elif bid == "btn-background":
            self.action_background()
        elif bid == "btn-export":
            self._export_results()
        elif bid == "btn-back":
            self.action_go_back()

    # ── actions ────────────────────────────────────────────────

    def action_start(self):
        if not self.is_training:
            self._start_training()

    def action_pause(self):
        if hasattr(self, "_run_dir") and self._run_dir:
            self._write_control("pause")
            self.is_paused = True
            self._set_button_states(started=True, paused=True)
            self.status_text = "Training PAUSED"
            self._log("Pause command sent to worker")

    def action_resume(self):
        if hasattr(self, "_run_dir") and self._run_dir:
            self._write_control("resume")
            self.is_paused = False
            self._set_button_states(started=True, paused=False)
            self.status_text = "Training RESUMED"
            self._log("Resume command sent to worker")

    def action_stop(self):
        if hasattr(self, "_run_dir") and self._run_dir:
            self._write_control("stop")
            self._stop_requested = True
            self._stopping_since = time.time()
            self._set_button_states(started=True, paused=True)  # Keep buttons disabled-ish
        self.status_text = "STOPPING... (waiting for worker)"
        self._log("Stop command sent to worker — waiting for batch to finish...")

    def action_background(self):
        self.app._active_run_dir = self._run_dir if hasattr(self, "_run_dir") else None
        self._log("Training continues in background (separate process).")
        self.notify("Training running in background", severity="information")
        self.dismiss()

    def action_go_back(self):
        if self.is_training and hasattr(self, "_worker_pid"):
            if self._is_pid_alive(self._worker_pid):
                self.action_background()
                return
        self.dismiss()

    # ── start ──────────────────────────────────────────────────

    def _start_training(self):
        self.is_training = True
        self.status_text = "Initializing..."

        data_cfg = self.app.config.get("data", {})
        training_cfg = self.app.config.get("training", {})

        run_name = training_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = training_cfg.get("output_dir", "./output")
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        self._run_dir = run_dir

        # Set run-specific paths in config
        self.app.config["training"]["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
        self.app.config["logging"]["log_dir"] = os.path.join(run_dir, "logs")

        # Save config so worker can load it
        from ...utils.config import save_config
        config_path = os.path.join(run_dir, "config.yaml")
        save_config(self.app.config, config_path)

        self._log(f"Run dir: {run_dir}")
        self._log(f"Config saved: {config_path}")
        self._log(f"Data: {data_cfg.get('dataset1_path', '?')}, {data_cfg.get('dataset2_path', '?')}")
        self._log(f"Batch size: {training_cfg.get('batch_size', '?')}")
        self._log(f"Epochs: {training_cfg.get('epochs', '?')}, LR: {training_cfg.get('learning_rate', '?')}")
        self._log(f"AMP: {training_cfg.get('use_amp', True)}")

        # Update UI
        self.query_one("#status-run-name", Static).update(f"Run: {run_name}")

        # Pre-set total batches from config for UI
        progress = self.query_one("#progress", TrainingProgress)
        progress.total_epochs = training_cfg.get("epochs", 100)
        progress.epoch = 0
        progress.batch = 0

        # Spawn worker subprocess (detached from console — survives terminal close)
        if sys.platform == "win32":
            creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creationflags = 0

        worker_cmd = [
            sys.executable, "-m", "trainer_app.main",
            "--worker",
            "--config", config_path,
            "--run-name", run_name,
        ]
        self._log(f"Spawning: {' '.join(worker_cmd)}")

        try:
            self._worker_proc = subprocess.Popen(
                worker_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._worker_pid = self._worker_proc.pid
            self._log(f"Worker PID: {self._worker_pid}")
            self.app._active_run_dir = run_dir
        except Exception as exc:
            self._log(f"ERROR spawning worker: {exc}")
            self.status_text = f"ERROR: {exc}"
            return

        self._epoch_history = []
        self._loss_vals = []
        self._acc_vals = []
        self._best_loss = float("inf")
        self._best_acc = 0.0
        self._best_epoch_loss = 0
        self._best_epoch_acc = 0
        self._epoch_start_time = time.time()
        self._last_seen_epoch = 0
        self._last_seen_batch = 0

        self._set_button_states(started=True, paused=False)
        self.status_text = f"RUNNING [{run_name}] (background process)"
        self.set_interval(0.5, self._refresh_ui)

    # ── metric callback ────────────────────────────────────────

    def _on_metric_update(self, metrics: dict):
        typ = metrics.get("type", "")
        epoch = metrics.get("epoch", 0)
        total_epochs = metrics.get("total_epochs", self.app.config["training"]["epochs"])

        if typ == "status":
            self._log(metrics.get("message", ""))
            return

        if typ == "batch":
            batch_idx = metrics.get("batch", 0)
            total_batches = metrics.get("total_batches", 0)
            self.query_one("#progress", TrainingProgress).batch = batch_idx
            self.query_one("#progress", TrainingProgress).total_batches = total_batches

            # Detailed per-batch status
            loss_v = metrics.get("loss", 0)
            acc_v = metrics.get("accuracy", 0)

            # Update status bar
            elapsed = time.time() - self._epoch_start_time if hasattr(self, "_epoch_start_time") else 0
            batch_pct = (batch_idx / total_batches * 100) if total_batches else 0
            self.query_one("#status-state", Static).update(
                f"[bold yellow]Training[/] Epoch {epoch}/{total_epochs}  "
                f"Batch {batch_idx}/{total_batches} ({batch_pct:.0f}%)"
            )
            self.query_one("#status-elapsed", Static).update(
                f"Elapsed: {elapsed:.0f}s"
            )

            # Update sparklines every batch
            self._loss_vals.append(loss_v)
            self._acc_vals.append(acc_v)
            self.query_one("#spark-loss", Sparkline).push(loss_v)
            self.query_one("#spark-acc", Sparkline).push(acc_v)

            # Current metrics display
            csi_val = metrics.get('iou', metrics.get('csi', 0))
            cur = (
                f"  Total Loss:   {metrics.get('total_loss', 0):.4f}\n"
                f"  MAE Loss:     {loss_v:.4f}\n"
                f"  Accuracy:     {acc_v:.4f}\n"
                f"  IoU/CSI:      {csi_val:.4f}\n"
                f"  F1:           {metrics.get('f1', 0):.4f}\n"
                f"  Precision:    {metrics.get('precision', 0):.4f}\n"
                f"  Recall:       {metrics.get('recall', 0):.4f}\n"
                f"  Grad Loss:    {metrics.get('grad_loss', 0):.4f}\n"
                f"  CSI Loss:     {metrics.get('csi_loss', 0):.4f}\n"
                f"  FSS Loss:     {metrics.get('fcsi_loss', 0):.4f}\n"
                f"  LPIPS:        {metrics.get('lpips_loss', 0):.4f}\n"
                f"  AUC Loss:     {metrics.get('auc_loss', 0):.4f}\n"
                f"  LR:           {metrics.get('lr', 0):.2e}\n"
                f"  Batch:        {batch_idx}/{total_batches}"
            )
            self.query_one("#cur-display", Static).update(cur)

            if batch_idx % 5 == 0:
                self._log(
                    f"[train] E{epoch:3d} B{batch_idx:4d}/{total_batches} | "
                    f"Loss={loss_v:.4f} Acc={acc_v:.3f} IoU={csi_val:.3f} F1={metrics.get('f1',0):.3f} "
                    f"CSI_L={metrics.get('csi_loss',0):.3f} FSS_L={metrics.get('fcsi_loss',0):.3f}"
                )

        elif typ == "epoch":
            phase = metrics.get("phase", "train")
            epoch_end = time.time()
            epoch_dur = epoch_end - self._epoch_start_time if hasattr(self, "_epoch_start_time") else 0
            self._epoch_start_time = time.time()

            loss_v = metrics.get("loss", 0)
            acc_v = metrics.get("accuracy", 0)

            self.query_one("#progress", TrainingProgress).epoch = epoch
            self.query_one("#progress", TrainingProgress).total_epochs = total_epochs

            # Track best metrics
            csi_val_e = metrics.get('iou', metrics.get('csi', 0))
            if phase == "val" and loss_v < self._best_loss:
                self._best_loss = loss_v
                self._best_epoch_loss = epoch
            if phase == "val" and csi_val_e > self._best_csi:
                self._best_csi = csi_val_e
                self._best_epoch_csi = epoch
            if phase == "val" and acc_v > self._best_acc:
                self._best_acc = acc_v
                self._best_epoch_acc = epoch

            # Update best display
            best_text = (
                f"  Best Loss:    {self._best_loss:.4f}  (epoch {self._best_epoch_loss})\n"
                f"  Best CSI:     {self._best_csi:.4f}  (epoch {self._best_epoch_csi})\n"
                f"  Best Acc:     {self._best_acc:.4f}  (epoch {self._best_epoch_acc})"
            )
            self.query_one("#best-display", Static).update(best_text)

            # Epoch history
            row = {
                "epoch": epoch,
                "train_loss": metrics.get("loss", 0) if phase == "train" else None,
                "train_acc": metrics.get("accuracy", 0) if phase == "train" else None,
                "train_csi": csi_val_e if phase == "train" else None,
                "train_f1": metrics.get("f1", 0) if phase == "train" else None,
                "val_loss": metrics.get("loss", 0) if phase == "val" else None,
                "val_acc": metrics.get("accuracy", 0) if phase == "val" else None,
                "val_csi": csi_val_e if phase == "val" else None,
                "val_f1": metrics.get("f1", 0) if phase == "val" else None,
                "lr": metrics.get("lr", 0),
            }
            # Merge with existing row for this epoch
            existing = next((r for r in self._epoch_history if r["epoch"] == epoch), None)
            if existing:
                existing.update({k: v for k, v in row.items() if v is not None})
            else:
                self._epoch_history.append(row)
            self.query_one("#history-table", HistoryTable).rows = self._epoch_history

            dur_str = f"{epoch_dur:.1f}s" if epoch_dur > 0 else "?"
            self._log(
                f"[{phase}] Epoch {epoch}/{total_epochs} done ({dur_str}) | "
                f"Loss={loss_v:.4f} Acc={acc_v:.3f} CSI={csi_val_e:.3f} F1={metrics.get('f1',0):.3f}"
            )
            self.query_one("#status-state", Static).update(
                f"[bold green]{phase.upper()}[/] Epoch {epoch}/{total_epochs} complete"
            )

    # ── periodic refresh ───────────────────────────────────────

    def _refresh_ui(self):
        if not hasattr(self, "_run_dir") or not self._run_dir:
            return

        status = self._read_status()
        if status is None:
            return

        worker_status = status.get("status", "unknown")
        epoch = status.get("epoch", 0)
        total_epochs = status.get("total_epochs", 0)
        batch = status.get("batch", 0)
        total_batches = status.get("total_batches", 0)
        last_loss = status.get("last_loss")
        last_accuracy = status.get("last_accuracy")
        message = status.get("message", "")

        # Progress bars
        progress_w = self.query_one("#progress", TrainingProgress)
        if total_batches > 0:
            progress_w.total_batches = total_batches
        if total_epochs > 0:
            progress_w.total_epochs = total_epochs
        progress_w.epoch = epoch
        progress_w.batch = batch

        # Elapsed
        start_time = status.get("timestamp", time.time())
        elapsed = time.time() - start_time if worker_status == "running" else 0
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        progress_w.elapsed = f"{h:02d}:{m:02d}:{s:02d}"
        self.query_one("#status-elapsed", Static).update(f"Elapsed: {h:02d}:{m:02d}:{s:02d}")

        # ETA
        self.query_one("#status-epoch-progress", Static).update(
            f"Epoch: {epoch}/{total_epochs}"
        )
        if epoch > 0 and total_epochs > 0 and elapsed > 1:
            sec_per_epoch = elapsed / epoch
            remaining = sec_per_epoch * (total_epochs - epoch)
            rh, rm = int(remaining // 3600), int((remaining % 3600) // 60)
            self.query_one("#status-eta", Static).update(f"ETA: {rh:02d}:{rm:02d}:00")
        else:
            self.query_one("#status-eta", Static).update("ETA: --")

        # Status text
        status_label = {
            "starting": "[bold yellow]STARTING[/]",
            "running": "[bold yellow]RUNNING[/]",
            "completed": "[bold green]COMPLETE[/]",
            "failed": "[bold red]FAILED[/]",
            "interrupted": "[bold red]INTERRUPTED[/]",
        }.get(worker_status, f"[bold]{worker_status.upper()}[/]")
        self.query_one("#status-state", Static).update(
            f"{status_label}  {message}"
        )

        # Update sparklines and current metrics when new data arrives
        import math
        loss_ok = last_loss is not None and not (isinstance(last_loss, float) and math.isnan(last_loss))
        acc_ok = last_accuracy is not None and not (isinstance(last_accuracy, float) and math.isnan(last_accuracy))
        if loss_ok and (batch != self._last_seen_batch or epoch != self._last_seen_epoch):
            self._last_seen_batch = batch
            self._last_seen_epoch = epoch
            self._loss_vals.append(last_loss)
            self.query_one("#spark-loss", Sparkline).push(last_loss)
            if acc_ok:
                self._acc_vals.append(last_accuracy)
                self.query_one("#spark-acc", Sparkline).push(last_accuracy)
            acc_str = f"{last_accuracy:.4f}" if acc_ok else "--"
            cur = (
                f"  Loss:     {last_loss:.4f}\n"
                f"  Accuracy: {acc_str}"
            )
            self.query_one("#cur-display", Static).update(cur)

        # Best metrics
        best_loss = status.get("best_loss")
        if best_loss is not None and best_loss < self._best_loss:
            self._best_loss = best_loss
            self._best_epoch_loss = epoch
        best_text = (
            f"  Best Loss:    {self._best_loss:.4f}  (epoch {self._best_epoch_loss})\n"
            f"  Best CSI:     {self._best_csi:.4f}  (epoch {self._best_epoch_csi})\n"
            f"  Best Acc:     {self._best_acc:.4f}  (epoch {self._best_epoch_acc})"
        )
        self.query_one("#best-display", Static).update(best_text)

        # Log epoch transitions
        if epoch > 0 and epoch != self._last_seen_epoch and self._last_seen_epoch > 0:
            # Epoch changed — read from SQLite for history table
            self._load_epoch_history()
            self._log(f"Epoch {self._last_seen_epoch} → {epoch}")

        # GPU
        self.query_one("#gpu", GPUMonitor).refresh_status()

        # Handle stopping state
        stopping = getattr(self, "_stop_requested", False)
        if stopping and worker_status in ("completed", "failed", "interrupted"):
            self._stop_requested = False
            self.is_training = False
            self._set_button_states(started=False, paused=False)
            self.status_text = "STOPPED"
            self.query_one("#status-state", Static).update("[bold red]STOPPED[/]")
            self._log("Worker confirmed stop")
        elif stopping and time.time() - getattr(self, "_stopping_since", time.time()) > 30:
            # Worker didn't respond to control.json — force kill
            self._log("Worker not responding, force killing...")
            if hasattr(self, "_worker_proc"):
                self._worker_proc.kill()
            self._stop_requested = False
            self.is_training = False
            self._set_button_states(started=False, paused=False)
            self.status_text = "STOPPED (forced)"
            self.query_one("#status-state", Static).update("[bold red]STOPPED (forced)[/]")
        elif stopping:
            self.query_one("#status-state", Static).update(
                "[bold yellow]STOPPING...[/] waiting for batch to finish"
            )

        # Terminal states
        if not stopping and worker_status in ("completed", "failed", "interrupted"):
            if self.is_training:
                self.is_training = False
                self._set_button_states(started=False, paused=False)
                self.status_text = worker_status.upper()
                if worker_status == "completed":
                    self.query_one("#status-state", Static).update("[bold green]COMPLETE[/]")
                    self.query_one("#btn-export", Button).disabled = False
                    self._log("Training completed!")
                    self._load_epoch_history()
                elif worker_status == "failed":
                    self.query_one("#status-state", Static).update(
                        f"[bold red]FAILED: {message}[/]"
                    )
                    self._log(f"Training failed: {message}")

        # Check worker liveness
        if worker_status == "running" and hasattr(self, "_worker_pid"):
            if not self._is_pid_alive(self._worker_pid):
                self._log("WARNING: Worker process not found — may have crashed")
                self.query_one("#status-state", Static).update(
                    "[bold red]WORKER LOST[/] — process not found"
                )

    # ── helpers ────────────────────────────────────────────────

    def _read_status(self) -> dict | None:
        """Read status.json from the worker's run directory."""
        if not hasattr(self, "_run_dir") or not self._run_dir:
            return None
        status_path = os.path.join(self._run_dir, "status.json")
        if not os.path.exists(status_path):
            return None
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError, OSError):
            return None

    def _write_control(self, action: str):
        """Write a control command to control.json for the worker."""
        if not hasattr(self, "_run_dir") or not self._run_dir:
            return
        path = os.path.join(self._run_dir, "control.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"action": action, "timestamp": time.time()}, f)
            os.replace(tmp, path)  # Atomic on POSIX; may need retry on Windows
        except PermissionError:
            time.sleep(0.05)
            try:
                os.replace(tmp, path)
            except OSError:
                pass
        except OSError as exc:
            self._log(f"Error writing control file: {exc}")

    def _is_pid_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is still running (Windows-safe)."""
        if sys.platform != "win32":
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # SYNCHRONIZE access allows WaitForSingleObject
            handle = kernel32.OpenProcess(0x100000, False, pid)
            if not handle:
                return False
            # WaitForSingleObject with timeout 0 — if alive, returns WAIT_TIMEOUT (0x102)
            WAIT_TIMEOUT = 0x00000102
            result = kernel32.WaitForSingleObject(handle, 0)
            kernel32.CloseHandle(handle)
            return result == WAIT_TIMEOUT
        except Exception:
            return False

    def _load_epoch_history(self):
        """Load epoch-level metrics from SQLite history for the history table."""
        if not hasattr(self, "_run_dir") or not self._run_dir:
            return
        try:
            from ...history.storage import HistoryStorage
            storage = HistoryStorage()
            runs = storage._get_conn()
            # Find the run that matches our log_dir
            with storage._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM training_runs WHERE log_dir LIKE ? ORDER BY id DESC LIMIT 1",
                    (f"%{os.path.basename(self._run_dir)}%",),
                ).fetchall()
                if not rows:
                    return
                run_id = rows[0][0]
                metrics = storage.get_run_metrics(run_id)
                if metrics:
                    self._epoch_history = []
                    for m in metrics:
                        row = {
                            "epoch": m["epoch"],
                            "train_loss": m["loss"] if m["phase"] == "train" else None,
                            "train_acc": m["accuracy"] if m["phase"] == "train" else None,
                            "val_loss": m["loss"] if m["phase"] == "val" else None,
                            "val_acc": m["accuracy"] if m["phase"] == "val" else None,
                            "lr": m.get("lr"),
                        }
                        existing = next((r for r in self._epoch_history if r["epoch"] == m["epoch"]), None)
                        if existing:
                            existing.update({k: v for k, v in row.items() if v is not None})
                        else:
                            self._epoch_history.append(row)
                    self.query_one("#history-table", HistoryTable).rows = self._epoch_history
                    # Update best metrics
                    for row in self._epoch_history:
                        if row.get("val_loss") and row["val_loss"] < self._best_loss:
                            self._best_loss = row["val_loss"]
                            self._best_epoch_loss = row["epoch"]
                        if row.get("val_csi") and row["val_csi"] > self._best_csi:
                            self._best_csi = row["val_csi"]
                            self._best_epoch_csi = row["epoch"]
                        if row.get("val_acc") and row["val_acc"] > self._best_acc:
                            self._best_acc = row["val_acc"]
                            self._best_epoch_acc = row["epoch"]
        except Exception:
            pass  # Silently handle

    def _set_button_states(self, started: bool, paused: bool):
        self.query_one("#btn-start-training", Button).disabled = started
        self.query_one("#btn-pause", Button).disabled = not started or paused
        self.query_one("#btn-resume", Button).disabled = not started or not paused
        self.query_one("#btn-stop", Button).disabled = not started
        self.query_one("#btn-background", Button).disabled = not started
        self.query_one("#btn-export", Button).disabled = started

    def _log(self, msg: str):
        self.query_one("#log-stream", Log).write_line(msg)

    def _export_results(self):
        from ...history.query import HistoryQuery
        from ...history.storage import HistoryStorage
        storage = HistoryStorage()
        query = HistoryQuery(storage)
        runs = query.list_runs(status="completed", limit=1)
        if runs:
            query.export_metrics_csv(runs[0]["id"], "./export_metrics.csv")
            self._log("Exported to ./export_metrics.csv")
            self.notify("Export: ./export_metrics.csv")
        else:
            self._log("No completed runs to export")

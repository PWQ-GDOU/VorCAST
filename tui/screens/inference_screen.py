"""Full-screen model inference monitor — dedicated prediction view."""
import time
import threading
from pathlib import Path

import numpy as np
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input, Button, Log, Switch
from textual.reactive import reactive

from ..widgets.sparkline import Sparkline


class InferenceScreen(Screen):
    """Full-screen inference monitor with detailed per-step prediction display.

    Layout:
      Top:    Status bar (model, checkpoint, step progress, elapsed, device)
      Row 1:  Model/Input summary | Current step stats
      Row 2:  Vorticity sparklines | Step history table
      Bottom: Log stream
      Dock:   Control buttons + shortcut bar
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("ctrl+r", "infer", "Run"),
    ]

    CSS = """
    InferenceScreen {
        layout: vertical;
    }

    #inf-status-bar {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary;
        dock: top;
    }
    #inf-status-model {
        text-style: bold;
        color: $accent;
    }
    #inf-status-device {
        color: $success;
    }
    #inf-status-elapsed {
        color: $warning;
    }

    #inf-main-grid {
        height: 1fr;
        layout: grid;
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: auto 1fr;
        padding: 0 1;
    }

    #inf-model-cell {
        border-right: dashed $surface;
        padding: 0 1 0 0;
    }
    #inf-step-cell {
        padding: 0 0 0 1;
    }
    #inf-spark-cell {
        border-right: dashed $surface;
        padding: 0 1 0 0;
        border-top: dashed $surface;
    }
    #inf-history-cell {
        padding: 0 0 0 1;
        border-top: dashed $surface;
    }

    #inf-bottom-bar {
        dock: bottom;
        height: auto;
        min-height: 8;
        max-height: 15;
        border-top: solid $primary;
        padding: 0 1;
    }
    #inf-log-stream {
        height: 1fr;
        min-height: 4;
    }

    #inf-control-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        align: center middle;
    }
    #inf-shortcut-bar {
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

    status_text = reactive("Ready")
    inference_running = reactive(False)

    def compose(self) -> ComposeResult:
        yield Header()

        # Status bar
        with Horizontal(id="inf-status-bar"):
            yield Static("Model: --", id="inf-status-model")
            yield Static("", id="inf-status-checkpoint")
            yield Static("", id="inf-status-steps")
            yield Static("", id="inf-status-elapsed")
            yield Static("", id="inf-status-device")

        # Main grid
        with Container(id="inf-main-grid"):
            with Vertical(id="inf-model-cell"):
                yield Static("Model & Input Summary", classes="section-title")
                yield Static("Load a checkpoint and input to begin", id="inf-model-info")

            with Vertical(id="inf-step-cell"):
                yield Static("Current Step", classes="section-title")
                yield Static("Waiting...", id="inf-step-info")

            with Vertical(id="inf-spark-cell"):
                yield Static("Vorticity Trend (per step)", classes="section-title")
                yield Sparkline(label="Max ζ", id="inf-spark-max")
                yield Sparkline(label="Mean ζ", id="inf-spark-mean")

            with Vertical(id="inf-history-cell"):
                yield Static("Step History", classes="section-title")
                yield Static("No data yet", id="inf-history-table")

        # Bottom: log + config inputs (compact)
        with Vertical(id="inf-bottom-bar"):
            with Horizontal():
                yield Static("Checkpoint:", classes="section-title")
                yield Input(placeholder=".pt / .pth path...", id="ckpt-input")
                yield Button("Browse", id="btn-browse-ckpt")
            with Horizontal():
                yield Static("Input:", classes="section-title")
                yield Input(placeholder=".npz file or directory...", id="input-data")
                yield Button("Browse", id="btn-browse-input")
            with Horizontal():
                yield Static("Output:", classes="section-title")
                yield Input(placeholder="./predictions", id="output-dir", value="./predictions")
                yield Static("  Steps:", classes="section-title")
                yield Input(value="36", id="pred-steps", type="integer")
                yield Static("  GPU:", classes="section-title")
                yield Switch(value=True, id="gpu-switch")
            yield Log(id="inf-log-stream", auto_scroll=True, max_lines=300)

        # Controls
        with Horizontal(id="inf-control-bar"):
            yield Button("Run Inference (Ctrl+R)", id="btn-infer", variant="primary")
            yield Button("Stop", id="btn-stop", disabled=True)
            yield Button("Back (ESC)", id="btn-back")
        yield Static(
            "Ctrl+R=Run  ESC=Back",
            id="inf-shortcut-bar",
        )
        yield Footer()

    # ── lifecycle ──────────────────────────────────────────────

    def on_mount(self):
        self._engine = None
        self._step_history = []  # list of dicts for step history table
        self._max_vals = []       # max zeta per step for sparkline
        self._mean_vals = []      # mean zeta per step for sparkline

    # ── button dispatch ────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "btn-browse-ckpt":
            self._browse_file("ckpt-input", ".pt")
        elif bid == "btn-browse-input":
            self._browse_file("input-data", ".npz")
        elif bid == "btn-infer":
            self.action_infer()
        elif bid == "btn-stop":
            self._stop_inference()
        elif bid == "btn-back":
            self.action_go_back()

    # ── actions ────────────────────────────────────────────────

    def action_infer(self):
        if not self.inference_running:
            self._start_inference()

    def action_go_back(self):
        if self.inference_running:
            self._stop_inference()
        self.dismiss()

    # ── file browser ───────────────────────────────────────────

    def _browse_file(self, input_id: str, ext: str):
        current = self.query_one(f"#{input_id}", Input).value.strip()
        start = current if current and Path(current).exists() else "."
        ext_filter = [".pt", ".pth"] if ext == ".pt" else [ext]
        self._browser_target = input_id
        from .file_browser import FileBrowserScreen
        self.app.push_screen(FileBrowserScreen(start, ext_filter=ext_filter))

    def on_screen_resume(self):
        result = getattr(self.app, '_browser_result', None)
        if result is not None and hasattr(self, '_browser_target'):
            input_id = self._browser_target
            self.query_one(f"#{input_id}", Input).value = result
            self._log(f"Selected: {result}")
            self.app._browser_result = None
            del self._browser_target

    # ── start inference ────────────────────────────────────────

    def _start_inference(self):
        ckpt = self.query_one("#ckpt-input", Input).value.strip()
        input_data = self.query_one("#input-data", Input).value.strip()
        output = self.query_one("#output-dir", Input).value.strip() or "./predictions"

        if not ckpt:
            self.status_text = "Error: Specify checkpoint path"
            return
        if not input_data:
            self.status_text = "Error: Specify input data path"
            return
        if not Path(ckpt).exists():
            self.status_text = f"Error: checkpoint not found: {ckpt}"
            return
        if not Path(input_data).exists():
            self.status_text = f"Error: input data not found: {input_data}"
            return

        try:
            pred_steps = int(self.query_one("#pred-steps", Input).value or "36")
        except ValueError:
            pred_steps = 36
        use_gpu = self.query_one("#gpu-switch", Switch).value

        self.inference_running = True
        self.status_text = "Loading model..."
        self._set_button_states(running=True)

        self._step_history = []
        self._max_vals = []
        self._mean_vals = []
        self._start_time = time.time()

        thread = threading.Thread(
            target=self._run_inference,
            args=(ckpt, input_data, output, pred_steps, use_gpu),
            daemon=True,
        )
        thread.start()

    def _stop_inference(self):
        self.inference_running = False
        self.status_text = "Stopped"
        self._set_button_states(running=False)

    # ── inference runner ───────────────────────────────────────

    def _run_inference(self, ckpt: str, input_data: str, output: str,
                       pred_steps: int, use_gpu: bool):
        from ...models.inference import InferenceEngine

        try:
            self._log("Loading model...")
            engine = InferenceEngine(
                self.app.config,
                prediction_steps=pred_steps,
                time_step_seconds=300.0,
                gpu_enabled=use_gpu,
            )
            self._engine = engine

            info = engine.load_checkpoint(ckpt)

            # Update model summary display
            model_cfg = self.app.config.get("model", {})
            data_cfg = self.app.config.get("data", {})
            depth = model_cfg.get("depth", "?")
            base_ch = model_cfg.get("base_channels", "?")
            in_ch = data_cfg.get("in_channels", "?")

            model_summary = (
                f"  Architecture: ResUNet3D\n"
                f"  Encoder Depth: {depth}\n"
                f"  Base Channels: {base_ch}\n"
                f"  Input Channels: {in_ch}\n"
                f"  Checkpoint: {info['source']}\n"
                f"  Train Epoch: {info['epoch'] or 'N/A'}\n"
                f"  Modules Loaded: {', '.join(info['loaded'])}\n"
                f"  Device: {engine.device}\n"
                f"  Prediction Steps: {pred_steps}\n"
                f"  Time Step: 300s (5 min)\n"
                f"  Total Duration: {pred_steps * 5} min ({pred_steps * 5 / 60:.1f} hr)"
            )
            self.app.call_from_thread(
                self.query_one("#inf-model-info", Static).update, model_summary
            )
            self.app.call_from_thread(
                self.query_one("#inf-status-model", Static).update,
                f"Model: ResUNet3D (depth={depth}, ch={base_ch})"
            )
            self.app.call_from_thread(
                self.query_one("#inf-status-checkpoint", Static).update,
                f"CKPT: {info['source']}"
            )
            self.app.call_from_thread(
                self.query_one("#inf-status-device", Static).update,
                f"Device: {engine.device}"
            )

            self._log(f"Loaded: {info['source']} | Epoch: {info['epoch']} | Device: {engine.device}")
            self._log(f"Modules: {', '.join(info['loaded'])}")

            # Load input data summary
            input_path = Path(input_data)
            if input_path.is_dir():
                files = sorted(input_path.glob("*.npz"))
                self._log(f"Batch mode: {len(files)} files in {input_path}")
            else:
                data = np.load(input_path, allow_pickle=True)
                input_arr = data["input"]
                self._log(f"Input shape: {input_arr.shape} (T_in={input_arr.shape[0]}, "
                         f"H={input_arr.shape[1]}, W={input_arr.shape[2]}, "
                         f"L={input_arr.shape[3]}, C={input_arr.shape[4]})")
                if "channel_names" in data:
                    self._log(f"Channels: {list(data['channel_names'])}")

            # Run prediction
            self.app.call_from_thread(
                self.query_one("#inf-status-steps", Static).update,
                f"Step: 0/{pred_steps}"
            )

            def on_step(step_idx, zeta_np):
                self._on_step_complete(step_idx, zeta_np, pred_steps)

            self._log(f"Starting prediction: {pred_steps} steps...")

            if input_path.is_dir():
                # Batch mode — predict first file with detailed display
                files = sorted(input_path.glob("*.npz"))
                total = len(files)
                self._log(f"Batch prediction: {total} files")
                out_dir = Path(output)
                out_dir.mkdir(parents=True, exist_ok=True)

                for fi, npz_file in enumerate(files):
                    self.app.call_from_thread(
                        self.query_one("#inf-status-steps", Static).update,
                        f"File: {fi+1}/{total}"
                    )
                    self._log(f"[{fi+1}/{total}] {npz_file.name}")
                    try:
                        out_file = out_dir / f"{npz_file.stem}_pred.npz"
                        engine.predict_from_file(
                            str(npz_file), str(out_file),
                            num_steps=pred_steps,
                            step_callback=on_step if fi == 0 else None,
                        )
                    except Exception as e:
                        self._log(f"  Error: {e}")

                self._log(f"Batch done! {total} files -> {output}")
                self.app.call_from_thread(
                    self.query_one("#inf-status-steps", Static).update,
                    f"Complete: {total} files"
                )
            else:
                out_path = Path(output) / f"{input_path.stem}_pred.npz"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                predictions = engine.predict_from_file(
                    str(input_path), str(out_path),
                    num_steps=pred_steps,
                    step_callback=on_step,
                )

                self._log(f"Output: {out_path}")
                self._log(f"Prediction shape: {predictions.shape} "
                         f"(T={predictions.shape[0]}, H={predictions.shape[1]}, "
                         f"W={predictions.shape[2]}, L={predictions.shape[3]})")
                self._log(f"Final zeta range: [{predictions.min():.6f}, {predictions.max():.6f}]")
                self._log(f"Final zeta mean: {predictions.mean():.6f}")

            total_elapsed = time.time() - self._start_time
            self._log(f"Total inference time: {total_elapsed:.1f}s")
            self.status_text = "Complete"
            self.inference_running = False
            self.app.call_from_thread(self._set_button_states, running=False)
            self.app.call_from_thread(
                self.query_one("#inf-status-steps", Static).update,
                f"Step: {pred_steps}/{pred_steps} DONE"
            )

        except Exception as e:
            self._log(f"Error: {e}")
            self.status_text = f"Inference failed: {e}"
            self.inference_running = False
            self.app.call_from_thread(self._set_button_states, running=False)

    # ── per-step callback ──────────────────────────────────────

    def _on_step_complete(self, step_idx: int, zeta: np.ndarray, total_steps: int):
        """Called from inference thread after each integration step."""
        zeta_flat = zeta.ravel()
        z_max = float(zeta_flat.max())
        z_min = float(zeta_flat.min())
        z_mean = float(zeta_flat.mean())
        z_std = float(zeta_flat.std())

        self._max_vals.append(z_max)
        self._mean_vals.append(z_mean)

        step_num = step_idx + 1
        elapsed = time.time() - self._start_time
        step_pct = step_num / total_steps * 100

        # Update sparklines
        self.app.call_from_thread(
            self.query_one("#inf-spark-max", Sparkline).push, z_max
        )
        self.app.call_from_thread(
            self.query_one("#inf-spark-mean", Sparkline).push, z_mean
        )

        # Update step info
        step_text = (
            f"  Step:          {step_num}/{total_steps} ({step_pct:.0f}%)\n"
            f"  Elapsed:       {elapsed:.1f}s\n"
            f"  Zeta Max:      {z_max:.6f}\n"
            f"  Zeta Min:      {z_min:.6f}\n"
            f"  Zeta Mean:     {z_mean:.6f}\n"
            f"  Zeta Std:      {z_std:.6f}\n"
            f"  Shape:         {zeta.shape}"
        )
        self.app.call_from_thread(
            self.query_one("#inf-step-info", Static).update, step_text
        )

        # Update status bar
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self.app.call_from_thread(
            self.query_one("#inf-status-steps", Static).update,
            f"Step: {step_num}/{total_steps}"
        )
        self.app.call_from_thread(
            self.query_one("#inf-status-elapsed", Static).update,
            f"Elapsed: {h:02d}:{m:02d}:{s:02d}"
        )

        # ETA
        if step_num > 0 and elapsed > 0.5:
            sec_per_step = elapsed / step_num
            remaining = sec_per_step * (total_steps - step_num)
            rh = int(remaining // 3600)
            rm = int((remaining % 3600) // 60)
            eta_str = f"ETA: {rh:02d}:{rm:02d}:{int(remaining % 60):02d}"
        else:
            eta_str = "ETA: --"
        self.app.call_from_thread(
            self.query_one("#inf-status-elapsed", Static).update,
            f"Elapsed: {h:02d}:{m:02d}:{s:02d}  {eta_str}"
        )

        # Step history table
        row = {
            "step": step_num,
            "max": z_max,
            "min": z_min,
            "mean": z_mean,
            "std": z_std,
        }
        self._step_history.append(row)
        self._update_history_table()

        # Log every 5 steps or first/last
        if step_num <= 3 or step_num % 5 == 0 or step_num == total_steps:
            self._log(
                f"Step {step_num:3d}/{total_steps} | "
                f"max={z_max:+.6f} min={z_min:+.6f} "
                f"mean={z_mean:+.6f} std={z_std:.6f}"
            )

    def _update_history_table(self):
        rows = self._step_history
        lines = ["Step │ Max ζ     │ Min ζ     │ Mean ζ    │ Std ζ"]
        lines.append("─────┼───────────┼───────────┼───────────┼──────────")
        for r in rows[-30:]:
            lines.append(
                f"  {r['step']:3d} │ {r['max']:+.6f} │ {r['min']:+.6f} │ "
                f"{r['mean']:+.6f} │ {r['std']:.6f}"
            )
        self.app.call_from_thread(
            self.query_one("#inf-history-table", Static).update, "\n".join(lines)
        )

    # ── helpers ────────────────────────────────────────────────

    def _set_button_states(self, running: bool):
        self.query_one("#btn-infer", Button).disabled = running
        self.query_one("#btn-stop", Button).disabled = not running
        self.query_one("#ckpt-input", Input).disabled = running
        self.query_one("#input-data", Input).disabled = running
        self.query_one("#output-dir", Input).disabled = running
        self.query_one("#pred-steps", Input).disabled = running

    def _log(self, msg: str):
        try:
            self.query_one("#inf-log-stream", Log).write_line(msg)
        except Exception:
            pass

"""Standalone training worker process.

Runs independently of the TUI — survives terminal close.

IPC via files in output/<run_name>/:
  status.json  — written by worker, read by TUI
  control.json — written by TUI, read & cleared by worker
  worker.pid   — PID file for liveness checks
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# 防止 DataLoader 子进程在 Windows 上弹出控制台窗口
if sys.platform == "win32":
    import multiprocessing
    pyw = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')
    if os.path.exists(pyw):
        multiprocessing.set_executable(pyw)

STATUS_FILE = "status.json"
CONTROL_FILE = "control.json"
PID_FILE = "worker.pid"


def _atomic_write(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    # os.replace is atomic on POSIX but may fail on Windows if the
    # destination is open in another process (e.g. TUI reading it).
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def write_status(run_dir: str, *, status: str = "running", **kwargs):
    payload = {
        "status": status,
        "epoch": 0,
        "total_epochs": 0,
        "batch": 0,
        "total_batches": 0,
        "best_loss": None,
        "last_loss": None,
        "last_accuracy": None,
        "message": "",
        "timestamp": time.time(),
    }
    payload.update(kwargs)
    _atomic_write(os.path.join(run_dir, STATUS_FILE), payload)


def read_control(run_dir: str) -> dict | None:
    path = os.path.join(run_dir, CONTROL_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def clear_control(run_dir: str):
    try:
        os.remove(os.path.join(run_dir, CONTROL_FILE))
    except OSError:
        pass


def run_worker(config_path: str, run_name: str):
    from ..utils.config import load_config
    from ..data.dataset import scan_processed_files, create_dataloaders
    from ..data.split import merge_datasets
    from .trainer import Trainer

    config = load_config(config_path)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    # Print key config values for user verification
    print(f"Config: batch_size={training_cfg.get('batch_size')}, "
          f"epochs={training_cfg.get('epochs')}, "
          f"lr={training_cfg.get('learning_rate')}, "
          f"num_workers={training_cfg.get('num_workers')}, "
          f"AMP={training_cfg.get('use_amp')}")
    print(f"Data: in_channels={data_cfg.get('in_channels')}, "
          f"grid_size={data_cfg.get('grid_size')}, "
          f"history_steps={data_cfg.get('history_steps')}, "
          f"future_steps={data_cfg.get('future_steps')}")
    model_cfg = config.get("model", {})
    print(f"Model: depth={model_cfg.get('depth')}, "
          f"base_channels={model_cfg.get('base_channels')}, "
          f"dropout={model_cfg.get('dropout')}")

    output_dir = training_cfg.get("output_dir", "./output")
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, PID_FILE), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    write_status(run_dir, status="starting", message="Loading data...")

    processed_dir = data_cfg.get("processed_dir", "./processed")
    files = scan_processed_files(processed_dir)
    if not files:
        write_status(run_dir, status="failed", message="No preprocessed data found")
        return

    train_files, val_files, test_files = merge_datasets(files, [])
    train_loader, val_loader, _ = create_dataloaders(
        train_files, val_files, test_files, config,
    )

    total_epochs_val = training_cfg.get("epochs", 100)

    write_status(
        run_dir, status="starting", message="Creating trainer...",
        total_epochs=total_epochs_val, total_batches=len(train_loader),
    )

    trainer = Trainer(config)

    # Remember total_epochs from initial status (batch callbacks don't include it)
    _total_epochs = total_epochs_val
    _total_batches = len(train_loader)

    def _safe(v):
        """Filter out NaN/Inf values that are invalid in JSON."""
        if v is None:
            return None
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    # Register callback that writes progress to status.json
    _current_batch = 0
    _current_loss = None
    _current_accuracy = None

    def status_callback(metrics: dict):
        nonlocal _total_epochs, _total_batches
        nonlocal _current_batch, _current_loss, _current_accuracy
        typ = metrics.get("type", "")
        if metrics.get("total_epochs"):
            _total_epochs = metrics["total_epochs"]
        if metrics.get("total_batches"):
            _total_batches = metrics["total_batches"]
        base = {
            "epoch": metrics.get("epoch", 0),
            "total_epochs": _total_epochs,
            "total_batches": _total_batches,
        }
        if typ == "batch":
            _current_batch = metrics.get("batch", 0)
            _current_loss = _safe(metrics.get("loss"))
            _current_accuracy = _safe(metrics.get("accuracy"))
            write_status(
                run_dir, status="running",
                batch=_current_batch,
                last_loss=_current_loss,
                last_accuracy=_current_accuracy,
                message=f"Epoch {metrics.get('epoch')}/{_total_epochs} "
                        f"Batch {_current_batch}/{_total_batches}",
                **base,
            )
        elif typ == "epoch":
            write_status(
                run_dir, status="running",
                batch=_current_batch,
                last_loss=_current_loss,
                last_accuracy=_current_accuracy,
                message=f"Epoch {metrics.get('epoch')}/{_total_epochs} "
                        f"({metrics.get('phase', '?')}) done",
                **base,
            )
        elif typ == "status":
            write_status(run_dir, message=metrics.get("message", ""), **base)

    trainer.on_metric_update(status_callback)

    # Background thread to poll control.json for TUI commands
    def control_poller():
        last_ts = 0
        while trainer.is_running:
            ctrl = read_control(run_dir)
            if ctrl:
                ts = ctrl.get("timestamp", 0)
                if ts > last_ts:
                    last_ts = ts
                    action = ctrl.get("action")
                    if action == "pause":
                        trainer.pause()
                    elif action == "resume":
                        trainer.resume()
                    elif action == "stop":
                        trainer.stop()
            time.sleep(1)

    poller = threading.Thread(target=control_poller, daemon=True)
    poller.start()

    write_status(
        run_dir, status="running", message="Training started",
        total_epochs=total_epochs_val, total_batches=len(train_loader),
    )

    try:
        result = trainer.train(train_loader, val_loader, run_name)
        write_status(
            run_dir,
            status=result.get("status", "completed"),
            best_loss=result.get("best_val_loss"),
            message=f"Training finished: {result.get('status')}",
            epoch=total_epochs_val,
            total_epochs=total_epochs_val,
        )
    except Exception as exc:
        import traceback
        write_status(
            run_dir,
            status="failed",
            message=f"Error: {exc}",
        )
        # Write full traceback to a crash log
        crash_path = os.path.join(run_dir, "crash.log")
        with open(crash_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
    finally:
        try:
            os.remove(os.path.join(run_dir, PID_FILE))
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Training worker process")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", required=True, help="Run name for output directory")
    args = parser.parse_args()
    run_worker(args.config, args.run_name)


if __name__ == "__main__":
    main()

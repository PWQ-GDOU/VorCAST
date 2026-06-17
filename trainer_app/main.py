#!/usr/bin/env python
"""龙卷风垂直涡度预测训练器 — 入口模块。

用法:
    python -m trainer_app.main                    # 启动 TUI
    python -m trainer_app.main --config my.yaml   # 指定配置文件
    python -m trainer_app.main --cli --dataset1 /path/to/d1 --dataset2 /path/to/d2  # CLI 模式
"""
import argparse
import multiprocessing
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer_app.utils.config import load_config
from trainer_app.utils.device import check_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="龙卷风垂直涡度预测训练器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m trainer_app.main
  python -m trainer_app.main --config my_config.yaml
  python -m trainer_app.main --cli --dataset1 ./data/d1 --dataset2 ./data/d2
        """,
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="用户配置文件路径 (YAML)")
    parser.add_argument("--cli", action="store_true", help="使用纯命令行模式（不启动 TUI）")
    parser.add_argument("--verbose", "-v", action="store_true", help="CLI 详细模式：每 50 batch 输出完整指标行")
    parser.add_argument("--dataset1", type=str, default=None, help="数据集1路径")
    parser.add_argument("--dataset2", type=str, default=None, help="数据集2路径")
    parser.add_argument("--gpu", action="store_true", default=None, help="启用 GPU 加速")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--worker", action="store_true", help="后台训练 Worker 进程（由 TUI 自动调用）")
    parser.add_argument("--run-name", type=str, default=None, help="Worker: 训练运行名称")
    parser.add_argument("--kill-workers", action="store_true", help="强制终止所有后台训练 Worker 进程")
    parser.add_argument("--infer", action="store_true", help="运行推理预测模式")
    parser.add_argument("--ckpt", type=str, default=None, help="推理: checkpoint 路径 (.pth)")
    parser.add_argument("--input", type=str, default=None, help="推理: 输入数据 (.npz 或目录)")
    parser.add_argument("--output", type=str, default="./predictions", help="推理: 输出目录")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    print("正在加载配置...")
    config = load_config(args.config)

    # GPU 检测
    gpu_info = check_gpu(config["device"]["gpu_enabled"])
    print(f"GPU 状态: {gpu_info['device']}")
    if gpu_info.get("device_name"):
        print(f"GPU 设备: {gpu_info['device_name']}")

    if args.gpu:
        config["device"]["gpu_enabled"] = True
    if args.cpu:
        config["device"]["gpu_enabled"] = False

    # 更新数据集路径
    if args.dataset1:
        config["data"]["dataset1_path"] = args.dataset1
    if args.dataset2:
        config["data"]["dataset2_path"] = args.dataset2

    if args.kill_workers:
        import json
        import signal
        killed = 0
        output_dir = config.get("training", {}).get("output_dir", "./output")
        if os.path.isdir(output_dir):
            for entry in os.scandir(output_dir):
                if not entry.is_dir():
                    continue
                pid_file = os.path.join(entry.path, "worker.pid")
                if not os.path.exists(pid_file):
                    continue
                try:
                    with open(pid_file, encoding="utf-8") as f:
                        pid = int(f.read().strip())
                    print(f"Terminating worker PID {pid} ({entry.path})...")
                    if sys.platform == "win32":
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        handle = kernel32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
                        if handle:
                            kernel32.TerminateProcess(handle, 1)
                            kernel32.CloseHandle(handle)
                            killed += 1
                    else:
                        os.kill(pid, signal.SIGTERM)
                        killed += 1
                    os.remove(pid_file)
                except Exception as exc:
                    print(f"  Failed: {exc}")
        print(f"Done: {killed} worker(s) terminated")
        return
    elif args.worker:
        if not args.config or not args.run_name:
            print("错误: --worker 需要 --config 和 --run-name 参数")
            return
        from trainer_app.training.worker import run_worker
        run_worker(args.config, args.run_name)
    elif args.cli:
        from trainer_app.training.trainer import run_training_cli
        run_training_cli(config, verbose=args.verbose)
    elif args.infer:
        if not args.ckpt or not args.input:
            print("错误: --infer 需要 --ckpt 和 --input 参数")
            return
        from trainer_app.models.inference import InferenceEngine
        engine = InferenceEngine(config)
        print(f"加载 checkpoint: {args.ckpt}")
        info = engine.load_checkpoint(args.ckpt)
        print(f"已加载: {info['loaded']}")
        input_path = args.input
        if os.path.isdir(input_path):
            engine.predict_batch(input_path, args.output,
                                 progress_callback=lambda c, t, m: print(f"[{c}/{t}] {m}"))
        else:
            pred = engine.predict_from_file(input_path)
            out = os.path.join(args.output, f"{os.path.splitext(os.path.basename(input_path))[0]}_pred.npz")
            os.makedirs(args.output, exist_ok=True)
            engine.save_predictions(pred, out)
        print(f"推理完成，输出: {args.output}")
    else:
        from trainer_app.tui.app import run_app
        run_app(config)


if __name__ == "__main__":
    # 防止 multiprocessing spawn 时子进程重复执行 main()
    multiprocessing.freeze_support()
    main()

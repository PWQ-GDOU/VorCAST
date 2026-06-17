"""GPU 监控组件 — 使用 pynvml 跨进程查询实时状态。"""
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive


class GPUMonitor(Vertical):
    """GPU 使用状态显示。"""

    gpu_name = reactive("N/A")
    memory_used = reactive("N/A")
    utilization = reactive("N/A")

    def compose(self) -> ComposeResult:
        yield Static("GPU Status", classes="section-title")
        yield Static(f"Device: {self.gpu_name}", id="gpu-name")
        yield Static(f"Memory: {self.memory_used}", id="gpu-memory")
        yield Static(f"Utilization: {self.utilization}", id="gpu-util")

    def refresh_status(self):
        """通过 pynvml 获取真实 GPU 状态（跨进程）。"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            self.gpu_name = pynvml.nvmlDeviceGetName(handle)

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.memory_used = f"{mem.used / 1024**3:.1f} / {mem.total / 1024**3:.1f} GB"

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.utilization = f"GPU: {util.gpu}%  MEM: {util.memory}%"

            pynvml.nvmlShutdown()
        except Exception:
            # Fallback to torch (current process only)
            import torch
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                self.gpu_name = torch.cuda.get_device_name(device_id)
                mem_alloc = torch.cuda.memory_allocated(device_id) / 1024**3
                mem_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                self.memory_used = f"{mem_alloc:.1f} / {mem_total:.1f} GB"
                self.utilization = "(nvidia-smi driver unavailable)"
            else:
                self.gpu_name = "CPU (no GPU)"
                self.memory_used = "--"
                self.utilization = "--"

    def watch_gpu_name(self, val: str):
        if self.is_mounted:
            self.query_one("#gpu-name", Static).update(f"Device: {val}")

    def watch_memory_used(self, val: str):
        if self.is_mounted:
            self.query_one("#gpu-memory", Static).update(f"Memory: {val}")

    def watch_utilization(self, val: str):
        if self.is_mounted:
            self.query_one("#gpu-util", Static).update(f"Utilization: {val}")

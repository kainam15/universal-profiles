"""在测量窗口外解析物理 GPU；容器与主机采集器共享同一 UUID。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import csv
import os
import subprocess


_SELECTED: ContextVar[dict | None] = ContextVar("acprof_gpu_device", default=None)


@contextmanager
def gpu_device_scope():
    """Limit a pinned selection to one CLI invocation, including embedded callers."""
    token = _SELECTED.set(None)
    try:
        yield
    finally:
        _SELECTED.reset(token)


def selected_gpu_device() -> dict:
    return dict(_SELECTED.get() or {})


def resolve_gpu_device(selector: str | None = None) -> dict:
    if selector is None and _SELECTED.get() is not None:
        return selected_gpu_device()
    selector = str(selector if selector is not None else
                   os.environ.get("ACPROF_GPU_DEVICE", os.environ.get("DEVICE_INDEX", "0"))).strip()
    if not (selector.isdecimal() or selector.startswith("GPU-")) or any(c in selector for c in ",\n\r "):
        raise ValueError("GPU selector must be one physical GPU index or UUID; all and MIG are unsupported")
    result = subprocess.run(
        ["nvidia-smi", f"--id={selector}",
         "--query-gpu=uuid,index,pci.bus_id,name,memory.total,mig.mode.current",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot resolve GPU {selector}: {result.stderr.strip()}")
    rows = list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeError(f"Expected exactly one physical GPU for {selector}")
    uuid, index, bus, name, memory_mib, mig = (value.strip() for value in rows[0])
    if not uuid.startswith("GPU-") or mig.lower() == "enabled":
        raise RuntimeError("MIG devices cannot be attributed using physical-device NVML energy")
    return {"uuid": uuid, "index": int(index), "pci_bus_id": bus, "name": name,
            "memory_total_bytes": int(float(memory_mib) * 1024 ** 2)}


def pin_gpu_device(selector: str | None = None) -> dict:
    device = resolve_gpu_device(selector)
    _SELECTED.set(device)
    return dict(device)


def gpu_docker_args(device: dict | None = None) -> list[str]:
    device = resolve_gpu_device() if device is None else device
    return ["--gpus", f"device={device['uuid']}",
            "-e", f"NVIDIA_VISIBLE_DEVICES={device['uuid']}", "-e", "CUDA_VISIBLE_DEVICES=0"]

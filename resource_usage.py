"""Container CPU/memory and NVML GPU utilization monitoring."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

try:
    import pynvml
except Exception:  # pragma: no cover - exercised by runtime fallback
    pynvml = None


BYTES_PER_GIB = 1024 ** 3


@dataclass
class ResourceUsageSample:
    timestamp: float
    container_cpu_s: Optional[float]
    container_mem_usage_bytes: Optional[int]
    gpu_util_pct: Optional[float]
    gpu_mem_used_bytes: Optional[int]
    gpu_mem_total_bytes: Optional[int]


@dataclass
class ResourceUsageResult:
    resource_usage_iters: int
    container_cpu_util_avg_pct: float
    container_cpu_util_peak_pct: float
    container_mem_usage_avg_bytes: float
    container_mem_usage_peak_bytes: float
    container_mem_util_avg_pct: float
    container_mem_util_peak_pct: float
    gpu_util_avg_pct: float
    gpu_util_peak_pct: float
    gpu_mem_used_avg_bytes: float
    gpu_mem_used_peak_bytes: float
    gpu_mem_util_avg_pct: float
    gpu_mem_util_peak_pct: float
    gpu_mem_total_bytes: float


def _nan_result(resource_usage_iters: int = 0) -> ResourceUsageResult:
    nan = float("nan")
    return ResourceUsageResult(
        resource_usage_iters,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
    )


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _read_int(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return int(f.read().strip())


def _read_cpu_stat_usage_s(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key, _, value = line.strip().partition(" ")
            if key == "usage_usec":
                return float(value) / 1_000_000.0
    raise RuntimeError(f"usage_usec missing from {path}")


def _docker_container_pid(container_name: str) -> int:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"docker inspect failed for {container_name}")

    try:
        pid = int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid container pid: {result.stdout.strip()!r}") from exc
    if pid <= 0:
        raise RuntimeError(f"container is not running: {container_name}")
    return pid


def _join_cgroup_path(root: str, relative: str, leaf: str) -> str:
    rel = relative.strip("/")
    return os.path.join(root, rel, leaf) if rel else os.path.join(root, leaf)


def _first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _v1_candidates(
    cgroup_root: str,
    controllers: str,
    relative: str,
    leaf: str,
) -> List[str]:
    controller_parts = [part for part in controllers.split(",") if part]
    roots = [os.path.join(cgroup_root, controllers)]
    roots.extend(os.path.join(cgroup_root, part) for part in controller_parts)
    roots.append(cgroup_root)

    candidates: List[str] = []
    for root in roots:
        path = _join_cgroup_path(root, relative, leaf)
        if path not in candidates:
            candidates.append(path)
    return candidates


def _resolve_container_readers(
    container_name: str,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_root: str = "/proc",
) -> Tuple[Optional[Callable[[], float]], Optional[Callable[[], int]]]:
    if not container_name:
        return None, None

    pid = _docker_container_pid(container_name)
    cgroup_file = os.path.join(proc_root, str(pid), "cgroup")
    with open(cgroup_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    cpu_reader: Optional[Callable[[], float]] = None
    mem_reader: Optional[Callable[[], int]] = None

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            cpu_path = _join_cgroup_path(cgroup_root, parts[2], "cpu.stat")
            mem_path = _join_cgroup_path(cgroup_root, parts[2], "memory.current")
            if os.path.exists(cpu_path):
                cpu_reader = lambda path=cpu_path: _read_cpu_stat_usage_s(path)
            if os.path.exists(mem_path):
                mem_reader = lambda path=mem_path: _read_int(path)
            return cpu_reader, mem_reader

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue

        controllers = set(parts[1].split(","))
        if cpu_reader is None and "cpuacct" in controllers:
            cpu_path = _first_existing(
                _v1_candidates(cgroup_root, parts[1], parts[2], "cpuacct.usage")
            )
            if cpu_path is not None:
                cpu_reader = lambda path=cpu_path: float(_read_int(path)) / 1_000_000_000.0

        if mem_reader is None and "memory" in controllers:
            mem_path = _first_existing(
                _v1_candidates(cgroup_root, parts[1], parts[2], "memory.usage_in_bytes")
            )
            if mem_path is not None:
                mem_reader = lambda path=mem_path: _read_int(path)

    return cpu_reader, mem_reader


def _result_from_samples(
    samples: List[ResourceUsageSample],
    cpu_cores: float,
    mem_limit_bytes: float,
    min_cpu_interval_s: float = 0.0,
) -> ResourceUsageResult:
    if not samples:
        return _nan_result(0)

    samples = sorted(samples, key=lambda sample: sample.timestamp)
    result = _nan_result(len(samples))

    cpu_interval_utils: List[float] = []
    cpu_elapsed_s = 0.0
    cpu_delta_s = 0.0
    if cpu_cores > 0:
        cpu_samples = [
            (sample.timestamp, sample.container_cpu_s)
            for sample in samples
            if sample.container_cpu_s is not None
        ]
        for idx in range(1, len(cpu_samples)):
            prev_t, prev_cpu = cpu_samples[idx - 1]
            curr_t, curr_cpu = cpu_samples[idx]
            dt = curr_t - prev_t
            delta = float(curr_cpu) - float(prev_cpu)
            if dt <= 0 or delta < 0:
                continue
            cpu_elapsed_s += dt
            cpu_delta_s += delta
            if dt >= min_cpu_interval_s:
                util = (delta / (dt * cpu_cores)) * 100.0
                cpu_interval_utils.append(util)

    if cpu_elapsed_s > 0:
        result.container_cpu_util_avg_pct = (cpu_delta_s / (cpu_elapsed_s * cpu_cores)) * 100.0
        if cpu_interval_utils:
            result.container_cpu_util_peak_pct = max(cpu_interval_utils)

    mem_values = [
        float(sample.container_mem_usage_bytes)
        for sample in samples
        if sample.container_mem_usage_bytes is not None
    ]
    if mem_values:
        result.container_mem_usage_avg_bytes = _mean(mem_values)
        result.container_mem_usage_peak_bytes = max(mem_values)
        if mem_limit_bytes > 0:
            result.container_mem_util_avg_pct = (
                result.container_mem_usage_avg_bytes / mem_limit_bytes
            ) * 100.0
            result.container_mem_util_peak_pct = (
                result.container_mem_usage_peak_bytes / mem_limit_bytes
            ) * 100.0

    gpu_utils = [
        float(sample.gpu_util_pct)
        for sample in samples
        if sample.gpu_util_pct is not None
    ]
    if gpu_utils:
        result.gpu_util_avg_pct = _mean(gpu_utils)
        result.gpu_util_peak_pct = max(gpu_utils)

    gpu_mem_used = [
        float(sample.gpu_mem_used_bytes)
        for sample in samples
        if sample.gpu_mem_used_bytes is not None
    ]
    gpu_mem_totals = [
        float(sample.gpu_mem_total_bytes)
        for sample in samples
        if sample.gpu_mem_total_bytes is not None and sample.gpu_mem_total_bytes > 0
    ]
    if gpu_mem_used:
        result.gpu_mem_used_avg_bytes = _mean(gpu_mem_used)
        result.gpu_mem_used_peak_bytes = max(gpu_mem_used)
    if gpu_mem_totals:
        result.gpu_mem_total_bytes = max(gpu_mem_totals)

    gpu_mem_utils = [
        (float(sample.gpu_mem_used_bytes) / float(sample.gpu_mem_total_bytes)) * 100.0
        for sample in samples
        if sample.gpu_mem_used_bytes is not None
        and sample.gpu_mem_total_bytes is not None
        and sample.gpu_mem_total_bytes > 0
    ]
    if gpu_mem_utils:
        result.gpu_mem_util_avg_pct = _mean(gpu_mem_utils)
        result.gpu_mem_util_peak_pct = max(gpu_mem_utils)

    return result


class ResourceUsageMonitor:
    def __init__(
        self,
        sample_hz: float = 20.0,
        container_name: str = "",
        cpu_cores: float = 1.0,
        mem_cap_gb: float = 1.0,
        use_gpu: bool = False,
        device_index: int = 0,
        cgroup_root: str = "/sys/fs/cgroup",
        proc_root: str = "/proc",
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.container_name = container_name
        self.cpu_cores = float(cpu_cores)
        self.mem_limit_bytes = float(mem_cap_gb) * float(BYTES_PER_GIB)
        self.use_gpu = bool(use_gpu)
        self.device_index = int(device_index)
        self.dt = 1.0 / self.sample_hz

        self.samples: List[ResourceUsageSample] = []
        self._cpu_reader: Optional[Callable[[], float]] = None
        self._mem_reader: Optional[Callable[[], int]] = None
        self._gpu_handle = None
        self._gpu_initialized = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._t_start: Optional[float] = None
        self._t_end: Optional[float] = None
        self._init_error = ""
        self._runtime_error = ""
        self._closed = False

        try:
            self._cpu_reader, self._mem_reader = _resolve_container_readers(
                container_name,
                cgroup_root=cgroup_root,
                proc_root=proc_root,
            )
        except Exception as exc:
            self._init_error = str(exc)

        if self.use_gpu:
            if pynvml is None:
                self._runtime_error = "pynvml unavailable"
            else:
                try:
                    pynvml.nvmlInit()
                    self._gpu_initialized = True
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                except Exception as exc:
                    self._runtime_error = str(exc)
                    self._gpu_handle = None

    @property
    def available(self) -> bool:
        return (
            self._cpu_reader is not None
            or self._mem_reader is not None
            or self._gpu_handle is not None
        )

    def start(self) -> None:
        if not self.available:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Resource usage monitor is already running")

        self.samples = []
        self._stop_event = threading.Event()
        self._t_start = time.perf_counter()
        self._t_end = None
        self._append_sample(self._t_start)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[ResourceUsageResult, str, List[ResourceUsageSample]]:
        if not self.available:
            return _nan_result(), self._error_message(), []

        if self._t_start is None:
            return _nan_result(0), self._error_message(), []

        self._t_end = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self._append_sample(self._t_end)
        samples = [
            sample
            for sample in self.samples
            if self._t_start <= sample.timestamp <= self._t_end
        ]
        samples.sort(key=lambda item: item.timestamp)
        self.samples = samples
        return (
            _result_from_samples(
                samples,
                self.cpu_cores,
                self.mem_limit_bytes,
                min_cpu_interval_s=0.5 * self.dt,
            ),
            self._error_message(),
            samples,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        if self._gpu_initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._closed = True

    def _error_message(self) -> str:
        errors = [error for error in (self._init_error, self._runtime_error) if error]
        return "; ".join(errors)

    def _sample_loop(self) -> None:
        next_t = (self._t_start if self._t_start is not None else time.perf_counter()) + self.dt
        while not self._stop_event.is_set():
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0 and self._stop_event.wait(sleep_s):
                break
            if self._stop_event.is_set():
                break
            t = time.perf_counter()
            if self._t_start is not None and t >= self._t_start:
                self._append_sample(t)
            next_t += self.dt

    def _read_sample(self, timestamp: float) -> ResourceUsageSample:
        container_cpu_s = None
        container_mem_usage_bytes = None
        gpu_util_pct = None
        gpu_mem_used_bytes = None
        gpu_mem_total_bytes = None

        if self._cpu_reader is not None:
            try:
                container_cpu_s = self._cpu_reader()
            except Exception as exc:
                self._runtime_error = str(exc)

        if self._mem_reader is not None:
            try:
                container_mem_usage_bytes = self._mem_reader()
            except Exception as exc:
                self._runtime_error = str(exc)

        if self._gpu_handle is not None and pynvml is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_util_pct = float(util.gpu)
                gpu_mem_used_bytes = int(mem.used)
                gpu_mem_total_bytes = int(mem.total)
            except Exception as exc:
                self._runtime_error = str(exc)

        return ResourceUsageSample(
            timestamp,
            container_cpu_s,
            container_mem_usage_bytes,
            gpu_util_pct,
            gpu_mem_used_bytes,
            gpu_mem_total_bytes,
        )

    def _append_sample(self, timestamp: float) -> None:
        self.samples.append(self._read_sample(timestamp))


def measure_usage_threaded(
    fn: Callable[[], object],
    sample_hz: float = 20.0,
    container_name: str = "",
    cpu_cores: float = 1.0,
    mem_cap_gb: float = 1.0,
    use_gpu: bool = False,
    device_index: int = 0,
) -> Tuple[ResourceUsageResult, str, List[ResourceUsageSample]]:
    monitor = ResourceUsageMonitor(
        sample_hz=sample_hz,
        container_name=container_name,
        cpu_cores=cpu_cores,
        mem_cap_gb=mem_cap_gb,
        use_gpu=use_gpu,
        device_index=device_index,
    )
    try:
        monitor.start()
        try:
            fn()
        finally:
            result = monitor.stop()
        return result
    finally:
        monitor.close()

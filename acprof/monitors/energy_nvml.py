import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pynvml

from acprof.config import DEFAULT_IDLE_SECONDS


@dataclass
class EnergyResult:
    energy_iters: int
    idle_power_w: float
    avg_power_total_w: float
    peak_power_total_w: float
    energy_total_j: float
    avg_power_eff_w: float
    peak_power_eff_w: float
    energy_eff_j: float


def _get_total_power_w(handle):
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def _median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _time_weighted_average_power(samples: List[Tuple[float, float]]) -> float:
    if not samples:
        return float("nan")
    if len(samples) == 1:
        return float(samples[0][1])

    energy_j = 0.0
    for idx in range(1, len(samples)):
        t0, p0 = samples[idx - 1]
        t1, p1 = samples[idx]
        dt = t1 - t0
        if dt > 0.0:
            energy_j += 0.5 * (p0 + p1) * dt

    duration_s = samples[-1][0] - samples[0][0]
    return energy_j / duration_s if duration_s > 0.0 else float("nan")


def _nan_result(energy_iters: int = 0, idle_power_w: float = float("nan")) -> EnergyResult:
    return EnergyResult(
        energy_iters,
        idle_power_w,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )


def _result_from_samples(
    samples: List[Tuple[float, float]],
    idle_power_w: float,
) -> EnergyResult:
    if len(samples) < 2:
        return _nan_result(len(samples), idle_power_w)

    powers = [p for _, p in samples]
    peak_power_total = max(powers)

    energy_total = 0.0
    for i in range(1, len(samples)):
        t0, p0 = samples[i - 1]
        t1, p1 = samples[i]
        energy_total += 0.5 * (p0 + p1) * (t1 - t0)
    duration_s = samples[-1][0] - samples[0][0]
    if duration_s <= 0.0:
        return _nan_result(len(samples), idle_power_w)
    avg_power_total = energy_total / duration_s

    powers_eff = [p - idle_power_w for p in powers]
    peak_power_eff = max(powers_eff)

    energy_eff = 0.0
    for i in range(1, len(samples)):
        t0, _ = samples[i - 1]
        t1, _ = samples[i]
        energy_eff += 0.5 * (powers_eff[i - 1] + powers_eff[i]) * (t1 - t0)
    avg_power_eff = energy_eff / duration_s

    return EnergyResult(
        len(samples),
        idle_power_w,
        avg_power_total,
        peak_power_total,
        energy_total,
        avg_power_eff,
        peak_power_eff,
        energy_eff,
    )


class GPUEnergyMonitor:
    def __init__(
        self,
        sample_hz: float = 20.0,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        device_index: int = 0,
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.idle_seconds = float(idle_seconds)
        self.device_index = int(device_index)
        self.dt = 1.0 / self.sample_hz

        self.handle = None
        self.gpu_name = "unknown"
        self.idle_power_w = float("nan")
        self.idle_trace: Dict[str, Any] = {}
        self.samples: List[Tuple[float, float]] = []

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._t_start: Optional[float] = None
        self._t_end: Optional[float] = None
        self._init_error = ""
        self._runtime_error = ""
        self._closed = False

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8", errors="ignore")
            self.gpu_name = str(gpu_name)
        except Exception as exc:
            self._init_error = str(exc)
            self.handle = None
            self.gpu_name = "unknown"

    def measure_idle(self, trace: bool = False) -> float:
        self.idle_trace = {}
        if self._init_error or self.handle is None:
            if trace:
                self.idle_trace = {
                    "gpu_idle_trace_schema": "nvml_gpu_idle_v1",
                    "gpu_idle_error": self._init_error or "NVML handle is unavailable",
                }
            return float("nan")

        idle_samples: List[Tuple[float, float]] = []
        t_start = time.perf_counter()
        t_idle_end = t_start + self.idle_seconds
        t_last = t_start
        try:
            while True:
                now = time.perf_counter()
                t_last = now
                if now >= t_idle_end:
                    break
                idle_samples.append((now, _get_total_power_w(self.handle)))
                time.sleep(self.dt)
        except Exception as exc:
            self._runtime_error = str(exc)

        powers = [power for _, power in idle_samples]
        self.idle_power_w = _time_weighted_average_power(idle_samples)
        if trace:
            self.idle_trace = self._build_idle_trace(
                idle_samples=idle_samples,
                t_start=t_start,
                t_end=t_last,
            )
        return self.idle_power_w

    def _build_idle_trace(
        self,
        *,
        idle_samples: List[Tuple[float, float]],
        t_start: float,
        t_end: float,
    ) -> Dict[str, Any]:
        powers = [power for _, power in idle_samples]
        trace: Dict[str, Any] = {
            "gpu_idle_trace_schema": "nvml_gpu_idle_v1",
            "gpu_idle_baseline_method": "time_weighted_mean",
            "gpu_idle_requested_duration_s": self.idle_seconds,
            "gpu_idle_actual_duration_s": max(0.0, t_end - t_start),
            "gpu_idle_sample_hz": self.sample_hz,
            "gpu_idle_sample_count": len(idle_samples),
            "gpu_idle_power_w": self.idle_power_w,
            "gpu_idle_power_samples": [
                {"t_s": timestamp - t_start, "power_w": power}
                for timestamp, power in idle_samples
            ],
        }
        if powers:
            trace.update({
                "gpu_idle_power_min_w": min(powers),
                "gpu_idle_power_max_w": max(powers),
                "gpu_idle_power_mean_w": sum(powers) / len(powers),
                "gpu_idle_power_time_weighted_mean_w": (
                    _time_weighted_average_power(idle_samples)
                ),
                "gpu_idle_power_p50_w": _median(powers),
            })
        else:
            trace.update({
                "gpu_idle_power_min_w": float("nan"),
                "gpu_idle_power_max_w": float("nan"),
                "gpu_idle_power_mean_w": float("nan"),
                "gpu_idle_power_time_weighted_mean_w": float("nan"),
                "gpu_idle_power_p50_w": float("nan"),
            })
        if self._runtime_error:
            trace["gpu_idle_error"] = self._runtime_error
        return trace

    def apply_control_baseline(
        self,
        result: EnergyResult,
        samples: List[Tuple[float, float]],
        *,
        trace: bool = False,
    ) -> float:
        """Use a monitor-matched blank window as the workload baseline."""
        self.idle_power_w = float(result.avg_power_total_w)
        self.idle_trace = {}
        if trace and samples:
            self.idle_trace = self._build_idle_trace(
                idle_samples=samples,
                t_start=samples[0][0],
                t_end=samples[-1][0],
            )
            self.idle_trace.update({
                "gpu_idle_trace_schema": "nvml_gpu_control_v1",
                "gpu_idle_baseline_method": "matched_control_time_weighted_mean",
            })
        return self.idle_power_w

    def start(self) -> None:
        if self._init_error or self.handle is None:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("GPU energy monitor is already running")

        self.samples = []
        self._runtime_error = ""
        self._stop_event = threading.Event()
        self._t_start = time.perf_counter()
        self._t_end = None

        self._append_sample(self._t_start)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[EnergyResult, str, str, List[Tuple[float, float]]]:
        if self._init_error or self.handle is None:
            return _nan_result(), self.gpu_name, self._init_error, []

        if self._t_start is None:
            return _nan_result(0, self.idle_power_w), self.gpu_name, self._runtime_error, []

        self._t_end = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self._append_sample(self._t_end)

        samples = [
            (t, p)
            for t, p in self.samples
            if self._t_start <= t <= self._t_end
        ]
        samples.sort(key=lambda item: item[0])
        self.samples = samples

        return (
            _result_from_samples(samples, self.idle_power_w),
            self.gpu_name,
            self._runtime_error,
            samples,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        self._closed = True

    def _sample_loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            t = time.perf_counter()
            if self._t_start is not None and t >= self._t_start:
                self._append_sample(t)
            next_t += self.dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _append_sample(self, timestamp: float) -> None:
        if self.handle is None:
            return
        try:
            self.samples.append((timestamp, _get_total_power_w(self.handle)))
        except Exception as exc:
            self._runtime_error = str(exc)


def measure_energy_threaded(
    fn: Callable[[], Any],
    sample_hz: float = 20.0,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    device_index: int = 0,
    align_to_fn: bool = True,
) -> Tuple[EnergyResult, str, str, List[Tuple[float, float]]]:
    del align_to_fn

    monitor = GPUEnergyMonitor(
        sample_hz=sample_hz,
        idle_seconds=idle_seconds,
        device_index=device_index,
    )
    try:
        monitor.measure_idle()
        monitor.start()
        try:
            fn()
        finally:
            result = monitor.stop()
        return result
    finally:
        monitor.close()

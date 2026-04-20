import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import pynvml


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
    avg_power_total = sum(powers) / len(powers)
    peak_power_total = max(powers)

    energy_total = 0.0
    for i in range(1, len(samples)):
        t0, p0 = samples[i - 1]
        t1, p1 = samples[i]
        energy_total += 0.5 * (p0 + p1) * (t1 - t0)

    powers_eff = [p - idle_power_w for p in powers]
    avg_power_eff = sum(powers_eff) / len(powers_eff)
    peak_power_eff = max(powers_eff)

    energy_eff = 0.0
    for i in range(1, len(samples)):
        t0, _ = samples[i - 1]
        t1, _ = samples[i]
        energy_eff += 0.5 * (powers_eff[i - 1] + powers_eff[i]) * (t1 - t0)

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
        idle_seconds: float = 3.0,
        device_index: int = 0,
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.idle_seconds = float(idle_seconds)
        self.device_index = int(device_index)
        self.dt = 1.0 / self.sample_hz

        self.handle = None
        self.gpu_name = "unknown"
        self.idle_power_w = float("nan")
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

    def measure_idle(self) -> float:
        if self._init_error or self.handle is None:
            return float("nan")

        idle_samples = []
        t_idle_end = time.perf_counter() + self.idle_seconds
        try:
            while time.perf_counter() < t_idle_end:
                idle_samples.append(_get_total_power_w(self.handle))
                time.sleep(self.dt)
        except Exception as exc:
            self._runtime_error = str(exc)

        self.idle_power_w = _median(idle_samples)
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
    idle_seconds: float = 3.0,
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

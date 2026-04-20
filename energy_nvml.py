import time
import threading
from typing import Callable, Any, Tuple, List
import pynvml
from dataclasses import dataclass


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


def measure_energy_threaded(
    fn: Callable[[], Any],
    sample_hz: float = 20.0,
    idle_seconds: float = 3.0,
    device_index: int = 0,
    align_to_fn: bool = True,
) -> Tuple[EnergyResult, str, str, List[Tuple[float, float]]]:

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="ignore")
    except Exception as e:
        return EnergyResult(
            0, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan")
        ), "unknown", str(e), []

    dt = 1.0 / sample_hz

    # ---------------- idle baseline ----------------
    idle_samples = []
    t_idle_end = time.perf_counter() + idle_seconds
    while time.perf_counter() < t_idle_end:
        idle_samples.append(_get_total_power_w(handle))
        time.sleep(dt)

    idle_power_w = _median(idle_samples)

    # ---------------- measurement ----------------
    samples: List[Tuple[float, float]] = []

    t_start = time.perf_counter()

    stop = threading.Event()

    def sampler():
        next_t = time.perf_counter()
        while not stop.is_set():
            t = time.perf_counter()
            if t >= t_start:  # 严格窗口起点
                samples.append((t, _get_total_power_w(handle)))
            next_t += dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    # 强制起点采样
    samples.append((t_start, _get_total_power_w(handle)))

    fn()

    t_end = time.perf_counter()

    # 强制终点采样
    samples.append((t_end, _get_total_power_w(handle)))

    stop.set()
    th.join(timeout=1.0)

    # 裁剪窗口
    samples = [(t, p) for t, p in samples if t_start <= t <= t_end]
    samples.sort(key=lambda x: x[0])

    if len(samples) < 2:
        pynvml.nvmlShutdown()
        return EnergyResult(
            len(samples), idle_power_w,
            float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan")
        ), gpu_name, "", samples

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

    pynvml.nvmlShutdown()

    return EnergyResult(
        len(samples),
        idle_power_w,
        avg_power_total,
        peak_power_total,
        energy_total,
        avg_power_eff,
        peak_power_eff,
        energy_eff,
    ), gpu_name, "", samples

"""Host CPU package and estimated vCPU energy monitoring.

The monitor is Linux-first and intentionally conservative: it only reports
real CPU package energy when RAPL powercap counters are available. Estimated
vCPU energy is derived from the container cgroup CPU time share in the same
sampling window.
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from acprof.config import DEFAULT_IDLE_SECONDS
from acprof.monitors.rapl_topology import discover_rapl_topology, dram_policy


@dataclass
class RaplDomain:
    name: str
    energy_path: str
    max_range_uj: int


@dataclass
class CPUSample:
    timestamp: float
    energy_uj: List[int]
    host_active_s: float
    container_cpu_s: Optional[float]
    dram_energy_uj: Optional[List[int]] = None


@dataclass
class DRAMEnergyResult:
    window_energy_j: float = float("nan")
    window_duration_s: float = float("nan")
    avg_power_w: float = float("nan")
    peak_power_w: float = float("nan")
    idle_power_w: float = float("nan")
    window_effective_energy_j: float = float("nan")
    status: str = "unavailable"
    error: str = ""


@dataclass
class CPUEnergyResult:
    cpu_energy_iters: int
    cpu_idle_power_w: float
    cpu_avg_power_total_w: float
    cpu_peak_power_total_w: float
    cpu_energy_total_j: float
    cpu_avg_power_eff_w: float
    cpu_peak_power_eff_w: float
    cpu_energy_eff_j: float
    vcpu_cpu_share: float
    vcpu_cpu_time_s: float
    vcpu_avg_power_total_w: float
    vcpu_peak_power_total_w: float
    vcpu_energy_total_j: float
    vcpu_avg_power_eff_w: float
    vcpu_peak_power_eff_w: float
    vcpu_energy_eff_j: float
    dram: DRAMEnergyResult = field(default_factory=DRAMEnergyResult)


def _nan_result(cpu_energy_iters: int = 0, cpu_idle_power_w: float = float("nan")) -> CPUEnergyResult:
    return CPUEnergyResult(
        cpu_energy_iters,
        cpu_idle_power_w,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )


def _read_int(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return int(f.read().strip())


def _clock_ticks_per_second() -> float:
    if hasattr(os, "sysconf"):
        return float(os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK")))
    return 100.0


def _discover_rapl_domains(powercap_root: str = "/sys/class/powercap") -> List[RaplDomain]:
    return _topology_domains(discover_rapl_topology(powercap_root), "package")


def _topology_domains(topology: dict, kind: str) -> List[RaplDomain]:
    return [RaplDomain(entry["id"], entry["energy_path"], entry["max_energy_range_uj"] or 0)
            for entry in topology["domains"] if entry["selected"] and entry["kind"] == kind]


def detect_cpu_power_source(powercap_root: str = "/sys/class/powercap") -> str:
    return "rapl" if _discover_rapl_domains(powercap_root) else "unavailable"


def detect_vcpu_power_method(
    powercap_root: str = "/sys/class/powercap",
    cgroup_root: str = "/sys/fs/cgroup",
) -> str:
    if detect_cpu_power_source(powercap_root) != "rapl":
        return "unavailable"
    if not os.path.isdir(cgroup_root):
        return "unavailable"
    return "rapl_cgroup_cpu_share"


def _read_host_active_s(proc_stat_path: str = "/proc/stat") -> float:
    with open(proc_stat_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    parts = first_line.split()
    if not parts or parts[0] != "cpu":
        raise RuntimeError(f"invalid proc stat line: {first_line!r}")

    values = [float(v) for v in parts[1:]]
    if len(values) < 5:
        raise RuntimeError(f"invalid proc stat cpu fields: {first_line!r}")

    idle = values[3] + values[4]
    active = sum(values) - idle
    return active / _clock_ticks_per_second()


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * pct
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _idle_power_windows(
    samples: List[CPUSample],
    domains: List[RaplDomain],
) -> List[Dict[str, Any]]:
    if len(samples) < 2:
        return []

    start_t = samples[0].timestamp
    windows: List[Dict[str, Any]] = []
    for idx in range(1, len(samples)):
        prev = samples[idx - 1]
        curr = samples[idx]
        dt = curr.timestamp - prev.timestamp
        if dt <= 0:
            continue
        energy_j = _energy_delta_j(prev, curr, domains)
        windows.append({
            "index": len(windows),
            "t0_s": prev.timestamp - start_t,
            "t1_s": curr.timestamp - start_t,
            "duration_s": dt,
            "energy_j": energy_j,
            "power_w": energy_j / dt,
        })
    return windows


def _build_idle_trace(
    samples: List[CPUSample],
    domains: List[RaplDomain],
    trace_interval_s: float,
) -> Dict[str, Any]:
    if len(samples) < 2:
        return {}

    start = samples[0]
    end = samples[-1]
    duration_s = end.timestamp - start.timestamp
    energy_j = _energy_delta_j(start, end, domains)
    power_windows = _idle_power_windows(samples, domains)
    powers = [row["power_w"] for row in power_windows]
    top_windows = sorted(power_windows, key=lambda row: row["power_w"], reverse=True)[:5]

    container_delta_s = float("nan")
    if start.container_cpu_s is not None and end.container_cpu_s is not None:
        container_delta_s = end.container_cpu_s - start.container_cpu_s

    trace: Dict[str, Any] = {
        "idle_trace_schema": "cpu_rapl_control_v1",
        "idle_trace_interval_s": trace_interval_s,
        "actual_idle_duration_s": duration_s,
        "rapl_domain_names": [domain.name for domain in domains],
        "rapl_start_uj": start.energy_uj,
        "rapl_end_uj": end.energy_uj,
        "rapl_energy_delta_j": energy_j,
        "rapl_avg_power_w": energy_j / duration_s if duration_s > 0 else float("nan"),
        "rapl_trace": {
            "interval_s": trace_interval_s,
            "sample_count": len(samples),
            "window_count": len(power_windows),
            "power_min_w": min(powers) if powers else float("nan"),
            "power_p50_w": _percentile(powers, 0.50),
            "power_p95_w": _percentile(powers, 0.95),
            "power_max_w": max(powers) if powers else float("nan"),
            "power_windows": power_windows,
            "top_power_windows": top_windows,
        },
        "idle_host_active_delta_s": end.host_active_s - start.host_active_s,
        "idle_container_cpu_delta_s": container_delta_s,
    }
    return trace


def _join_cgroup_path(root: str, relative: str, leaf: str) -> str:
    rel = relative.strip("/")
    return os.path.join(root, rel, leaf) if rel else os.path.join(root, leaf)


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


def _resolve_container_cpu_reader(
    container_name: str,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_root: str = "/proc",
) -> Optional[Callable[[], float]]:
    if not container_name:
        return None

    pid = _docker_container_pid(container_name)
    cgroup_file = os.path.join(proc_root, str(pid), "cgroup")
    with open(cgroup_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            stat_path = _join_cgroup_path(cgroup_root, parts[2], "cpu.stat")
            if os.path.exists(stat_path):
                def read_v2(path: str = stat_path) -> float:
                    with open(path, "r", encoding="utf-8") as stat_f:
                        for stat_line in stat_f:
                            key, _, value = stat_line.strip().partition(" ")
                            if key == "usage_usec":
                                return float(value) / 1_000_000.0
                    raise RuntimeError(f"usage_usec missing from {path}")

                return read_v2

    return None


def _energy_delta_j(prev: CPUSample, curr: CPUSample, domains: List[RaplDomain]) -> float:
    total_uj = 0
    for index, domain in enumerate(domains):
        old = prev.energy_uj[index]
        new = curr.energy_uj[index]
        if new >= old:
            delta = new - old
        elif domain.max_range_uj > 0:
            delta = (domain.max_range_uj - old) + new
        else:
            return float("nan")
        if old < 0 or new < 0 or delta < 0:
            return float("nan")
        total_uj += delta
    return float(total_uj) / 1_000_000.0


def _result_from_samples(
    samples: List[CPUSample],
    idle_power_w: float,
    domains: List[RaplDomain],
    min_power_interval_s: float = 0.0,
) -> CPUEnergyResult:
    if len(samples) < 2 or not domains:
        return _nan_result(len(samples), idle_power_w)

    samples = sorted(samples, key=lambda sample: sample.timestamp)
    duration_s = samples[-1].timestamp - samples[0].timestamp
    if duration_s <= 0:
        return _nan_result(len(samples), idle_power_w)

    total_energy_j = 0.0
    energy_eff_j = 0.0
    total_container_cpu_s = 0.0
    total_host_active_s = 0.0
    vcpu_energy_total_j = 0.0
    vcpu_energy_eff_j = 0.0
    total_power_peaks: List[float] = []
    eff_power_peaks: List[float] = []
    vcpu_total_power_peaks: List[float] = []
    vcpu_eff_power_peaks: List[float] = []
    has_total_interval = False
    has_eff_interval = False

    for idx in range(1, len(samples)):
        prev = samples[idx - 1]
        curr = samples[idx]
        dt = curr.timestamp - prev.timestamp
        if dt <= 0:
            continue

        energy_j = _energy_delta_j(prev, curr, domains)
        power_w = energy_j / dt
        eff_power_w = power_w - idle_power_w if idle_power_w == idle_power_w else float("nan")

        total_energy_j += energy_j
        has_total_interval = True
        if eff_power_w == eff_power_w:
            energy_eff_j += eff_power_w * dt
            has_eff_interval = True
        use_peak_interval = dt >= min_power_interval_s
        if use_peak_interval:
            total_power_peaks.append(power_w)
            if eff_power_w == eff_power_w:
                eff_power_peaks.append(eff_power_w)

        if prev.container_cpu_s is None or curr.container_cpu_s is None:
            continue

        host_delta = curr.host_active_s - prev.host_active_s
        container_delta = curr.container_cpu_s - prev.container_cpu_s
        if host_delta <= 0 or container_delta < 0:
            continue

        share = min(1.0, max(0.0, container_delta / host_delta))
        total_container_cpu_s += container_delta
        total_host_active_s += host_delta

        vcpu_power_w = power_w * share
        vcpu_energy_total_j += energy_j * share
        if use_peak_interval:
            vcpu_total_power_peaks.append(vcpu_power_w)
        if eff_power_w == eff_power_w:
            vcpu_eff_power_w = eff_power_w * share
            vcpu_energy_eff_j += eff_power_w * dt * share
            if use_peak_interval:
                vcpu_eff_power_peaks.append(vcpu_eff_power_w)

    if not has_total_interval:
        return _nan_result(len(samples), idle_power_w)

    avg_power_total = total_energy_j / duration_s
    peak_power_total = max(total_power_peaks) if total_power_peaks else float("nan")
    avg_power_eff = energy_eff_j / duration_s if has_eff_interval else float("nan")
    peak_power_eff = max(eff_power_peaks) if eff_power_peaks else float("nan")

    if total_host_active_s > 0:
        vcpu_share = min(1.0, max(0.0, total_container_cpu_s / total_host_active_s))
        vcpu_avg_power_total = vcpu_energy_total_j / duration_s
        vcpu_peak_power_total = (
            max(vcpu_total_power_peaks) if vcpu_total_power_peaks else float("nan")
        )
        vcpu_avg_power_eff = vcpu_energy_eff_j / duration_s if has_eff_interval else float("nan")
        vcpu_peak_power_eff = (
            max(vcpu_eff_power_peaks) if vcpu_eff_power_peaks else float("nan")
        )
    else:
        vcpu_share = float("nan")
        total_container_cpu_s = float("nan")
        vcpu_avg_power_total = float("nan")
        vcpu_peak_power_total = float("nan")
        vcpu_energy_total_j = float("nan")
        vcpu_avg_power_eff = float("nan")
        vcpu_peak_power_eff = float("nan")
        vcpu_energy_eff_j = float("nan")

    return CPUEnergyResult(
        len(samples),
        idle_power_w,
        avg_power_total,
        peak_power_total,
        total_energy_j,
        avg_power_eff,
        peak_power_eff,
        energy_eff_j if has_eff_interval else float("nan"),
        vcpu_share,
        total_container_cpu_s,
        vcpu_avg_power_total,
        vcpu_peak_power_total,
        vcpu_energy_total_j,
        vcpu_avg_power_eff,
        vcpu_peak_power_eff,
        vcpu_energy_eff_j,
    )


def _dram_result_from_samples(samples, domains, idle_power_w, status="available", error="",
                              min_power_interval_s=0.0) -> DRAMEnergyResult:
    result = DRAMEnergyResult(idle_power_w=idle_power_w, status=status, error=error)
    if status != "available":
        result.error = error or ("DRAM not requested" if status == "not_requested" else "incomplete DRAM domain coverage")
        return result
    if error or len(samples) < 2 or not domains or any(
        sample.dram_energy_uj is None or len(sample.dram_energy_uj) != len(domains) for sample in samples
    ):
        result.status = "error"
        result.error = error or "missing DRAM samples"
        return result
    duration = samples[-1].timestamp - samples[0].timestamp
    if duration <= 0:
        result.status, result.error = "error", "invalid DRAM window duration"
        return result
    energy = 0.0
    peaks = []
    for previous, current in zip(samples, samples[1:]):
        dt = current.timestamp - previous.timestamp
        if dt <= 0:
            result.status, result.error = "error", "invalid DRAM sample interval"
            return result
        before = CPUSample(previous.timestamp, previous.dram_energy_uj, 0, None)
        after = CPUSample(current.timestamp, current.dram_energy_uj, 0, None)
        interval_energy = _energy_delta_j(before, after, domains)
        if not math.isfinite(interval_energy):
            result.status, result.error = "error", "invalid DRAM counter or unknown wrap range"
            return result
        energy += interval_energy
        if dt >= min_power_interval_s:
            peaks.append(interval_energy / dt)
    result.window_energy_j = energy
    result.window_duration_s = duration
    result.avg_power_w = energy / duration
    result.peak_power_w = max(peaks) if peaks else float("nan")
    if math.isfinite(idle_power_w):
        result.window_effective_energy_j = energy - idle_power_w * duration
    result.status = "verified"
    return result


class CPUEnergyMonitor:
    def __init__(
        self,
        sample_hz: float = 20.0,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        container_name: str = "",
        powercap_root: str = "/sys/class/powercap",
        cgroup_root: str = "/sys/fs/cgroup",
        proc_root: str = "/proc",
        dram_energy: str = "auto",
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.idle_seconds = float(idle_seconds)
        self.container_name = container_name
        self.powercap_root = powercap_root
        self.cgroup_root = cgroup_root
        self.proc_root = proc_root
        self.dt = 1.0 / self.sample_hz

        dram_policy("full", dram_energy)
        self.topology = discover_rapl_topology(powercap_root)
        self.domains = _topology_domains(self.topology, "package")
        self.dram_energy = dram_energy
        self.dram_domains = _topology_domains(self.topology, "dram") if dram_energy != "off" else []
        self.dram_status = self.topology["dram_status"] if dram_energy != "off" else "not_requested"
        self.dram_idle_power_w = float("nan")
        self._dram_runtime_error = ""
        if dram_energy == "required" and self.dram_status != "available":
            raise RuntimeError(f"required DRAM RAPL unavailable: {self.dram_status}")
        self.idle_power_w = float("nan")
        self.idle_trace: Dict[str, Any] = {}
        self.samples: List[CPUSample] = []
        self._container_cpu_reader: Optional[Callable[[], float]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._t_start: Optional[float] = None
        self._t_end: Optional[float] = None
        self._init_error = ""
        self._runtime_error = ""

        if not self.domains:
            self._init_error = "RAPL powercap energy_uj unavailable"
            return

        try:
            self._container_cpu_reader = _resolve_container_cpu_reader(
                container_name,
                cgroup_root=cgroup_root,
                proc_root=proc_root,
            )
        except Exception as exc:
            self._runtime_error = f"container cgroup unavailable: {exc}"
            self._container_cpu_reader = None

    @property
    def available(self) -> bool:
        return bool(self.domains) and not self._init_error


    def apply_control_baseline(
        self,
        result: CPUEnergyResult,
        samples: List[CPUSample],
        *,
        trace: bool = False,
    ) -> float:
        """Use a monitor-matched blank window as the workload baseline."""
        self.idle_power_w = float(result.cpu_avg_power_total_w)
        self.dram_idle_power_w = getattr(getattr(result, "dram", None), "avg_power_w", float("nan"))
        self.idle_trace = {}
        if trace and len(samples) >= 2:
            self.idle_trace = _build_idle_trace(
                samples,
                self.domains,
                self.dt,
            )
            self.idle_trace.update({
                "cpu_idle_baseline_method": "matched_control_energy_delta",
                "idle_proc_cpu_top": [],
                "idle_proc_cpu_note": (
                    "per-process snapshots are collected after the matched control "
                    "window so they do not perturb its RAPL baseline"
                ),
            })
        return self.idle_power_w


    def start(self) -> None:
        if not self.available:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("CPU energy monitor is already running")

        self.samples = []
        self._dram_runtime_error = ""
        self._stop_event = threading.Event()
        self._t_start = time.perf_counter()
        self._t_end = None
        self._append_sample(self._t_start)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[CPUEnergyResult, str, List[CPUSample]]:
        if not self.available:
            return _nan_result(), self._init_error, []

        if self._t_start is None:
            return _nan_result(0, self.idle_power_w), self._runtime_error, []

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
        result = _result_from_samples(samples, self.idle_power_w, self.domains,
                                      min_power_interval_s=0.5 * self.dt)
        result.dram = _dram_result_from_samples(
            samples, self.dram_domains, self.dram_idle_power_w, self.dram_status,
            self._dram_runtime_error, min_power_interval_s=0.5 * self.dt)
        return result, self._runtime_error, samples


    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.stop()

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

    def _read_sample(self, timestamp: float) -> CPUSample:
        energy_uj = [_read_int(domain.energy_path) for domain in self.domains]
        host_active_s = _read_host_active_s(os.path.join(self.proc_root, "stat"))
        container_cpu_s = None
        if self._container_cpu_reader is not None:
            container_cpu_s = self._container_cpu_reader()
        dram_values = None
        if self.dram_status == "available":
            try:
                dram_values = [_read_int(domain.energy_path) for domain in self.dram_domains]
            except (OSError, ValueError) as exc:
                self._dram_runtime_error = str(exc)
        return CPUSample(timestamp, energy_uj, host_active_s, container_cpu_s, dram_values)

    def _append_sample(self, timestamp: float) -> None:
        try:
            self.samples.append(self._read_sample(timestamp))
        except Exception as exc:
            self._runtime_error = str(exc)

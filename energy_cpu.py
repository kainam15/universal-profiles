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
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


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


def _discover_rapl_domains(powercap_root: str = "/sys/class/powercap") -> List[RaplDomain]:
    if not os.path.isdir(powercap_root):
        return []

    domains: List[RaplDomain] = []
    for entry in sorted(os.scandir(powercap_root), key=lambda item: item.name):
        if not entry.is_dir(follow_symlinks=True):
            continue
        if entry.name.count(":") != 1:
            continue

        name_path = os.path.join(entry.path, "name")
        if os.path.exists(name_path):
            try:
                with open(name_path, "r", encoding="utf-8") as f:
                    domain_name = f.read().strip()
            except Exception:
                continue
            if not domain_name.startswith("package-"):
                continue

        energy_path = os.path.join(entry.path, "energy_uj")
        if not os.path.isfile(energy_path):
            continue

        max_path = os.path.join(entry.path, "max_energy_range_uj")
        try:
            max_range = _read_int(max_path) if os.path.exists(max_path) else 0
            _read_int(energy_path)
        except Exception:
            continue

        domains.append(RaplDomain(entry.name, energy_path, max_range))

    return domains


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
    if hasattr(os, "sysconf"):
        ticks = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))
    else:
        ticks = 100
    return active / float(ticks)


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

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = set(parts[1].split(","))
        if "cpuacct" not in controllers:
            continue

        candidates = [
            _join_cgroup_path(os.path.join(cgroup_root, parts[1]), parts[2], "cpuacct.usage"),
            _join_cgroup_path(os.path.join(cgroup_root, "cpuacct"), parts[2], "cpuacct.usage"),
            _join_cgroup_path(cgroup_root, parts[2], "cpuacct.usage"),
        ]
        for usage_path in candidates:
            if os.path.exists(usage_path):
                def read_v1(path: str = usage_path) -> float:
                    return float(_read_int(path)) / 1_000_000_000.0

                return read_v1

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
            delta = new
        total_uj += max(0, delta)
    return float(total_uj) / 1_000_000.0


def _result_from_samples(
    samples: List[CPUSample],
    idle_power_w: float,
    domains: List[RaplDomain],
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
    total_powers: List[float] = []
    eff_powers: List[float] = []
    vcpu_total_powers: List[float] = []
    vcpu_eff_powers: List[float] = []

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
        total_powers.append(power_w)
        if eff_power_w == eff_power_w:
            energy_eff_j += eff_power_w * dt
            eff_powers.append(eff_power_w)

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
        vcpu_total_powers.append(vcpu_power_w)
        if eff_power_w == eff_power_w:
            vcpu_eff_power_w = eff_power_w * share
            vcpu_energy_eff_j += eff_power_w * dt * share
            vcpu_eff_powers.append(vcpu_eff_power_w)

    if not total_powers:
        return _nan_result(len(samples), idle_power_w)

    avg_power_total = total_energy_j / duration_s
    peak_power_total = max(total_powers)
    avg_power_eff = energy_eff_j / duration_s if eff_powers else float("nan")
    peak_power_eff = max(eff_powers) if eff_powers else float("nan")

    if total_host_active_s > 0 and vcpu_total_powers:
        vcpu_share = min(1.0, max(0.0, total_container_cpu_s / total_host_active_s))
        vcpu_avg_power_total = vcpu_energy_total_j / duration_s
        vcpu_peak_power_total = max(vcpu_total_powers)
        vcpu_avg_power_eff = vcpu_energy_eff_j / duration_s if vcpu_eff_powers else float("nan")
        vcpu_peak_power_eff = max(vcpu_eff_powers) if vcpu_eff_powers else float("nan")
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
        energy_eff_j if eff_powers else float("nan"),
        vcpu_share,
        total_container_cpu_s,
        vcpu_avg_power_total,
        vcpu_peak_power_total,
        vcpu_energy_total_j,
        vcpu_avg_power_eff,
        vcpu_peak_power_eff,
        vcpu_energy_eff_j,
    )


class CPUEnergyMonitor:
    def __init__(
        self,
        sample_hz: float = 20.0,
        idle_seconds: float = 3.0,
        container_name: str = "",
        powercap_root: str = "/sys/class/powercap",
        cgroup_root: str = "/sys/fs/cgroup",
        proc_root: str = "/proc",
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.idle_seconds = float(idle_seconds)
        self.container_name = container_name
        self.powercap_root = powercap_root
        self.cgroup_root = cgroup_root
        self.proc_root = proc_root
        self.dt = 1.0 / self.sample_hz

        self.domains = _discover_rapl_domains(powercap_root)
        self.idle_power_w = float("nan")
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

    def measure_idle(self) -> float:
        if not self.available:
            return float("nan")

        try:
            start = self._read_sample(time.perf_counter())
            sleep_s = max(0.0, self.idle_seconds)
            if sleep_s > 0:
                time.sleep(sleep_s)
            end = self._read_sample(time.perf_counter())
        except Exception as exc:
            self._runtime_error = str(exc)
            self.idle_power_w = float("nan")
            return self.idle_power_w

        duration_s = end.timestamp - start.timestamp
        if duration_s > 0:
            self.idle_power_w = _energy_delta_j(start, end, self.domains) / duration_s
        else:
            self.idle_power_w = float("nan")
        return self.idle_power_w

    def start(self) -> None:
        if not self.available:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("CPU energy monitor is already running")

        self.samples = []
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
        return _result_from_samples(samples, self.idle_power_w, self.domains), self._runtime_error, samples

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.stop()

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

    def _read_sample(self, timestamp: float) -> CPUSample:
        energy_uj = [_read_int(domain.energy_path) for domain in self.domains]
        host_active_s = _read_host_active_s(os.path.join(self.proc_root, "stat"))
        container_cpu_s = None
        if self._container_cpu_reader is not None:
            container_cpu_s = self._container_cpu_reader()
        return CPUSample(timestamp, energy_uj, host_active_s, container_cpu_s)

    def _append_sample(self, timestamp: float) -> None:
        try:
            self.samples.append(self._read_sample(timestamp))
        except Exception as exc:
            self._runtime_error = str(exc)


def measure_energy_threaded(
    fn: Callable[[], object],
    sample_hz: float = 20.0,
    idle_seconds: float = 3.0,
    container_name: str = "",
) -> Tuple[CPUEnergyResult, str, List[CPUSample]]:
    monitor = CPUEnergyMonitor(
        sample_hz=sample_hz,
        idle_seconds=idle_seconds,
        container_name=container_name,
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

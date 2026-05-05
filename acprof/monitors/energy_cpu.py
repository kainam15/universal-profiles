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
from typing import Any, Callable, Dict, List, Optional, Tuple


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
class ProcCPUSample:
    pid: int
    ppid: int
    comm: str
    cpu_s: float
    cmdline: str


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


def _clock_ticks_per_second() -> float:
    if hasattr(os, "sysconf"):
        return float(os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK")))
    return 100.0


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
    return active / _clock_ticks_per_second()


def _read_proc_cmdline(pid_dir: str, fallback: str) -> str:
    try:
        with open(os.path.join(pid_dir, "cmdline"), "rb") as f:
            raw = f.read()
    except OSError:
        return fallback

    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    return cmdline or fallback


def _parse_proc_stat_process(stat_text: str, ticks: float) -> ProcCPUSample:
    left = stat_text.find("(")
    right = stat_text.rfind(")")
    if left < 0 or right < left:
        raise ValueError(f"invalid /proc stat process line: {stat_text!r}")

    pid = int(stat_text[:left].strip())
    comm = stat_text[left + 1:right]
    fields = stat_text[right + 2:].split()
    if len(fields) <= 12:
        raise ValueError(f"short /proc stat process line: {stat_text!r}")

    ppid = int(fields[1])
    utime_ticks = float(fields[11])
    stime_ticks = float(fields[12])
    return ProcCPUSample(
        pid=pid,
        ppid=ppid,
        comm=comm,
        cpu_s=(utime_ticks + stime_ticks) / ticks,
        cmdline=comm,
    )


def _read_proc_cpu_snapshot(proc_root: str = "/proc") -> Dict[int, ProcCPUSample]:
    if not os.path.isdir(proc_root):
        return {}

    ticks = _clock_ticks_per_second()
    snapshot: Dict[int, ProcCPUSample] = {}
    for entry in os.scandir(proc_root):
        if not entry.name.isdigit():
            continue
        pid_dir = entry.path
        try:
            with open(os.path.join(pid_dir, "stat"), "r", encoding="utf-8") as f:
                sample = _parse_proc_stat_process(f.read().strip(), ticks)
            sample.cmdline = _read_proc_cmdline(pid_dir, sample.comm)
        except (OSError, ValueError):
            continue
        snapshot[sample.pid] = sample
    return snapshot


def _proc_cpu_delta_top(
    start: Dict[int, Any],
    end: Dict[int, Any],
    limit: int = 15,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pid, curr in end.items():
        prev = start.get(pid)
        prev_cpu_s = float(prev.cpu_s) if prev is not None else 0.0
        delta_s = float(curr.cpu_s) - prev_cpu_s
        if delta_s <= 0:
            continue
        rows.append({
            "pid": int(curr.pid),
            "ppid": int(curr.ppid),
            "comm": str(curr.comm),
            "cpu_time_ms": delta_s * 1000.0,
            "cmd": str(curr.cmdline),
            "started_during_idle": prev is None,
        })

    rows.sort(key=lambda row: (row["cpu_time_ms"], row["pid"]), reverse=True)
    return rows[:limit]


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
    proc_start: Dict[int, Any],
    proc_end: Dict[int, Any],
    proc_snapshot_error: str = "",
    idle_started_at_unix_s: Optional[float] = None,
    idle_ended_at_unix_s: Optional[float] = None,
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
        "idle_trace_schema": "cpu_rapl_idle_v1",
        "idle_trace_interval_s": trace_interval_s,
        "idle_started_at_unix_s": idle_started_at_unix_s,
        "idle_ended_at_unix_s": idle_ended_at_unix_s,
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
        "idle_proc_cpu_top": _proc_cpu_delta_top(proc_start, proc_end),
        "idle_proc_cpu_note": (
            "proc delta is start/end sampling; short-lived processes that exit before "
            "idle end require bpftrace to attribute"
        ),
    }
    if proc_snapshot_error:
        trace["idle_proc_cpu_error"] = proc_snapshot_error
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

    def measure_idle(self, trace_interval_s: Optional[float] = None) -> float:
        self.idle_trace = {}
        if not self.available:
            return float("nan")

        try:
            if trace_interval_s is not None and trace_interval_s > 0:
                start, end, _trace_samples = self._measure_idle_with_trace(float(trace_interval_s))
            else:
                start = self._read_sample(time.perf_counter())
                sleep_s = max(0.0, self.idle_seconds)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                end = self._read_sample(time.perf_counter())
        except Exception as exc:
            self._runtime_error = str(exc)
            self.idle_power_w = float("nan")
            self.idle_trace = {}
            return self.idle_power_w

        duration_s = end.timestamp - start.timestamp
        if duration_s > 0:
            self.idle_power_w = _energy_delta_j(start, end, self.domains) / duration_s
        else:
            self.idle_power_w = float("nan")
        return self.idle_power_w

    def _measure_idle_with_trace(self, trace_interval_s: float) -> Tuple[CPUSample, CPUSample, List[CPUSample]]:
        sleep_s = max(0.0, self.idle_seconds)
        interval_s = max(0.001, trace_interval_s)
        proc_start: Dict[int, Any] = {}
        proc_end: Dict[int, Any] = {}
        proc_snapshot_error = ""

        idle_started_at_unix_s = time.time()
        start = self._read_sample(time.perf_counter())
        try:
            proc_start = _read_proc_cpu_snapshot(self.proc_root)
        except Exception as exc:
            proc_snapshot_error = repr(exc)

        trace_samples = [start]
        deadline = start.timestamp + sleep_s
        while True:
            now = time.perf_counter()
            remaining_s = deadline - now
            if remaining_s <= 0:
                break
            time.sleep(min(interval_s, remaining_s))
            sample_t = time.perf_counter()
            if sample_t > trace_samples[-1].timestamp:
                trace_samples.append(self._read_sample(sample_t))

        if len(trace_samples) == 1 or trace_samples[-1].timestamp < deadline:
            end_t = time.perf_counter()
            if end_t > trace_samples[-1].timestamp:
                trace_samples.append(self._read_sample(end_t))

        end = trace_samples[-1]
        try:
            proc_end = _read_proc_cpu_snapshot(self.proc_root)
        except Exception as exc:
            proc_snapshot_error = "; ".join(item for item in [proc_snapshot_error, repr(exc)] if item)
        idle_ended_at_unix_s = time.time()

        self.idle_trace = _build_idle_trace(
            trace_samples,
            self.domains,
            interval_s,
            proc_start,
            proc_end,
            proc_snapshot_error=proc_snapshot_error,
            idle_started_at_unix_s=idle_started_at_unix_s,
            idle_ended_at_unix_s=idle_ended_at_unix_s,
        )
        return start, end, trace_samples

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
        return (
            _result_from_samples(
                samples,
                self.idle_power_w,
                self.domains,
                min_power_interval_s=0.5 * self.dt,
            ),
            self._runtime_error,
            samples,
        )

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

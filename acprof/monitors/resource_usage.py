"""Container CPU/memory and NVML GPU utilization monitoring."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    cpu_freq_avg_hz: Optional[float] = None
    cpu_freq_peak_hz: Optional[float] = None
    gpu_sm_clock_mhz: Optional[float] = None
    gpu_memory_clock_mhz: Optional[float] = None
    gpu_pstate: Optional[str] = None
    gpu_temp_c: Optional[float] = None
    container_swap_usage_bytes: Optional[int] = None


@dataclass
class ResourceUsageResult:
    resource_usage_iters: int
    container_cpu_util_avg_pct: float
    container_cpu_util_peak_pct: float
    cpu_freq_avg_hz: float
    cpu_freq_peak_hz: float
    container_mem_usage_avg_bytes: float
    container_mem_usage_peak_bytes: float
    container_mem_util_avg_pct: float
    container_mem_util_peak_pct: float
    gpu_util_avg_pct: float
    gpu_util_peak_pct: float
    gpu_sm_clock_mhz: float
    gpu_memory_clock_mhz: float
    gpu_pstate: str
    gpu_temp_c: float
    gpu_mem_used_avg_bytes: float
    gpu_mem_used_peak_bytes: float
    gpu_mem_util_avg_pct: float
    gpu_mem_util_peak_pct: float
    gpu_mem_total_bytes: float
    container_swap_limit_bytes: float = float("nan")
    container_swap_usage_avg_bytes: float = float("nan")
    container_swap_usage_peak_bytes: float = float("nan")
    container_io_read_bytes: float = float("nan")
    container_io_write_bytes: float = float("nan")
    container_cpu_nr_periods_delta: float = float("nan")
    container_cpu_nr_throttled_delta: float = float("nan")
    container_cpu_throttled_period_ratio_pct: float = float("nan")
    container_cpu_throttled_time_s: float = float("nan")
    container_cpu_pressure_some_stall_pct: float = float("nan")
    container_cpu_pressure_full_stall_pct: float = float("nan")
    container_mem_high_events_delta: float = float("nan")
    container_mem_max_events_delta: float = float("nan")
    container_mem_oom_events_delta: float = float("nan")
    container_mem_oom_kill_events_delta: float = float("nan")
    container_mem_pressure_some_stall_pct: float = float("nan")
    container_mem_pressure_full_stall_pct: float = float("nan")
    container_mem_peak_cgroup_bytes: float = float("nan")
    container_mem_anon_bytes_end: float = float("nan")
    container_mem_file_bytes_end: float = float("nan")
    container_mem_slab_bytes_end: float = float("nan")
    container_mem_pgfault_delta: float = float("nan")
    container_mem_pgmajfault_delta: float = float("nan")
    container_mem_workingset_refault_delta: float = float("nan")
    container_io_read_ops: float = float("nan")
    container_io_write_ops: float = float("nan")
    container_io_pressure_some_stall_pct: float = float("nan")
    container_io_pressure_full_stall_pct: float = float("nan")
    container_pids_current_end: float = float("nan")
    container_pids_peak_cgroup: float = float("nan")
    container_pids_max_events_delta: float = float("nan")


@dataclass
class _ContainerReaders:
    cpu: Optional[Callable[[], float]] = None
    memory: Optional[Callable[[], int]] = None
    swap: Optional[Callable[[], int]] = None
    swap_limit: Optional[Callable[[], int]] = None
    io: Optional[Callable[[], Tuple[int, int]]] = None
    cpu_throttle: Optional[Callable[[], Dict[str, float]]] = None
    memory_events: Optional[Callable[[], Dict[str, float]]] = None
    cpu_pressure: Optional[Callable[[], Dict[str, float]]] = None
    memory_pressure: Optional[Callable[[], Dict[str, float]]] = None
    io_pressure: Optional[Callable[[], Dict[str, float]]] = None
    memory_peak: Optional[Callable[[], Dict[str, float]]] = None
    memory_stat: Optional[Callable[[], Dict[str, float]]] = None
    io_operations: Optional[Callable[[], Dict[str, float]]] = None
    pids: Optional[Callable[[], Dict[str, float]]] = None


@dataclass
class _WindowCounterSnapshots:
    cpu_throttle: Optional[Dict[str, float]] = None
    memory_events: Optional[Dict[str, float]] = None
    cpu_pressure: Optional[Dict[str, float]] = None
    memory_pressure: Optional[Dict[str, float]] = None
    io_pressure: Optional[Dict[str, float]] = None
    memory_peak: Optional[Dict[str, float]] = None
    memory_stat: Optional[Dict[str, float]] = None
    io_operations: Optional[Dict[str, float]] = None
    pids: Optional[Dict[str, float]] = None


def _nan_result(resource_usage_iters: int = 0) -> ResourceUsageResult:
    nan = float("nan")
    return ResourceUsageResult(
        resource_usage_iters=resource_usage_iters,
        container_cpu_util_avg_pct=nan,
        container_cpu_util_peak_pct=nan,
        cpu_freq_avg_hz=nan,
        cpu_freq_peak_hz=nan,
        container_mem_usage_avg_bytes=nan,
        container_mem_usage_peak_bytes=nan,
        container_mem_util_avg_pct=nan,
        container_mem_util_peak_pct=nan,
        gpu_util_avg_pct=nan,
        gpu_util_peak_pct=nan,
        gpu_sm_clock_mhz=nan,
        gpu_memory_clock_mhz=nan,
        gpu_pstate="nan",
        gpu_temp_c=nan,
        gpu_mem_used_avg_bytes=nan,
        gpu_mem_used_peak_bytes=nan,
        gpu_mem_util_avg_pct=nan,
        gpu_mem_util_peak_pct=nan,
        gpu_mem_total_bytes=nan,
        container_swap_limit_bytes=nan,
        container_swap_usage_avg_bytes=nan,
        container_swap_usage_peak_bytes=nan,
        container_io_read_bytes=nan,
        container_io_write_bytes=nan,
    )


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _normalize_pstate(value: object) -> Optional[str]:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized.startswith("P") and normalized[1:].isdigit():
            numeric = int(normalized[1:])
            return normalized if 0 <= numeric <= 15 else None
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return f"P{numeric}" if 0 <= numeric <= 15 else None


def _dominant_pstate(values: List[str]) -> str:
    normalized = [state for value in values if (state := _normalize_pstate(value))]
    if not normalized:
        return "nan"
    counts = Counter(normalized)
    highest_count = max(counts.values())
    candidates = [state for state, count in counts.items() if count == highest_count]
    return min(candidates, key=lambda state: int(state[1:]))


def _read_int(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return int(f.read().strip())


def _read_cgroup_limit(path: str) -> int:
    """Read a cgroup byte limit, using -1 for the kernel's unlimited value."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip().lower()
    if raw == "max":
        return -1
    value = int(raw)
    # cgroup v1 represents an unlimited limit with a very large page-aligned
    # integer rather than the cgroup v2 ``max`` token.
    return -1 if value >= (1 << 60) else value


def _read_v1_swap_usage(memsw_path: str, memory_path: str) -> int:
    return max(0, _read_int(memsw_path) - _read_int(memory_path))


def _read_v1_swap_limit(memsw_path: str, memory_path: str) -> int:
    memsw_limit = _read_cgroup_limit(memsw_path)
    memory_limit = _read_cgroup_limit(memory_path)
    if memsw_limit < 0:
        return -1
    if memory_limit < 0:
        return memsw_limit
    return max(0, memsw_limit - memory_limit)


def _read_cgroup_v2_io_stat(path: str) -> Tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            for token in parts[1:]:
                key, separator, raw_value = token.partition("=")
                if not separator:
                    continue
                if key == "rbytes":
                    read_bytes += int(raw_value)
                elif key == "wbytes":
                    write_bytes += int(raw_value)
    return read_bytes, write_bytes


def _read_cgroup_v2_io_operations(path: str) -> Dict[str, float]:
    read_ops = 0
    write_ops = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            for token in parts[1:]:
                key, separator, raw_value = token.partition("=")
                if not separator:
                    continue
                if key == "rios":
                    read_ops += int(raw_value)
                elif key == "wios":
                    write_ops += int(raw_value)
    return {"read_ops": float(read_ops), "write_ops": float(write_ops)}


def _read_cgroup_v1_io_service_bytes(path: str) -> Tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3 or parts[0].lower() == "total":
                continue
            operation = parts[1].strip().lower()
            if operation == "read":
                read_bytes += int(parts[2])
            elif operation == "write":
                write_bytes += int(parts[2])
    return read_bytes, write_bytes


def _read_cpu_stat_usage_s(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key, _, value = line.strip().partition(" ")
            if key == "usage_usec":
                return float(value) / 1_000_000.0
    raise RuntimeError(f"usage_usec missing from {path}")


def _read_flat_cgroup_stats(path: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) != 2:
                continue
            try:
                stats[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return stats


def _read_cgroup_memory_peak(path: str) -> Dict[str, float]:
    return {"peak": float(_read_int(path))}


def _read_cgroup_memory_stat(path: str) -> Dict[str, float]:
    stats = _read_flat_cgroup_stats(path)
    slab = stats.get("slab", float("nan"))
    if slab != slab:
        slab_reclaimable = stats.get("slab_reclaimable", float("nan"))
        slab_unreclaimable = stats.get("slab_unreclaimable", float("nan"))
        if slab_reclaimable == slab_reclaimable and slab_unreclaimable == slab_unreclaimable:
            slab = slab_reclaimable + slab_unreclaimable

    refault_anon = stats.get("workingset_refault_anon", float("nan"))
    refault_file = stats.get("workingset_refault_file", float("nan"))
    refault_total = stats.get("workingset_refault", float("nan"))
    if refault_total != refault_total and refault_anon == refault_anon and refault_file == refault_file:
        refault_total = refault_anon + refault_file

    return {
        "anon": stats.get("anon", float("nan")),
        "file": stats.get("file", float("nan")),
        "slab": slab,
        "pgfault": stats.get("pgfault", float("nan")),
        "pgmajfault": stats.get("pgmajfault", float("nan")),
        "workingset_refault": refault_total,
    }


def _read_cgroup_pids(
    current_path: Optional[str],
    peak_path: Optional[str],
    events_path: Optional[str],
) -> Dict[str, float]:
    result = {
        "current": float("nan"),
        "peak": float("nan"),
        "max_events": float("nan"),
    }
    if current_path is not None:
        result["current"] = float(_read_int(current_path))
    if peak_path is not None:
        result["peak"] = float(_read_int(peak_path))
    if events_path is not None:
        result["max_events"] = _read_flat_cgroup_stats(events_path).get(
            "max",
            float("nan"),
        )
    return result


def _read_cgroup_v2_cpu_throttle(path: str) -> Dict[str, float]:
    stats = _read_flat_cgroup_stats(path)
    return {
        "nr_periods": stats.get("nr_periods", float("nan")),
        "nr_throttled": stats.get("nr_throttled", float("nan")),
        "throttled_usec": stats.get("throttled_usec", float("nan")),
    }


def _read_cgroup_v1_cpu_throttle(path: str) -> Dict[str, float]:
    stats = _read_flat_cgroup_stats(path)
    throttled_time_ns = stats.get("throttled_time", float("nan"))
    return {
        "nr_periods": stats.get("nr_periods", float("nan")),
        "nr_throttled": stats.get("nr_throttled", float("nan")),
        "throttled_usec": (
            throttled_time_ns / 1_000.0
            if throttled_time_ns == throttled_time_ns
            else float("nan")
        ),
    }


def _read_cgroup_memory_events(path: str) -> Dict[str, float]:
    stats = _read_flat_cgroup_stats(path)
    return {
        key: stats.get(key, float("nan"))
        for key in ("high", "max", "oom", "oom_kill")
    }


def _read_pressure_totals(path: str) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if not parts:
                continue
            scope = parts[0]
            if scope not in {"some", "full"}:
                continue
            for token in parts[1:]:
                key, separator, value = token.partition("=")
                if key != "total" or not separator:
                    continue
                try:
                    totals[scope] = float(value)
                except ValueError:
                    pass
                break
    return totals


def _counter_delta(
    start: Optional[Dict[str, float]],
    end: Optional[Dict[str, float]],
    key: str,
) -> float:
    if start is None or end is None:
        return float("nan")
    start_value = float(start.get(key, float("nan")))
    end_value = float(end.get(key, float("nan")))
    if (
        start_value != start_value
        or end_value != end_value
        or end_value < start_value
    ):
        return float("nan")
    return end_value - start_value


def _snapshot_value(snapshot: Optional[Dict[str, float]], key: str) -> float:
    if snapshot is None:
        return float("nan")
    try:
        return float(snapshot.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _pressure_stall_pct(
    start: Optional[Dict[str, float]],
    end: Optional[Dict[str, float]],
    scope: str,
    elapsed_s: float,
) -> float:
    delta_usec = _counter_delta(start, end, scope)
    if delta_usec != delta_usec or elapsed_s <= 0.0:
        return float("nan")
    return (delta_usec / (elapsed_s * 1_000_000.0)) * 100.0


def _apply_window_counter_metrics(
    result: ResourceUsageResult,
    start: _WindowCounterSnapshots,
    end: _WindowCounterSnapshots,
    elapsed_s: float,
) -> None:
    nr_periods = _counter_delta(
        start.cpu_throttle,
        end.cpu_throttle,
        "nr_periods",
    )
    nr_throttled = _counter_delta(
        start.cpu_throttle,
        end.cpu_throttle,
        "nr_throttled",
    )
    throttled_usec = _counter_delta(
        start.cpu_throttle,
        end.cpu_throttle,
        "throttled_usec",
    )
    result.container_cpu_nr_periods_delta = nr_periods
    result.container_cpu_nr_throttled_delta = nr_throttled
    if nr_periods == nr_periods and nr_periods > 0.0 and nr_throttled == nr_throttled:
        result.container_cpu_throttled_period_ratio_pct = (
            nr_throttled / nr_periods
        ) * 100.0
    if throttled_usec == throttled_usec:
        result.container_cpu_throttled_time_s = throttled_usec / 1_000_000.0

    result.container_cpu_pressure_some_stall_pct = _pressure_stall_pct(
        start.cpu_pressure,
        end.cpu_pressure,
        "some",
        elapsed_s,
    )
    result.container_cpu_pressure_full_stall_pct = _pressure_stall_pct(
        start.cpu_pressure,
        end.cpu_pressure,
        "full",
        elapsed_s,
    )

    memory_event_fields = {
        "high": "container_mem_high_events_delta",
        "max": "container_mem_max_events_delta",
        "oom": "container_mem_oom_events_delta",
        "oom_kill": "container_mem_oom_kill_events_delta",
    }
    for event, field in memory_event_fields.items():
        setattr(
            result,
            field,
            _counter_delta(start.memory_events, end.memory_events, event),
        )

    result.container_mem_pressure_some_stall_pct = _pressure_stall_pct(
        start.memory_pressure,
        end.memory_pressure,
        "some",
        elapsed_s,
    )
    result.container_mem_pressure_full_stall_pct = _pressure_stall_pct(
        start.memory_pressure,
        end.memory_pressure,
        "full",
        elapsed_s,
    )

    result.container_mem_peak_cgroup_bytes = _snapshot_value(
        end.memory_peak,
        "peak",
    )
    result.container_mem_anon_bytes_end = _snapshot_value(
        end.memory_stat,
        "anon",
    )
    result.container_mem_file_bytes_end = _snapshot_value(
        end.memory_stat,
        "file",
    )
    result.container_mem_slab_bytes_end = _snapshot_value(
        end.memory_stat,
        "slab",
    )
    result.container_mem_pgfault_delta = _counter_delta(
        start.memory_stat,
        end.memory_stat,
        "pgfault",
    )
    result.container_mem_pgmajfault_delta = _counter_delta(
        start.memory_stat,
        end.memory_stat,
        "pgmajfault",
    )
    result.container_mem_workingset_refault_delta = _counter_delta(
        start.memory_stat,
        end.memory_stat,
        "workingset_refault",
    )

    result.container_io_read_ops = _counter_delta(
        start.io_operations,
        end.io_operations,
        "read_ops",
    )
    result.container_io_write_ops = _counter_delta(
        start.io_operations,
        end.io_operations,
        "write_ops",
    )
    result.container_io_pressure_some_stall_pct = _pressure_stall_pct(
        start.io_pressure,
        end.io_pressure,
        "some",
        elapsed_s,
    )
    result.container_io_pressure_full_stall_pct = _pressure_stall_pct(
        start.io_pressure,
        end.io_pressure,
        "full",
        elapsed_s,
    )
    result.container_pids_current_end = _snapshot_value(end.pids, "current")
    result.container_pids_peak_cgroup = _snapshot_value(end.pids, "peak")
    result.container_pids_max_events_delta = _counter_delta(
        start.pids,
        end.pids,
        "max_events",
    )


def _parse_online_cpu_ids(text: str) -> List[int]:
    cpu_ids: List[int] = []
    for raw_part in text.replace("\n", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end >= start:
                cpu_ids.extend(range(start, end + 1))
        else:
            cpu_ids.append(int(part))
    return sorted(set(cpu_ids))


def _discover_cpu_ids(cpu_sysfs_root: str) -> List[int]:
    online_path = os.path.join(cpu_sysfs_root, "online")
    if os.path.exists(online_path):
        try:
            with open(online_path, "r", encoding="utf-8") as f:
                cpu_ids = _parse_online_cpu_ids(f.read())
            if cpu_ids:
                return cpu_ids
        except Exception:
            pass

    cpu_ids = []
    try:
        for entry in os.scandir(cpu_sysfs_root):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("cpu") and name[3:].isdigit():
                cpu_ids.append(int(name[3:]))
    except Exception:
        return []
    return sorted(set(cpu_ids))


def _read_proc_cpuinfo_freqs_hz(proc_cpuinfo_path: str) -> List[float]:
    freqs: List[float] = []
    try:
        with open(proc_cpuinfo_path, "r", encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.partition(":")
                if sep and key.strip().lower() == "cpu mhz":
                    mhz = float(value.strip())
                    if mhz > 0:
                        freqs.append(mhz * 1_000_000.0)
    except Exception:
        return []
    return freqs


def _read_cpu_frequency_hz(
    cpu_sysfs_root: str = "/sys/devices/system/cpu",
    proc_cpuinfo_path: str = "/proc/cpuinfo",
) -> Tuple[Optional[float], Optional[float]]:
    freqs: List[float] = []
    for cpu_id in _discover_cpu_ids(cpu_sysfs_root):
        cpufreq_root = os.path.join(cpu_sysfs_root, f"cpu{cpu_id}", "cpufreq")
        for leaf in ("scaling_cur_freq", "cpuinfo_cur_freq"):
            freq_path = os.path.join(cpufreq_root, leaf)
            if not os.path.exists(freq_path):
                continue
            try:
                with open(freq_path, "r", encoding="utf-8") as f:
                    khz = float(f.read().strip())
            except Exception:
                continue
            if khz > 0:
                freqs.append(khz * 1_000.0)
                break

    if not freqs:
        freqs = _read_proc_cpuinfo_freqs_hz(proc_cpuinfo_path)

    if not freqs:
        return None, None
    return _mean(freqs), max(freqs)


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


def _resolve_container_metric_readers(
    container_name: str,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_root: str = "/proc",
) -> _ContainerReaders:
    if not container_name:
        return _ContainerReaders()

    pid = _docker_container_pid(container_name)
    cgroup_file = os.path.join(proc_root, str(pid), "cgroup")
    with open(cgroup_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    readers = _ContainerReaders()

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            cpu_path = _join_cgroup_path(cgroup_root, parts[2], "cpu.stat")
            mem_path = _join_cgroup_path(cgroup_root, parts[2], "memory.current")
            swap_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.swap.current",
            )
            swap_limit_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.swap.max",
            )
            io_path = _join_cgroup_path(cgroup_root, parts[2], "io.stat")
            memory_peak_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.peak",
            )
            memory_stat_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.stat",
            )
            memory_events_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.events",
            )
            cpu_pressure_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "cpu.pressure",
            )
            memory_pressure_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "memory.pressure",
            )
            io_pressure_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "io.pressure",
            )
            pids_current_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "pids.current",
            )
            pids_peak_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "pids.peak",
            )
            pids_events_path = _join_cgroup_path(
                cgroup_root,
                parts[2],
                "pids.events",
            )
            if os.path.exists(cpu_path):
                readers.cpu = lambda path=cpu_path: _read_cpu_stat_usage_s(path)
                readers.cpu_throttle = (
                    lambda path=cpu_path: _read_cgroup_v2_cpu_throttle(path)
                )
            if os.path.exists(mem_path):
                readers.memory = lambda path=mem_path: _read_int(path)
            if os.path.exists(swap_path):
                readers.swap = lambda path=swap_path: _read_int(path)
            if os.path.exists(swap_limit_path):
                readers.swap_limit = (
                    lambda path=swap_limit_path: _read_cgroup_limit(path)
                )
            if os.path.exists(io_path):
                readers.io = lambda path=io_path: _read_cgroup_v2_io_stat(path)
                readers.io_operations = (
                    lambda path=io_path: _read_cgroup_v2_io_operations(path)
                )
            if os.path.exists(memory_peak_path):
                readers.memory_peak = (
                    lambda path=memory_peak_path: _read_cgroup_memory_peak(path)
                )
            if os.path.exists(memory_stat_path):
                readers.memory_stat = (
                    lambda path=memory_stat_path: _read_cgroup_memory_stat(path)
                )
            if os.path.exists(memory_events_path):
                readers.memory_events = (
                    lambda path=memory_events_path: _read_cgroup_memory_events(path)
                )
            if os.path.exists(cpu_pressure_path):
                readers.cpu_pressure = (
                    lambda path=cpu_pressure_path: _read_pressure_totals(path)
                )
            if os.path.exists(memory_pressure_path):
                readers.memory_pressure = (
                    lambda path=memory_pressure_path: _read_pressure_totals(path)
                )
            if os.path.exists(io_pressure_path):
                readers.io_pressure = (
                    lambda path=io_pressure_path: _read_pressure_totals(path)
                )
            existing_pids_paths = [
                path
                for path in (
                    pids_current_path,
                    pids_peak_path,
                    pids_events_path,
                )
                if os.path.exists(path)
            ]
            if existing_pids_paths:
                readers.pids = (
                    lambda current=(
                        pids_current_path
                        if os.path.exists(pids_current_path)
                        else None
                    ), peak=(
                        pids_peak_path
                        if os.path.exists(pids_peak_path)
                        else None
                    ), events=(
                        pids_events_path
                        if os.path.exists(pids_events_path)
                        else None
                    ): _read_cgroup_pids(current, peak, events)
                )
            return readers

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue

        controllers = set(parts[1].split(","))
        if readers.cpu_throttle is None and "cpu" in controllers:
            cpu_stat_path = _first_existing(
                _v1_candidates(cgroup_root, parts[1], parts[2], "cpu.stat")
            )
            if cpu_stat_path is not None:
                readers.cpu_throttle = (
                    lambda path=cpu_stat_path: _read_cgroup_v1_cpu_throttle(path)
                )

        if readers.cpu is None and "cpuacct" in controllers:
            cpu_path = _first_existing(
                _v1_candidates(cgroup_root, parts[1], parts[2], "cpuacct.usage")
            )
            if cpu_path is not None:
                readers.cpu = (
                    lambda path=cpu_path: float(_read_int(path)) / 1_000_000_000.0
                )

        if readers.memory is None and "memory" in controllers:
            mem_path = _first_existing(
                _v1_candidates(cgroup_root, parts[1], parts[2], "memory.usage_in_bytes")
            )
            if mem_path is not None:
                readers.memory = lambda path=mem_path: _read_int(path)

            memsw_path = _first_existing(
                _v1_candidates(
                    cgroup_root,
                    parts[1],
                    parts[2],
                    "memory.memsw.usage_in_bytes",
                )
            )
            if memsw_path is not None and mem_path is not None:
                readers.swap = (
                    lambda memsw=memsw_path, memory=mem_path: _read_v1_swap_usage(
                        memsw,
                        memory,
                    )
                )

            memsw_limit_path = _first_existing(
                _v1_candidates(
                    cgroup_root,
                    parts[1],
                    parts[2],
                    "memory.memsw.limit_in_bytes",
                )
            )
            memory_limit_path = _first_existing(
                _v1_candidates(
                    cgroup_root,
                    parts[1],
                    parts[2],
                    "memory.limit_in_bytes",
                )
            )
            if memsw_limit_path is not None and memory_limit_path is not None:
                readers.swap_limit = (
                    lambda memsw=memsw_limit_path, memory=memory_limit_path: (
                        _read_v1_swap_limit(memsw, memory)
                    )
                )

        if readers.io is None and "blkio" in controllers:
            io_path = _first_existing(
                _v1_candidates(
                    cgroup_root,
                    parts[1],
                    parts[2],
                    "blkio.throttle.io_service_bytes",
                )
                + _v1_candidates(
                    cgroup_root,
                    parts[1],
                    parts[2],
                    "blkio.io_service_bytes",
                )
            )
            if io_path is not None:
                readers.io = (
                    lambda path=io_path: _read_cgroup_v1_io_service_bytes(path)
                )

    return readers


def _resolve_container_readers(
    container_name: str,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_root: str = "/proc",
) -> Tuple[Optional[Callable[[], float]], Optional[Callable[[], int]]]:
    """Compatibility wrapper for callers that only need CPU and memory."""
    readers = _resolve_container_metric_readers(
        container_name,
        cgroup_root=cgroup_root,
        proc_root=proc_root,
    )
    return readers.cpu, readers.memory


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

    cpu_freq_avgs = [
        float(sample.cpu_freq_avg_hz)
        for sample in samples
        if sample.cpu_freq_avg_hz is not None
    ]
    cpu_freq_peaks = [
        float(sample.cpu_freq_peak_hz)
        for sample in samples
        if sample.cpu_freq_peak_hz is not None
    ]
    if cpu_freq_avgs:
        result.cpu_freq_avg_hz = _mean(cpu_freq_avgs)
    if cpu_freq_peaks:
        result.cpu_freq_peak_hz = max(cpu_freq_peaks)

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

    swap_values = [
        float(sample.container_swap_usage_bytes)
        for sample in samples
        if sample.container_swap_usage_bytes is not None
    ]
    if swap_values:
        result.container_swap_usage_avg_bytes = _mean(swap_values)
        result.container_swap_usage_peak_bytes = max(swap_values)

    gpu_utils = [
        float(sample.gpu_util_pct)
        for sample in samples
        if sample.gpu_util_pct is not None
    ]
    if gpu_utils:
        result.gpu_util_avg_pct = _mean(gpu_utils)
        result.gpu_util_peak_pct = max(gpu_utils)

    gpu_sm_clocks = [
        float(sample.gpu_sm_clock_mhz)
        for sample in samples
        if sample.gpu_sm_clock_mhz is not None
    ]
    gpu_memory_clocks = [
        float(sample.gpu_memory_clock_mhz)
        for sample in samples
        if sample.gpu_memory_clock_mhz is not None
    ]
    gpu_pstates = [
        sample.gpu_pstate
        for sample in samples
        if sample.gpu_pstate is not None
    ]
    gpu_temperatures = [
        float(sample.gpu_temp_c)
        for sample in samples
        if sample.gpu_temp_c is not None
    ]
    if gpu_sm_clocks:
        result.gpu_sm_clock_mhz = _mean(gpu_sm_clocks)
    if gpu_memory_clocks:
        result.gpu_memory_clock_mhz = _mean(gpu_memory_clocks)
    result.gpu_pstate = _dominant_pstate(gpu_pstates)
    if gpu_temperatures:
        result.gpu_temp_c = _mean(gpu_temperatures)

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
        cpu_sysfs_root: str = "/sys/devices/system/cpu",
        proc_cpuinfo_path: str = "/proc/cpuinfo",
    ) -> None:
        self.sample_hz = float(sample_hz)
        self.container_name = container_name
        self.cpu_cores = float(cpu_cores)
        self.mem_limit_bytes = float(mem_cap_gb) * float(BYTES_PER_GIB)
        self.use_gpu = bool(use_gpu)
        self.device_index = int(device_index)
        self.dt = 1.0 / self.sample_hz
        self.cpu_sysfs_root = cpu_sysfs_root
        self.proc_cpuinfo_path = proc_cpuinfo_path

        self.samples: List[ResourceUsageSample] = []
        self._cpu_reader: Optional[Callable[[], float]] = None
        self._mem_reader: Optional[Callable[[], int]] = None
        self._swap_reader: Optional[Callable[[], int]] = None
        self._swap_limit_reader: Optional[Callable[[], int]] = None
        self._io_reader: Optional[Callable[[], Tuple[int, int]]] = None
        self._cpu_throttle_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._memory_events_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._cpu_pressure_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._memory_pressure_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._io_pressure_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._memory_peak_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._memory_stat_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._io_operations_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._pids_reader: Optional[
            Callable[[], Dict[str, float]]
        ] = None
        self._swap_limit_bytes: Optional[int] = None
        self._io_start: Optional[Tuple[int, int]] = None
        self._io_end: Optional[Tuple[int, int]] = None
        self._window_counters_start = _WindowCounterSnapshots()
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
            readers = _resolve_container_metric_readers(
                container_name,
                cgroup_root=cgroup_root,
                proc_root=proc_root,
            )
            self._cpu_reader = readers.cpu
            self._mem_reader = readers.memory
            self._swap_reader = readers.swap
            self._swap_limit_reader = readers.swap_limit
            self._io_reader = readers.io
            self._cpu_throttle_reader = readers.cpu_throttle
            self._memory_events_reader = readers.memory_events
            self._cpu_pressure_reader = readers.cpu_pressure
            self._memory_pressure_reader = readers.memory_pressure
            self._io_pressure_reader = readers.io_pressure
            self._memory_peak_reader = readers.memory_peak
            self._memory_stat_reader = readers.memory_stat
            self._io_operations_reader = readers.io_operations
            self._pids_reader = readers.pids
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
            or self._swap_reader is not None
            or self._swap_limit_reader is not None
            or self._io_reader is not None
            or self._cpu_throttle_reader is not None
            or self._memory_events_reader is not None
            or self._cpu_pressure_reader is not None
            or self._memory_pressure_reader is not None
            or self._io_pressure_reader is not None
            or self._memory_peak_reader is not None
            or self._memory_stat_reader is not None
            or self._io_operations_reader is not None
            or self._pids_reader is not None
            or self._gpu_handle is not None
        )

    def start(self) -> None:
        if not self.available:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Resource usage monitor is already running")

        self.samples = []
        self._stop_event = threading.Event()
        self._swap_limit_bytes = None
        self._io_start = None
        self._io_end = None
        self._window_counters_start = _WindowCounterSnapshots()
        if self._swap_limit_reader is not None:
            try:
                self._swap_limit_bytes = self._swap_limit_reader()
            except Exception as exc:
                self._runtime_error = str(exc)
        if self._io_reader is not None:
            try:
                self._io_start = self._io_reader()
            except Exception as exc:
                self._runtime_error = str(exc)
        self._window_counters_start = self._read_window_counter_snapshots()
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
        window_counters_end = self._read_window_counter_snapshots()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self._append_sample(self._t_end)
        if self._io_reader is not None:
            try:
                self._io_end = self._io_reader()
            except Exception as exc:
                self._runtime_error = str(exc)
        samples = [
            sample
            for sample in self.samples
            if self._t_start <= sample.timestamp <= self._t_end
        ]
        samples.sort(key=lambda item: item.timestamp)
        self.samples = samples
        result = _result_from_samples(
            samples,
            self.cpu_cores,
            self.mem_limit_bytes,
            min_cpu_interval_s=0.5 * self.dt,
        )
        if self._swap_limit_bytes is not None:
            result.container_swap_limit_bytes = float(self._swap_limit_bytes)
        if self._io_start is not None and self._io_end is not None:
            result.container_io_read_bytes = float(
                max(0, self._io_end[0] - self._io_start[0])
            )
            result.container_io_write_bytes = float(
                max(0, self._io_end[1] - self._io_start[1])
            )
        _apply_window_counter_metrics(
            result,
            self._window_counters_start,
            window_counters_end,
            max(0.0, self._t_end - self._t_start),
        )
        return (
            result,
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

    def _read_window_counter_snapshots(self) -> _WindowCounterSnapshots:
        snapshots = _WindowCounterSnapshots()
        readers = (
            ("cpu_throttle", self._cpu_throttle_reader),
            ("memory_events", self._memory_events_reader),
            ("cpu_pressure", self._cpu_pressure_reader),
            ("memory_pressure", self._memory_pressure_reader),
            ("io_pressure", self._io_pressure_reader),
            ("memory_peak", self._memory_peak_reader),
            ("memory_stat", self._memory_stat_reader),
            ("io_operations", self._io_operations_reader),
            ("pids", self._pids_reader),
        )
        for field, reader in readers:
            if reader is None:
                continue
            try:
                setattr(snapshots, field, reader())
            except Exception as exc:
                self._runtime_error = str(exc)
        return snapshots

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
        container_swap_usage_bytes = None
        gpu_util_pct = None
        gpu_mem_used_bytes = None
        gpu_mem_total_bytes = None
        cpu_freq_avg_hz = None
        cpu_freq_peak_hz = None
        gpu_sm_clock_mhz = None
        gpu_memory_clock_mhz = None
        gpu_pstate = None
        gpu_temp_c = None

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

        if self._swap_reader is not None:
            try:
                container_swap_usage_bytes = self._swap_reader()
            except Exception as exc:
                self._runtime_error = str(exc)

        try:
            cpu_freq_avg_hz, cpu_freq_peak_hz = _read_cpu_frequency_hz(
                cpu_sysfs_root=self.cpu_sysfs_root,
                proc_cpuinfo_path=self.proc_cpuinfo_path,
            )
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
            try:
                gpu_sm_clock_mhz = float(
                    pynvml.nvmlDeviceGetClockInfo(
                        self._gpu_handle,
                        pynvml.NVML_CLOCK_SM,
                    )
                )
            except Exception as exc:
                self._runtime_error = str(exc)
            try:
                gpu_memory_clock_mhz = float(
                    pynvml.nvmlDeviceGetClockInfo(
                        self._gpu_handle,
                        pynvml.NVML_CLOCK_MEM,
                    )
                )
            except Exception as exc:
                self._runtime_error = str(exc)
            try:
                gpu_pstate = _normalize_pstate(
                    pynvml.nvmlDeviceGetPerformanceState(self._gpu_handle)
                )
            except Exception as exc:
                self._runtime_error = str(exc)
            try:
                gpu_temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._gpu_handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                )
            except Exception as exc:
                self._runtime_error = str(exc)

        return ResourceUsageSample(
            timestamp,
            container_cpu_s,
            container_mem_usage_bytes,
            gpu_util_pct,
            gpu_mem_used_bytes,
            gpu_mem_total_bytes,
            cpu_freq_avg_hz,
            cpu_freq_peak_hz,
            gpu_sm_clock_mhz,
            gpu_memory_clock_mhz,
            gpu_pstate,
            gpu_temp_c,
            container_swap_usage_bytes,
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
    cpu_sysfs_root: str = "/sys/devices/system/cpu",
    proc_cpuinfo_path: str = "/proc/cpuinfo",
) -> Tuple[ResourceUsageResult, str, List[ResourceUsageSample]]:
    monitor = ResourceUsageMonitor(
        sample_hz=sample_hz,
        container_name=container_name,
        cpu_cores=cpu_cores,
        mem_cap_gb=mem_cap_gb,
        use_gpu=use_gpu,
        device_index=device_index,
        cpu_sysfs_root=cpu_sysfs_root,
        proc_cpuinfo_path=proc_cpuinfo_path,
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

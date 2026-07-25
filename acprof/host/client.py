"""AC-Prof Universal Client - workload generation, latency measurement, energy monitoring.

Runs on HOST (not inside container). Generalized from example-code/client.py.
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional


def _ensure_local_proxy_bypass() -> None:
    local_hosts = ("localhost", "127.0.0.1", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        known = {part.lower() for part in parts}
        missing = [host for host in local_hosts if host.lower() not in known]
        if missing:
            os.environ[key] = ",".join(parts + missing)


_ensure_local_proxy_bypass()

import requests

from acprof.config import (
    CSV_FIELDS,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_TASK_PARAMS,
    SCALING_DIMENSIONS,
)
from acprof.host.compute_profile_plan import (
    compute_mflops as _compute_mflops,
    find_compute_profile_entry as _find_compute_profile_entry,
    load_compute_profile_plan as _load_compute_profile_plan,
)

# ─────────────────────────────────────────────
# Config from env
# ─────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
TASK_FAMILY = os.getenv("TASK_FAMILY", "nlp")
PIPELINE_TAG = os.getenv("PIPELINE_TAG", "text-generation")
RUNTIME_BACKEND = os.getenv("RUNTIME_BACKEND", "transformers_pipeline")
IMAGE_TAG = os.getenv("IMAGE_TAG", "")

CPU_CORES = os.getenv("CPU_CORES", "")
MEM_CAP_GB = os.getenv("MEM_CAP_GB", "")
GPU_MODE = os.getenv("GPU_MODE", "off").lower()
GPU_MODE = "on" if GPU_MODE == "on" else "off"

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8002").rstrip("/")
ENDPOINT = os.getenv("ENDPOINT", "/predict")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
WARMUP = int(os.getenv("WARMUP", "2"))
REPEAT = int(os.getenv("REPEAT", "5"))
REPEAT_IN_WINDOW = int(os.getenv("REPEAT_IN_WINDOW", str(DEFAULT_REPEAT_IN_WINDOW)))
REPEAT_WINDOW_SECONDS = float(os.getenv("REPEAT_WINDOW_SECONDS", str(DEFAULT_REPEAT_WINDOW_SECONDS)))
AUTO_WARMUP_REQUESTS = int(os.getenv("AUTO_WARMUP_REQUESTS", "5"))
SLOW_LATENCY_THRESHOLD_S = float(os.getenv("SLOW_LATENCY_THRESHOLD_S", "0.06"))

COLD_START_S = os.getenv("COLD_START_S", "nan")
OUT_CSV = os.getenv("OUT_CSV", "result.csv")
CASE_NAME = os.getenv("CASE_NAME", "").strip()
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "").strip()
SNIFF_GROUPS_PATH = os.getenv("SNIFF_GROUPS_PATH", "").strip()
IDLE_DEBUG = os.getenv("IDLE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
IDLE_DIAG_PATH = os.getenv("IDLE_DIAG_PATH", "").strip()
IDLE_DEBUG_TRACE_INTERVAL_S = float(os.getenv("IDLE_DEBUG_TRACE_INTERVAL_S", "0.1"))
USE_MIPS = os.getenv("USE_MIPS", "").strip().lower() in {"1", "true", "yes", "on"}

SAMPLE_HZ = float(os.getenv("SAMPLE_HZ", "20"))
IDLE_SECONDS = float(os.getenv("IDLE_SECONDS", "3"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))
IDLE_COOLDOWN_SECONDS = float(
    os.getenv("IDLE_COOLDOWN_SECONDS", str(DEFAULT_IDLE_COOLDOWN_SECONDS))
)

# Input scales from task family config
INPUT_SCALES_STR = os.getenv("INPUT_SCALES", "")
INPUT_SCALE_PLAN_FILE = os.getenv("INPUT_SCALE_PLAN_FILE", "").strip()
COMPUTE_PROFILE_PLAN_FILE = os.getenv("COMPUTE_PROFILE_PLAN_FILE", "").strip()
TASK_PARAM_STR = os.getenv("TASK_PARAM", "")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _is_file_empty(path: str) -> bool:
    try:
        return (not os.path.exists(path)) or os.path.getsize(path) == 0
    except Exception:
        return True


def _sniff_groups_path(csv_path: str) -> str:
    return SNIFF_GROUPS_PATH or f"{csv_path}.sniff_groups.jsonl"


def _idle_diag_path(csv_path: str) -> str:
    return IDLE_DIAG_PATH or f"{csv_path}.idle_diag.jsonl"


def _to_float_or_nan(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _fmt_float(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.6f}"


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _finite_positive(value: Any) -> bool:
    number = _to_float_or_nan(value)
    return math.isfinite(number) and number > 0.0


class EnergyAbort(RuntimeError):
    """Raised when energy measurement prerequisites are not stable enough."""


class MIPSAbort(RuntimeError):
    """Raised when perf MIPS profiling cannot continue."""


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _percentile_nearest_rank(xs: List[float], percentile: float) -> float:
    values = sorted(
        value
        for value in (_to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    )
    if not values:
        return float("nan")
    rank = int(math.ceil((float(percentile) / 100.0) * len(values)))
    index = min(max(rank - 1, 0), len(values) - 1)
    return values[index]


def _slow_ratio(xs: List[float]) -> float:
    values = [
        value
        for value in (_to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    ]
    if not values:
        return float("nan")
    return sum(value > SLOW_LATENCY_THRESHOLD_S for value in values) / float(len(values))


def _latency_distribution_metrics(prefix: str, latencies: List[float]) -> Dict[str, float]:
    return {
        f"{prefix}_p50_s": _percentile_nearest_rank(latencies, 50.0),
        f"{prefix}_p90_s": _percentile_nearest_rank(latencies, 90.0),
        f"{prefix}_p95_s": _percentile_nearest_rank(latencies, 95.0),
        f"{prefix}_slow_ratio": _slow_ratio(latencies),
    }


def _eff_negative_warnings(
    *,
    avg_power_eff_w: float,
    peak_power_eff_w: float,
    energy_eff_j: float,
) -> List[str]:
    metrics = [
        ("gpu_avg_power_eff_w", avg_power_eff_w),
        ("gpu_peak_power_eff_w", peak_power_eff_w),
        ("gpu_energy_eff_j", energy_eff_j),
    ]
    return [f"{name}<0" for name, value in metrics if value == value and value < 0.0]


def _named_negative_warnings(metrics: Dict[str, float]) -> List[str]:
    return [f"{name}<0" for name, value in metrics.items() if value == value and value < 0.0]


def _parse_effective_input_scale(resp: Dict[str, Any]) -> Optional[float]:
    if not isinstance(resp, dict):
        return None
    value = resp.get("effective_input_scale")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_effective_input_scale(
    current: Optional[float],
    candidate: Optional[float],
    requested_scale: float,
) -> float:
    resolved = float(requested_scale) if candidate is None else float(candidate)
    if current is None:
        return resolved
    if not math.isclose(current, resolved, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"inconsistent effective_input_scale for requested_scale={requested_scale}: "
            f"{current} vs {resolved}"
        )
    return current


def _estimate_cpu_cycles(
    latency_s: float,
    cpu_freq_avg_hz: float,
    cpu_cores: float,
    container_cpu_util_avg_pct: float,
) -> float:
    latency = _to_float_or_nan(latency_s)
    freq = _to_float_or_nan(cpu_freq_avg_hz)
    cores = _to_float_or_nan(cpu_cores)
    cpu_util_pct = _to_float_or_nan(container_cpu_util_avg_pct)
    if (
        latency == latency
        and freq == freq
        and cores == cores
        and cpu_util_pct == cpu_util_pct
        and latency > 0.0
        and freq > 0.0
        and cores > 0.0
        and cpu_util_pct >= 0.0
    ):
        return latency * freq * cores * (cpu_util_pct / 100.0)
    return float("nan")


def _prepare_repeat_window(
    scale_value: float,
    scale_label: str,
    payload_override: Optional[Dict[str, Any]],
) -> int:
    if REPEAT_IN_WINDOW > 0:
        return REPEAT_IN_WINDOW
    if REPEAT_WINDOW_SECONDS <= 0.0:
        raise EnergyAbort(
            f"invalid REPEAT_WINDOW_SECONDS={REPEAT_WINDOW_SECONDS!r}; expected a positive value"
        )
    if AUTO_WARMUP_REQUESTS < 0:
        raise EnergyAbort(
            "invalid AUTO_WARMUP_REQUESTS="
            f"{AUTO_WARMUP_REQUESTS!r}; expected >= 0"
        )

    for idx in range(AUTO_WARMUP_REQUESTS):
        req_id = f"{CASE_NAME}_{scale_label}_auto_warmup{idx}"
        _one_request(scale_value, req_id=req_id, payload_override=payload_override)

    print(
        "[client] auto repeat-window "
        f"scale={scale_value:g} warmup_requests={AUTO_WARMUP_REQUESTS} "
        f"target_window_s={REPEAT_WINDOW_SECONDS:.3f}",
        flush=True,
    )
    return 1


def _should_send_window_request(
    completed_requests: int,
    latency_sum_s: float,
    repeat_request_limit: int,
) -> bool:
    if REPEAT_IN_WINDOW > 0:
        return completed_requests < repeat_request_limit
    if completed_requests <= 0:
        return True
    return latency_sum_s < REPEAT_WINDOW_SECONDS


# ─────────────────────────────────────────────
# Determine input scales
# ─────────────────────────────────────────────
if INPUT_SCALES_STR:
    input_scales = sorted(set(_parse_float_list(INPUT_SCALES_STR)))
else:
    scaling_cfg = SCALING_DIMENSIONS.get(TASK_FAMILY)
    if scaling_cfg:
        input_scales = scaling_cfg.values
    else:
        input_scales = [1.0]

# Task param (secondary parameter)
task_param = TASK_PARAM_STR
if not task_param:
    defaults = DEFAULT_TASK_PARAMS.get(TASK_FAMILY, {})
    # Filter out params not applicable to this task type
    _GENERATIVE_TAGS = {"text-generation", "text2text-generation", "summarization",
                        "translation", "conversational"}
    if PIPELINE_TAG not in _GENERATIVE_TAGS:
        defaults = {k: v for k, v in defaults.items() if k != "max_new_tokens"}
    task_param = json.dumps(defaults) if defaults else ""

print(f"[client] PIPELINE_TAG={PIPELINE_TAG}, task_param={task_param!r}", flush=True)


# ─────────────────────────────────────────────
# Workload generator
# ─────────────────────────────────────────────
from acprof.workloads import get_generator  # noqa: E402

workload_gen = get_generator(TASK_FAMILY, MODEL_ID, PIPELINE_TAG, BATCH_SIZE)


def _one_request(scale_value: float, req_id: str, payload_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload_override if payload_override is not None else workload_gen.generate(scale_value)

    headers = {
        "Connection": "close",
        "X-Req-Id": req_id,
    }

    t0 = time.perf_counter()
    r = requests.post(BASE_URL + ENDPOINT, json=payload, headers=headers, timeout=300)
    t1 = time.perf_counter()
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", "")
        except Exception:
            detail = r.text[:500]
        raise RuntimeError(f"HTTP {r.status_code}: {detail or r.reason}")
    resp = r.json()
    return {
        "latency_app_s": (t1 - t0),
        "resp": resp,
        "effective_input_scale": _parse_effective_input_scale(resp),
    }


def _load_input_scale_entries() -> List[Dict[str, Any]]:
    if INPUT_SCALE_PLAN_FILE:
        with open(INPUT_SCALE_PLAN_FILE, "r", encoding="utf-8") as f:
            plan = json.load(f)

        entries = plan.get("entries")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"invalid input scale plan file: {INPUT_SCALE_PLAN_FILE}")

        loaded_entries: List[Dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"invalid input scale plan entry at index {idx}: {entry!r}"
                )

            raw_scale = entry.get("input_scale")
            payload = entry.get("payload")
            if raw_scale is None or not isinstance(payload, dict):
                raise RuntimeError(
                    f"input scale plan entry missing input_scale/payload at index {idx}"
                )

            scale_value = float(raw_scale)
            scale_label = str(entry.get("scale_label") or workload_gen.scale_label(scale_value))
            loaded_entries.append({
                "input_scale": scale_value,
                "scale_label": scale_label,
                "payload": payload,
            })

        return loaded_entries

    return [
        {
            "input_scale": float(scale_value),
            "scale_label": workload_gen.scale_label(scale_value),
            "payload": None,
        }
        for scale_value in input_scales
    ]


input_scale_entries = _load_input_scale_entries()


# ─────────────────────────────────────────────
# Optional NVML energy
# ─────────────────────────────────────────────
USE_ENERGY = (GPU_MODE == "on")
energy_mod = None
if USE_ENERGY:
    try:
        from acprof.monitors import energy_nvml as energy_mod
    except Exception as _e:
        energy_mod = None
        print(f"[WARN] GPU energy monitoring unavailable: {_e.__class__.__name__}: {_e}",
              file=__import__('sys').stderr)
        print("[WARN] Install pynvml: pip install pynvml", file=__import__('sys').stderr)

cpu_energy_mod = None
try:
    from acprof.monitors import energy_cpu as cpu_energy_mod
except Exception as _e:
    cpu_energy_mod = None
    print(f"[WARN] CPU energy monitoring unavailable: {_e.__class__.__name__}: {_e}",
          file=__import__('sys').stderr)

resource_usage_mod = None
try:
    from acprof.monitors import resource_usage as resource_usage_mod
except Exception as _e:
    resource_usage_mod = None
    print(f"[WARN] Resource usage monitoring unavailable: {_e.__class__.__name__}: {_e}",
          file=__import__('sys').stderr)

perf_mips_mod = None
try:
    from acprof.monitors import perf_mips as perf_mips_mod
except Exception as _e:
    perf_mips_mod = None
    if USE_MIPS:
        print(f"[WARN] MIPS monitoring unavailable: {_e.__class__.__name__}: {_e}",
              file=__import__('sys').stderr)


GPU_METRIC_FIELDS = [
    "gpu_idle_power_w",
    "gpu_energy_iters",
    "gpu_avg_power_total_w",
    "gpu_peak_power_total_w",
    "gpu_energy_total_j",
    "gpu_avg_power_eff_w",
    "gpu_peak_power_eff_w",
    "gpu_energy_eff_j",
]

CPU_METRIC_FIELDS = [
    "cpu_idle_power_w",
    "cpu_energy_iters",
    "cpu_avg_power_total_w",
    "cpu_peak_power_total_w",
    "cpu_energy_total_j",
    "cpu_avg_power_eff_w",
    "cpu_peak_power_eff_w",
    "cpu_energy_eff_j",
    "vcpu_cpu_share",
    "vcpu_cpu_time_s",
    "vcpu_avg_power_total_w",
    "vcpu_peak_power_total_w",
    "vcpu_energy_total_j",
    "vcpu_avg_power_eff_w",
    "vcpu_peak_power_eff_w",
    "vcpu_energy_eff_j",
]

RESOURCE_USAGE_METRIC_FIELDS = [
    "resource_usage_iters",
    "container_cpu_util_avg_pct",
    "container_cpu_util_peak_pct",
    "cpu_freq_avg_hz",
    "cpu_freq_peak_hz",
    "container_mem_usage_avg_bytes",
    "container_mem_usage_peak_bytes",
    "container_mem_util_avg_pct",
    "container_mem_util_peak_pct",
    "gpu_util_avg_pct",
    "gpu_util_peak_pct",
    "gpu_mem_used_avg_bytes",
    "gpu_mem_used_peak_bytes",
    "gpu_mem_util_avg_pct",
    "gpu_mem_util_peak_pct",
]

LATENCY_PACKET_DISTRIBUTION_FIELDS = [
    "latency_p50_s",
    "latency_p90_s",
    "latency_p95_s",
    "latency_slow_ratio",
]

LATENCY_APP_DISTRIBUTION_FIELDS = [
    "latency_app_p50_s",
    "latency_app_p90_s",
    "latency_app_p95_s",
    "latency_app_slow_ratio",
]

MIPS_METRIC_FIELDS = [
    "cpu_instructions_per_request",
    "cpu_mips_app",
    "cpu_mips_packet",
    "cpu_perf_elapsed_s",
]


def _nan_metrics(fields: List[str]) -> Dict[str, float]:
    return {field: float("nan") for field in fields}


def _divide_if_number(value: float, divisor: float) -> float:
    value = _to_float_or_nan(value)
    if value == value:
        return value / divisor
    return value


def _gpu_metrics_from_result(result: Any, repeat_in_window: int) -> Dict[str, float]:
    return {
        "gpu_idle_power_w": _to_float_or_nan(result.idle_power_w),
        "gpu_energy_iters": float(result.energy_iters),
        "gpu_avg_power_total_w": _to_float_or_nan(result.avg_power_total_w),
        "gpu_peak_power_total_w": _to_float_or_nan(result.peak_power_total_w),
        "gpu_energy_total_j": _divide_if_number(result.energy_total_j, float(repeat_in_window)),
        "gpu_avg_power_eff_w": _to_float_or_nan(result.avg_power_eff_w),
        "gpu_peak_power_eff_w": _to_float_or_nan(result.peak_power_eff_w),
        "gpu_energy_eff_j": _divide_if_number(result.energy_eff_j, float(repeat_in_window)),
    }


def _cpu_metrics_from_result(result: Any, repeat_in_window: int) -> Dict[str, float]:
    return {
        "cpu_idle_power_w": _to_float_or_nan(result.cpu_idle_power_w),
        "cpu_energy_iters": float(result.cpu_energy_iters),
        "cpu_avg_power_total_w": _to_float_or_nan(result.cpu_avg_power_total_w),
        "cpu_peak_power_total_w": _to_float_or_nan(result.cpu_peak_power_total_w),
        "cpu_energy_total_j": _divide_if_number(result.cpu_energy_total_j, float(repeat_in_window)),
        "cpu_avg_power_eff_w": _to_float_or_nan(result.cpu_avg_power_eff_w),
        "cpu_peak_power_eff_w": _to_float_or_nan(result.cpu_peak_power_eff_w),
        "cpu_energy_eff_j": _divide_if_number(result.cpu_energy_eff_j, float(repeat_in_window)),
        "vcpu_cpu_share": _to_float_or_nan(result.vcpu_cpu_share),
        "vcpu_cpu_time_s": _divide_if_number(result.vcpu_cpu_time_s, float(repeat_in_window)),
        "vcpu_avg_power_total_w": _to_float_or_nan(result.vcpu_avg_power_total_w),
        "vcpu_peak_power_total_w": _to_float_or_nan(result.vcpu_peak_power_total_w),
        "vcpu_energy_total_j": _divide_if_number(result.vcpu_energy_total_j, float(repeat_in_window)),
        "vcpu_avg_power_eff_w": _to_float_or_nan(result.vcpu_avg_power_eff_w),
        "vcpu_peak_power_eff_w": _to_float_or_nan(result.vcpu_peak_power_eff_w),
        "vcpu_energy_eff_j": _divide_if_number(result.vcpu_energy_eff_j, float(repeat_in_window)),
    }


def _resource_usage_metrics_from_result(result: Any) -> Dict[str, float]:
    return {
        "resource_usage_iters": float(result.resource_usage_iters),
        "container_cpu_util_avg_pct": _to_float_or_nan(result.container_cpu_util_avg_pct),
        "container_cpu_util_peak_pct": _to_float_or_nan(result.container_cpu_util_peak_pct),
        "cpu_freq_avg_hz": _to_float_or_nan(
            getattr(result, "cpu_freq_avg_hz", float("nan"))
        ),
        "cpu_freq_peak_hz": _to_float_or_nan(
            getattr(result, "cpu_freq_peak_hz", float("nan"))
        ),
        "container_mem_usage_avg_bytes": _to_float_or_nan(result.container_mem_usage_avg_bytes),
        "container_mem_usage_peak_bytes": _to_float_or_nan(result.container_mem_usage_peak_bytes),
        "container_mem_util_avg_pct": _to_float_or_nan(result.container_mem_util_avg_pct),
        "container_mem_util_peak_pct": _to_float_or_nan(result.container_mem_util_peak_pct),
        "gpu_util_avg_pct": _to_float_or_nan(result.gpu_util_avg_pct),
        "gpu_util_peak_pct": _to_float_or_nan(result.gpu_util_peak_pct),
        "gpu_mem_used_avg_bytes": _to_float_or_nan(result.gpu_mem_used_avg_bytes),
        "gpu_mem_used_peak_bytes": _to_float_or_nan(result.gpu_mem_used_peak_bytes),
        "gpu_mem_util_avg_pct": _to_float_or_nan(result.gpu_mem_util_avg_pct),
        "gpu_mem_util_peak_pct": _to_float_or_nan(result.gpu_mem_util_peak_pct),
    }


def _mips_metrics_from_result(result: Any) -> Dict[str, float]:
    return {
        "cpu_instructions_per_request": _to_float_or_nan(
            result.instructions_per_request
        ),
        "cpu_mips_app": _to_float_or_nan(result.cpu_mips_app),
        "cpu_mips_packet": float("nan"),
        "cpu_perf_elapsed_s": _to_float_or_nan(result.perf_elapsed_s),
    }


def _is_mips_error(exc: Exception) -> bool:
    if isinstance(exc, MIPSAbort):
        return True
    if perf_mips_mod is None:
        return False
    mips_error_cls = getattr(perf_mips_mod, "MIPSProfilingError", None)
    return bool(mips_error_cls is not None and isinstance(exc, mips_error_cls))


def _append_sniff_group(sidecar_f, sniff_group_id: str) -> None:
    sidecar_f.write(json.dumps({"sniff_group_id": sniff_group_id}, ensure_ascii=True) + "\n")
    sidecar_f.flush()
    os.fsync(sidecar_f.fileno())


def _idle_power_debug_stats(values: List[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_idle_valid_count": 0,
            f"{prefix}_idle_min_w": float("nan"),
            f"{prefix}_idle_max_w": float("nan"),
            f"{prefix}_idle_mean_w": float("nan"),
            f"{prefix}_idle_rel_range_so_far": float("nan"),
        }

    mean_idle = sum(values) / len(values)
    relative_range = (
        (max(values) - min(values)) / mean_idle
        if mean_idle > 0.0
        else float("nan")
    )
    return {
        f"{prefix}_idle_valid_count": len(values),
        f"{prefix}_idle_min_w": min(values),
        f"{prefix}_idle_max_w": max(values),
        f"{prefix}_idle_mean_w": mean_idle,
        f"{prefix}_idle_rel_range_so_far": relative_range,
    }


def _idle_debug_stats(values: List[float]) -> Dict[str, float]:
    return _idle_power_debug_stats(values, "cpu")


def _sleep_before_idle_baseline() -> None:
    if IDLE_COOLDOWN_SECONDS > 0.0:
        time.sleep(IDLE_COOLDOWN_SECONDS)


def _supports_matched_control(*monitors: Any) -> bool:
    active = [monitor for monitor in monitors if monitor is not None]
    return bool(active) and all(
        callable(getattr(monitor, "apply_control_baseline", None))
        for monitor in active
    )


def _run_matched_control_window(
    gpu_monitor: Any,
    cpu_monitor: Any,
    resource_usage_monitor: Any,
    mips_monitor: Any,
) -> None:
    """Run a blank window with the same monitor lifecycle as the workload."""
    gpu_started = False
    cpu_started = False
    resource_started = False
    mips_started = False
    gpu_result = None
    gpu_samples = []
    cpu_result = None
    cpu_samples = []
    primary_error: Optional[Exception] = None
    stop_error: Optional[Exception] = None

    try:
        if gpu_monitor is not None:
            gpu_monitor.start()
            gpu_started = True
        if cpu_monitor is not None:
            cpu_monitor.start()
            cpu_started = True
        if resource_usage_monitor is not None:
            resource_usage_monitor.start()
            resource_started = True
        if mips_monitor is not None:
            mips_monitor.start()
            mips_started = True
        if IDLE_SECONDS > 0.0:
            time.sleep(IDLE_SECONDS)
    except Exception as exc:
        primary_error = exc
    finally:
        if mips_started:
            try:
                mips_monitor.stop(1, max(IDLE_SECONDS, 1e-9))
            except Exception as exc:
                stop_error = stop_error or exc
        if resource_started:
            try:
                resource_usage_monitor.stop()
            except Exception as exc:
                stop_error = stop_error or exc
        if gpu_started:
            try:
                gpu_result, _gpu_name, _gpu_error, gpu_samples = gpu_monitor.stop()
            except Exception as exc:
                stop_error = stop_error or exc
        if cpu_started:
            try:
                cpu_result, _cpu_error, cpu_samples = cpu_monitor.stop()
            except Exception as exc:
                stop_error = stop_error or exc

    if primary_error is not None:
        raise primary_error
    if stop_error is not None:
        raise stop_error

    if gpu_monitor is not None and gpu_result is not None:
        gpu_monitor.apply_control_baseline(
            gpu_result,
            gpu_samples,
            trace=IDLE_DEBUG,
        )
    if cpu_monitor is not None and cpu_result is not None:
        cpu_monitor.apply_control_baseline(
            cpu_result,
            cpu_samples,
            trace=IDLE_DEBUG,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _run_json_lines(cmd: List[str], timeout: float = 2.0) -> List[Dict[str, Any]]:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())
    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _collect_top_cpu_processes(limit: int = 10) -> List[Dict[str, Any]]:
    result = subprocess.run(
        [
            "ps",
            "-eo",
            "pid=,ppid=,user=,comm=,%cpu=,%mem=,args=",
            "--sort=-%cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=2.0,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())

    processes = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, user, comm, cpu_pct, mem_pct, args = parts
        processes.append({
            "pid": int(pid),
            "ppid": int(ppid),
            "user": user,
            "comm": comm,
            "cpu_pct": _to_float_or_nan(cpu_pct),
            "mem_pct": _to_float_or_nan(mem_pct),
            "args": args,
        })
        if len(processes) >= limit:
            break
    return processes


def _run_text(cmd: List[str], timeout: float = 2.0) -> str:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())
    return result.stdout


def _float_or_none(value: str) -> Optional[float]:
    number = _to_float_or_nan(value.strip())
    return number if math.isfinite(number) else None


def _int_or_none(value: str) -> Optional[int]:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _collect_nvidia_smi_gpu_snapshot(device_index: int = 0) -> Dict[str, Any]:
    query_fields = [
        "index",
        "name",
        "pstate",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "clocks.gr",
        "clocks.video",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
    ]
    output = _run_text([
        "nvidia-smi",
        f"--id={device_index}",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ])
    line = next((row.strip() for row in output.splitlines() if row.strip()), "")
    values = [part.strip() for part in line.split(",")]
    if len(values) != len(query_fields):
        raise RuntimeError(f"unexpected nvidia-smi gpu row: {line!r}")

    return {
        "index": _int_or_none(values[0]),
        "name": values[1],
        "pstate": values[2],
        "power_draw_w": _float_or_none(values[3]),
        "power_limit_w": _float_or_none(values[4]),
        "clocks_sm_mhz": _float_or_none(values[5]),
        "clocks_mem_mhz": _float_or_none(values[6]),
        "clocks_gr_mhz": _float_or_none(values[7]),
        "clocks_video_mhz": _float_or_none(values[8]),
        "temperature_gpu_c": _float_or_none(values[9]),
        "utilization_gpu_pct": _float_or_none(values[10]),
        "utilization_memory_pct": _float_or_none(values[11]),
        "memory_used_mib": _float_or_none(values[12]),
        "memory_total_mib": _float_or_none(values[13]),
    }


def _collect_nvidia_smi_compute_apps(device_index: int = 0) -> List[Dict[str, Any]]:
    output = _run_text([
        "nvidia-smi",
        f"--id={device_index}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    apps: List[Dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "No running processes found" in line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        apps.append({
            "pid": _int_or_none(parts[0]),
            "process_name": parts[1],
            "used_memory_mib": _float_or_none(parts[2]),
        })
    return apps


def _collect_nvidia_smi_pmon(device_index: int = 0) -> List[Dict[str, Any]]:
    output = _run_text(["nvidia-smi", "pmon", "-c", "1", "-i", str(device_index)])
    rows: List[Dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        gpu, pid, proc_type, sm, mem, enc, dec, jpg, ofa, *command = parts
        rows.append({
            "gpu": _int_or_none(gpu),
            "pid": _int_or_none(pid),
            "type": proc_type,
            "sm_pct": _float_or_none(sm),
            "mem_pct": _float_or_none(mem),
            "enc_pct": _float_or_none(enc),
            "dec_pct": _float_or_none(dec),
            "jpg_pct": _float_or_none(jpg),
            "ofa_pct": _float_or_none(ofa),
            "command": " ".join(command),
        })
    return rows


def _collect_gpu_idle_debug_snapshot(device_index: int = DEVICE_INDEX) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"gpu_snapshot_scope": "after_gpu_idle"}
    try:
        snapshot["nvidia_smi_gpu"] = _collect_nvidia_smi_gpu_snapshot(device_index)
    except Exception as exc:
        snapshot["nvidia_smi_gpu_error"] = repr(exc)

    try:
        snapshot["nvidia_smi_pmon"] = _collect_nvidia_smi_pmon(device_index)
    except Exception as exc:
        snapshot["nvidia_smi_pmon_error"] = repr(exc)

    try:
        snapshot["nvidia_smi_compute_apps"] = _collect_nvidia_smi_compute_apps(device_index)
    except Exception as exc:
        snapshot["nvidia_smi_compute_apps_error"] = repr(exc)

    return snapshot


def _collect_idle_debug_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"snapshot_scope": "after_idle"}
    try:
        snapshot["loadavg"] = list(os.getloadavg())
    except Exception as exc:
        snapshot["loadavg_error"] = repr(exc)

    try:
        snapshot["top_cpu_processes"] = _collect_top_cpu_processes()
    except Exception as exc:
        snapshot["top_cpu_processes_error"] = repr(exc)

    try:
        snapshot["docker_containers"] = _run_json_lines([
            "docker",
            "ps",
            "--format",
            "{{json .}}",
        ])
    except Exception as exc:
        snapshot["docker_containers_error"] = repr(exc)

    try:
        snapshot["docker_stats"] = _run_json_lines([
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
        ])
    except Exception as exc:
        snapshot["docker_stats_error"] = repr(exc)

    return snapshot


def _append_idle_diag(diag_f, record: Dict[str, Any]) -> None:
    diag_f.write(json.dumps(_json_safe(record), ensure_ascii=True, sort_keys=True) + "\n")
    diag_f.flush()
    os.fsync(diag_f.fileno())


def _append_row(
    writer: csv.DictWriter,
    row: Dict[str, Any],
    f,
    sidecar_f,
    sniff_group_id: str,
    diag_f=None,
    idle_diag_record: Optional[Dict[str, Any]] = None,
) -> None:
    out = {k: row.get(k, "") for k in CSV_FIELDS}
    writer.writerow(out)
    f.flush()
    os.fsync(f.fileno())
    _append_sniff_group(sidecar_f, sniff_group_id)
    if diag_f is not None and idle_diag_record is not None:
        _append_idle_diag(diag_f, idle_diag_record)


def main() -> None:
    if USE_ENERGY and energy_mod is None:
        raise EnergyAbort(
            "GPU energy monitoring is required for gpu_mode=on but NVML/pynvml is unavailable. "
            "Install pynvml, verify NVIDIA driver access, or rerun with --gpus off."
        )
    if USE_MIPS and perf_mips_mod is None:
        raise MIPSAbort(
            "MIPS profiling is enabled but perf_mips.py could not be imported."
        )

    compute_profile_plan = _load_compute_profile_plan(COMPUTE_PROFILE_PLAN_FILE)
    need_header = _is_file_empty(OUT_CSV)
    sidecar_mode = "w" if need_header else "a"
    diag_context = (
        open(_idle_diag_path(OUT_CSV), sidecar_mode, encoding="utf-8")
        if IDLE_DEBUG
        else nullcontext(None)
    )
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f, open(
        _sniff_groups_path(OUT_CSV),
        sidecar_mode,
        encoding="utf-8",
    ) as sidecar_f, diag_context as diag_f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
            quoting=csv.QUOTE_MINIMAL,
        )
        if need_header:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        # /ready check
        try:
            rr = requests.get(BASE_URL + "/ready", timeout=60, headers={"Connection": "close"})
            if rr.status_code >= 400:
                raise RuntimeError(f"/ready HTTP {rr.status_code}: {rr.text[:200]}")
        except Exception as e:
            row = {k: "nan" for k in CSV_FIELDS}
            row.update({
                "cpu_cores": CPU_CORES,
                "mem_cap_gb": MEM_CAP_GB,
                "gpu_mode": GPU_MODE,
                "cold_start_s": COLD_START_S,
                "status": "error",
                "error": f"ready_failed: {repr(e)}",
            })
            _append_row(writer, row, f, sidecar_f, "")
            return

        cpu_idle_values_so_far: List[float] = []
        gpu_idle_values_so_far: List[float] = []
        for scale_entry in input_scale_entries:
            scale_val = float(scale_entry["input_scale"])
            payload_override = scale_entry.get("payload")
            repeat_request_limit = _prepare_repeat_window(
                scale_val,
                str(scale_entry["scale_label"]),
                payload_override,
            )
            for idx in range(WARMUP + REPEAT):
                warmup_flag = 1 if idx < WARMUP else 0
                repeat_idx = idx if warmup_flag else (idx - WARMUP)

                scale_label = str(scale_entry["scale_label"])
                phase = "w" if warmup_flag else "r"
                sniff_group_id = f"{CASE_NAME}_{scale_label}_{phase}{repeat_idx}"

                latency_app_s = float("nan")
                status = "ok"
                err_msg = ""
                gpu_metrics = _nan_metrics(GPU_METRIC_FIELDS)
                cpu_metrics = _nan_metrics(CPU_METRIC_FIELDS)
                resource_usage_metrics = _nan_metrics(RESOURCE_USAGE_METRIC_FIELDS)
                mips_metrics = _nan_metrics(MIPS_METRIC_FIELDS)
                latency_packet_distribution_metrics = _nan_metrics(LATENCY_PACKET_DISTRIBUTION_FIELDS)
                latency_app_distribution_metrics = _nan_metrics(LATENCY_APP_DISTRIBUTION_FIELDS)
                effective_input_scale: Optional[float] = None
                cpu_idle_measured_at = "nan"
                gpu_idle_measured_at = "nan"
                idle_debug_snapshot: Optional[Dict[str, Any]] = None
                gpu_idle_debug_snapshot: Optional[Dict[str, Any]] = None
                idle_trace: Dict[str, Any] = {}
                gpu_idle_trace: Dict[str, Any] = {}

                try:
                    gpu_monitor = None
                    cpu_monitor = None
                    resource_usage_monitor = None
                    mips_monitor = None
                    gpu_result = None
                    cpu_result = None
                    resource_usage_result = None
                    mips_result = None
                    actual_repeat_in_window = 0
                    lat_sum = 0.0
                    latency_app_values: List[float] = []
                    gpu_monitor_started = False
                    cpu_monitor_started = False
                    resource_usage_monitor_started = False
                    mips_monitor_started = False
                    try:
                        if USE_ENERGY and (energy_mod is not None):
                            gpu_monitor = energy_mod.GPUEnergyMonitor(
                                sample_hz=SAMPLE_HZ,
                                idle_seconds=IDLE_SECONDS,
                                device_index=DEVICE_INDEX,
                            )

                        if cpu_energy_mod is not None:
                            cpu_monitor = cpu_energy_mod.CPUEnergyMonitor(
                                sample_hz=SAMPLE_HZ,
                                idle_seconds=IDLE_SECONDS,
                                container_name=CONTAINER_NAME,
                            )

                        if resource_usage_mod is not None:
                            resource_usage_monitor = resource_usage_mod.ResourceUsageMonitor(
                                sample_hz=SAMPLE_HZ,
                                container_name=CONTAINER_NAME,
                                cpu_cores=_to_float_or_nan(CPU_CORES),
                                mem_cap_gb=_to_float_or_nan(MEM_CAP_GB),
                                use_gpu=USE_ENERGY,
                                device_index=DEVICE_INDEX,
                            )

                        if USE_MIPS:
                            mips_monitor = perf_mips_mod.PerfMIPSMonitor(CONTAINER_NAME)

                        if gpu_monitor is not None or cpu_monitor is not None:
                            _sleep_before_idle_baseline()
                            if _supports_matched_control(gpu_monitor, cpu_monitor):
                                _run_matched_control_window(
                                    gpu_monitor,
                                    cpu_monitor,
                                    resource_usage_monitor,
                                    mips_monitor,
                                )
                                measured_at = _now_iso()
                                if gpu_monitor is not None:
                                    gpu_idle_measured_at = measured_at
                                    gpu_idle_trace = dict(
                                        getattr(gpu_monitor, "idle_trace", {}) or {}
                                    )
                                if cpu_monitor is not None:
                                    cpu_idle_measured_at = measured_at
                                    idle_trace = dict(
                                        getattr(cpu_monitor, "idle_trace", {}) or {}
                                    )
                            else:
                                # Compatibility path for third-party/legacy monitors.
                                if gpu_monitor is not None:
                                    if IDLE_DEBUG:
                                        gpu_monitor.measure_idle(trace=True)
                                    else:
                                        gpu_monitor.measure_idle()
                                    gpu_idle_measured_at = _now_iso()
                                    gpu_idle_trace = dict(
                                        getattr(gpu_monitor, "idle_trace", {}) or {}
                                    )
                                if cpu_monitor is not None:
                                    if IDLE_DEBUG:
                                        cpu_monitor.measure_idle(
                                            trace_interval_s=IDLE_DEBUG_TRACE_INTERVAL_S
                                        )
                                    else:
                                        cpu_monitor.measure_idle()
                                    cpu_idle_measured_at = _now_iso()
                                    idle_trace = dict(
                                        getattr(cpu_monitor, "idle_trace", {}) or {}
                                    )

                            if IDLE_DEBUG:
                                if gpu_monitor is not None:
                                    gpu_idle_debug_snapshot = _collect_gpu_idle_debug_snapshot()
                                if cpu_monitor is not None:
                                    idle_debug_snapshot = _collect_idle_debug_snapshot()

                        if gpu_monitor is not None:
                            gpu_monitor.start()
                            gpu_monitor_started = True
                        if cpu_monitor is not None:
                            cpu_monitor.start()
                            cpu_monitor_started = True
                        if resource_usage_monitor is not None:
                            resource_usage_monitor.start()
                            resource_usage_monitor_started = True
                        if mips_monitor is not None:
                            mips_monitor.start()
                            mips_monitor_started = True

                        while _should_send_window_request(
                            actual_repeat_in_window,
                            lat_sum,
                            repeat_request_limit,
                        ):
                            req_id = f"{sniff_group_id}:{actual_repeat_in_window}"
                            out = _one_request(scale_val, req_id=req_id, payload_override=payload_override)
                            request_latency_app_s = float(out["latency_app_s"])
                            lat_sum += request_latency_app_s
                            latency_app_values.append(request_latency_app_s)
                            actual_repeat_in_window += 1
                            effective_input_scale = _merge_effective_input_scale(
                                effective_input_scale,
                                out.get("effective_input_scale"),
                                scale_val,
                            )
                    finally:
                        mips_stop_error = None
                        if mips_monitor is not None and mips_monitor_started:
                            try:
                                mips_latency_app_s = (
                                    lat_sum / float(actual_repeat_in_window)
                                    if actual_repeat_in_window > 0
                                    else float("nan")
                                )
                                mips_result = mips_monitor.stop(
                                    actual_repeat_in_window,
                                    mips_latency_app_s,
                                )
                            except Exception as exc:
                                mips_stop_error = exc
                        if resource_usage_monitor is not None and resource_usage_monitor_started:
                            resource_usage_result, _resource_usage_err, _resource_usage_samples = (
                                resource_usage_monitor.stop()
                            )
                        if gpu_monitor is not None:
                            try:
                                if gpu_monitor_started:
                                    gpu_result, _gpu_name_ret, _gpu_err, _gpu_samples = (
                                        gpu_monitor.stop()
                                    )
                            finally:
                                gpu_monitor.close()
                        if cpu_monitor is not None:
                            try:
                                if cpu_monitor_started:
                                    cpu_result, _cpu_err, _cpu_samples = cpu_monitor.stop()
                            finally:
                                cpu_monitor.close()
                        if resource_usage_monitor is not None:
                            resource_usage_monitor.close()
                        if mips_monitor is not None:
                            mips_monitor.close()
                        if mips_stop_error is not None:
                            raise mips_stop_error

                    latency_app_s = _mean(latency_app_values)
                    latency_app_distribution_metrics = _latency_distribution_metrics(
                        "latency_app",
                        latency_app_values,
                    )

                    if gpu_result is not None:
                        gpu_metrics = _gpu_metrics_from_result(
                            gpu_result,
                            actual_repeat_in_window,
                        )
                    if cpu_result is not None:
                        cpu_metrics = _cpu_metrics_from_result(
                            cpu_result,
                            actual_repeat_in_window,
                        )
                    if resource_usage_result is not None:
                        resource_usage_metrics = _resource_usage_metrics_from_result(resource_usage_result)
                    if mips_result is not None:
                        mips_metrics = _mips_metrics_from_result(mips_result)

                    warnings = []
                    warnings.extend(_eff_negative_warnings(
                        avg_power_eff_w=gpu_metrics["gpu_avg_power_eff_w"],
                        peak_power_eff_w=gpu_metrics["gpu_peak_power_eff_w"],
                        energy_eff_j=gpu_metrics["gpu_energy_eff_j"],
                    ))
                    warnings.extend(_named_negative_warnings({
                        "cpu_avg_power_eff_w": cpu_metrics["cpu_avg_power_eff_w"],
                        "cpu_peak_power_eff_w": cpu_metrics["cpu_peak_power_eff_w"],
                        "cpu_energy_eff_j": cpu_metrics["cpu_energy_eff_j"],
                        "vcpu_avg_power_eff_w": cpu_metrics["vcpu_avg_power_eff_w"],
                        "vcpu_peak_power_eff_w": cpu_metrics["vcpu_peak_power_eff_w"],
                        "vcpu_energy_eff_j": cpu_metrics["vcpu_energy_eff_j"],
                    }))
                    if warnings:
                        status = "warn"
                        err_msg = "; ".join(warnings)

                    if latency_app_s == latency_app_s and latency_app_s > 0:
                        throughput = float(BATCH_SIZE) / float(latency_app_s)
                    else:
                        throughput = float("nan")

                except EnergyAbort:
                    raise
                except MIPSAbort:
                    raise
                except Exception as e:
                    if _is_mips_error(e):
                        raise MIPSAbort(str(e)) from None
                    status = "error"
                    err_msg = repr(e)
                    throughput = float("nan")

                gpu_idle_stats = _idle_power_debug_stats(gpu_idle_values_so_far, "gpu")
                if IDLE_DEBUG and _finite_positive(gpu_metrics["gpu_idle_power_w"]):
                    gpu_idle_values_so_far.append(_to_float_or_nan(gpu_metrics["gpu_idle_power_w"]))
                    gpu_idle_stats = _idle_power_debug_stats(gpu_idle_values_so_far, "gpu")

                idle_stats = _idle_debug_stats(cpu_idle_values_so_far)
                if IDLE_DEBUG and _finite_positive(cpu_metrics["cpu_idle_power_w"]):
                    cpu_idle_values_so_far.append(_to_float_or_nan(cpu_metrics["cpu_idle_power_w"]))
                    idle_stats = _idle_debug_stats(cpu_idle_values_so_far)

                cpu_cycles_est_app = _estimate_cpu_cycles(
                    latency_app_s,
                    resource_usage_metrics["cpu_freq_avg_hz"],
                    _to_float_or_nan(CPU_CORES),
                    resource_usage_metrics["container_cpu_util_avg_pct"],
                )

                row = {
                    "cpu_cores": CPU_CORES,
                    "mem_cap_gb": MEM_CAP_GB,
                    "gpu_mode": GPU_MODE,
                    "input_scale": str(
                        effective_input_scale if effective_input_scale is not None else scale_val
                    ),
                    "task_param": task_param,
                    "repeat_idx": str(repeat_idx),
                    "warmup": str(warmup_flag),
                    "repeat_in_window": str(actual_repeat_in_window),
                    "latency_s": "nan",  # Placeholder: filled by merge_packet_latency
                    **{field: _fmt_float(latency_packet_distribution_metrics[field]) for field in LATENCY_PACKET_DISTRIBUTION_FIELDS},
                    "latency_app_s": _fmt_float(latency_app_s),
                    **{field: _fmt_float(latency_app_distribution_metrics[field]) for field in LATENCY_APP_DISTRIBUTION_FIELDS},
                    "throughput_samples_per_s": _fmt_float(throughput),
                    **{field: _fmt_float(gpu_metrics[field]) for field in GPU_METRIC_FIELDS},
                    "gpu_idle_measured_at": gpu_idle_measured_at if IDLE_DEBUG else "nan",
                    "gpu_idle_rel_range_so_far": (
                        _fmt_float(gpu_idle_stats["gpu_idle_rel_range_so_far"])
                        if IDLE_DEBUG
                        else "nan"
                    ),
                    **{field: _fmt_float(cpu_metrics[field]) for field in CPU_METRIC_FIELDS},
                    "cpu_idle_measured_at": cpu_idle_measured_at if IDLE_DEBUG else "nan",
                    "cpu_idle_rel_range_so_far": (
                        _fmt_float(idle_stats["cpu_idle_rel_range_so_far"])
                        if IDLE_DEBUG
                        else "nan"
                    ),
                    **{field: _fmt_float(resource_usage_metrics[field]) for field in RESOURCE_USAGE_METRIC_FIELDS},
                    "cpu_cycles_est_app": _fmt_float(cpu_cycles_est_app),
                    "cpu_cycles_est_packet": "nan",
                    **{field: _fmt_float(mips_metrics[field]) for field in MIPS_METRIC_FIELDS},
                    "cold_start_s": COLD_START_S if COLD_START_S else "nan",
                    "status": status,
                    "error": err_msg,
                }
                row_scale = _to_float_or_nan(row["input_scale"])
                compute_profile = _find_compute_profile_entry(
                    compute_profile_plan,
                    GPU_MODE,
                    row_scale,
                )
                model_mflop_per_request = _to_float_or_nan(
                    compute_profile["model_mflop_per_request"]
                )
                compute_mflops_app = _compute_mflops(
                    model_mflop_per_request,
                    latency_app_s,
                )
                row.update({
                    "compute_profile_tool": compute_profile["tool"],
                    "model_mflop_per_request": _fmt_float(model_mflop_per_request),
                    "compute_mflops_app": _fmt_float(compute_mflops_app),
                    "compute_mflops": _fmt_float(compute_mflops_app),
                    "compute_profile_error": compute_profile["error"],
                })
                idle_diag_record = None
                if IDLE_DEBUG:
                    if idle_debug_snapshot is None:
                        idle_debug_snapshot = _collect_idle_debug_snapshot()
                    idle_diag_record = {
                        "case_name": CASE_NAME,
                        "gpu_mode": GPU_MODE,
                        "cpu_cores": CPU_CORES,
                        "mem_cap_gb": MEM_CAP_GB,
                        "input_scale": row["input_scale"],
                        "warmup": row["warmup"],
                        "repeat_idx": row["repeat_idx"],
                        "repeat_in_window": row["repeat_in_window"],
                        "sniff_group_id": sniff_group_id,
                        "gpu_idle_measured_at": row["gpu_idle_measured_at"],
                        "gpu_idle_power_w": _to_float_or_nan(row["gpu_idle_power_w"]),
                        **gpu_idle_stats,
                        **gpu_idle_trace,
                        **(gpu_idle_debug_snapshot or {}),
                        "cpu_idle_measured_at": row["cpu_idle_measured_at"],
                        "cpu_idle_power_w": _to_float_or_nan(row["cpu_idle_power_w"]),
                        **idle_stats,
                        **idle_trace,
                        **idle_debug_snapshot,
                    }
                _append_row(
                    writer,
                    row,
                    f,
                    sidecar_f,
                    sniff_group_id,
                    diag_f=diag_f,
                    idle_diag_record=idle_diag_record,
                )


def run_cli() -> None:
    try:
        main()
    except EnergyAbort as exc:
        print(f"[energy][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except MIPSAbort as exc:
        message = str(exc)
        if message.startswith("[mips][ERROR]"):
            print(message, file=sys.stderr)
        else:
            print(f"[mips][ERROR] {message}", file=sys.stderr)
        exit_code = getattr(perf_mips_mod, "MIPS_EXIT_CODE", 8) if perf_mips_mod else 8
        raise SystemExit(exit_code) from None


if __name__ == "__main__":
    run_cli()

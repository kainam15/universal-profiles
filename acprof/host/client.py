"""AC-Prof Universal Client - workload generation, latency measurement, energy monitoring.

Runs on HOST (not inside container). Generalized from example-code/client.py.
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import re
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
    CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
    CSV_FIELDS,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    GPU_RUNTIME_STATE_FIELDS,
    IDLE_DIAG_DIRNAME,
    SCALING_DIMENSIONS,
)
from acprof.host.compute_profile_plan import (
    NCU_ERROR_FIELD,
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    NCU_TOTAL_MFLOP_FIELD,
    TORCH_ERROR_FIELD,
    TORCH_LOGICAL_MFLOP_FIELD,
    compute_mflops as _compute_mflops,
    find_compute_profile_entry as _find_compute_profile_entry,
    load_compute_profile_plan as _load_compute_profile_plan,
)
from acprof.host.execution_profile_plan import (
    find_execution_profile_entry as _find_execution_profile_entry,
    load_execution_profile_plan as _load_execution_profile_plan,
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
REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS))
)
SLOW_LATENCY_THRESHOLD_S = float(os.getenv("SLOW_LATENCY_THRESHOLD_S", "0.06"))

COLD_START_S = os.getenv("COLD_START_S", "nan")
COLD_START_STARTED_AT = os.getenv("COLD_START_STARTED_AT", "nan")
COLD_START_READY_AT = os.getenv("COLD_START_READY_AT", "nan")
COLD_START_CONTAINER_LAUNCH_S = os.getenv(
    "COLD_START_CONTAINER_LAUNCH_S",
    "nan",
)
COLD_START_SERVER_SETUP_S = os.getenv("COLD_START_SERVER_SETUP_S", "nan")
COLD_START_CUDA_INIT_S = os.getenv("COLD_START_CUDA_INIT_S", "nan")
COLD_START_MODEL_LOAD_S = os.getenv("COLD_START_MODEL_LOAD_S", "nan")
COLD_START_READY_WAIT_S = os.getenv("COLD_START_READY_WAIT_S", "nan")
OUT_CSV = os.getenv("OUT_CSV", "result.csv")
CASE_NAME = os.getenv("CASE_NAME", "").strip()
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "").strip()
SNIFF_GROUPS_PATH = os.getenv("SNIFF_GROUPS_PATH", "").strip()
IDLE_DEBUG = os.getenv("IDLE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
IDLE_DIAG_PATH = os.getenv("IDLE_DIAG_PATH", "").strip()
CLIENT_ERROR_PATH = os.getenv("CLIENT_ERROR_PATH", "").strip()
IDLE_DEBUG_TRACE_INTERVAL_S = float(os.getenv("IDLE_DEBUG_TRACE_INTERVAL_S", "0.1"))
USE_MIPS = os.getenv("USE_MIPS", "").strip().lower() in {"1", "true", "yes", "on"}

_FIRST_PREDICT_APP_S = float("nan")

SAMPLE_HZ = float(os.getenv("SAMPLE_HZ", "20"))
IDLE_SECONDS = float(os.getenv("IDLE_SECONDS", str(DEFAULT_IDLE_SECONDS)))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))
IDLE_COOLDOWN_SECONDS = float(
    os.getenv("IDLE_COOLDOWN_SECONDS", str(DEFAULT_IDLE_COOLDOWN_SECONDS))
)

# Input scales from task family config
INPUT_SCALES_STR = os.getenv("INPUT_SCALES", "")
INPUT_SCALE_PLAN_FILE = os.getenv("INPUT_SCALE_PLAN_FILE", "").strip()
COMPUTE_PROFILE_PLAN_FILE = os.getenv("COMPUTE_PROFILE_PLAN_FILE", "").strip()
EXECUTION_PROFILE_PLAN_FILE = os.getenv(
    "EXECUTION_PROFILE_PLAN_FILE",
    "",
).strip()


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
    if IDLE_DIAG_PATH:
        return IDLE_DIAG_PATH
    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.basename(csv_path)
    return os.path.join(
        csv_dir,
        IDLE_DIAG_DIRNAME,
        f"{csv_name}.idle_diag.jsonl",
    )


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


def _input_units_per_request(input_scale: Any, batch_size: Any) -> float:
    scale = _to_float_or_nan(input_scale)
    batch = _to_float_or_nan(batch_size)
    if math.isfinite(scale) and scale > 0.0 and math.isfinite(batch) and batch > 0.0:
        return scale * batch
    return float("nan")


def _per_positive_denominator(value: Any, denominator: Any) -> float:
    numerator = _to_float_or_nan(value)
    divisor = _to_float_or_nan(denominator)
    if math.isfinite(numerator) and math.isfinite(divisor) and divisor > 0.0:
        return numerator / divisor
    return float("nan")


def _cold_start_row_metrics() -> Dict[str, str]:
    return {
        "cold_start_started_at": COLD_START_STARTED_AT or "nan",
        "cold_start_ready_at": COLD_START_READY_AT or "nan",
        "cold_start_container_launch_s": COLD_START_CONTAINER_LAUNCH_S or "nan",
        "cold_start_server_setup_s": COLD_START_SERVER_SETUP_S or "nan",
        "cold_start_cuda_init_s": COLD_START_CUDA_INIT_S or "nan",
        "cold_start_model_load_s": COLD_START_MODEL_LOAD_S or "nan",
        "cold_start_ready_wait_s": COLD_START_READY_WAIT_S or "nan",
        "cold_start_first_predict_app_s": _fmt_float(_FIRST_PREDICT_APP_S),
        "cold_start_s": COLD_START_S or "nan",
    }


def _compute_profile_row_metrics(
    profile: Dict[str, Any],
    latency_app_s: Any,
) -> Dict[str, str]:
    """Format independent Torch-eager and NCU metrics for one live row."""
    logical_mflop = _to_float_or_nan(profile.get(TORCH_LOGICAL_MFLOP_FIELD))
    logical_mflops_app = _compute_mflops(logical_mflop, latency_app_s)

    ncu_total_mflop = _to_float_or_nan(profile.get(NCU_TOTAL_MFLOP_FIELD))
    ncu_mflops_app = _compute_mflops(ncu_total_mflop, latency_app_s)
    return {
        # Explicit Torch eager logical FLOPs.
        TORCH_LOGICAL_MFLOP_FIELD: _fmt_float(logical_mflop),
        "model_logical_mflops_app_torch_profiler_eager": _fmt_float(
            logical_mflops_app
        ),
        # Packet latency is not known until merge_packet_latency runs, so use
        # the application-denominator value until the packet merge completes.
        "model_logical_mflops_packet_torch_profiler_eager": _fmt_float(
            logical_mflops_app
        ),
        TORCH_ERROR_FIELD: str(profile.get(TORCH_ERROR_FIELD) or ""),
        # Explicit NCU GPU-executed FLOPs and launch metadata.
        NCU_TOTAL_MFLOP_FIELD: _fmt_float(ncu_total_mflop),
        NCU_TENSOR_MFLOP_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_TENSOR_MFLOP_FIELD))
        ),
        NCU_SCALAR_MFLOP_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_SCALAR_MFLOP_FIELD))
        ),
        NCU_TENSOR_SHARE_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_TENSOR_SHARE_FIELD))
        ),
        "gpu_executed_mflops_app_ncu": _fmt_float(ncu_mflops_app),
        "gpu_executed_mflops_packet_ncu": _fmt_float(ncu_mflops_app),
        NCU_KERNEL_COUNT_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_KERNEL_COUNT_FIELD))
        ),
        NCU_KERNEL_TIME_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_KERNEL_TIME_FIELD))
        ),
        NCU_ERROR_FIELD: str(profile.get(NCU_ERROR_FIELD) or ""),
    }


EXECUTION_PROFILE_NUMERIC_FIELDS = (
    "cpu_heap_peak_bytes_massif",
    "cpu_heap_extra_peak_bytes_massif",
    "cpu_stack_peak_bytes_massif",
    "cpu_heap_peak_total_bytes_massif",
    "cpu_heap_peak_at_ms_massif",
    "host_inference_wall_time_ms_per_request_nsys",
    "cuda_api_time_sum_ms_per_request_nsys",
    "cuda_api_call_count_per_request_nsys",
    "gpu_kernel_time_sum_ms_per_request_nsys",
    "gpu_kernel_launch_count_per_request_nsys",
    "gpu_memcpy_time_sum_ms_per_request_nsys",
    "gpu_memcpy_count_per_request_nsys",
    "gpu_memcpy_bytes_per_request_nsys",
)
EXECUTION_PROFILE_ERROR_FIELDS = (
    "compute_profile_error_massif",
    "compute_profile_error_nsys",
)


def _execution_profile_row_metrics(
    profile: Dict[str, Any],
) -> Dict[str, str]:
    """Format intrusive profiler summaries without changing their semantics."""
    result = {
        field: _fmt_float(_to_float_or_nan(profile.get(field)))
        for field in EXECUTION_PROFILE_NUMERIC_FIELDS
    }
    result.update({
        field: str(profile.get(field) or "")
        for field in EXECUTION_PROFILE_ERROR_FIELDS
    })
    return result


class EnergyAbort(RuntimeError):
    """Raised when energy measurement prerequisites are not stable enough."""


class MIPSAbort(RuntimeError):
    """Raised when perf MIPS profiling cannot continue."""


class RequestTimeoutAbort(RuntimeError):
    """Raised when the current resource case cannot finish inference in time."""

    def __init__(
        self,
        message: str,
        *,
        input_scale: Optional[float] = None,
        request_id: str = "",
        timeout_s: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.input_scale = input_scale
        self.request_id = request_id
        self.timeout_s = timeout_s


def _request_phase_context(request_id: str) -> Dict[str, Any]:
    auto_match = re.search(r"_auto_warmup(?P<request_idx>\d+)$", request_id)
    if auto_match:
        return {
            "request_phase": "auto_repeat_window_warmup",
            "request_index_in_window": int(auto_match.group("request_idx")),
        }

    measurement_match = re.search(
        r"_(?P<phase>[wr])(?P<repeat_idx>\d+):(?P<request_idx>\d+)$",
        request_id,
    )
    if measurement_match:
        phase = measurement_match.group("phase")
        return {
            "request_phase": (
                "measurement_warmup" if phase == "w" else "measurement_repeat"
            ),
            "measurement_repeat_idx": int(measurement_match.group("repeat_idx")),
            "request_index_in_window": int(measurement_match.group("request_idx")),
        }

    return {"request_phase": "unknown"}


def _write_client_error_sidecar(exc: RequestTimeoutAbort) -> None:
    if not CLIENT_ERROR_PATH:
        return

    payload = {
        "schema_version": 1,
        "error_type": "client_request_timeout",
        "message": str(exc),
        "input_scale": exc.input_scale,
        "request_id": exc.request_id,
        "request_timeout_s": exc.timeout_s,
        "measurement_completed": False,
        **_request_phase_context(exc.request_id),
    }
    os.makedirs(os.path.dirname(CLIENT_ERROR_PATH) or ".", exist_ok=True)
    tmp_path = f"{CLIENT_ERROR_PATH}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, CLIENT_ERROR_PATH)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _mean_finite(xs: List[float]) -> float:
    values = [
        value
        for value in (_to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    ]
    return _mean(values)


def _canonical_task_param(payload: Optional[Dict[str, Any]]) -> str:
    """Serialize the parameters that the server actually receives."""
    params: Any = {}
    if isinstance(payload, dict):
        nested_params = payload.get("params")
        if isinstance(nested_params, dict):
            params = dict(nested_params)
        elif nested_params is not None:
            params = {"params": nested_params}

        # Time-series handlers consume prediction_length at the top level,
        # rather than through the common nested params object.
        if "prediction_length" in payload:
            params["prediction_length"] = payload["prediction_length"]
    return json.dumps(
        params,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _prepared_body_size_bytes(response: Any) -> float:
    """Return the encoded body size from requests' PreparedRequest."""
    prepared = getattr(response, "request", None)
    body = getattr(prepared, "body", None)
    if isinstance(body, str):
        return float(len(body.encode("utf-8")))
    if isinstance(body, (bytes, bytearray, memoryview)):
        return float(len(body))
    return float("nan")


def _derive_input_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recover v1 audio metadata without changing its legacy payload."""
    metadata: Dict[str, Any] = {}
    audio_samples = payload.get("audio_samples")
    if isinstance(audio_samples, (list, tuple)):
        metadata["input_num_samples"] = len(audio_samples)
        sample_rate = payload.get("sample_rate")
        if sample_rate is not None:
            metadata["sample_rate"] = sample_rate
    return metadata


def _input_num_samples(input_metadata: Any) -> float:
    if not isinstance(input_metadata, dict):
        return float("nan")
    for key in ("input_num_samples", "num_samples", "actual_num_samples"):
        value = _to_float_or_nan(input_metadata.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


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
    values = [
        value
        for value in (_to_float_or_nan(item) for item in latencies)
        if math.isfinite(value)
    ]
    mean = _mean(values)
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        if len(values) >= 2
        else float("nan")
    )
    return {
        f"{prefix}_request_count": float(len(values)),
        f"{prefix}_p50_s": _percentile_nearest_rank(values, 50.0),
        f"{prefix}_p90_s": _percentile_nearest_rank(values, 90.0),
        f"{prefix}_p95_s": _percentile_nearest_rank(values, 95.0),
        f"{prefix}_std_s": std,
        f"{prefix}_cv": (
            std / mean
            if math.isfinite(std) and math.isfinite(mean) and mean > 0.0
            else float("nan")
        ),
        f"{prefix}_iqr_s": (
            _percentile_nearest_rank(values, 75.0)
            - _percentile_nearest_rank(values, 25.0)
            if len(values) >= 2
            else float("nan")
        ),
        f"{prefix}_max_s": max(values) if values else float("nan"),
        f"{prefix}_slow_ratio": _slow_ratio(values),
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

print(f"[client] PIPELINE_TAG={PIPELINE_TAG}", flush=True)


# ─────────────────────────────────────────────
# Workload generator
# ─────────────────────────────────────────────
from acprof.workloads import get_generator  # noqa: E402

workload_gen = (
    None
    if INPUT_SCALE_PLAN_FILE
    else get_generator(TASK_FAMILY, MODEL_ID, PIPELINE_TAG, BATCH_SIZE)
)


def _generic_scale_label(scale_value: float) -> str:
    return f"scale{float(scale_value):g}"


def _one_request(scale_value: float, req_id: str, payload_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global _FIRST_PREDICT_APP_S
    if payload_override is not None:
        payload = payload_override
    else:
        if workload_gen is None:
            raise RuntimeError(
                "input scale plan entry has no payload; the plan must be self-contained"
            )
        payload = workload_gen.generate(scale_value)

    headers = {
        "Connection": "close",
        "X-Req-Id": req_id,
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(
            BASE_URL + ENDPOINT,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout as exc:
        raise RequestTimeoutAbort(
            "inference request timed out after "
            f"{REQUEST_TIMEOUT_SECONDS:g}s "
            f"(input_scale={scale_value:g}, req_id={req_id})",
            input_scale=float(scale_value),
            request_id=req_id,
            timeout_s=float(REQUEST_TIMEOUT_SECONDS),
        ) from exc
    t1 = time.perf_counter()
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", "")
        except Exception:
            detail = r.text[:500]
        raise RuntimeError(f"HTTP {r.status_code}: {detail or r.reason}")
    resp = r.json()
    request_latency_s = t1 - t0
    if not math.isfinite(_FIRST_PREDICT_APP_S):
        _FIRST_PREDICT_APP_S = request_latency_s
    return {
        "latency_app_s": request_latency_s,
        "resp": resp,
        "effective_input_scale": _parse_effective_input_scale(resp),
        "request_payload_bytes": _prepared_body_size_bytes(r),
        "output_length": _to_float_or_nan(resp.get("output_length")),
        "output_token_count": _to_float_or_nan(resp.get("output_token_count")),
        "task_param": _canonical_task_param(payload),
    }


def _load_input_scale_entries() -> List[Dict[str, Any]]:
    if INPUT_SCALE_PLAN_FILE:
        with open(INPUT_SCALE_PLAN_FILE, "r", encoding="utf-8") as f:
            plan = json.load(f)

        raw_schema_version = plan.get("schema_version", 1)
        try:
            schema_version = int(raw_schema_version)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid input scale plan schema_version: {raw_schema_version!r}"
            ) from exc
        if schema_version not in {1, 2}:
            raise RuntimeError(
                f"unsupported input scale plan schema_version={schema_version}"
            )

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
            scale_label = str(
                entry.get("scale_label") or _generic_scale_label(scale_value)
            )
            raw_input_metadata = entry.get("input_metadata")
            if raw_input_metadata is None:
                input_metadata = _derive_input_metadata(payload)
            elif isinstance(raw_input_metadata, dict):
                input_metadata = dict(raw_input_metadata)
                if "input_num_samples" not in input_metadata:
                    derived_metadata = _derive_input_metadata(payload)
                    if "input_num_samples" in derived_metadata:
                        input_metadata["input_num_samples"] = derived_metadata[
                            "input_num_samples"
                        ]
            else:
                raise RuntimeError(
                    f"invalid input_metadata at input scale plan entry {idx}"
                )
            loaded_entries.append({
                "input_scale": scale_value,
                "scale_label": scale_label,
                "payload": payload,
                "input_metadata": input_metadata,
            })

        return loaded_entries

    if workload_gen is None:
        raise RuntimeError("legacy input scales require a workload generator")
    return [
        {
            "input_scale": float(scale_value),
            "scale_label": workload_gen.scale_label(scale_value),
            "payload": None,
            "input_metadata": {},
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
    "container_cpu_nr_periods_delta",
    "container_cpu_nr_throttled_delta",
    "container_cpu_throttled_period_ratio_pct",
    "container_cpu_throttled_time_s_per_request",
    "container_cpu_pressure_some_stall_pct",
    "container_cpu_pressure_full_stall_pct",
    "cpu_freq_avg_hz",
    "cpu_freq_peak_hz",
    "container_mem_usage_avg_bytes",
    "container_mem_usage_peak_bytes",
    "container_mem_util_avg_pct",
    "container_mem_util_peak_pct",
    "container_mem_peak_cgroup_bytes",
    "container_mem_anon_bytes_end",
    "container_mem_file_bytes_end",
    "container_mem_slab_bytes_end",
    "container_mem_pgfault_delta",
    "container_mem_pgmajfault_delta",
    "container_mem_workingset_refault_delta",
    "container_mem_high_events_delta",
    "container_mem_max_events_delta",
    "container_mem_oom_events_delta",
    "container_mem_oom_kill_events_delta",
    "container_mem_pressure_some_stall_pct",
    "container_mem_pressure_full_stall_pct",
    "container_swap_limit_bytes",
    "container_swap_usage_avg_bytes",
    "container_swap_usage_peak_bytes",
    "container_io_read_bytes_per_request",
    "container_io_write_bytes_per_request",
    "container_io_read_ops_per_request",
    "container_io_write_ops_per_request",
    "container_io_pressure_some_stall_pct",
    "container_io_pressure_full_stall_pct",
    "container_pids_current_end",
    "container_pids_peak_cgroup",
    "container_pids_max_events_delta",
    "gpu_util_avg_pct",
    "gpu_util_peak_pct",
    "gpu_mem_used_avg_bytes",
    "gpu_mem_used_peak_bytes",
    "gpu_mem_util_avg_pct",
    "gpu_mem_util_peak_pct",
]

LATENCY_PACKET_DISTRIBUTION_FIELDS = [
    "latency_request_count",
    "latency_p50_s",
    "latency_p90_s",
    "latency_p95_s",
    "latency_std_s",
    "latency_cv",
    "latency_iqr_s",
    "latency_max_s",
    "latency_slow_ratio",
]

LATENCY_APP_DISTRIBUTION_FIELDS = [
    "latency_app_request_count",
    "latency_app_p50_s",
    "latency_app_p90_s",
    "latency_app_p95_s",
    "latency_app_std_s",
    "latency_app_cv",
    "latency_app_iqr_s",
    "latency_app_max_s",
    "latency_app_slow_ratio",
]

EFFICIENCY_METRIC_FIELDS = [
    "container_attributed_energy_eff_j",
    "container_attributed_samples_per_j",
    "container_attributed_edp_app_js",
    "output_tokens_per_s_app",
    "container_attributed_j_per_output_token",
    "container_attributed_j_per_input_unit",
]

MIPS_METRIC_FIELDS = [
    "cpu_instructions_per_request",
    "cpu_mips_app",
    "cpu_mips_packet",
    "cpu_perf_elapsed_s",
    "cpu_cache_references_per_request",
    "cpu_cache_misses_per_request",
    "cpu_cache_miss_rate_pct",
    "cpu_dtlb_loads_per_request",
    "cpu_dtlb_load_misses_per_request",
    "cpu_dtlb_load_miss_rate_pct",
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


def _derived_efficiency_metrics(
    *,
    gpu_mode: str,
    batch_size: int,
    latency_app_s: float,
    output_token_count_avg: float,
    gpu_energy_eff_j: float,
    vcpu_energy_eff_j: float,
    input_units_per_request: float = float("nan"),
) -> Dict[str, float]:
    latency = _to_float_or_nan(latency_app_s)
    output_tokens = _to_float_or_nan(output_token_count_avg)
    gpu_energy = _to_float_or_nan(gpu_energy_eff_j)
    vcpu_energy = _to_float_or_nan(vcpu_energy_eff_j)

    required_energy = [vcpu_energy]
    if str(gpu_mode).strip().lower() == "on":
        required_energy.append(gpu_energy)
    attributed_energy = (
        sum(required_energy)
        if all(math.isfinite(value) and value >= 0.0 for value in required_energy)
        else float("nan")
    )

    samples_per_j = (
        float(batch_size) / attributed_energy
        if math.isfinite(attributed_energy)
        and attributed_energy > 0.0
        and batch_size > 0
        else float("nan")
    )
    edp_app = (
        attributed_energy * latency
        if math.isfinite(attributed_energy)
        and attributed_energy >= 0.0
        and math.isfinite(latency)
        and latency > 0.0
        else float("nan")
    )
    output_tokens_per_s = (
        output_tokens / latency
        if math.isfinite(output_tokens)
        and output_tokens > 0.0
        and math.isfinite(latency)
        and latency > 0.0
        else float("nan")
    )
    joules_per_output_token = (
        attributed_energy / output_tokens
        if math.isfinite(attributed_energy)
        and attributed_energy >= 0.0
        and math.isfinite(output_tokens)
        and output_tokens > 0.0
        else float("nan")
    )
    joules_per_input_unit = _per_positive_denominator(
        attributed_energy,
        input_units_per_request,
    )
    return {
        "container_attributed_energy_eff_j": attributed_energy,
        "container_attributed_samples_per_j": samples_per_j,
        "container_attributed_edp_app_js": edp_app,
        "output_tokens_per_s_app": output_tokens_per_s,
        "container_attributed_j_per_output_token": joules_per_output_token,
        "container_attributed_j_per_input_unit": joules_per_input_unit,
    }


def _resource_usage_metrics_from_result(
    result: Any,
    repeat_in_window: int,
) -> Dict[str, float]:
    io_read_bytes = _to_float_or_nan(
        getattr(result, "container_io_read_bytes", float("nan"))
    )
    io_write_bytes = _to_float_or_nan(
        getattr(result, "container_io_write_bytes", float("nan"))
    )
    io_read_ops = _to_float_or_nan(
        getattr(result, "container_io_read_ops", float("nan"))
    )
    io_write_ops = _to_float_or_nan(
        getattr(result, "container_io_write_ops", float("nan"))
    )
    if repeat_in_window > 0:
        io_read_bytes /= float(repeat_in_window)
        io_write_bytes /= float(repeat_in_window)
        io_read_ops /= float(repeat_in_window)
        io_write_ops /= float(repeat_in_window)
    else:
        io_read_bytes = float("nan")
        io_write_bytes = float("nan")
        io_read_ops = float("nan")
        io_write_ops = float("nan")
    return {
        "resource_usage_iters": float(result.resource_usage_iters),
        "container_cpu_util_avg_pct": _to_float_or_nan(result.container_cpu_util_avg_pct),
        "container_cpu_util_peak_pct": _to_float_or_nan(result.container_cpu_util_peak_pct),
        "container_cpu_nr_periods_delta": _to_float_or_nan(
            getattr(result, "container_cpu_nr_periods_delta", float("nan"))
        ),
        "container_cpu_nr_throttled_delta": _to_float_or_nan(
            getattr(result, "container_cpu_nr_throttled_delta", float("nan"))
        ),
        "container_cpu_throttled_period_ratio_pct": _to_float_or_nan(
            getattr(
                result,
                "container_cpu_throttled_period_ratio_pct",
                float("nan"),
            )
        ),
        "container_cpu_throttled_time_s_per_request": (
            _divide_if_number(
                getattr(result, "container_cpu_throttled_time_s", float("nan")),
                float(repeat_in_window),
            )
            if repeat_in_window > 0
            else float("nan")
        ),
        "container_cpu_pressure_some_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_cpu_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_cpu_pressure_full_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_cpu_pressure_full_stall_pct",
                float("nan"),
            )
        ),
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
        "container_mem_peak_cgroup_bytes": _to_float_or_nan(
            getattr(result, "container_mem_peak_cgroup_bytes", float("nan"))
        ),
        "container_mem_anon_bytes_end": _to_float_or_nan(
            getattr(result, "container_mem_anon_bytes_end", float("nan"))
        ),
        "container_mem_file_bytes_end": _to_float_or_nan(
            getattr(result, "container_mem_file_bytes_end", float("nan"))
        ),
        "container_mem_slab_bytes_end": _to_float_or_nan(
            getattr(result, "container_mem_slab_bytes_end", float("nan"))
        ),
        "container_mem_pgfault_delta": _to_float_or_nan(
            getattr(result, "container_mem_pgfault_delta", float("nan"))
        ),
        "container_mem_pgmajfault_delta": _to_float_or_nan(
            getattr(result, "container_mem_pgmajfault_delta", float("nan"))
        ),
        "container_mem_workingset_refault_delta": _to_float_or_nan(
            getattr(
                result,
                "container_mem_workingset_refault_delta",
                float("nan"),
            )
        ),
        "container_mem_high_events_delta": _to_float_or_nan(
            getattr(result, "container_mem_high_events_delta", float("nan"))
        ),
        "container_mem_max_events_delta": _to_float_or_nan(
            getattr(result, "container_mem_max_events_delta", float("nan"))
        ),
        "container_mem_oom_events_delta": _to_float_or_nan(
            getattr(result, "container_mem_oom_events_delta", float("nan"))
        ),
        "container_mem_oom_kill_events_delta": _to_float_or_nan(
            getattr(result, "container_mem_oom_kill_events_delta", float("nan"))
        ),
        "container_mem_pressure_some_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_mem_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_mem_pressure_full_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_mem_pressure_full_stall_pct",
                float("nan"),
            )
        ),
        "container_swap_limit_bytes": _to_float_or_nan(
            getattr(result, "container_swap_limit_bytes", float("nan"))
        ),
        "container_swap_usage_avg_bytes": _to_float_or_nan(
            getattr(result, "container_swap_usage_avg_bytes", float("nan"))
        ),
        "container_swap_usage_peak_bytes": _to_float_or_nan(
            getattr(result, "container_swap_usage_peak_bytes", float("nan"))
        ),
        "container_io_read_bytes_per_request": io_read_bytes,
        "container_io_write_bytes_per_request": io_write_bytes,
        "container_io_read_ops_per_request": io_read_ops,
        "container_io_write_ops_per_request": io_write_ops,
        "container_io_pressure_some_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_io_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_io_pressure_full_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_io_pressure_full_stall_pct",
                float("nan"),
            )
        ),
        "container_pids_current_end": _to_float_or_nan(
            getattr(result, "container_pids_current_end", float("nan"))
        ),
        "container_pids_peak_cgroup": _to_float_or_nan(
            getattr(result, "container_pids_peak_cgroup", float("nan"))
        ),
        "container_pids_max_events_delta": _to_float_or_nan(
            getattr(result, "container_pids_max_events_delta", float("nan"))
        ),
        "gpu_util_avg_pct": _to_float_or_nan(result.gpu_util_avg_pct),
        "gpu_util_peak_pct": _to_float_or_nan(result.gpu_util_peak_pct),
        "gpu_mem_used_avg_bytes": _to_float_or_nan(result.gpu_mem_used_avg_bytes),
        "gpu_mem_used_peak_bytes": _to_float_or_nan(result.gpu_mem_used_peak_bytes),
        "gpu_mem_util_avg_pct": _to_float_or_nan(result.gpu_mem_util_avg_pct),
        "gpu_mem_util_peak_pct": _to_float_or_nan(result.gpu_mem_util_peak_pct),
    }


def _gpu_runtime_metrics_from_result(resource_usage_result: Any) -> Dict[str, Any]:
    pstate = str(
        getattr(resource_usage_result, "gpu_pstate", "nan")
        if resource_usage_result is not None
        else "nan"
    ).strip().upper()
    if not (
        pstate.startswith("P")
        and pstate[1:].isdigit()
        and 0 <= int(pstate[1:]) <= 15
    ):
        pstate = "nan"
    return {
        "gpu_sm_clock_mhz": _to_float_or_nan(
            getattr(resource_usage_result, "gpu_sm_clock_mhz", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
        "gpu_memory_clock_mhz": _to_float_or_nan(
            getattr(resource_usage_result, "gpu_memory_clock_mhz", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
        "gpu_pstate": pstate,
        "gpu_temp_c": _to_float_or_nan(
            getattr(resource_usage_result, "gpu_temp_c", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
    }


def _mips_metrics_from_result(result: Any) -> Dict[str, float]:
    return {
        "cpu_instructions_per_request": _to_float_or_nan(
            result.instructions_per_request
        ),
        "cpu_mips_app": _to_float_or_nan(result.cpu_mips_app),
        "cpu_mips_packet": float("nan"),
        "cpu_perf_elapsed_s": _to_float_or_nan(result.perf_elapsed_s),
        "cpu_cache_references_per_request": _to_float_or_nan(
            getattr(result, "cache_references_per_request", float("nan"))
        ),
        "cpu_cache_misses_per_request": _to_float_or_nan(
            getattr(result, "cache_misses_per_request", float("nan"))
        ),
        "cpu_cache_miss_rate_pct": _to_float_or_nan(
            getattr(result, "cache_miss_rate_pct", float("nan"))
        ),
        "cpu_dtlb_loads_per_request": _to_float_or_nan(
            getattr(result, "dtlb_loads_per_request", float("nan"))
        ),
        "cpu_dtlb_load_misses_per_request": _to_float_or_nan(
            getattr(result, "dtlb_load_misses_per_request", float("nan"))
        ),
        "cpu_dtlb_load_miss_rate_pct": _to_float_or_nan(
            getattr(result, "dtlb_load_miss_rate_pct", float("nan"))
        ),
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
    if str(out.get("status") or "").strip().lower() == "error" and not str(
        out.get("error") or ""
    ).strip():
        raise RuntimeError("refusing to write status=error without an error diagnostic")
    writer.writerow(out)
    f.flush()
    os.fsync(f.fileno())
    _append_sniff_group(sidecar_f, sniff_group_id)
    if diag_f is not None and idle_diag_record is not None:
        _append_idle_diag(diag_f, idle_diag_record)


def main() -> None:
    global _FIRST_PREDICT_APP_S
    _FIRST_PREDICT_APP_S = float("nan")

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
    execution_profile_plan = _load_execution_profile_plan(
        EXECUTION_PROFILE_PLAN_FILE
    )
    need_header = _is_file_empty(OUT_CSV)
    sidecar_mode = "w" if need_header else "a"
    if IDLE_DEBUG:
        diag_path = _idle_diag_path(OUT_CSV)
        os.makedirs(os.path.dirname(diag_path) or ".", exist_ok=True)
        diag_context = open(diag_path, sidecar_mode, encoding="utf-8")
    else:
        diag_context = nullcontext(None)
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
                **_cold_start_row_metrics(),
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
            input_num_samples = _input_num_samples(
                scale_entry.get("input_metadata")
            )
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
                resolved_input_scale = scale_val
                input_units_per_request = _input_units_per_request(
                    resolved_input_scale,
                    BATCH_SIZE,
                )
                latency_app_s_per_input_unit = float("nan")
                throughput_per_cpu_core = float("nan")
                request_payload_bytes = float("nan")
                output_length_avg = float("nan")
                output_token_count_avg = float("nan")
                executed_task_param: Optional[str] = None
                status = "ok"
                err_msg = ""
                gpu_metrics = _nan_metrics(GPU_METRIC_FIELDS)
                cpu_metrics = _nan_metrics(CPU_METRIC_FIELDS)
                efficiency_metrics = _nan_metrics(EFFICIENCY_METRIC_FIELDS)
                resource_usage_metrics = _nan_metrics(RESOURCE_USAGE_METRIC_FIELDS)
                gpu_runtime_metrics: Dict[str, Any] = _nan_metrics(
                    GPU_RUNTIME_STATE_FIELDS
                )
                gpu_runtime_metrics["gpu_pstate"] = "nan"
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
                    request_payload_bytes_values: List[float] = []
                    output_length_values: List[float] = []
                    output_token_count_values: List[float] = []
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
                            request_payload_bytes_values.append(
                                _to_float_or_nan(out.get("request_payload_bytes"))
                            )
                            output_length_values.append(
                                _to_float_or_nan(out.get("output_length"))
                            )
                            output_token_count_values.append(
                                _to_float_or_nan(out.get("output_token_count"))
                            )
                            request_task_param = out.get("task_param")
                            if (
                                request_task_param is not None
                                and executed_task_param is None
                            ):
                                executed_task_param = str(request_task_param)
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
                    request_payload_bytes = _mean_finite(
                        request_payload_bytes_values
                    )
                    output_length_avg = _mean_finite(output_length_values)
                    output_token_count_avg = _mean_finite(
                        output_token_count_values
                    )
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
                        resource_usage_metrics = _resource_usage_metrics_from_result(
                            resource_usage_result,
                            actual_repeat_in_window,
                        )
                    resolved_input_scale = (
                        effective_input_scale
                        if effective_input_scale is not None
                        else scale_val
                    )
                    input_units_per_request = _input_units_per_request(
                        resolved_input_scale,
                        BATCH_SIZE,
                    )
                    efficiency_metrics = _derived_efficiency_metrics(
                        gpu_mode=GPU_MODE,
                        batch_size=BATCH_SIZE,
                        latency_app_s=latency_app_s,
                        output_token_count_avg=output_token_count_avg,
                        gpu_energy_eff_j=gpu_metrics["gpu_energy_eff_j"],
                        vcpu_energy_eff_j=cpu_metrics["vcpu_energy_eff_j"],
                        input_units_per_request=input_units_per_request,
                    )
                    gpu_runtime_metrics = _gpu_runtime_metrics_from_result(
                        resource_usage_result
                    )
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
                    latency_app_s_per_input_unit = _per_positive_denominator(
                        latency_app_s,
                        input_units_per_request,
                    )
                    throughput_per_cpu_core = _per_positive_denominator(
                        throughput,
                        CPU_CORES,
                    )

                except RequestTimeoutAbort:
                    raise
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
                    throughput_per_cpu_core = float("nan")

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
                    "input_scale": str(resolved_input_scale),
                    "input_units_per_request": _fmt_float(
                        input_units_per_request
                    ),
                    "input_num_samples": _fmt_float(input_num_samples),
                    "request_payload_bytes": _fmt_float(request_payload_bytes),
                    "packet_request_wire_bytes_per_request": "nan",
                    "packet_response_wire_bytes_per_request": "nan",
                    "packet_total_wire_bytes_per_request": "nan",
                    "packet_tcp_payload_bytes_per_request": "nan",
                    "packet_protocol_overhead_bytes_per_request": "nan",
                    "packet_protocol_overhead_ratio": "nan",
                    "task_param": (
                        executed_task_param
                        if executed_task_param is not None
                        else _canonical_task_param(payload_override)
                    ),
                    "output_length_avg": _fmt_float(output_length_avg),
                    "output_token_count_avg": _fmt_float(
                        output_token_count_avg
                    ),
                    "repeat_idx": str(repeat_idx),
                    "warmup": str(warmup_flag),
                    "repeat_in_window": str(actual_repeat_in_window),
                    "latency_s": "nan",  # Placeholder: filled by merge_packet_latency
                    "latency_s_per_input_unit": "nan",
                    **{field: _fmt_float(latency_packet_distribution_metrics[field]) for field in LATENCY_PACKET_DISTRIBUTION_FIELDS},
                    "latency_app_s": _fmt_float(latency_app_s),
                    "latency_app_s_per_input_unit": _fmt_float(
                        latency_app_s_per_input_unit
                    ),
                    **{field: _fmt_float(latency_app_distribution_metrics[field]) for field in LATENCY_APP_DISTRIBUTION_FIELDS},
                    "throughput_samples_per_s": _fmt_float(throughput),
                    "throughput_samples_per_s_per_cpu_core": _fmt_float(
                        throughput_per_cpu_core
                    ),
                    **{field: _fmt_float(gpu_metrics[field]) for field in GPU_METRIC_FIELDS},
                    "gpu_idle_measured_at": gpu_idle_measured_at if IDLE_DEBUG else "nan",
                    "gpu_idle_rel_range_so_far": (
                        _fmt_float(gpu_idle_stats["gpu_idle_rel_range_so_far"])
                        if IDLE_DEBUG
                        else "nan"
                    ),
                    **{field: _fmt_float(cpu_metrics[field]) for field in CPU_METRIC_FIELDS},
                    **{
                        field: _fmt_float(efficiency_metrics[field])
                        for field in EFFICIENCY_METRIC_FIELDS
                    },
                    "cpu_idle_measured_at": cpu_idle_measured_at if IDLE_DEBUG else "nan",
                    "cpu_idle_rel_range_so_far": (
                        _fmt_float(idle_stats["cpu_idle_rel_range_so_far"])
                        if IDLE_DEBUG
                        else "nan"
                    ),
                    **{field: _fmt_float(resource_usage_metrics[field]) for field in RESOURCE_USAGE_METRIC_FIELDS},
                    **{
                        field: (
                            str(gpu_runtime_metrics[field])
                            if field == "gpu_pstate"
                            else _fmt_float(gpu_runtime_metrics[field])
                        )
                        for field in GPU_RUNTIME_STATE_FIELDS
                    },
                    "cpu_cycles_est_app": _fmt_float(cpu_cycles_est_app),
                    "cpu_cycles_est_packet": "nan",
                    **{field: _fmt_float(mips_metrics[field]) for field in MIPS_METRIC_FIELDS},
                    **_cold_start_row_metrics(),
                    "status": status,
                    "error": err_msg,
                }
                row_scale = _to_float_or_nan(row["input_scale"])
                compute_profile = _find_compute_profile_entry(
                    compute_profile_plan,
                    GPU_MODE,
                    row_scale,
                )
                row.update(
                    _compute_profile_row_metrics(
                        compute_profile,
                        latency_app_s,
                    )
                )
                execution_profile = _find_execution_profile_entry(
                    execution_profile_plan,
                    CPU_CORES,
                    MEM_CAP_GB,
                    GPU_MODE,
                    row_scale,
                )
                row.update(
                    _execution_profile_row_metrics(execution_profile)
                )
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
    except RequestTimeoutAbort as exc:
        try:
            _write_client_error_sidecar(exc)
        except OSError as sidecar_exc:
            print(
                "[case][WARN] failed to persist structured timeout context: "
                f"{sidecar_exc}",
                file=sys.stderr,
            )
        print(f"[case][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(CLIENT_REQUEST_TIMEOUT_EXIT_CODE) from None
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

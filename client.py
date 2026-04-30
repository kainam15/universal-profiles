"""AC-Prof Universal Client - workload generation, latency measurement, energy monitoring.

Runs on HOST (not inside container). Generalized from example-code/client.py.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
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

from config import (
    CSV_FIELDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_TASK_PARAMS,
    SCALING_DIMENSIONS,
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
CALIBRATION_REQUESTS = 3

COLD_START_S = os.getenv("COLD_START_S", "nan")
OUT_CSV = os.getenv("OUT_CSV", "result.csv")
CASE_NAME = os.getenv("CASE_NAME", "").strip()
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "").strip()
SNIFF_GROUPS_PATH = os.getenv("SNIFF_GROUPS_PATH", "").strip()

SAMPLE_HZ = float(os.getenv("SAMPLE_HZ", "20"))
IDLE_SECONDS = float(os.getenv("IDLE_SECONDS", "3"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))
COOLDOWN_SECONDS = 3

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


class EnergyAbort(RuntimeError):
    """Raised when energy measurement prerequisites are not stable enough."""


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    values = sorted(xs)
    n = len(values)
    if n % 2:
        return values[n // 2]
    return 0.5 * (values[n // 2 - 1] + values[n // 2])


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


def _load_compute_profile_plan(path: str) -> Dict[str, Any]:
    if not path:
        return {"profiles": {}, "_load_error": "compute_profile_disabled"}
    if not os.path.exists(path):
        return {"profiles": {}, "_load_error": f"compute_profile_plan_not_found:{path}"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as exc:
        return {"profiles": {}, "_load_error": f"compute_profile_plan_invalid:{exc!r}"}
    if not isinstance(plan, dict):
        return {"profiles": {}, "_load_error": "compute_profile_plan_invalid:not_dict"}
    profiles = plan.get("profiles")
    if not isinstance(profiles, dict):
        return {"profiles": {}, "_load_error": "compute_profile_plan_invalid:missing_profiles"}
    return plan


def _find_compute_profile_entry(
    plan: Dict[str, Any],
    gpu_mode: str,
    input_scale: float,
) -> Dict[str, Any]:
    profile_key = "gpu" if gpu_mode == "on" else "cpu"
    load_error = str(plan.get("_load_error", "") or "")
    if load_error:
        return {
            "tool": "nan",
            "model_mflop_per_request": float("nan"),
            "error": load_error,
        }

    profile = plan.get("profiles", {}).get(profile_key)
    if not isinstance(profile, dict):
        return {
            "tool": "nan",
            "model_mflop_per_request": float("nan"),
            "error": f"compute_profile_missing_profile:{profile_key}",
        }

    tool = str(profile.get("tool") or "nan")
    profile_error = str(profile.get("error") or "")
    entries = profile.get("entries")
    if not isinstance(entries, list):
        return {
            "tool": tool,
            "model_mflop_per_request": float("nan"),
            "error": profile_error or f"compute_profile_missing_entries:{profile_key}",
        }

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            entry_scale = float(entry.get("input_scale"))
        except (TypeError, ValueError):
            continue
        if math.isclose(entry_scale, float(input_scale), rel_tol=0.0, abs_tol=1e-6):
            return {
                "tool": str(entry.get("tool") or tool or "nan"),
                "model_mflop_per_request": _to_float_or_nan(
                    entry.get("model_mflop_per_request")
                ),
                "error": str(entry.get("error") or profile_error),
            }

    return {
        "tool": tool,
        "model_mflop_per_request": float("nan"),
        "error": f"compute_profile_missing_scale:{input_scale:g}",
    }


def _compute_mflops(model_mflop_per_request: float, latency_s: float) -> float:
    mflop = _to_float_or_nan(model_mflop_per_request)
    latency = _to_float_or_nan(latency_s)
    if mflop == mflop and latency == latency and latency > 0:
        return mflop / latency
    return float("nan")


def _calibrate_repeat_in_window(
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

    latencies: List[float] = []
    for idx in range(CALIBRATION_REQUESTS):
        req_id = f"{CASE_NAME}_{scale_label}_calib{idx}"
        out = _one_request(scale_value, req_id=req_id, payload_override=payload_override)
        latency = _to_float_or_nan(out.get("latency_app_s"))
        if not math.isfinite(latency) or latency <= 0.0:
            raise EnergyAbort(
                "repeat-in-window auto calibration failed: calibration request returned "
                f"invalid latency_app_s={latency!r} for input_scale={scale_value:g}"
            )
        latencies.append(latency)

    median_latency = _median(latencies)
    repeat_in_window = max(1, int(math.ceil(REPEAT_WINDOW_SECONDS / median_latency)))
    print(
        "[client] auto repeat-in-window "
        f"scale={scale_value:g} median_latency_s={median_latency:.6f} "
        f"target_window_s={REPEAT_WINDOW_SECONDS:.3f} repeat_in_window={repeat_in_window}",
        flush=True,
    )
    return repeat_in_window


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
from workloads import get_generator  # noqa: E402

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
        import energy_nvml as energy_mod
    except Exception as _e:
        energy_mod = None
        print(f"[WARN] GPU energy monitoring unavailable: {_e.__class__.__name__}: {_e}",
              file=__import__('sys').stderr)
        print("[WARN] Install pynvml: pip install pynvml", file=__import__('sys').stderr)

cpu_energy_mod = None
try:
    import energy_cpu as cpu_energy_mod
except Exception as _e:
    cpu_energy_mod = None
    print(f"[WARN] CPU energy monitoring unavailable: {_e.__class__.__name__}: {_e}",
          file=__import__('sys').stderr)

resource_usage_mod = None
try:
    import resource_usage as resource_usage_mod
except Exception as _e:
    resource_usage_mod = None
    print(f"[WARN] Resource usage monitoring unavailable: {_e.__class__.__name__}: {_e}",
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


def _append_sniff_group(sidecar_f, sniff_group_id: str) -> None:
    sidecar_f.write(json.dumps({"sniff_group_id": sniff_group_id}, ensure_ascii=True) + "\n")
    sidecar_f.flush()
    os.fsync(sidecar_f.fileno())


def _append_row(
    writer: csv.DictWriter,
    row: Dict[str, Any],
    f,
    sidecar_f,
    sniff_group_id: str,
) -> None:
    out = {k: row.get(k, "") for k in CSV_FIELDS}
    writer.writerow(out)
    f.flush()
    os.fsync(f.fileno())
    _append_sniff_group(sidecar_f, sniff_group_id)


def main() -> None:
    if USE_ENERGY and energy_mod is None:
        raise EnergyAbort(
            "GPU energy monitoring is required for gpu_mode=on but NVML/pynvml is unavailable. "
            "Install pynvml, verify NVIDIA driver access, or rerun with --gpus off."
        )

    compute_profile_plan = _load_compute_profile_plan(COMPUTE_PROFILE_PLAN_FILE)
    need_header = _is_file_empty(OUT_CSV)
    sidecar_mode = "w" if need_header else "a"
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f, open(
        _sniff_groups_path(OUT_CSV),
        sidecar_mode,
        encoding="utf-8",
    ) as sidecar_f:
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

        for scale_entry in input_scale_entries:
            scale_val = float(scale_entry["input_scale"])
            payload_override = scale_entry.get("payload")
            actual_repeat_in_window = _calibrate_repeat_in_window(
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
                effective_input_scale: Optional[float] = None

                try:
                    gpu_monitor = None
                    cpu_monitor = None
                    resource_usage_monitor = None
                    gpu_result = None
                    cpu_result = None
                    resource_usage_result = None
                    try:
                        if USE_ENERGY and (energy_mod is not None):
                            # Preserve the existing GPU cooldown behavior.
                            time.sleep(COOLDOWN_SECONDS)
                            gpu_monitor = energy_mod.GPUEnergyMonitor(
                                sample_hz=SAMPLE_HZ,
                                idle_seconds=IDLE_SECONDS,
                                device_index=DEVICE_INDEX,
                            )
                            gpu_monitor.measure_idle()

                        if cpu_energy_mod is not None:
                            cpu_monitor = cpu_energy_mod.CPUEnergyMonitor(
                                sample_hz=SAMPLE_HZ,
                                idle_seconds=IDLE_SECONDS,
                                container_name=CONTAINER_NAME,
                            )
                            cpu_monitor.measure_idle()

                        if resource_usage_mod is not None:
                            resource_usage_monitor = resource_usage_mod.ResourceUsageMonitor(
                                sample_hz=SAMPLE_HZ,
                                container_name=CONTAINER_NAME,
                                cpu_cores=_to_float_or_nan(CPU_CORES),
                                mem_cap_gb=_to_float_or_nan(MEM_CAP_GB),
                                use_gpu=USE_ENERGY,
                                device_index=DEVICE_INDEX,
                            )

                        if gpu_monitor is not None:
                            gpu_monitor.start()
                        if cpu_monitor is not None:
                            cpu_monitor.start()
                        if resource_usage_monitor is not None:
                            resource_usage_monitor.start()

                        lat_sum = 0.0
                        for k in range(actual_repeat_in_window):
                            req_id = f"{sniff_group_id}:{k}"
                            out = _one_request(scale_val, req_id=req_id, payload_override=payload_override)
                            lat_sum += float(out["latency_app_s"])
                            effective_input_scale = _merge_effective_input_scale(
                                effective_input_scale,
                                out.get("effective_input_scale"),
                                scale_val,
                            )
                    finally:
                        if resource_usage_monitor is not None:
                            resource_usage_result, _resource_usage_err, _resource_usage_samples = (
                                resource_usage_monitor.stop()
                            )
                        if gpu_monitor is not None:
                            try:
                                gpu_result, _gpu_name_ret, _gpu_err, _gpu_samples = gpu_monitor.stop()
                            finally:
                                gpu_monitor.close()
                        if cpu_monitor is not None:
                            try:
                                cpu_result, _cpu_err, _cpu_samples = cpu_monitor.stop()
                            finally:
                                cpu_monitor.close()
                        if resource_usage_monitor is not None:
                            resource_usage_monitor.close()

                    latency_app_s = lat_sum / float(actual_repeat_in_window)

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
                except Exception as e:
                    status = "error"
                    err_msg = repr(e)
                    throughput = float("nan")

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
                    "latency_app_s": _fmt_float(latency_app_s),
                    "throughput_samples_per_s": _fmt_float(throughput),
                    **{field: _fmt_float(gpu_metrics[field]) for field in GPU_METRIC_FIELDS},
                    **{field: _fmt_float(cpu_metrics[field]) for field in CPU_METRIC_FIELDS},
                    **{field: _fmt_float(resource_usage_metrics[field]) for field in RESOURCE_USAGE_METRIC_FIELDS},
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
                _append_row(writer, row, f, sidecar_f, sniff_group_id)


def run_cli() -> None:
    try:
        main()
    except EnergyAbort as exc:
        print(f"[energy][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    run_cli()

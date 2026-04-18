"""AC-Prof Universal Client - workload generation, latency measurement, energy monitoring.

Runs on HOST (not inside container). Generalized from example-code/client.py.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import requests

from config import CSV_FIELDS, SCALING_DIMENSIONS, DEFAULT_TASK_PARAMS

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
REPEAT_IN_WINDOW = int(os.getenv("REPEAT_IN_WINDOW", "20"))

COLD_START_S = os.getenv("COLD_START_S", "nan")
OUT_CSV = os.getenv("OUT_CSV", "result.csv")
CASE_NAME = os.getenv("CASE_NAME", "").strip()

SAMPLE_HZ = float(os.getenv("SAMPLE_HZ", "20"))
IDLE_SECONDS = float(os.getenv("IDLE_SECONDS", "3"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))
COOLDOWN_SECONDS = 3

EFF_POWER_EPS_W = float(os.getenv("EFF_POWER_EPS_W", "0.01"))
EFF_ENERGY_EPS_J = float(os.getenv("EFF_ENERGY_EPS_J", "0.001"))
MIN_ENERGY_ITERS = int(os.getenv("MIN_ENERGY_ITERS", "6"))

# Input scales from task family config
INPUT_SCALES_STR = os.getenv("INPUT_SCALES", "")
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


def _one_request(scale_value: float, req_id: str) -> Dict[str, Any]:
    payload = workload_gen.generate(scale_value)

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


def _append_row(writer: csv.DictWriter, row: Dict[str, Any], f) -> None:
    out = {k: row.get(k, "") for k in CSV_FIELDS}
    writer.writerow(out)
    f.flush()
    os.fsync(f.fileno())


def main() -> None:
    need_header = _is_file_empty(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
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
            _append_row(writer, row, f)
            return

        for scale_val in input_scales:
            for idx in range(WARMUP + REPEAT):
                warmup_flag = 1 if idx < WARMUP else 0
                repeat_idx = idx if warmup_flag else (idx - WARMUP)

                scale_label = workload_gen.scale_label(scale_val)
                phase = "w" if warmup_flag else "r"
                sniff_group_id = f"{CASE_NAME}_{scale_label}_{phase}{repeat_idx}"

                latency_app_s = float("nan")
                status = "ok"
                err_msg = ""

                idle_power_w = float("nan")
                energy_iters = float("nan")
                avg_power_total_w = float("nan")
                peak_power_total_w = float("nan")
                energy_total_j = float("nan")
                avg_power_eff_w = float("nan")
                peak_power_eff_w = float("nan")
                energy_eff_j = float("nan")
                effective_input_scale: Optional[float] = None

                try:
                    if USE_ENERGY and (energy_mod is not None):
                        holder: Dict[str, Any] = {
                            "lat_app_sum": 0.0,
                            "effective_input_scale": None,
                        }

                        def fn():
                            lat_sum = 0.0
                            for k in range(REPEAT_IN_WINDOW):
                                req_id = f"{sniff_group_id}:{k}"
                                out = _one_request(scale_val, req_id=req_id)
                                lat_sum += float(out["latency_app_s"])
                                holder["effective_input_scale"] = _merge_effective_input_scale(
                                    holder.get("effective_input_scale"),
                                    out.get("effective_input_scale"),
                                    scale_val,
                                )
                            holder["lat_app_sum"] = lat_sum

                        time.sleep(COOLDOWN_SECONDS)
                        er, _gpu_name_ret, err, _samples = energy_mod.measure_energy_threaded(
                            fn=fn,
                            sample_hz=SAMPLE_HZ,
                            idle_seconds=IDLE_SECONDS,
                            device_index=DEVICE_INDEX,
                            align_to_fn=True,
                        )

                        latency_total_app_s = _to_float_or_nan(holder.get("lat_app_sum"))
                        effective_input_scale = holder.get("effective_input_scale")
                        if latency_total_app_s == latency_total_app_s and latency_total_app_s > 0:
                            latency_app_s = latency_total_app_s / float(REPEAT_IN_WINDOW)

                        if err:
                            status = "error"
                            err_msg = err
                        else:
                            idle_power_w = _to_float_or_nan(er.idle_power_w)
                            energy_iters = float(er.energy_iters)

                            avg_power_total_w = _to_float_or_nan(er.avg_power_total_w)
                            peak_power_total_w = _to_float_or_nan(er.peak_power_total_w)
                            energy_total_j = _to_float_or_nan(er.energy_total_j)
                            if energy_total_j == energy_total_j:
                                energy_total_j = energy_total_j / float(REPEAT_IN_WINDOW)

                            avg_power_eff_w = _to_float_or_nan(er.avg_power_eff_w)
                            peak_power_eff_w = _to_float_or_nan(er.peak_power_eff_w)
                            energy_eff_j = _to_float_or_nan(er.energy_eff_j)
                            if energy_eff_j == energy_eff_j:
                                energy_eff_j = energy_eff_j / float(REPEAT_IN_WINDOW)

                            eff_low_signal = (
                                (energy_iters == energy_iters and energy_iters < MIN_ENERGY_ITERS)
                                or (avg_power_eff_w == avg_power_eff_w and abs(avg_power_eff_w) <= EFF_POWER_EPS_W)
                                or (energy_eff_j == energy_eff_j and abs(energy_eff_j) <= EFF_ENERGY_EPS_J)
                            )
                            if (
                                peak_power_eff_w == peak_power_eff_w
                                and abs(peak_power_eff_w) <= EFF_POWER_EPS_W
                                and eff_low_signal
                            ):
                                status = "warn"
                                err_msg = "eff_energy_near_zero_clamped; consider using total_energy_* or heavier workload"
                                avg_power_eff_w = float("nan")
                                peak_power_eff_w = float("nan")
                                energy_eff_j = float("nan")
                    else:
                        # GPU off: still send REPEAT_IN_WINDOW requests for sniffing
                        lat_sum = 0.0
                        for k in range(REPEAT_IN_WINDOW):
                            req_id = f"{sniff_group_id}:{k}"
                            out = _one_request(scale_val, req_id=req_id)
                            lat_sum += float(out["latency_app_s"])
                            effective_input_scale = _merge_effective_input_scale(
                                effective_input_scale,
                                out.get("effective_input_scale"),
                                scale_val,
                            )
                        latency_app_s = lat_sum / float(REPEAT_IN_WINDOW)

                    if latency_app_s == latency_app_s and latency_app_s > 0:
                        throughput = float(BATCH_SIZE) / float(latency_app_s)
                    else:
                        throughput = float("nan")

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
                    "sniff_group_id": sniff_group_id,
                    "repeat_in_window": str(REPEAT_IN_WINDOW),
                    "latency_s": "nan",  # Placeholder: filled by merge_packet_latency
                    "latency_app_s": _fmt_float(latency_app_s),
                    "throughput_samples_per_s": _fmt_float(throughput),
                    "idle_power_w": _fmt_float(idle_power_w),
                    "energy_iters": _fmt_float(energy_iters),
                    "avg_power_total_w": _fmt_float(avg_power_total_w),
                    "peak_power_total_w": _fmt_float(peak_power_total_w),
                    "energy_total_j": _fmt_float(energy_total_j),
                    "avg_power_eff_w": _fmt_float(avg_power_eff_w),
                    "peak_power_eff_w": _fmt_float(peak_power_eff_w),
                    "energy_eff_j": _fmt_float(energy_eff_j),
                    "cold_start_s": COLD_START_S if COLD_START_S else "nan",
                    "status": status,
                    "error": err_msg,
                }
                _append_row(writer, row, f)


if __name__ == "__main__":
    main()

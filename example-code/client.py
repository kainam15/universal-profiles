# client.py (run on HOST, not inside container)
from __future__ import annotations

import csv
import math
import os
import random
import time
from typing import Any, Dict, List

import requests

# -------------------------
# Config from env
# -------------------------
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("MODEL_ID", "amazon/chronos-bolt-base"))
CPU_CORES = os.getenv("CPU_CORES", "")
MEM_CAP_GB = os.getenv("MEM_CAP_GB", "")
GPU_MODE = os.getenv("GPU_MODE", "off").lower()
GPU_MODE = "on" if GPU_MODE == "on" else "off"

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8002").rstrip("/")
ENDPOINT = os.getenv("ENDPOINT", "/predict")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
PRED_LEN = int(os.getenv("PRED_LEN", "24"))
CONTEXT_LENS = os.getenv("CONTEXT_LENS", "64,128,256,512")
WARMUP = int(os.getenv("WARMUP", "2"))
REPEAT = int(os.getenv("REPEAT", "5"))

REPEAT_IN_WINDOW = int(os.getenv("REPEAT_IN_WINDOW", "20"))

COLD_START_S = os.getenv("COLD_START_S", os.getenv("COLD_START", "nan"))
OUT_CSV = os.getenv("OUT_CSV", "result.csv")
CASE_NAME = os.getenv("CASE_NAME", "").strip()

SAMPLE_HZ = float(os.getenv("SAMPLE_HZ", "20"))
IDLE_SECONDS = float(os.getenv("IDLE_SECONDS", "3"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))
COOLDOWN_SECONDS = 3

EFF_POWER_EPS_W = float(os.getenv("EFF_POWER_EPS_W", "0.01"))
EFF_ENERGY_EPS_J = float(os.getenv("EFF_ENERGY_EPS_J", "0.001"))
MIN_ENERGY_ITERS = int(os.getenv("MIN_ENERGY_ITERS", "6"))

# -------------------------
# CSV schema
# -------------------------
FIELDS = [
    "cpu_cores",
    "mem_cap_gb",
    "gpu_mode",
    "batch_size",
    "context_length",
    "prediction_length",
    "repeat_idx",
    "warmup",

    # 抓包回填关键
    "sniff_group_id",
    "repeat_in_window",

    "latency_s",        # packet-level（后处理 merge 回填）
    "latency_app_s",    # app-level（实时写，debug/主参考）
    "throughput_samples_per_s",

    "idle_power_w",
    "energy_iters",

    "avg_power_total_w",
    "peak_power_total_w",
    "energy_total_j",

    "avg_power_eff_w",
    "peak_power_eff_w",
    "energy_eff_j",

    "cold_start_s",
    "status",
    "error",
]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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


BASE_SEED = 12345
context_lens = sorted(set(_parse_int_list(CONTEXT_LENS)))

MAX_LEN = max(context_lens)
random.seed(BASE_SEED)
BASE_SEQ = [random.random() for _ in range(MAX_LEN)]
CONTEXT_CACHE = {L: BASE_SEQ[:L] for L in context_lens}


def _one_request(context_len: int, req_id: str) -> Dict[str, Any]:
    one = CONTEXT_CACHE[context_len]
    context_batch = []
    for b in range(BATCH_SIZE):
        context_batch.append([x + 1e-6 * (b + 1) for x in one])

    payload = {
        "context": context_batch,
        "prediction_length": PRED_LEN,
        "num_samples": 1,
    }

    headers = {
        "Connection": "close",   # 关键：避免 keep-alive 混在一个 TCP stream
        "X-Req-Id": req_id,      # 关键：pcap 解析用这个识别每个请求属于哪个 group
    }

    t0 = time.perf_counter()
    r = requests.post(BASE_URL + ENDPOINT, json=payload, headers=headers, timeout=300)
    t1 = time.perf_counter()
    r.raise_for_status()
    return {"latency_app_s": (t1 - t0), "resp": r.json()}


# -------------------------
# Optional NVML energy
# -------------------------
USE_ENERGY = (GPU_MODE == "on")
energy_mod = None
if USE_ENERGY:
    try:
        import energy_nvml as energy_mod
    except Exception:
        energy_mod = None


def _append_row(writer: csv.DictWriter, row: Dict[str, Any], f) -> None:
    out = {k: row.get(k, "") for k in FIELDS}
    writer.writerow(out)
    f.flush()
    os.fsync(f.fileno())


def main() -> None:
    need_header = _is_file_empty(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
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
            row = {k: "nan" for k in FIELDS}
            row.update({
                "cpu_cores": CPU_CORES,
                "mem_cap_gb": MEM_CAP_GB,
                "gpu_mode": GPU_MODE,
                "batch_size": str(BATCH_SIZE),
                "prediction_length": str(PRED_LEN),
                "cold_start_s": COLD_START_S,
                "status": "error",
                "error": f"ready_failed: {repr(e)}",
            })
            _append_row(writer, row, f)
            return

        for L in context_lens:
            for idx in range(WARMUP + REPEAT):
                warmup_flag = 1 if idx < WARMUP else 0
                repeat_idx = idx if warmup_flag else (idx - WARMUP)

                # ✅ 关键：每个 context_length + repeat 独立一个 group
                sniff_group_id = f"{CASE_NAME}_ctx{L}_rep{repeat_idx}"

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

                try:
                    if USE_ENERGY and (energy_mod is not None):
                        holder: Dict[str, Any] = {"lat_app_sum": 0.0}

                        def fn():
                            lat_sum = 0.0
                            for k in range(REPEAT_IN_WINDOW):
                                # ✅ 关键：X-Req-Id 前缀=sniff_group_id
                                req_id = f"{sniff_group_id}:{k}"
                                out = _one_request(L, req_id=req_id)
                                lat_sum += float(out["latency_app_s"])
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
                                (energy_iters == energy_iters and energy_iters < MIN_ENERGY_ITERS) or
                                (avg_power_eff_w == avg_power_eff_w and abs(avg_power_eff_w) <= EFF_POWER_EPS_W) or
                                (energy_eff_j == energy_eff_j and abs(energy_eff_j) <= EFF_ENERGY_EPS_J)
                            )
                            if (
                                peak_power_eff_w == peak_power_eff_w and
                                abs(peak_power_eff_w) <= EFF_POWER_EPS_W and
                                eff_low_signal
                            ):
                                status = "warn"
                                err_msg = "eff_energy_near_zero_clamped; consider using total_energy_* or heavier workload"
                                avg_power_eff_w = float("nan")
                                peak_power_eff_w = float("nan")
                                energy_eff_j = float("nan")
                    else:
                        # GPU off：也照样发 REPEAT_IN_WINDOW 次，保证抓包有足够样本
                        lat_sum = 0.0
                        for k in range(REPEAT_IN_WINDOW):
                            req_id = f"{sniff_group_id}:{k}"
                            out = _one_request(L, req_id=req_id)
                            lat_sum += float(out["latency_app_s"])
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
                    "batch_size": str(BATCH_SIZE),
                    "context_length": str(L),
                    "prediction_length": str(PRED_LEN),
                    "repeat_idx": str(repeat_idx),
                    "warmup": str(warmup_flag),

                    "sniff_group_id": sniff_group_id,
                    "repeat_in_window": str(REPEAT_IN_WINDOW),

                    # ✅ 占位：packet-level 由 sniff_parse_pcap + merge 回填
                    "latency_s": "nan",
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
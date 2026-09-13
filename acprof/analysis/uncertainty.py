"""按测量窗口聚合的不确定性；独立 profiler 与冷启动复用值不参与。"""
from __future__ import annotations

from collections import defaultdict
import math
import random
import statistics

from acprof.metric_registry import METRICS
from acprof.result_csv import measurement_key


def summarize_windows(rows, metrics, *, confidence=0.95, resamples=5000, seed=0, block_size=1):
    if not 0 < confidence < 1 or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("confidence 必须在 0 与 1 之间，resamples 必须为正整数")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size 必须为正整数")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("请选择不重复的数值指标")
    for name in metrics:
        metric = METRICS.get(name)
        if metric is None or metric.kind != "number" or metric.window not in {"request_window", "matched_control"}:
            raise ValueError(f"{name} 不属于独立测量窗口；不对复用的 profiler/生命周期值计算区间")
    cases, keys = defaultdict(list), set()
    for row in rows:
        key = measurement_key(row)
        if key in keys:
            raise ValueError(f"duplicate measurement: {key}")
        keys.add(key)
        if str(row.get("status", "")).strip().lower() == "ok" and key[4] == "0":
            cases[key[:4]].append(row)
    groups = []
    for case in sorted(cases):
        ordered = sorted(cases[case], key=lambda row: float(row["repeat_idx"]))
        for name in metrics:
            values = []
            for row in ordered:
                try:
                    value = float(row.get(name, "nan"))
                except (ValueError, TypeError):
                    value = float("nan")
                if math.isfinite(value):
                    values.append(value)
            count = len(values)
            result = {"cpu_cores": float(case[0]), "mem_cap_gb": float(case[1]), "gpu_mode": case[2],
                      "input_scale": float(case[3]), "metric": name, "unit": METRICS[name].unit,
                      "n_windows": count, "missing_windows": len(ordered) - count,
                      "mean": statistics.fmean(values) if values else None,
                      "std": statistics.stdev(values) if count > 1 else None,
                      "ci_low": None, "ci_high": None, "reason": ""}
            if count < max(3, 3 * block_size):
                result["reason"] = "insufficient_windows"
            elif block_size > 1 and count != len(ordered):
                result["reason"] = "missing_windows_break_blocks"
            elif block_size > 1 and any(float(b["repeat_idx"]) != float(a["repeat_idx"]) + 1
                                        for a, b in zip(ordered, ordered[1:])):
                result["reason"] = "nonconsecutive_windows"
            else:
                result["ci_low"], result["ci_high"] = bootstrap_mean_interval(
                    values, confidence=confidence, resamples=resamples, seed=seed, block_size=block_size)
            groups.append(result)
    return {"schema_version": 1, "confidence": confidence, "resamples": resamples, "seed": seed,
            "block_size": block_size, "method": "circular_block_percentile_bootstrap",
            "resampling_unit": "csv_request_window", "filter": "status=ok and warmup=0",
            "assumption": "窗口（或选定连续块）之间可视为独立；区间仅描述本实验内变异", "groups": groups}


def bootstrap_mean_interval(values, *, confidence=0.95, resamples=5000, seed=0, block_size=1):
    """有限窗口均值的 percentile 区间；不把单窗口当作可估计区间。"""
    if not 0 < confidence < 1 or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("invalid bootstrap settings")
    if not isinstance(block_size, int) or block_size < 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("invalid bootstrap values or block size")
    count = len(values)
    if count < max(3, 3 * block_size):
        return None, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        # 循环移动块；block_size=1 即普通 percentile bootstrap。
        sampled = []
        while len(sampled) < count:
            start = rng.randrange(count)
            sampled.extend(values[(start + offset) % count] for offset in range(block_size))
        estimates.append(statistics.fmean(sampled[:count]))
    estimates.sort()

    def quantile(probability):
        index = (len(estimates) - 1) * probability
        lower = math.floor(index)
        fraction = index - lower
        return estimates[lower] * (1 - fraction) + estimates[min(lower + 1, len(estimates) - 1)] * fraction

    tail = (1 - confidence) / 2
    return quantile(tail), quantile(1 - tail)

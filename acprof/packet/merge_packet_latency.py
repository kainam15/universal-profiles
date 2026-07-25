import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Sequence

SNIFF_GROUP_FIELD = "sniff_group_id"
SLOW_LATENCY_THRESHOLD_S = float(os.getenv("SLOW_LATENCY_THRESHOLD_S", "0.06"))


def _read_static_batch_size(csv_path: str) -> float:
    static_meta_path = os.path.join(os.path.dirname(csv_path) or ".", "static_meta.csv")
    if not os.path.exists(static_meta_path):
        return float("nan")

    with open(static_meta_path, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f), None)

    if not row:
        return float("nan")

    try:
        return float(row.get("batch_size", "nan"))
    except Exception:
        return float("nan")


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return float("nan")
    rank = int(math.ceil((float(percentile) / 100.0) * len(finite_values)))
    index = min(max(rank - 1, 0), len(finite_values) - 1)
    return finite_values[index]


def _slow_ratio(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return sum(value > SLOW_LATENCY_THRESHOLD_S for value in finite_values) / float(len(finite_values))


def _fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _estimate_cpu_cycles(row: dict, latency_s: float) -> float:
    freq_hz = _to_float(row.get("cpu_freq_avg_hz", "nan"))
    cpu_cores = _to_float(row.get("cpu_cores", "nan"))
    cpu_util_pct = _to_float(row.get("container_cpu_util_avg_pct", "nan"))
    if (
        latency_s == latency_s
        and freq_hz == freq_hz
        and cpu_cores == cpu_cores
        and cpu_util_pct == cpu_util_pct
        and latency_s > 0.0
        and freq_hz > 0.0
        and cpu_cores > 0.0
        and cpu_util_pct >= 0.0
    ):
        return latency_s * freq_hz * cpu_cores * (cpu_util_pct / 100.0)
    return float("nan")


def _compute_cpu_mips(row: dict, latency_s: float) -> float:
    instructions_per_request = _to_float(row.get("cpu_instructions_per_request", "nan"))
    if (
        latency_s == latency_s
        and instructions_per_request == instructions_per_request
        and latency_s > 0.0
        and instructions_per_request >= 0.0
    ):
        return instructions_per_request / latency_s / 1_000_000.0
    return float("nan")


def _read_sidecar_groups(csv_path: str) -> list[str]:
    sidecar_path = f"{csv_path}.sniff_groups.jsonl"
    if not os.path.exists(sidecar_path):
        return []

    groups = []
    with open(sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                groups.append("")
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                groups.append("")
                continue
            groups.append(str(payload.get(SNIFF_GROUP_FIELD, "") or ""))
    return groups


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        raise SystemExit(
            "usage: python -m acprof.packet.merge_packet_latency "
            "<in_csv> <lat_json> <out_csv>"
        )

    in_csv, lat_json, out_csv = args
    with open(lat_json, "r", encoding="utf-8") as f:
        lat_map = json.load(f)
    static_batch_size = _read_static_batch_size(in_csv)

    group_lats = defaultdict(list)
    for req_id, lat in lat_map.items():
        group_id = req_id.split(":", 1)[0]
        group_lats[group_id].append(float(lat))

    with open(in_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    if fields is None:
        raise SystemExit("empty csv")

    sidecar_groups = _read_sidecar_groups(in_csv)

    for idx, r in enumerate(rows):
        gid = r.get(SNIFF_GROUP_FIELD, "")
        if not gid and idx < len(sidecar_groups):
            gid = sidecar_groups[idx]
        if gid and gid in group_lats and group_lats[gid]:
            latencies = group_lats[gid]
            r["latency_s"] = _fmt_float(_mean(latencies))
            r["latency_p50_s"] = _fmt_float(_percentile_nearest_rank(latencies, 50.0))
            r["latency_p90_s"] = _fmt_float(_percentile_nearest_rank(latencies, 90.0))
            r["latency_p95_s"] = _fmt_float(_percentile_nearest_rank(latencies, 95.0))
            r["latency_slow_ratio"] = _fmt_float(_slow_ratio(latencies))
            try:
                bs = static_batch_size
                if bs != bs:
                    bs = float(r.get("batch_size", "nan"))
                lat = float(r["latency_s"])
                if lat > 0:
                    r["throughput_samples_per_s"] = f"{(bs / lat):.6f}"
            except Exception:
                pass
            try:
                lat = float(r["latency_s"])
                model_mflop_per_request = float(r.get("model_mflop_per_request", "nan"))
                if lat > 0 and model_mflop_per_request == model_mflop_per_request:
                    r["compute_mflops"] = f"{(model_mflop_per_request / lat):.6f}"
            except Exception:
                pass
            try:
                lat = float(r["latency_s"])
                cpu_cycles_est_packet = _estimate_cpu_cycles(r, lat)
                if cpu_cycles_est_packet == cpu_cycles_est_packet:
                    r["cpu_cycles_est_packet"] = f"{cpu_cycles_est_packet:.6f}"
            except Exception:
                pass
            try:
                lat = float(r["latency_s"])
                cpu_mips_packet = _compute_cpu_mips(r, lat)
                if cpu_mips_packet == cpu_mips_packet:
                    r["cpu_mips_packet"] = f"{cpu_mips_packet:.6f}"
            except Exception:
                pass

    fields = [field for field in fields if field != SNIFF_GROUP_FIELD]
    for r in rows:
        r.pop(SNIFF_GROUP_FIELD, None)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

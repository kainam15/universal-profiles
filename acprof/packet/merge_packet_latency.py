import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Sequence

SNIFF_GROUP_FIELD = "sniff_group_id"
SLOW_LATENCY_THRESHOLD_S = float(os.getenv("SLOW_LATENCY_THRESHOLD_S", "0.06"))
NETWORK_RECORD_TO_CSV_FIELD = {
    "request_wire_bytes": "packet_request_wire_bytes_per_request",
    "response_wire_bytes": "packet_response_wire_bytes_per_request",
    "total_wire_bytes": "packet_total_wire_bytes_per_request",
    "tcp_payload_bytes": "packet_tcp_payload_bytes_per_request",
    "protocol_overhead_bytes": "packet_protocol_overhead_bytes_per_request",
}


def _read_static_batch_size(csv_path: str) -> float:
    result_dir = os.path.dirname(csv_path) or "."
    static_meta_json = os.path.join(result_dir, "static_meta.json")
    row = None
    if os.path.exists(static_meta_json):
        with open(static_meta_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            row = payload
    else:
        legacy_csv = os.path.join(result_dir, "static_meta.csv")
        if os.path.exists(legacy_csv):
            with open(legacy_csv, "r", encoding="utf-8", newline="") as f:
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


def _request_records(payload: object) -> dict[str, dict[str, float]]:
    """Normalize both the legacy flat map and packet schema v2."""
    if not isinstance(payload, dict):
        return {}
    raw_requests = payload.get("requests")
    if isinstance(raw_requests, dict):
        records = {}
        for request_id, raw_record in raw_requests.items():
            if not isinstance(raw_record, dict):
                continue
            records[str(request_id)] = {
                key: _to_float(value)
                for key, value in raw_record.items()
            }
        return records

    records = {}
    for request_id, latency in payload.items():
        parsed = _to_float(latency)
        if math.isfinite(parsed):
            records[str(request_id)] = {"latency_s": parsed}
    return records


def _input_units_per_request(row: dict, batch_size: float) -> float:
    explicit = _to_float(row.get("input_units_per_request", "nan"))
    if math.isfinite(explicit) and explicit > 0.0:
        return explicit
    input_scale = _to_float(row.get("input_scale", "nan"))
    if (
        math.isfinite(input_scale)
        and input_scale > 0.0
        and math.isfinite(batch_size)
        and batch_size > 0.0
    ):
        return input_scale * batch_size
    return float("nan")


def _set_packet_network_metrics(row: dict, records: list[dict[str, float]]) -> None:
    for record_field, csv_field in NETWORK_RECORD_TO_CSV_FIELD.items():
        value = _mean([record.get(record_field, float("nan")) for record in records])
        if math.isfinite(value):
            row[csv_field] = _fmt_float(value)

    overhead_values = [
        record.get("protocol_overhead_bytes", float("nan"))
        for record in records
    ]
    finite_overhead_values = [
        value for value in overhead_values if math.isfinite(value)
    ]
    wire_values = [
        record.get("total_wire_bytes", float("nan")) for record in records
    ]
    finite_wire_values = [value for value in wire_values if math.isfinite(value)]
    total_wire_sum = sum(finite_wire_values)
    if finite_overhead_values and finite_wire_values and total_wire_sum > 0.0:
        row["packet_protocol_overhead_ratio"] = _fmt_float(
            sum(finite_overhead_values) / total_wire_sum
        )


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


def _distribution_metrics(values: list[float]) -> dict[str, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    mean = _mean(finite_values)
    std = (
        math.sqrt(
            sum((value - mean) ** 2 for value in finite_values)
            / len(finite_values)
        )
        if len(finite_values) >= 2
        else float("nan")
    )
    return {
        "latency_request_count": float(len(finite_values)),
        "latency_p50_s": _percentile_nearest_rank(finite_values, 50.0),
        "latency_p90_s": _percentile_nearest_rank(finite_values, 90.0),
        "latency_p95_s": _percentile_nearest_rank(finite_values, 95.0),
        "latency_std_s": std,
        "latency_cv": (
            std / mean
            if math.isfinite(std) and math.isfinite(mean) and mean > 0.0
            else float("nan")
        ),
        "latency_iqr_s": (
            _percentile_nearest_rank(finite_values, 75.0)
            - _percentile_nearest_rank(finite_values, 25.0)
            if len(finite_values) >= 2
            else float("nan")
        ),
        "latency_max_s": max(finite_values) if finite_values else float("nan"),
        "latency_slow_ratio": _slow_ratio(finite_values),
    }


def _fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _mflops_for_latency(work_mflop: object, latency_s: float) -> float:
    work = _to_float(work_mflop)
    if math.isfinite(work) and math.isfinite(latency_s) and latency_s > 0.0:
        return work / latency_s
    return float("nan")


def _recompute_packet_flop_rates(row: dict, latency_s: float) -> None:
    """Recompute packet-denominator rates without changing app rates."""
    legacy_work = _to_float(row.get("model_mflop_per_request", "nan"))
    logical_work = _to_float(
        row.get(
            "model_logical_mflop_per_request_torch_profiler_eager",
            "nan",
        )
    )
    ncu_work = _to_float(
        row.get("gpu_executed_mflop_per_request_ncu", "nan")
    )
    tool = str(row.get("compute_profile_tool", "") or "").strip().lower()

    # Legacy CSVs can carry generic aliases; new CSVs use explicit fields.
    generic_work = (
        legacy_work if math.isfinite(legacy_work) else logical_work
    )
    generic_rate = _mflops_for_latency(generic_work, latency_s)
    if "compute_mflops" in row and math.isfinite(generic_rate):
        row["compute_mflops"] = _fmt_float(generic_rate)

    if not math.isfinite(logical_work) and "torch" in tool:
        logical_work = legacy_work
    logical_rate = _mflops_for_latency(logical_work, latency_s)
    if (
        "model_logical_mflops_packet_torch_profiler_eager" in row
        and math.isfinite(logical_rate)
    ):
        row[
            "model_logical_mflops_packet_torch_profiler_eager"
        ] = _fmt_float(logical_rate)

    if not math.isfinite(ncu_work) and (
        tool == "ncu" or "nsight" in tool
    ):
        ncu_work = legacy_work
    ncu_rate = _mflops_for_latency(ncu_work, latency_s)
    if (
        "gpu_executed_mflops_packet_ncu" in row
        and math.isfinite(ncu_rate)
    ):
        row["gpu_executed_mflops_packet_ncu"] = _fmt_float(ncu_rate)


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
        packet_payload = json.load(f)
    request_records = _request_records(packet_payload)
    static_batch_size = _read_static_batch_size(in_csv)

    group_records = defaultdict(list)
    for req_id, record in request_records.items():
        group_id = req_id.split(":", 1)[0]
        group_records[group_id].append(record)

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
        if gid and gid in group_records and group_records[gid]:
            records = group_records[gid]
            latencies = [record.get("latency_s", float("nan")) for record in records]
            r["latency_s"] = _fmt_float(_mean(latencies))
            for field, value in _distribution_metrics(latencies).items():
                r[field] = _fmt_float(value)
            try:
                bs = static_batch_size
                if bs != bs:
                    bs = float(r.get("batch_size", "nan"))
                lat = float(r["latency_s"])
                if lat > 0:
                    r["throughput_samples_per_s"] = f"{(bs / lat):.6f}"
                    input_units = _input_units_per_request(r, bs)
                    if math.isfinite(input_units) and input_units > 0.0:
                        r["latency_s_per_input_unit"] = _fmt_float(
                            lat / input_units
                        )
                    cpu_cores = _to_float(r.get("cpu_cores", "nan"))
                    if math.isfinite(cpu_cores) and cpu_cores > 0.0:
                        r["throughput_samples_per_s_per_cpu_core"] = _fmt_float(
                            (bs / lat) / cpu_cores
                        )
            except Exception:
                pass
            try:
                _set_packet_network_metrics(r, records)
            except Exception:
                pass
            try:
                _recompute_packet_flop_rates(r, float(r["latency_s"]))
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

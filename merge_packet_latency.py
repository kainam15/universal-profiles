import csv
import json
import os
import sys
from collections import defaultdict

in_csv = sys.argv[1]
lat_json = sys.argv[2]
out_csv = sys.argv[3]

lat_map = json.load(open(lat_json, "r", encoding="utf-8"))
SNIFF_GROUP_FIELD = "sniff_group_id"


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


static_batch_size = _read_static_batch_size(in_csv)


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


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

# group_id -> list of latencies
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
        r["latency_s"] = f"{(sum(group_lats[gid]) / len(group_lats[gid])):.6f}"
        # 你也可以选择同时用 packet latency 更新 throughput
        try:
            bs = static_batch_size
            if bs != bs:
                bs = float(r.get("batch_size", "nan"))
            lat = float(r["latency_s"])
            if lat > 0:
                r["throughput_samples_per_s"] = f"{(bs/lat):.6f}"
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
    else:
        # 没匹配到就保持 nan，方便你发现抓包/解析问题
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

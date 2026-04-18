import csv
import json
import os
import sys
from collections import defaultdict

in_csv = sys.argv[1]
lat_json = sys.argv[2]
out_csv = sys.argv[3]

lat_map = json.load(open(lat_json, "r", encoding="utf-8"))


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

for r in rows:
    gid = r.get("sniff_group_id", "")
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
    else:
        # 没匹配到就保持 nan，方便你发现抓包/解析问题
        pass

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=fields,
        quoting=csv.QUOTE_MINIMAL,
    )
    w.writeheader()
    w.writerows(rows)

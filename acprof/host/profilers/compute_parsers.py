"""Advisor 与 NCU 原生 CSV 解析；只依赖标准库。"""
from __future__ import annotations

import csv
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


NCU_FLOAT_TYPE_PATTERN = r"(?:bf\d+|fp\d+|tf\d+)"


NCU_TENSOR_METRIC_RE = re.compile(
    rf"^sm__ops_path_tensor_src_{NCU_FLOAT_TYPE_PATTERN}"
    rf"(?:_{NCU_FLOAT_TYPE_PATTERN})*_dst_{NCU_FLOAT_TYPE_PATTERN}\.sum$"
)


NCU_SASS_FLOP_WEIGHTS = {
    "smsp__sass_thread_inst_executed_op_dadd_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_dmul_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_dfma_pred_on": 2.0,
    "smsp__sass_thread_inst_executed_op_fadd_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_ffma_pred_on": 2.0,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on": 2.0,
}


NCU_DURATION_METRIC = "gpu__time_duration.sum"


def _clean_numeric_text(value: str) -> str:
    value = value.strip().replace(",", "")
    if not value:
        return value
    suffix = value[-1].lower()
    factors = {
        "k": 1_000.0,
        "m": 1_000_000.0,
        "g": 1_000_000_000.0,
        "t": 1_000_000_000_000.0,
    }
    if suffix in factors:
        return str(float(value[:-1]) * factors[suffix])
    return value


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        if isinstance(value, str):
            value = _clean_numeric_text(value)
        return float(value)
    except Exception:
        return float("nan")


def _finite_or_none(value: Any) -> Optional[float]:
    number = _to_float(value)
    return number if math.isfinite(number) else None


def parse_advisor_self_gflop_csv(report_path: str) -> float:
    """Return summed Intel Advisor Self GFLOP from a CSV report."""
    total_gflop = 0.0
    with open(report_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        fieldnames = None
        for row in reader:
            field_map = {field.strip().lower(): field for field in row}
            if "self gflop" in field_map:
                fieldnames = row
                break
        if fieldnames is None:
            return float("nan")
        field_map = {field.strip().lower(): field for field in fieldnames}
        gflop_field = field_map.get("self gflop")
        if gflop_field is None:
            return float("nan")
        found = False
        for row in csv.DictReader(f, fieldnames=fieldnames):
            value = _to_float(row.get(gflop_field))
            if value == value:
                total_gflop += value
                found = True
    return total_gflop if found else float("nan")


def _metric_name_from_row(row: Dict[str, str]) -> str:
    for key in ("Metric Name", "Metric", "Name"):
        value = row.get(key)
        if value:
            return value.strip()
    for key, value in row.items():
        if key and key.strip().lower() == "metric name" and value:
            return value.strip()
    return ""


def _metric_value_from_row(row: Dict[str, str]) -> float:
    for key in ("Metric Value", "Value"):
        if key in row:
            return _to_float(row.get(key))
    for key, value in row.items():
        if key and key.strip().lower() in {"metric value", "value"}:
            return _to_float(value)
    return float("nan")


def _metric_unit_from_row(row: Dict[str, str]) -> str:
    for key in ("Metric Unit", "Unit"):
        if key in row and row.get(key):
            return str(row[key]).strip()
    for key, value in row.items():
        if key and key.strip().lower() in {"metric unit", "unit"} and value:
            return str(value).strip()
    return ""


def _ncu_metric_flop_weight(metric_name: str) -> Optional[float]:
    if NCU_TENSOR_METRIC_RE.match(metric_name):
        return 1.0
    if metric_name in NCU_SASS_FLOP_WEIGHTS:
        return NCU_SASS_FLOP_WEIGHTS[metric_name]
    if metric_name.endswith(".sum"):
        base_name = metric_name[:-len(".sum")]
        if base_name in NCU_SASS_FLOP_WEIGHTS:
            return NCU_SASS_FLOP_WEIGHTS[base_name]
    return None


def _is_ncu_flop_metric(metric_name: str) -> bool:
    return _ncu_metric_flop_weight(metric_name) is not None


def _is_ncu_tensor_metric(metric_name: str) -> bool:
    return NCU_TENSOR_METRIC_RE.match(metric_name) is not None


def _is_ncu_sass_metric(metric_name: str) -> bool:
    base_name = (
        metric_name[:-len(".sum")]
        if metric_name.endswith(".sum")
        else metric_name
    )
    return base_name in NCU_SASS_FLOP_WEIGHTS


def _duration_to_ms(value: float, unit: str) -> float:
    normalized = str(unit or "").strip().lower().replace(" ", "")
    factors = {
        "s": 1_000.0,
        "sec": 1_000.0,
        "second": 1_000.0,
        "seconds": 1_000.0,
        "ms": 1.0,
        "msec": 1.0,
        "msecond": 1.0,
        "millisecond": 1.0,
        "milliseconds": 1.0,
        "us": 0.001,
        "usec": 0.001,
        "usecond": 0.001,
        "µs": 0.001,
        "microsecond": 0.001,
        "microseconds": 0.001,
        "ns": 0.000001,
        "nsec": 0.000001,
        "nsecond": 0.000001,
        "nanosecond": 0.000001,
        "nanoseconds": 0.000001,
    }
    return value * factors.get(normalized, 0.000001)


def _ncu_csv_rows(report_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read an NCU CSV while skipping profiler diagnostics before the header."""
    with open(report_path, "r", encoding="utf-8-sig", newline="") as f:
        filtered_lines = [
            line
            for line in f
            if line.strip() and not line.lstrip().startswith("==PROF==")
        ]
    parsed_rows = list(csv.reader(filtered_lines))

    header_index = None
    for idx, row in enumerate(parsed_rows):
        normalized = {field.strip().lower() for field in row if field}
        if "metric name" in normalized or "kernel name" in normalized:
            header_index = idx
            break
        if any(
            _is_ncu_flop_metric(field.strip())
            or field.strip() == NCU_DURATION_METRIC
            for field in row
        ):
            header_index = idx
            break
    if header_index is None:
        return [], []

    fieldnames = parsed_rows[header_index]
    rows = [
        {
            fieldnames[column]: value
            for column, value in enumerate(row[:len(fieldnames)])
        }
        for row in parsed_rows[header_index + 1:]
    ]
    return fieldnames, rows


def _first_finite(rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> float:
    for row in rows:
        for field in fields:
            value = _to_float(row.get(field))
            if value == value:
                return value
    return float("nan")


def _first_text(rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> str:
    for row in rows:
        for field in fields:
            value = str(row.get(field) or "").strip()
            if value:
                return value
    return ""


def _ncu_launch_count_wide(rows: Sequence[Dict[str, str]]) -> int:
    count = 0
    for row in rows:
        launch_id = str(row.get("ID") or "").strip()
        kernel_name = str(
            row.get("Kernel Name")
            or row.get("Kernel")
            or row.get("launch__kernel_name")
            or ""
        ).strip()
        if launch_id or kernel_name:
            count += 1
    return count


def _ncu_launch_count_long(rows: Sequence[Dict[str, str]]) -> int:
    launch_ids = set()
    for row in rows:
        launch_id = str(row.get("ID") or row.get("Launch ID") or "").strip()
        if launch_id:
            process_id = str(row.get("Process ID") or "").strip()
            launch_ids.add((process_id, launch_id))
    if launch_ids:
        return len(launch_ids)

    duration_rows = sum(
        1 for row in rows if _metric_name_from_row(row) == NCU_DURATION_METRIC
    )
    if duration_rows:
        return duration_rows

    metric_counts: Dict[str, int] = {}
    for row in rows:
        metric_name = _metric_name_from_row(row)
        if _is_ncu_flop_metric(metric_name):
            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
    return max(metric_counts.values(), default=0)


def parse_ncu_profile_csv(
    report_path: str,
    repeat: int = 1,
) -> Dict[str, Any]:
    """Parse wide or long NCU CSV into per-request hardware execution metrics."""
    fieldnames, rows = _ncu_csv_rows(report_path)
    normalized_repeat = max(1, int(repeat))
    if not fieldnames:
        return {
            "total_flops_per_request": float("nan"),
            "tensor_flops_per_request": float("nan"),
            "scalar_flops_per_request": float("nan"),
            "tensor_share_pct": float("nan"),
            "kernel_launch_count_per_request": float("nan"),
            "kernel_time_sum_ms_per_request": float("nan"),
            "gpu_compute_capability": "",
            "gpu_sm_count": float("nan"),
        }

    is_long = any(
        field.strip().lower() == "metric name"
        for field in fieldnames
    )
    metric_values: List[Tuple[str, float, str]] = []
    kernel_time_ms = 0.0
    duration_found = False

    if is_long:
        for row in rows:
            metric_name = _metric_name_from_row(row)
            value = _metric_value_from_row(row)
            if value != value:
                continue
            if _is_ncu_flop_metric(metric_name):
                metric_values.append((metric_name, value, _metric_unit_from_row(row)))
            elif metric_name == NCU_DURATION_METRIC:
                kernel_time_ms += _duration_to_ms(value, _metric_unit_from_row(row))
                duration_found = True
        launch_count = _ncu_launch_count_long(rows)
    else:
        metric_fields = [
            field
            for field in fieldnames
            if _is_ncu_flop_metric(field)
        ]
        duration_fields = [
            field for field in fieldnames if field == NCU_DURATION_METRIC
        ]
        unit_by_field: Dict[str, str] = {}
        for row in rows:
            for field in [*metric_fields, *duration_fields]:
                raw_value = str(row.get(field) or "").strip()
                if raw_value and _to_float(raw_value) != _to_float(raw_value):
                    unit_by_field.setdefault(field, raw_value)
        for row in rows:
            for field in metric_fields:
                value = _to_float(row.get(field))
                if value == value:
                    metric_values.append((field, value, unit_by_field.get(field, "")))
            for field in duration_fields:
                value = _to_float(row.get(field))
                if value == value:
                    kernel_time_ms += _duration_to_ms(
                        value,
                        unit_by_field.get(field, ""),
                    )
                    duration_found = True
        launch_count = _ncu_launch_count_wide(rows)

    tensor_flops = 0.0
    scalar_flops = 0.0
    tensor_found = False
    scalar_found = False
    for metric_name, value, _unit in metric_values:
        weight = _ncu_metric_flop_weight(metric_name)
        if weight is None:
            continue
        if _is_ncu_tensor_metric(metric_name):
            tensor_flops += value * weight
            tensor_found = True
        else:
            scalar_flops += value * weight
            scalar_found = True

    any_flops = tensor_found or scalar_found
    total_flops = tensor_flops + scalar_flops
    total_per_request = (
        total_flops / normalized_repeat if any_flops else float("nan")
    )
    tensor_per_request = (
        tensor_flops / normalized_repeat if any_flops else float("nan")
    )
    scalar_per_request = (
        scalar_flops / normalized_repeat if any_flops else float("nan")
    )
    tensor_share_pct = (
        tensor_flops / total_flops * 100.0
        if any_flops and total_flops > 0
        else float("nan")
    )

    compute_capability = _first_text(rows, ("CC", "Compute Capability"))
    if not compute_capability:
        major = _first_finite(
            rows,
            ("device__attribute_compute_capability_major",),
        )
        minor = _first_finite(
            rows,
            ("device__attribute_compute_capability_minor",),
        )
        if major == major and minor == minor:
            compute_capability = f"{int(major)}.{int(minor)}"
    sm_count = _first_finite(
        rows,
        ("launch__sm_count", "device__attribute_multiprocessor_count"),
    )

    return {
        "total_flops_per_request": total_per_request,
        "tensor_flops_per_request": tensor_per_request,
        "scalar_flops_per_request": scalar_per_request,
        "tensor_share_pct": tensor_share_pct,
        "kernel_launch_count_per_request": (
            float(launch_count) / normalized_repeat
            if launch_count > 0
            else float("nan")
        ),
        "kernel_time_sum_ms_per_request": (
            kernel_time_ms / normalized_repeat
            if duration_found
            else float("nan")
        ),
        "gpu_compute_capability": compute_capability,
        "gpu_sm_count": sm_count,
    }

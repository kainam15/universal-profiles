"""Massif snapshot 与 Nsys stats CSV 解析；不查找或运行外部工具。"""
from __future__ import annotations

import csv
import io
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


NSYS_REPORTS = (
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
)


def _finite_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if not value:
                return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_massif_output(report_path: str) -> Dict[str, Any]:
    """Parse native Massif snapshots and return the execution-plan fields.

    Component peaks are independent maxima.  The total peak and its timestamp
    come from the single snapshot maximizing heap + heap-extra + stack.
    """
    snapshots: List[Dict[str, float]] = []
    current: Optional[Dict[str, float]] = None
    time_unit = ""

    with open(report_path, "r", encoding="utf-8", errors="replace") as report:
        for raw_line in report:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("time_unit:"):
                time_unit = line.split(":", 1)[1].strip().lower()
                continue
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if key == "snapshot":
                if current is not None:
                    snapshots.append(current)
                current = {}
                continue
            if current is None or key not in {
                "time",
                "mem_heap_B",
                "mem_heap_extra_B",
                "mem_stacks_B",
            }:
                continue
            value = _finite_float(raw_value)
            if value is not None:
                current[key] = value

    if current is not None:
        snapshots.append(current)
    if time_unit != "ms":
        raise ValueError(
            f"massif_parse_failed:expected_time_unit_ms,got={time_unit or 'missing'}"
        )

    required = {"time", "mem_heap_B", "mem_heap_extra_B", "mem_stacks_B"}
    complete = [snapshot for snapshot in snapshots if required <= snapshot.keys()]
    if not complete:
        raise ValueError("massif_parse_failed:no_complete_snapshots")

    heap_peak = max(snapshot["mem_heap_B"] for snapshot in complete)
    extra_peak = max(snapshot["mem_heap_extra_B"] for snapshot in complete)
    stack_peak = max(snapshot["mem_stacks_B"] for snapshot in complete)
    total_snapshot = max(
        complete,
        key=lambda snapshot: (
            snapshot["mem_heap_B"]
            + snapshot["mem_heap_extra_B"]
            + snapshot["mem_stacks_B"]
        ),
    )
    total_peak = (
        total_snapshot["mem_heap_B"]
        + total_snapshot["mem_heap_extra_B"]
        + total_snapshot["mem_stacks_B"]
    )

    def _integer_if_exact(number: float) -> Any:
        return int(number) if float(number).is_integer() else number

    return {
        "cpu_heap_peak_bytes_massif": _integer_if_exact(heap_peak),
        "cpu_heap_extra_peak_bytes_massif": _integer_if_exact(extra_peak),
        "cpu_stack_peak_bytes_massif": _integer_if_exact(stack_peak),
        "cpu_heap_peak_total_bytes_massif": _integer_if_exact(total_peak),
        "cpu_heap_peak_at_ms_massif": _integer_if_exact(total_snapshot["time"]),
    }


def parse_massif_snapshots(report_path: str) -> Dict[str, Any]:
    """Compatibility alias with an explicit parser-oriented name."""
    return parse_massif_output(report_path)


def _header_base(header: str) -> str:
    value = re.sub(r"\([^)]*\)|\[[^]]*\]", "", str(header))
    value = value.split(":", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _duration_factor_to_ms(header: str) -> float:
    value = str(header).lower().replace("μ", "u").replace("µ", "u")
    tokens = re.findall(r"[a-z]+", value)
    for token in reversed(tokens):
        if token in {"ns", "nsec", "nsecs", "nanosecond", "nanoseconds"}:
            return 1e-6
        if token in {"us", "usec", "usecs", "microsecond", "microseconds"}:
            return 1e-3
        if token in {"ms", "msec", "msecs", "millisecond", "milliseconds"}:
            return 1.0
        if token in {"s", "sec", "secs", "second", "seconds"}:
            return 1000.0
    # Nsys report scripts use native nanoseconds when no formatter unit is
    # visible in the header.
    return 1e-6


def _memory_factor_to_bytes(header: str) -> float:
    value = str(header)
    candidates = re.findall(
        r"(?<![A-Za-z])(KiB|MiB|GiB|KB|MB|GB|B)(?![A-Za-z])",
        value,
        flags=re.IGNORECASE,
    )
    if not candidates:
        return 1.0
    unit = candidates[-1]
    binary = unit.lower().endswith("ib")
    prefix = unit[0].upper() if len(unit) > 1 else ""
    exponent = {"": 0, "K": 1, "M": 2, "G": 3}.get(prefix)
    if exponent is None:
        raise ValueError(f"nsys_parse_failed:unsupported_memory_unit:{unit}")
    return float((1024 if binary else 1000) ** exponent)


def _csv_table(
    csv_text: str,
    *,
    required_header_bases: Sequence[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    lines = (csv_text or "").splitlines()
    for index, line in enumerate(lines):
        try:
            candidate = next(csv.reader([line]))
        except (csv.Error, StopIteration):
            continue
        bases = {_header_base(header) for header in candidate}
        if not all(required in bases for required in required_header_bases):
            continue
        reader = csv.DictReader(io.StringIO("\n".join(lines[index:])))
        fieldnames = [str(field) for field in (reader.fieldnames or [])]
        return fieldnames, [
            {str(key): str(value or "") for key, value in row.items() if key is not None}
            for row in reader
        ]
    raise ValueError(
        "nsys_parse_failed:csv_header_missing:"
        + ",".join(required_header_bases)
    )


def _field_for_base(fieldnames: Sequence[str], bases: Sequence[str]) -> str:
    for expected in bases:
        for field in fieldnames:
            if _header_base(field) == expected:
                return field
    raise ValueError(
        "nsys_parse_failed:field_missing:" + ",".join(bases)
    )


def _sum_numeric_column(
    rows: Sequence[Mapping[str, str]],
    field: str,
    *,
    include_row: Optional[Any] = None,
) -> float:
    total = 0.0
    found = False
    eligible = False
    for row in rows:
        if include_row is not None and not include_row(row):
            continue
        eligible = True
        value = _finite_float(row.get(field))
        if value is None:
            continue
        total += value
        found = True
    if eligible and not found:
        raise ValueError(f"nsys_parse_failed:no_numeric_values:{field}")
    return total


def _nsys_memory_report_has_no_data(csv_text: str) -> bool:
    """Recognize Nsys' successful no-MemOps report without hiding real errors."""
    normalized = " ".join(str(csv_text or "").lower().split())
    return (
        "skipped" in normalized
        and "does not contain" in normalized
        and "memory data" in normalized
    )


def parse_nsys_stats_csv(
    csv_text: str,
    report_name: str,
    *,
    repeat: int = 1,
) -> Dict[str, float]:
    """Parse one Nsys stats CSV report, normalizing units and repetitions."""
    normalized_repeat = max(1, int(repeat))
    report_name = str(report_name).strip()
    if report_name not in NSYS_REPORTS:
        raise ValueError(f"unsupported nsys report: {report_name}")

    if (
        report_name.startswith("cuda_gpu_mem_")
        and _nsys_memory_report_has_no_data(csv_text)
    ):
        if report_name == "cuda_gpu_mem_size_sum":
            return {
                "total_bytes_per_request": 0.0,
                "count_per_request": 0.0,
            }
        return {
            "total_time_ms_per_request": 0.0,
            "count_per_request": 0.0,
        }

    if report_name == "cuda_api_sum":
        count_bases = ("num calls", "calls", "count")
        required = ("total time",)
    elif report_name == "cuda_gpu_kern_sum":
        count_bases = ("instances", "count", "num calls")
        required = ("total time",)
    elif report_name == "cuda_gpu_mem_time_sum":
        count_bases = ("operations", "count", "instances", "num calls")
        required = ("total time",)
    else:
        count_bases = ("operations", "count", "instances", "num calls")
        required = ("total",)

    fieldnames, rows = _csv_table(
        csv_text,
        required_header_bases=required,
    )
    count_field = _field_for_base(fieldnames, count_bases)

    include_row = None
    if report_name.startswith("cuda_gpu_mem_"):
        operation_fields = [
            field
            for field in fieldnames
            if _header_base(field) in {"operation", "name"}
        ]
        if operation_fields:
            operation_field = operation_fields[0]

            def include_row(row: Mapping[str, str]) -> bool:
                return "memcpy" in str(row.get(operation_field, "")).lower()

    count = _sum_numeric_column(
        rows,
        count_field,
        include_row=include_row,
    ) / normalized_repeat

    if report_name == "cuda_gpu_mem_size_sum":
        total_field = _field_for_base(fieldnames, ("total",))
        total = _sum_numeric_column(
            rows,
            total_field,
            include_row=include_row,
        )
        return {
            "total_bytes_per_request": (
                total * _memory_factor_to_bytes(total_field) / normalized_repeat
            ),
            "count_per_request": count,
        }

    total_field = _field_for_base(fieldnames, ("total time",))
    total = _sum_numeric_column(
        rows,
        total_field,
        include_row=include_row,
    )
    return {
        "total_time_ms_per_request": (
            total * _duration_factor_to_ms(total_field) / normalized_repeat
        ),
        "count_per_request": count,
    }


def parse_nsys_stats_reports(
    report_outputs: Mapping[str, str],
    *,
    repeat: int = 1,
) -> Dict[str, float]:
    """Combine the four Nsys summary reports into execution-plan fields."""
    missing = [report for report in NSYS_REPORTS if report not in report_outputs]
    if missing:
        raise ValueError("nsys_parse_failed:missing_reports:" + ",".join(missing))

    api = parse_nsys_stats_csv(
        report_outputs["cuda_api_sum"],
        "cuda_api_sum",
        repeat=repeat,
    )
    kernel = parse_nsys_stats_csv(
        report_outputs["cuda_gpu_kern_sum"],
        "cuda_gpu_kern_sum",
        repeat=repeat,
    )
    memcpy_time = parse_nsys_stats_csv(
        report_outputs["cuda_gpu_mem_time_sum"],
        "cuda_gpu_mem_time_sum",
        repeat=repeat,
    )
    memcpy_size = parse_nsys_stats_csv(
        report_outputs["cuda_gpu_mem_size_sum"],
        "cuda_gpu_mem_size_sum",
        repeat=repeat,
    )
    return {
        "cuda_api_time_sum_ms_per_request_nsys": api[
            "total_time_ms_per_request"
        ],
        "cuda_api_call_count_per_request_nsys": api["count_per_request"],
        "gpu_kernel_time_sum_ms_per_request_nsys": kernel[
            "total_time_ms_per_request"
        ],
        "gpu_kernel_launch_count_per_request_nsys": kernel[
            "count_per_request"
        ],
        "gpu_memcpy_time_sum_ms_per_request_nsys": memcpy_time[
            "total_time_ms_per_request"
        ],
        "gpu_memcpy_count_per_request_nsys": memcpy_time["count_per_request"],
        "gpu_memcpy_bytes_per_request_nsys": memcpy_size[
            "total_bytes_per_request"
        ],
    }

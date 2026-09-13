"""结果 CSV 的结构、测量唯一键与完整性校验，不初始化采集依赖。"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from acprof.artifacts import atomic_write
from acprof.config import CSV_FIELDS


KEY_FIELDS = ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx")
MeasurementKey = tuple[str, ...]


class ResultValidationError(ValueError):
    """输入不完整或无法唯一归属到计划中的测量。"""


def require_current_fields(fields: Iterable[str]) -> None:
    """拒绝已被明确归因字段替代的旧列，不猜测其测量来源。"""
    retired = {
        "energy_iters", "avg_power_total_w", "peak_power_total_w", "energy_total_j",
        "avg_power_eff_w", "peak_power_eff_w", "energy_eff_j", "model_mflop_per_request",
        "compute_mflops_app", "compute_mflops", "compute_profile_tool", "compute_profile_error",
    }
    unsupported = retired.intersection(fields)
    if unsupported:
        raise ResultValidationError(f"unsupported retired CSV fields: {sorted(unsupported)}; regenerate current results")


def measurement_key(row: Mapping[str, object]) -> MeasurementKey:
    values = []
    for field in KEY_FIELDS:
        value = str(row.get(field, "")).strip()
        if field == "gpu_mode":
            if value not in ("off", "on"):
                raise ResultValidationError(f"invalid gpu_mode: {value!r}")
        else:
            try:
                number = Decimal(value)
            except InvalidOperation as exc:
                raise ResultValidationError(f"invalid {field}: {value!r}") from exc
            if not number.is_finite() or number < 0:
                raise ResultValidationError(f"invalid {field}: {value!r}")
            if field in ("cpu_cores", "mem_cap_gb", "input_scale") and number <= 0:
                raise ResultValidationError(f"invalid {field}: {value!r}")
            if field in ("warmup", "repeat_idx") and number != number.to_integral_value():
                raise ResultValidationError(f"invalid {field}: {value!r}")
            if field == "warmup" and number not in (0, 1):
                raise ResultValidationError(f"invalid warmup: {value!r}")
            value = str(number.normalize())
        values.append(value)
    return tuple(values)


def expected_measurements(cpus: Sequence[int], mems: Sequence[int], gpus: Sequence[str],
                          scales: Sequence[float], warmup: int, repeat: int) -> set[MeasurementKey]:
    return {
        measurement_key(dict(zip(KEY_FIELDS, (cpu, mem, gpu, f"{float(scale):g}", is_warmup, index))))
        for cpu, mem, gpu, scale in product(cpus, mems, gpus, scales)
        for is_warmup, count in ((1, warmup), (0, repeat))
        for index in range(count)
    }


def read_result_csv(path: str | Path, *, expected: Iterable[MeasurementKey] | None = None
                    ) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        require_current_fields(fields)
        if not fields or len(fields) != len(set(fields)):
            raise ResultValidationError(f"missing or duplicate CSV columns: {path}")
        missing = set(KEY_FIELDS) - set(fields)
        if missing:
            raise ResultValidationError(f"missing identity columns {sorted(missing)}: {path}")
        rows, keys = [], set()
        for index, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ResultValidationError(f"malformed CSV row: {path}:{index}")
            try:
                key = measurement_key(row)
            except ResultValidationError as exc:
                raise ResultValidationError(f"{path}:{index}: {exc}") from exc
            if key in keys:
                raise ResultValidationError(f"duplicate measurement: {path}:{index}: {key}")
            if row.get("status", "").strip().lower() == "error" and not row.get("error", "").strip():
                raise ResultValidationError(f"status=error without an error diagnostic: {path}:{index}")
            keys.add(key)
            rows.append(row)
    if not rows:
        raise ResultValidationError(f"empty case CSV (no measurement rows): {path}")
    if expected is not None:
        planned = set(expected)
        if keys != planned:
            raise ResultValidationError(
                f"measurement plan mismatch: {path}; missing={len(planned - keys)}, "
                f"unexpected={len(keys - planned)}"
            )
    return fields, rows


def merge_result_csvs(paths: Sequence[str], destination: str, *,
                     expected: Iterable[MeasurementKey] | None = None) -> int:
    if not paths:
        raise ResultValidationError("no case files to merge")
    resolved = [Path(path).resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ResultValidationError("duplicate case CSV input path")
    if Path(destination).resolve() in resolved:
        raise ResultValidationError("final CSV cannot also be an input case")
    rows, keys = [], set()
    fields = list(CSV_FIELDS)
    for path in resolved:
        if not path.is_file():
            raise ResultValidationError(f"missing case CSV: {path}")
        source_fields, source_rows = read_result_csv(path)
        fields.extend(field for field in source_fields if field not in fields)
        for row in source_rows:
            key = measurement_key(row)
            if key in keys:
                raise ResultValidationError(f"duplicate measurement across case CSVs: {key}")
            keys.add(key)
            rows.append(row)
    if expected is not None:
        planned = set(expected)
        if keys != planned:
            raise ResultValidationError(
                f"measurement plan mismatch: missing={len(planned - keys)}, unexpected={len(keys - planned)}"
            )
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=fields, restval="nan")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(destination, write)
    return len(rows)

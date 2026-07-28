"""Backfill compute-profile fields in an existing AC-Prof result CSV."""
from __future__ import annotations

import argparse
import codecs
import csv
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from acprof.host.compute_profile_plan import (
    INPUT_SCALE_ABS_TOLERANCE,
    NCU_ERROR_FIELD,
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    NCU_TOTAL_MFLOP_FIELD,
    TORCH_ERROR_FIELD,
    TORCH_LOGICAL_MFLOP_FIELD,
    compute_mflops,
    find_compute_profile_entry,
    load_compute_profile_plan,
)


COMPUTE_PROFILE_FIELDS = (
    TORCH_LOGICAL_MFLOP_FIELD,
    "model_logical_mflops_app_torch_profiler_eager",
    "model_logical_mflops_packet_torch_profiler_eager",
    TORCH_ERROR_FIELD,
    NCU_TOTAL_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    "gpu_executed_mflops_app_ncu",
    "gpu_executed_mflops_packet_ncu",
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_ERROR_FIELD,
)
REQUIRED_INPUT_FIELDS = ("gpu_mode", "input_scale", "latency_app_s")
DIAGNOSTIC_FIELDS = (
    TORCH_ERROR_FIELD,
    NCU_ERROR_FIELD,
)
OBSOLETE_COMPUTE_PROFILE_FIELDS = (
    "compute_profile_tool",
    "model_mflop_per_request",
    "compute_mflops_app",
    "compute_mflops",
    "compute_profile_error",
    "gpu_profile_report_ncu",
)

NCU_SUMMARY_FIELD_ALIASES = {
    NCU_TOTAL_MFLOP_FIELD: (
        NCU_TOTAL_MFLOP_FIELD,
        "gpu_hw_mflop_per_request_ncu",
    ),
    NCU_TENSOR_MFLOP_FIELD: (
        NCU_TENSOR_MFLOP_FIELD,
        "gpu_hw_tensor_mflop_per_request_ncu",
    ),
    NCU_SCALAR_MFLOP_FIELD: (
        NCU_SCALAR_MFLOP_FIELD,
        "gpu_hw_scalar_mflop_per_request_ncu",
    ),
    NCU_TENSOR_SHARE_FIELD: (
        NCU_TENSOR_SHARE_FIELD,
        "gpu_hw_tensor_share_pct_ncu",
    ),
    NCU_KERNEL_COUNT_FIELD: (
        NCU_KERNEL_COUNT_FIELD,
        "ncu_kernel_count",
    ),
    NCU_KERNEL_TIME_FIELD: (
        NCU_KERNEL_TIME_FIELD,
        "ncu_kernel_time_sum_ms",
    ),
    NCU_ERROR_FIELD: (
        NCU_ERROR_FIELD,
        "error",
    ),
}


@dataclass(frozen=True)
class BackfillSummary:
    output_csv: str
    row_count: int
    diagnostic_count: int


def _to_float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt_float(value: Any) -> str:
    number = _to_float_or_nan(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6f}"


def _csv_encoding(path: str) -> str:
    with open(path, "rb") as f:
        has_utf8_bom = f.read(len(codecs.BOM_UTF8)) == codecs.BOM_UTF8
    return "utf-8-sig" if has_utf8_bom else "utf-8"


def _read_result_csv(path: str) -> Tuple[List[str], List[Dict[str, str]], str]:
    encoding = _csv_encoding(path)
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("input CSV is empty or has no header")
        rows = list(reader)

    missing_fields = [
        field for field in REQUIRED_INPUT_FIELDS if field not in fieldnames
    ]
    if missing_fields:
        raise ValueError(
            "input CSV is missing required fields: " + ", ".join(missing_fields)
        )
    if any(None in row for row in rows):
        raise ValueError("input CSV has rows with more values than header fields")

    for field in OBSOLETE_COMPUTE_PROFILE_FIELDS:
        if field in fieldnames:
            fieldnames.remove(field)
        for row in rows:
            row.pop(field, None)

    for field in COMPUTE_PROFILE_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames, rows, encoding


def _read_ncu_summary(path: str) -> List[Tuple[float, Dict[str, Any]]]:
    """Read the historical per-scale NCU sidecar without changing its paths."""
    with open(path, "r", encoding=_csv_encoding(path), newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("NCU summary CSV is empty or has no header")
        if "input_scale" not in fieldnames:
            raise ValueError("NCU summary CSV is missing required field: input_scale")
        if not any(
            alias in fieldnames
            for alias in NCU_SUMMARY_FIELD_ALIASES[NCU_TOTAL_MFLOP_FIELD]
        ):
            raise ValueError(
                "NCU summary CSV is missing total FLOP field "
                f"({', '.join(NCU_SUMMARY_FIELD_ALIASES[NCU_TOTAL_MFLOP_FIELD])})"
            )

        entries: List[Tuple[float, Dict[str, Any]]] = []
        for row_index, row in enumerate(reader, start=2):
            scale = _to_float_or_nan(row.get("input_scale"))
            if not math.isfinite(scale):
                raise ValueError(
                    f"NCU summary CSV has invalid input_scale at row {row_index}"
                )
            if any(
                math.isclose(
                    scale,
                    existing_scale,
                    rel_tol=0.0,
                    abs_tol=INPUT_SCALE_ABS_TOLERANCE,
                )
                for existing_scale, _entry in entries
            ):
                raise ValueError(
                    f"NCU summary CSV has duplicate input_scale: {scale:g}"
                )

            mapped: Dict[str, Any] = {}
            for destination, aliases in NCU_SUMMARY_FIELD_ALIASES.items():
                value: Any = None
                for alias in aliases:
                    if alias in row and row.get(alias) not in (None, ""):
                        value = row.get(alias)
                        break
                mapped[destination] = value
            entries.append((scale, mapped))
    return entries


def _find_ncu_summary_entry(
    entries: Sequence[Tuple[float, Dict[str, Any]]],
    input_scale: float,
) -> Dict[str, Any] | None:
    for entry_scale, entry in entries:
        if math.isclose(
            entry_scale,
            input_scale,
            rel_tol=0.0,
            abs_tol=INPUT_SCALE_ABS_TOLERANCE,
        ):
            return entry
    return None


def _apply_ncu_summary(
    profile: Dict[str, Any],
    *,
    entries: Sequence[Tuple[float, Dict[str, Any]]],
    input_scale: float,
) -> None:
    entry = _find_ncu_summary_entry(entries, input_scale)
    if entry is None:
        profile.update(
            {
                NCU_TOTAL_MFLOP_FIELD: float("nan"),
                NCU_TENSOR_MFLOP_FIELD: float("nan"),
                NCU_SCALAR_MFLOP_FIELD: float("nan"),
                NCU_TENSOR_SHARE_FIELD: float("nan"),
                NCU_KERNEL_COUNT_FIELD: float("nan"),
                NCU_KERNEL_TIME_FIELD: float("nan"),
                NCU_ERROR_FIELD: f"ncu_summary_missing_scale:{input_scale:g}",
            }
        )
        return

    profile.update(
        {
            NCU_TOTAL_MFLOP_FIELD: _to_float_or_nan(
                entry.get(NCU_TOTAL_MFLOP_FIELD)
            ),
            NCU_TENSOR_MFLOP_FIELD: _to_float_or_nan(
                entry.get(NCU_TENSOR_MFLOP_FIELD)
            ),
            NCU_SCALAR_MFLOP_FIELD: _to_float_or_nan(
                entry.get(NCU_SCALAR_MFLOP_FIELD)
            ),
            NCU_TENSOR_SHARE_FIELD: _to_float_or_nan(
                entry.get(NCU_TENSOR_SHARE_FIELD)
            ),
            NCU_KERNEL_COUNT_FIELD: _to_float_or_nan(
                entry.get(NCU_KERNEL_COUNT_FIELD)
            ),
            NCU_KERNEL_TIME_FIELD: _to_float_or_nan(
                entry.get(NCU_KERNEL_TIME_FIELD)
            ),
            NCU_ERROR_FIELD: str(entry.get(NCU_ERROR_FIELD) or ""),
        }
    )


def _backfill_row(
    row: Dict[str, str],
    plan: Dict[str, Any],
    ncu_summary: Sequence[Tuple[float, Dict[str, Any]]] | None = None,
) -> None:
    input_scale = _to_float_or_nan(row.get("input_scale"))
    gpu_mode = str(row.get("gpu_mode", "") or "").strip().lower()
    profile = find_compute_profile_entry(plan, gpu_mode, input_scale)

    if gpu_mode == "on" and ncu_summary is not None:
        _apply_ncu_summary(
            profile,
            entries=ncu_summary,
            input_scale=input_scale,
        )

    logical_mflop = _to_float_or_nan(profile.get(TORCH_LOGICAL_MFLOP_FIELD))
    logical_mflops_app = compute_mflops(
        logical_mflop,
        row.get("latency_app_s"),
    )
    logical_mflops_packet = compute_mflops(
        logical_mflop,
        row.get("latency_s"),
    )
    effective_logical_mflops_packet = (
        logical_mflops_packet
        if math.isfinite(logical_mflops_packet)
        else logical_mflops_app
    )

    ncu_total_mflop = _to_float_or_nan(profile.get(NCU_TOTAL_MFLOP_FIELD))
    ncu_mflops_app = compute_mflops(
        ncu_total_mflop,
        row.get("latency_app_s"),
    )
    ncu_mflops_packet = compute_mflops(
        ncu_total_mflop,
        row.get("latency_s"),
    )
    effective_ncu_mflops_packet = (
        ncu_mflops_packet
        if math.isfinite(ncu_mflops_packet)
        else ncu_mflops_app
    )

    row.update(
        {
            TORCH_LOGICAL_MFLOP_FIELD: _fmt_float(logical_mflop),
            "model_logical_mflops_app_torch_profiler_eager": _fmt_float(
                logical_mflops_app
            ),
            "model_logical_mflops_packet_torch_profiler_eager": _fmt_float(
                effective_logical_mflops_packet
            ),
            TORCH_ERROR_FIELD: str(profile.get(TORCH_ERROR_FIELD) or ""),
            NCU_TOTAL_MFLOP_FIELD: _fmt_float(ncu_total_mflop),
            NCU_TENSOR_MFLOP_FIELD: _fmt_float(
                profile.get(NCU_TENSOR_MFLOP_FIELD)
            ),
            NCU_SCALAR_MFLOP_FIELD: _fmt_float(
                profile.get(NCU_SCALAR_MFLOP_FIELD)
            ),
            NCU_TENSOR_SHARE_FIELD: _fmt_float(
                profile.get(NCU_TENSOR_SHARE_FIELD)
            ),
            "gpu_executed_mflops_app_ncu": _fmt_float(ncu_mflops_app),
            "gpu_executed_mflops_packet_ncu": _fmt_float(
                effective_ncu_mflops_packet
            ),
            NCU_KERNEL_COUNT_FIELD: _fmt_float(
                profile.get(NCU_KERNEL_COUNT_FIELD)
            ),
            NCU_KERNEL_TIME_FIELD: _fmt_float(
                profile.get(NCU_KERNEL_TIME_FIELD)
            ),
            NCU_ERROR_FIELD: str(profile.get(NCU_ERROR_FIELD) or ""),
        }
    )


def _publish_csv_atomically(
    *,
    input_csv: str,
    output_csv: str,
    fieldnames: Sequence[str],
    rows: Sequence[Dict[str, str]],
    encoding: str,
    overwrite: bool,
) -> None:
    output_path = os.path.abspath(output_csv)
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"output directory does not exist: {output_dir}")
    if os.path.lexists(output_path) and not overwrite:
        raise FileExistsError(
            f"output already exists (pass --overwrite to replace it): {output_csv}"
        )

    fd, temporary_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_MINIMAL,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

        input_mode = stat.S_IMODE(os.stat(input_csv).st_mode)
        os.chmod(temporary_path, input_mode)
        if overwrite:
            os.replace(temporary_path, output_path)
            temporary_path = ""
        else:
            # A hard-link publish is atomic and fails instead of clobbering a
            # destination created after the initial existence check.
            os.link(temporary_path, output_path)
            os.unlink(temporary_path)
            temporary_path = ""
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def backfill_compute_profile_csv(
    input_csv: str,
    compute_profile_plan: str,
    output_csv: str,
    *,
    ncu_summary: str | None = None,
    overwrite: bool = False,
) -> BackfillSummary:
    """Backfill compute fields without changing input row or non-compute values."""
    if not output_csv:
        raise ValueError("an explicit output CSV path is required")

    fieldnames, rows, encoding = _read_result_csv(input_csv)
    plan = load_compute_profile_plan(compute_profile_plan)
    ncu_summary_entries = (
        _read_ncu_summary(ncu_summary) if ncu_summary else None
    )
    for row in rows:
        _backfill_row(row, plan, ncu_summary_entries)

    _publish_csv_atomically(
        input_csv=input_csv,
        output_csv=output_csv,
        fieldnames=fieldnames,
        rows=rows,
        encoding=encoding,
        overwrite=overwrite,
    )
    return BackfillSummary(
        output_csv=output_csv,
        row_count=len(rows),
        diagnostic_count=sum(
            any(bool(str(row.get(field, "") or "")) for field in DIAGNOSTIC_FIELDS)
            for row in rows
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill result_all.csv compute-profile fields from "
            "compute_profile_plan.json."
        )
    )
    parser.add_argument("input_csv", help="Existing result_all.csv")
    parser.add_argument("compute_profile_plan", help="compute_profile_plan.json")
    parser.add_argument(
        "--ncu-summary",
        help=(
            "Optional per-scale NCU summary CSV. Values are applied only to "
            "gpu_mode=on rows using numeric input-scale matching."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Explicit output CSV path; existing files are not replaced by default",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing explicit output path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = backfill_compute_profile_csv(
            args.input_csv,
            args.compute_profile_plan,
            args.output,
            ncu_summary=args.ncu_summary,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        f"Backfilled {summary.row_count} rows to {summary.output_csv} "
        f"({summary.diagnostic_count} rows with compute-profile diagnostics)."
    )


if __name__ == "__main__":
    main()

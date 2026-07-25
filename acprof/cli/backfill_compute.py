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
    compute_mflops,
    find_compute_profile_entry,
    load_compute_profile_plan,
)


COMPUTE_PROFILE_FIELDS = (
    "compute_profile_tool",
    "model_mflop_per_request",
    "compute_mflops_app",
    "compute_mflops",
    "compute_profile_error",
)
REQUIRED_INPUT_FIELDS = ("gpu_mode", "input_scale", "latency_app_s")


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

    for field in COMPUTE_PROFILE_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames, rows, encoding


def _backfill_row(row: Dict[str, str], plan: Dict[str, Any]) -> None:
    input_scale = _to_float_or_nan(row.get("input_scale"))
    gpu_mode = str(row.get("gpu_mode", "") or "").strip().lower()
    profile = find_compute_profile_entry(plan, gpu_mode, input_scale)

    model_mflop_per_request = _to_float_or_nan(
        profile.get("model_mflop_per_request")
    )
    compute_mflops_app = compute_mflops(
        model_mflop_per_request,
        row.get("latency_app_s"),
    )
    compute_mflops_packet = compute_mflops(
        model_mflop_per_request,
        row.get("latency_s"),
    )
    effective_compute_mflops = (
        compute_mflops_packet
        if math.isfinite(compute_mflops_packet)
        else compute_mflops_app
    )

    row.update(
        {
            "compute_profile_tool": str(profile.get("tool") or "nan"),
            "model_mflop_per_request": _fmt_float(model_mflop_per_request),
            "compute_mflops_app": _fmt_float(compute_mflops_app),
            "compute_mflops": _fmt_float(effective_compute_mflops),
            "compute_profile_error": str(profile.get("error") or ""),
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
    overwrite: bool = False,
) -> BackfillSummary:
    """Backfill compute fields without changing input row or non-compute values."""
    if not output_csv:
        raise ValueError("an explicit output CSV path is required")

    fieldnames, rows, encoding = _read_result_csv(input_csv)
    plan = load_compute_profile_plan(compute_profile_plan)
    for row in rows:
        _backfill_row(row, plan)

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
            bool(row.get("compute_profile_error", "")) for row in rows
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

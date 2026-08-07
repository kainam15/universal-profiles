"""Profile-only Torch, NCU, Nsight Systems, and Massif CSV backfill.

The normal benchmark and the high-overhead profiler probes are intentionally
separate.  This command reuses an existing result directory, collects only
missing profiler data, and updates ``result_all.csv``/``static_meta.json`` in
place after creating a recoverable backup.
"""
from __future__ import annotations

import argparse
import codecs
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from acprof.host.compute_profile_plan import (
    INPUT_SCALE_ABS_TOLERANCE,
    NCU_ERROR_FIELD,
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_PROFILE_KEY,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    NCU_TOTAL_MFLOP_FIELD,
    TORCH_ERROR_FIELD,
    TORCH_LOGICAL_MFLOP_FIELD,
    TORCH_PROFILE_KEY,
    compute_mflops,
    find_compute_profile_entry,
)
from acprof.host.detect import TaskInfo
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.execution_profile_plan import (
    MASSIF_ERROR_FIELD,
    MASSIF_HEAP_EXTRA_PEAK_FIELD,
    MASSIF_HEAP_PEAK_FIELD,
    MASSIF_HEAP_PEAK_TOTAL_FIELD,
    MASSIF_METRIC_FIELDS,
    MASSIF_PEAK_AT_MS_FIELD,
    MASSIF_STACK_PEAK_FIELD,
    NSYS_CUDA_API_CALL_COUNT_FIELD,
    NSYS_CUDA_API_TIME_FIELD,
    NSYS_ERROR_FIELD,
    NSYS_GPU_KERNEL_LAUNCH_COUNT_FIELD,
    NSYS_GPU_KERNEL_TIME_FIELD,
    NSYS_GPU_MEMCPY_BYTES_FIELD,
    NSYS_GPU_MEMCPY_COUNT_FIELD,
    NSYS_GPU_MEMCPY_TIME_FIELD,
    NSYS_HOST_WALL_TIME_FIELD,
    NSYS_METRIC_FIELDS,
    find_execution_profile_entry,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_CSV_NAME = "result_all.csv"
STATIC_META_NAME = "static_meta.json"
INPUT_SCALE_PLAN_NAME = "input_scale_plan.json"
POSTHOC_DIRNAME = "posthoc_profiles"
BACKUP_DIRNAME = "posthoc_backups"
LOCK_FILENAME = ".posthoc.lock"
SUPPORTED_TOOLS = ("torch", "ncu", "nsys", "massif")

TORCH_FIELDS = (TORCH_LOGICAL_MFLOP_FIELD, TORCH_ERROR_FIELD)
NCU_DERIVED_APP_FIELD = "gpu_executed_mflops_app_ncu"
NCU_DERIVED_PACKET_FIELD = "gpu_executed_mflops_packet_ncu"
NCU_FIELDS = (
    NCU_TOTAL_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    NCU_DERIVED_APP_FIELD,
    NCU_DERIVED_PACKET_FIELD,
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_ERROR_FIELD,
)
MASSIF_FIELDS = (*MASSIF_METRIC_FIELDS, MASSIF_ERROR_FIELD)
NSYS_FIELDS = (*NSYS_METRIC_FIELDS, NSYS_ERROR_FIELD)
TOOL_FIELDS = {
    "torch": TORCH_FIELDS,
    "ncu": NCU_FIELDS,
    "massif": MASSIF_FIELDS,
    "nsys": NSYS_FIELDS,
}
TOOL_METRIC_FIELDS = {
    "torch": (TORCH_LOGICAL_MFLOP_FIELD,),
    "ncu": NCU_FIELDS[:-1],
    "massif": MASSIF_METRIC_FIELDS,
    "nsys": NSYS_METRIC_FIELDS,
}
TOOL_ERROR_FIELD = {
    "torch": TORCH_ERROR_FIELD,
    "ncu": NCU_ERROR_FIELD,
    "massif": MASSIF_ERROR_FIELD,
    "nsys": NSYS_ERROR_FIELD,
}
TOOL_GPU_MODES = {
    "torch": ("off", "on"),
    "ncu": ("on",),
    "nsys": ("on",),
    "massif": ("off",),
}

COMPUTE_PLAN_METRIC_FIELDS = {
    "torch": (TORCH_LOGICAL_MFLOP_FIELD,),
    "ncu": (
        NCU_TOTAL_MFLOP_FIELD,
        NCU_TENSOR_MFLOP_FIELD,
        NCU_SCALAR_MFLOP_FIELD,
        NCU_TENSOR_SHARE_FIELD,
        NCU_KERNEL_COUNT_FIELD,
        NCU_KERNEL_TIME_FIELD,
    ),
}


class PosthocError(RuntimeError):
    """Raised for a user-actionable post-hoc profiling failure."""


@dataclass
class ResultContext:
    result_dir: Path
    result_csv: Path
    static_meta_path: Path
    input_scale_plan_path: Path
    fieldnames: List[str]
    rows: List[Dict[str, str]]
    csv_encoding: str
    static_meta: Dict[str, Any]
    input_scale_plan: Dict[str, Any]
    task_info: TaskInfo
    image_tag: str
    resource_cases: List[Tuple[int, int, str]]
    scales_by_mode: Dict[str, List[float]]

    def cases_for_mode(self, gpu_mode: str) -> List[Tuple[int, int, str]]:
        return [case for case in self.resource_cases if case[2] == gpu_mode]


@dataclass(frozen=True)
class PosthocSummary:
    result_csv: str
    static_meta: str
    backup_dir: Optional[str]
    collected_tools: Tuple[str, ...]
    reused_tools: Tuple[str, ...]
    skipped_tools: Tuple[str, ...]
    updated_rows_by_tool: Dict[str, int]


def _finite_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return float("nan")
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _fmt_float(value: Any) -> str:
    number = _finite_float(value)
    return f"{number:.6f}" if math.isfinite(number) else "nan"


def _integer(value: Any, field: str) -> int:
    number = _finite_float(value)
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise PosthocError(f"invalid {field} in result CSV: {value!r}")
    return int(number)


def _csv_encoding(path: Path) -> str:
    with path.open("rb") as f:
        return (
            "utf-8-sig"
            if f.read(len(codecs.BOM_UTF8)) == codecs.BOM_UTF8
            else "utf-8"
        )


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise PosthocError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PosthocError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PosthocError(f"invalid {label} (expected JSON object): {path}")
    return payload


def _load_result_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]], str]:
    if not path.is_file():
        raise PosthocError(
            f"missing completed {RESULT_CSV_NAME}: {path}; wait for run.py to finish"
        )
    encoding = _csv_encoding(path)
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise PosthocError(f"empty result CSV: {path}")
    required = {"cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise PosthocError(
            "result CSV is missing required columns: " + ", ".join(missing)
        )
    if not rows:
        raise PosthocError(f"result CSV has no rows: {path}")
    if any(None in row for row in rows):
        raise PosthocError(f"result CSV has rows wider than its header: {path}")
    return fieldnames, rows, encoding


def _unique_scales(rows: Iterable[Mapping[str, Any]]) -> List[float]:
    scales: List[float] = []
    for row in rows:
        scale = _finite_float(row.get("input_scale"))
        if not math.isfinite(scale):
            raise PosthocError(
                f"invalid input_scale in result CSV: {row.get('input_scale')!r}"
            )
        if not any(
            math.isclose(
                scale,
                existing,
                rel_tol=0.0,
                abs_tol=INPUT_SCALE_ABS_TOLERANCE,
            )
            for existing in scales
        ):
            scales.append(scale)
    return sorted(scales)


def _input_plan_scales(plan: Mapping[str, Any]) -> List[float]:
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PosthocError("input_scale_plan.json has no entries")
    scales: List[float] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("payload"), dict):
            raise PosthocError(
                f"invalid input scale plan entry at index {index}: payload missing"
            )
        scale = _finite_float(entry.get("input_scale"))
        if not math.isfinite(scale):
            raise PosthocError(
                f"invalid input scale plan entry at index {index}: input_scale missing"
            )
        scales.append(scale)
    return scales


def _scale_is_present(scale: float, candidates: Iterable[float]) -> bool:
    return any(
        math.isclose(
            scale,
            candidate,
            rel_tol=0.0,
            abs_tol=INPUT_SCALE_ABS_TOLERANCE,
        )
        for candidate in candidates
    )


def load_result_context(result_dir: str | os.PathLike[str]) -> ResultContext:
    directory = Path(result_dir).expanduser().resolve()
    if not directory.is_dir():
        raise PosthocError(f"result directory does not exist: {directory}")

    result_csv = directory / RESULT_CSV_NAME
    static_meta_path = directory / STATIC_META_NAME
    input_scale_plan_path = directory / INPUT_SCALE_PLAN_NAME
    fieldnames, rows, encoding = _load_result_csv(result_csv)
    static_meta = _load_json_object(static_meta_path, STATIC_META_NAME)
    input_plan = _load_json_object(input_scale_plan_path, INPUT_SCALE_PLAN_NAME)

    model_id = str(
        static_meta.get("model_name") or input_plan.get("model_id") or ""
    ).strip()
    if not model_id:
        raise PosthocError("static_meta.json has no model_name")
    plan_model_id = str(input_plan.get("model_id") or "").strip()
    if plan_model_id and plan_model_id != model_id:
        raise PosthocError(
            f"model mismatch: static_meta={model_id}, input_plan={plan_model_id}"
        )

    expected_plan_hash = str(static_meta.get("input_scale_plan_sha256") or "").strip()
    if expected_plan_hash:
        actual_hash = hashlib.sha256(input_scale_plan_path.read_bytes()).hexdigest()
        if actual_hash != expected_plan_hash:
            raise PosthocError(
                "input_scale_plan.json hash does not match static_meta.json; "
                "refusing to profile a different payload"
            )

    resource_cases = sorted(
        {
            (
                _integer(row.get("cpu_cores"), "cpu_cores"),
                _integer(row.get("mem_cap_gb"), "mem_cap_gb"),
                str(row.get("gpu_mode") or "").strip().lower(),
            )
            for row in rows
        }
    )
    invalid_modes = sorted({case[2] for case in resource_cases} - {"off", "on"})
    if invalid_modes:
        raise PosthocError("invalid gpu_mode values: " + ", ".join(invalid_modes))

    scales_by_mode = {
        mode: _unique_scales(
            row
            for row in rows
            if str(row.get("gpu_mode") or "").strip().lower() == mode
        )
        for mode in ("off", "on")
        if any(case[2] == mode for case in resource_cases)
    }
    planned_scales = _input_plan_scales(input_plan)
    missing_scales = [
        scale
        for scales in scales_by_mode.values()
        for scale in scales
        if not _scale_is_present(scale, planned_scales)
    ]
    if missing_scales:
        labels = ", ".join(f"{scale:g}" for scale in sorted(set(missing_scales)))
        raise PosthocError(
            f"result CSV scales are missing from input_scale_plan.json: {labels}"
        )

    image_tag = str(static_meta.get("image_tag") or "").strip()
    if not image_tag:
        raise PosthocError("static_meta.json has no image_tag")

    task_info = TaskInfo(
        model_id=model_id,
        pipeline_tag=str(static_meta.get("pipeline_tag") or "").strip(),
        task_family=str(static_meta.get("task_family") or "").strip(),
        runtime_backend=str(static_meta.get("runtime_backend") or "").strip(),
        library_name="",
        model_revision=str(static_meta.get("model_revision") or "main").strip(),
        detection_method="posthoc_static_meta",
    )
    if not all(
        (task_info.pipeline_tag, task_info.task_family, task_info.runtime_backend)
    ):
        raise PosthocError(
            "static_meta.json is missing pipeline_tag/task_family/runtime_backend"
        )

    return ResultContext(
        result_dir=directory,
        result_csv=result_csv,
        static_meta_path=static_meta_path,
        input_scale_plan_path=input_scale_plan_path,
        fieldnames=fieldnames,
        rows=rows,
        csv_encoding=encoding,
        static_meta=static_meta,
        input_scale_plan=input_plan,
        task_info=task_info,
        image_tag=image_tag,
        resource_cases=resource_cases,
        scales_by_mode=scales_by_mode,
    )


def parse_tools(value: str | Iterable[str]) -> Tuple[str, ...]:
    raw_values = [value] if isinstance(value, str) else list(value)
    tools: List[str] = []
    for raw in raw_values:
        for token in str(raw).replace(";", ",").split(","):
            tool = token.strip().lower()
            if not tool:
                continue
            if tool not in SUPPORTED_TOOLS:
                raise PosthocError(
                    f"unsupported tool {tool!r}; choose from {', '.join(SUPPORTED_TOOLS)}"
                )
            if tool not in tools:
                tools.append(tool)
    if not tools:
        raise PosthocError("at least one profiler tool is required")
    return tuple(tool for tool in SUPPORTED_TOOLS if tool in tools)


def applicable_tools(
    context: ResultContext,
    tools: Iterable[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    modes = {case[2] for case in context.resource_cases}
    applicable: List[str] = []
    skipped: List[str] = []
    for tool in tools:
        if modes.intersection(TOOL_GPU_MODES[tool]):
            applicable.append(tool)
        else:
            skipped.append(tool)
    return tuple(applicable), tuple(skipped)


def _row_tool_complete(row: Mapping[str, Any], tool: str) -> bool:
    error = str(row.get(TOOL_ERROR_FIELD[tool]) or "").strip()
    return not error and all(
        math.isfinite(_finite_float(row.get(field)))
        for field in TOOL_METRIC_FIELDS[tool]
    )


def csv_tool_complete(context: ResultContext, tool: str) -> bool:
    modes = set(TOOL_GPU_MODES[tool])
    applicable_rows = [
        row
        for row in context.rows
        if str(row.get("gpu_mode") or "").strip().lower() in modes
    ]
    return bool(applicable_rows) and all(
        _row_tool_complete(row, tool) for row in applicable_rows
    )


def compute_plan_covers_ncu(
    plan: Mapping[str, Any],
    scales: Iterable[float],
) -> bool:
    if not isinstance(plan, Mapping):
        return False
    for scale in scales:
        profile = find_compute_profile_entry(dict(plan), "on", scale)
        required_fields = (
            NCU_TOTAL_MFLOP_FIELD,
            NCU_TENSOR_MFLOP_FIELD,
            NCU_SCALAR_MFLOP_FIELD,
            NCU_TENSOR_SHARE_FIELD,
            NCU_KERNEL_COUNT_FIELD,
            NCU_KERNEL_TIME_FIELD,
        )
        if not all(
            math.isfinite(_finite_float(profile.get(field)))
            for field in required_fields
        ):
            return False
        if str(profile.get(NCU_ERROR_FIELD) or "").strip():
            return False
    return True


def compute_plan_covers_tool(
    plan: Mapping[str, Any],
    context: ResultContext,
    tool: str,
) -> bool:
    if tool not in {"torch", "ncu"} or not isinstance(plan, Mapping):
        return False
    found_mode = False
    for mode in TOOL_GPU_MODES[tool]:
        scales = context.scales_by_mode.get(mode, [])
        if not scales:
            continue
        found_mode = True
        for scale in scales:
            profile = find_compute_profile_entry(dict(plan), mode, scale)
            if not all(
                math.isfinite(_finite_float(profile.get(field)))
                for field in COMPUTE_PLAN_METRIC_FIELDS[tool]
            ):
                return False
            if str(profile.get(TOOL_ERROR_FIELD[tool]) or "").strip():
                return False
    return found_mode


def execution_plan_covers_tool(
    plan: Mapping[str, Any],
    context: ResultContext,
    tool: str,
) -> bool:
    if tool not in {"massif", "nsys"} or not isinstance(plan, Mapping):
        return False
    mode = TOOL_GPU_MODES[tool][0]
    cases = context.cases_for_mode(mode)
    scales = context.scales_by_mode.get(mode, [])
    if not cases or not scales:
        return False
    for cpu, mem, gpu_mode in cases:
        for scale in scales:
            profile = find_execution_profile_entry(
                dict(plan), cpu, mem, gpu_mode, scale
            )
            if not all(
                math.isfinite(_finite_float(profile.get(field)))
                for field in TOOL_METRIC_FIELDS[tool]
            ):
                return False
            if str(profile.get(TOOL_ERROR_FIELD[tool]) or "").strip():
                return False
    return True


def _packet_mflops(total_mflop: Any, row: Mapping[str, Any]) -> float:
    packet = compute_mflops(total_mflop, row.get("latency_s"))
    return (
        packet
        if math.isfinite(packet)
        else compute_mflops(total_mflop, row.get("latency_app_s"))
    )


def _backfill_torch_row(
    row: Dict[str, str],
    plan: Mapping[str, Any],
) -> None:
    mode = str(row.get("gpu_mode") or "").strip().lower()
    scale = _finite_float(row.get("input_scale"))
    profile = find_compute_profile_entry(dict(plan), mode, scale)
    row[TORCH_LOGICAL_MFLOP_FIELD] = _fmt_float(
        profile.get(TORCH_LOGICAL_MFLOP_FIELD)
    )
    row[TORCH_ERROR_FIELD] = str(profile.get(TORCH_ERROR_FIELD) or "")


def _backfill_ncu_row(
    row: Dict[str, str],
    plan: Mapping[str, Any],
) -> None:
    scale = _finite_float(row.get("input_scale"))
    profile = find_compute_profile_entry(dict(plan), "on", scale)
    total = profile.get(NCU_TOTAL_MFLOP_FIELD)
    row.update(
        {
            NCU_TOTAL_MFLOP_FIELD: _fmt_float(total),
            NCU_TENSOR_MFLOP_FIELD: _fmt_float(
                profile.get(NCU_TENSOR_MFLOP_FIELD)
            ),
            NCU_SCALAR_MFLOP_FIELD: _fmt_float(
                profile.get(NCU_SCALAR_MFLOP_FIELD)
            ),
            NCU_TENSOR_SHARE_FIELD: _fmt_float(
                profile.get(NCU_TENSOR_SHARE_FIELD)
            ),
            NCU_DERIVED_APP_FIELD: _fmt_float(
                compute_mflops(total, row.get("latency_app_s"))
            ),
            NCU_DERIVED_PACKET_FIELD: _fmt_float(
                _packet_mflops(total, row)
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


def _backfill_execution_row(
    row: Dict[str, str],
    plan: Mapping[str, Any],
    tool: str,
) -> None:
    profile = find_execution_profile_entry(
        dict(plan),
        _integer(row.get("cpu_cores"), "cpu_cores"),
        _integer(row.get("mem_cap_gb"), "mem_cap_gb"),
        str(row.get("gpu_mode") or "").strip().lower(),
        _finite_float(row.get("input_scale")),
    )
    fields = MASSIF_METRIC_FIELDS if tool == "massif" else NSYS_METRIC_FIELDS
    for field in fields:
        row[field] = _fmt_float(profile.get(field))
    error_field = TOOL_ERROR_FIELD[tool]
    row[error_field] = str(profile.get(error_field) or "")


def backfill_rows(
    context: ResultContext,
    *,
    tools: Iterable[str],
    compute_plan: Optional[Mapping[str, Any]] = None,
    execution_plan: Optional[Mapping[str, Any]] = None,
    force: bool = False,
) -> Tuple[List[str], List[Dict[str, str]], Dict[str, int]]:
    selected = tuple(tools)
    fieldnames = list(context.fieldnames)
    for tool in selected:
        for field in TOOL_FIELDS[tool]:
            if field not in fieldnames:
                fieldnames.append(field)

    rows = [dict(row) for row in context.rows]
    updated = {tool: 0 for tool in selected}
    for row in rows:
        mode = str(row.get("gpu_mode") or "").strip().lower()
        for tool in selected:
            if mode not in TOOL_GPU_MODES[tool]:
                continue
            if not force and _row_tool_complete(row, tool):
                continue
            if tool in {"torch", "ncu"}:
                if compute_plan is None:
                    raise PosthocError(
                        f"compute plan is required for {tool} backfill"
                    )
                if tool == "torch":
                    _backfill_torch_row(row, compute_plan)
                else:
                    _backfill_ncu_row(row, compute_plan)
            else:
                if execution_plan is None:
                    raise PosthocError(
                        f"execution plan is required for {tool} backfill"
                    )
                _backfill_execution_row(row, execution_plan, tool)
            updated[tool] += 1
    return fieldnames, rows, updated


def _read_plan(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _plan_matches_context(
    plan: Mapping[str, Any],
    context: ResultContext,
) -> bool:
    model_id = str(plan.get("model_id") or "").strip()
    if model_id and model_id != context.task_info.model_id:
        return False
    revision = str(plan.get("model_revision") or "").strip()
    expected_revision = str(context.task_info.model_revision or "main").strip()
    if revision and revision != expected_revision:
        return False
    for key, expected in (
        ("task_family", context.task_info.task_family),
        ("pipeline_tag", context.task_info.pipeline_tag),
        ("runtime_backend", context.task_info.runtime_backend),
    ):
        value = str(plan.get(key) or "").strip()
        if value and value != expected:
            return False
    return True


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _compute_plan_candidates(
    context: ResultContext,
    workspace: Path,
    tool: str,
) -> List[Path]:
    return [
        context.result_dir / "compute_profile_plan.json",
        workspace / "compute_profile_plan.json",
        workspace / tool / "compute_profile_plan.json",
    ]


def _execution_plan_candidates(
    context: ResultContext,
    workspace: Path,
    tool: str,
) -> List[Path]:
    return [
        context.result_dir / "execution_profile_plan.json",
        workspace / "execution_profile_plan.json",
        workspace / tool / "execution_profile_plan.for_backfill.json",
        workspace / tool / "execution_profile_plan.json",
    ]


def _find_reusable_compute_plan(
    context: ResultContext,
    workspace: Path,
    tool: str,
) -> Optional[Dict[str, Any]]:
    for path in _compute_plan_candidates(context, workspace, tool):
        plan = _read_plan(path)
        if (
            plan is not None
            and _plan_matches_context(plan, context)
            and compute_plan_covers_tool(plan, context, tool)
        ):
            print(f"[profile][{tool}] Reusing complete plan: {path}")
            return plan
    return None


def _find_reusable_execution_plan(
    context: ResultContext,
    workspace: Path,
    tool: str,
) -> Optional[Dict[str, Any]]:
    for path in _execution_plan_candidates(context, workspace, tool):
        plan = _read_plan(path)
        if (
            plan is not None
            and _plan_matches_context(plan, context)
            and execution_plan_covers_tool(plan, context, tool)
        ):
            print(f"[profile][{tool}] Reusing complete plan: {path}")
            return plan
    return None


def _validate_profiler_runtime(context: ResultContext) -> None:
    # Post-hoc collection deliberately skips packet, RAPL, perf, and workload
    # preflights.  It only needs the same native Docker daemon and model image.
    from acprof.cli.run import require_native_docker, require_native_linux_host

    require_native_linux_host()
    require_native_docker()
    result = subprocess.run(
        ["docker", "image", "inspect", context.image_tag],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "image not found").strip()
        raise PosthocError(
            f"required model image is unavailable: {context.image_tag}: {detail}"
        )


def _collect_compute_plan(
    context: ResultContext,
    output_dir: Path,
    *,
    tool: str,
    ncu_root: Optional[str],
    torch_repeat: int,
    ncu_repeat: int,
    compute_profile_cpus: Optional[int],
    compute_profile_mem: Optional[int],
) -> Dict[str, Any]:
    from acprof.host.compute_profile import collect_compute_profile_plan

    modes = [
        mode
        for mode in ("off", "on")
        if mode in TOOL_GPU_MODES[tool] and context.cases_for_mode(mode)
    ]
    cases = [case for case in context.resource_cases if case[2] in modes]
    cpus = sorted({case[0] for case in cases}) or [1]
    mems = sorted({case[1] for case in cases}) or [1]
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = collect_compute_profile_plan(
        task_info=context.task_info,
        image_tag=context.image_tag,
        cpu_list=cpus,
        mem_list=mems,
        gpu_list=modes,
        output_dir=str(output_dir),
        input_scale_plan_file=str(context.input_scale_plan_path),
        advisor_root=None,
        ncu_root=ncu_root,
        advisor_repeat=1,
        torch_profiler_repeat=torch_repeat,
        ncu_repeat=ncu_repeat,
        keep_profiles=True,
        compute_profile_cpus=compute_profile_cpus,
        compute_profile_mem=compute_profile_mem,
        compute_profile_tool=tool,
    )
    plan = _read_plan(Path(plan_path))
    if plan is None:
        raise PosthocError(
            f"{tool} collector did not produce a valid compute plan: {plan_path}"
        )
    return plan


def _resolve_massif_reference(
    context: ResultContext,
    reference_cpu: Optional[int],
    reference_mem: Optional[int],
) -> Tuple[int, int]:
    cases = [(cpu, mem) for cpu, mem, mode in context.resource_cases if mode == "off"]
    candidates = [
        case
        for case in cases
        if (reference_cpu is None or case[0] == reference_cpu)
        and (reference_mem is None or case[1] == reference_mem)
    ]
    if not candidates:
        raise PosthocError(
            "Massif reference resource is not present in CPU result rows: "
            f"cpu={reference_cpu or 'auto'}, mem={reference_mem or 'auto'}"
        )
    return max(candidates, key=lambda case: (case[0], case[1]))


def expand_representative_massif_plan(
    plan: Mapping[str, Any],
    context: ResultContext,
    *,
    reference_cpu: int,
    reference_mem: int,
) -> Dict[str, Any]:
    expanded = copy.deepcopy(dict(plan))
    source_profile: Optional[Mapping[str, Any]] = None
    for profile in expanded.get("profiles", []):
        if not isinstance(profile, Mapping):
            continue
        if (
            int(profile.get("cpu_cores", -1)) == reference_cpu
            and int(profile.get("mem_cap_gb", -1)) == reference_mem
            and str(profile.get("gpu_mode") or "").strip().lower() == "off"
            and isinstance(profile.get("tools"), Mapping)
            and isinstance(profile["tools"].get("massif"), Mapping)
        ):
            source_profile = profile
            break
    if source_profile is None:
        raise PosthocError("representative Massif plan has no source profile")

    source_tool = copy.deepcopy(source_profile["tools"]["massif"])
    for entry in source_tool.get("entries", []):
        if isinstance(entry, dict):
            entry["profile_source_cpu_cores"] = reference_cpu
            entry["profile_source_mem_cap_gb"] = reference_mem
            entry["profile_sampling_strategy"] = "representative_per_scale"

    expanded_profiles: List[Dict[str, Any]] = []
    for cpu, mem, mode in context.resource_cases:
        if mode != "off":
            continue
        expanded_profiles.append(
            {
                "cpu_cores": cpu,
                "mem_cap_gb": mem,
                "gpu_mode": "off",
                "tools": {"massif": copy.deepcopy(source_tool)},
            }
        )
    expanded["profiles"] = expanded_profiles
    metadata = expanded.setdefault("static_metadata", {})
    if isinstance(metadata, dict):
        metadata.update(
            {
                "massif_sampling_strategy": "representative_per_scale",
                "massif_reference_cpu_cores": reference_cpu,
                "massif_reference_mem_cap_gb": reference_mem,
                "massif_reused_across_resource_cases": True,
            }
        )
    expanded["massif_sampling_strategy"] = "representative_per_scale"
    return expanded


def _collect_execution_plan(
    context: ResultContext,
    output_dir: Path,
    *,
    tool: str,
    massif_sampling: str,
    massif_reference_cpu: Optional[int],
    massif_reference_mem: Optional[int],
    nsys_sampling: str,
    nsys_reference_cpu: Optional[int],
    nsys_reference_mem: Optional[int],
    massif_repeat: int,
    nsys_repeat: int,
    nsys_root: Optional[str],
) -> Dict[str, Any]:
    from acprof.host.execution_profile import collect_execution_profile_plan

    mode = TOOL_GPU_MODES[tool][0]
    cases = context.cases_for_mode(mode)
    cpus = sorted({case[0] for case in cases})
    mems = sorted({case[1] for case in cases})

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan_path = collect_execution_profile_plan(
            task_info=context.task_info,
            image_tag=context.image_tag,
            cpu_list=cpus,
            mem_list=mems,
            gpu_list=[mode],
            output_dir=str(output_dir),
            input_scale_plan_file=str(context.input_scale_plan_path),
            project_dir=str(PROJECT_DIR),
            tool_mode=tool,
            massif_sampling=massif_sampling,
            massif_reference_cpu=massif_reference_cpu,
            massif_reference_mem=massif_reference_mem,
            massif_repeat=massif_repeat,
            nsys_sampling=nsys_sampling,
            nsys_reference_cpu=nsys_reference_cpu,
            nsys_reference_mem=nsys_reference_mem,
            nsys_repeat=nsys_repeat,
            nsys_root=nsys_root,
            keep_profiles=True,
        )
    except ValueError as exc:
        raise PosthocError(str(exc)) from exc
    plan = _read_plan(Path(plan_path))
    if plan is None:
        raise PosthocError(
            f"{tool} collector did not produce a valid plan: {plan_path}"
        )
    return plan


def merge_compute_plans(
    context: ResultContext,
    plans_by_tool: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    profiles: Dict[str, Dict[str, Any]] = {}
    static_metadata: Dict[str, Any] = {}
    enabled_tools: List[str] = []
    plan_keys = {"torch": TORCH_PROFILE_KEY, "ncu": NCU_PROFILE_KEY}

    for requested_tool, plan in plans_by_tool.items():
        plan_key = plan_keys[requested_tool]
        metadata = plan.get("static_metadata")
        if isinstance(metadata, Mapping):
            metadata_keys = ["compute_profiles_retained"]
            if requested_tool == "torch":
                metadata_keys.extend(
                    key
                    for key in metadata
                    if str(key).startswith("torch_profiler_eager_")
                )
                metadata_keys.extend(["torch_version", "transformers_version"])
            else:
                metadata_keys.extend(
                    key for key in metadata if str(key).startswith("ncu_")
                )
            metadata_keys.extend(["gpu_compute_capability", "gpu_sm_count"])
            for key in metadata_keys:
                if key in metadata:
                    static_metadata[key] = copy.deepcopy(metadata[key])

        source_profiles = plan.get("profiles")
        if isinstance(source_profiles, Mapping):
            for profile_name in ("cpu", "gpu"):
                source_group = source_profiles.get(profile_name)
                tool_profile = (
                    source_group.get(plan_key)
                    if isinstance(source_group, Mapping)
                    else None
                )
                if isinstance(tool_profile, Mapping):
                    profiles.setdefault(profile_name, {})[plan_key] = copy.deepcopy(
                        dict(tool_profile)
                    )
        if plan_key not in enabled_tools:
            enabled_tools.append(plan_key)

    static_metadata["compute_profile_tools"] = [
        tool
        for tool in (TORCH_PROFILE_KEY, NCU_PROFILE_KEY)
        if tool in enabled_tools
    ]
    static_metadata["compute_profiles_retained"] = True
    static_metadata["compute_profile_provenance"] = "posthoc_backfill"
    return {
        "model_id": context.task_info.model_id,
        "model_revision": context.task_info.model_revision or "main",
        "task_family": context.task_info.task_family,
        "pipeline_tag": context.task_info.pipeline_tag,
        "runtime_backend": context.task_info.runtime_backend,
        "compute_profile_tool_mode": "posthoc",
        "static_metadata": static_metadata,
        "profiles": profiles,
    }


def merge_execution_plans(
    context: ResultContext,
    plans_by_tool: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    profiles: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    static_metadata: Dict[str, Any] = {}
    enabled_tools: List[str] = []
    for requested_tool, plan in plans_by_tool.items():
        metadata = plan.get("static_metadata")
        if isinstance(metadata, Mapping):
            metadata_keys = [
                "execution_profile_schema_version",
                "execution_profiles_retained",
            ]
            if requested_tool == "massif":
                metadata_keys.extend(
                    key
                    for key in metadata
                    if str(key).startswith("massif_")
                )
            elif requested_tool == "nsys":
                metadata_keys.extend(
                    key for key in metadata if str(key).startswith("nsys_")
                )
            for key in metadata_keys:
                if key in metadata:
                    static_metadata[key] = copy.deepcopy(metadata[key])
        for profile in plan.get("profiles", []):
            if not isinstance(profile, Mapping):
                continue
            try:
                key = (
                    int(profile.get("cpu_cores")),
                    int(profile.get("mem_cap_gb")),
                    str(profile.get("gpu_mode") or "").strip().lower(),
                )
            except (TypeError, ValueError):
                continue
            tools = profile.get("tools")
            tool_profile = (
                tools.get(requested_tool) if isinstance(tools, Mapping) else None
            )
            if not isinstance(tool_profile, Mapping):
                continue
            target = profiles.setdefault(
                key,
                {
                    "cpu_cores": key[0],
                    "mem_cap_gb": key[1],
                    "gpu_mode": key[2],
                    "tools": {},
                },
            )
            target["tools"][requested_tool] = copy.deepcopy(dict(tool_profile))
        if requested_tool not in enabled_tools:
            enabled_tools.append(requested_tool)

    static_metadata["execution_profile_tools"] = [
        tool for tool in ("massif", "nsys") if tool in enabled_tools
    ]
    static_metadata["execution_profile_provenance"] = "posthoc_backfill"
    return {
        "schema_version": 1,
        "model_id": context.task_info.model_id,
        "model_revision": context.task_info.model_revision or "main",
        "task_family": context.task_info.task_family,
        "pipeline_tag": context.task_info.pipeline_tag,
        "runtime_backend": context.task_info.runtime_backend,
        "execution_profile_tool_mode": "posthoc",
        "static_metadata": static_metadata,
        "profiles": [profiles[key] for key in sorted(profiles)],
    }


def _normalize_tool_metadata(value: Any) -> List[str]:
    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    result: List[str] = []
    for value in values:
        tool = str(value).strip()
        if tool and tool not in result:
            result.append(tool)
    return result


def _merge_provenance(existing: Any, new_value: str) -> str:
    parts = [
        part.strip()
        for part in str(existing or "").replace("+", ",").split(",")
        if part.strip()
    ]
    if new_value not in parts:
        parts.append(new_value)
    return "+".join(parts)


def _static_flops_from_compute_plan(
    plan: Mapping[str, Any],
    context: ResultContext,
) -> Optional[Dict[str, Any]]:
    profiles = plan.get("profiles")
    if not isinstance(profiles, Mapping):
        return None
    metadata = plan.get("static_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}

    # Match the normal run.py metadata rule: prefer the GPU Torch profile when
    # present, otherwise use CPU. Logical FLOP is then keyed only by input scale.
    for profile_name in ("gpu", "cpu"):
        profile_group = profiles.get(profile_name)
        torch_profile = (
            profile_group.get(TORCH_PROFILE_KEY)
            if isinstance(profile_group, Mapping)
            else None
        )
        if not isinstance(torch_profile, Mapping):
            continue
        entries = torch_profile.get("entries")
        if not isinstance(entries, list):
            continue

        values: List[Dict[str, Any]] = []
        seen_scales: set[float] = set()
        for entry in entries:
            if not isinstance(entry, Mapping) or entry.get("error"):
                continue
            scale = _finite_float(entry.get("input_scale"))
            mflop = _finite_float(entry.get(TORCH_LOGICAL_MFLOP_FIELD))
            if not math.isfinite(scale) or not math.isfinite(mflop) or mflop < 0:
                continue
            normalized_scale: int | float = (
                int(scale) if scale.is_integer() else scale
            )
            if normalized_scale in seen_scales:
                continue
            seen_scales.add(normalized_scale)
            values.append(
                {
                    "input_scale": normalized_scale,
                    "flops_per_request": int(round(mflop * 1_000_000)),
                }
            )
        if values:
            values.sort(key=lambda item: float(item["input_scale"]))
            semantics = (
                torch_profile.get("flop_semantics")
                or metadata.get("torch_profiler_eager_flop_semantics")
                or "logical_operator_shape_flops"
            )
            return {
                "source": TORCH_PROFILE_KEY,
                "profile": profile_name,
                "semantics": semantics,
                "unit": "FLOP/request",
                "input_scale_type": context.static_meta.get("input_scale_type", ""),
                "batch_size": context.static_meta.get("batch_size", 1),
                "values": values,
            }
    return None


def update_static_meta(
    context: ResultContext,
    *,
    tools: Iterable[str],
    compute_plan: Optional[Mapping[str, Any]],
    execution_plan: Optional[Mapping[str, Any]],
    backup_dir: Path,
    massif_sampling: str,
    nsys_sampling: str,
) -> Dict[str, Any]:
    updated = copy.deepcopy(context.static_meta)
    selected = tuple(tools)

    compute_selected = [tool for tool in selected if tool in {"torch", "ncu"}]
    if compute_selected and compute_plan is not None:
        metadata = compute_plan.get("static_metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        compute_tools = _normalize_tool_metadata(updated.get("compute_profile_tools"))
        for tool in compute_selected:
            metadata_name = TORCH_PROFILE_KEY if tool == "torch" else NCU_PROFILE_KEY
            if metadata_name not in compute_tools:
                compute_tools.append(metadata_name)
        updated["compute_profile_tools"] = compute_tools
        metadata_keys: List[str] = ["gpu_compute_capability", "gpu_sm_count"]
        if "torch" in compute_selected:
            metadata_keys.extend(
                [
                    "torch_profiler_eager_flop_semantics",
                    "torch_profiler_eager_attention_implementation",
                    "torch_profiler_eager_repeat_cpu",
                    "torch_profiler_eager_repeat_gpu",
                    "torch_version",
                    "transformers_version",
                ]
            )
        if "ncu" in compute_selected:
            metadata_keys.extend(
                [
                    "ncu_flop_semantics",
                    "ncu_repeat",
                    "ncu_fma_flop_weight",
                    "ncu_metrics",
                    "ncu_version",
                ]
            )
        for key in metadata_keys:
            value = metadata.get(key)
            if value not in (None, "", "unknown"):
                updated[key] = copy.deepcopy(value)
        if "torch" in compute_selected:
            static_flops = _static_flops_from_compute_plan(compute_plan, context)
            if static_flops is not None:
                updated["static_flops"] = static_flops
        updated["compute_profiles_retained"] = True
        updated["compute_profile_provenance"] = _merge_provenance(
            updated.get("compute_profile_provenance"), "posthoc_backfill"
        )

    execution_tools = [tool for tool in selected if tool in {"massif", "nsys"}]
    if execution_tools and execution_plan is not None:
        metadata = execution_plan.get("static_metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        known_tools = _normalize_tool_metadata(updated.get("execution_profile_tools"))
        for tool in execution_tools:
            if tool not in known_tools:
                known_tools.append(tool)
        updated["execution_profile_tools"] = [
            tool for tool in ("massif", "nsys") if tool in known_tools
        ]
        metadata_keys: List[str] = ["execution_profile_schema_version"]
        if "massif" in execution_tools:
            metadata_keys.extend(
                [
                    "massif_peak_semantics",
                    "massif_repeat",
                    "massif_version",
                    "massif_sampling_strategy",
                    "massif_reference_cpu_cores",
                    "massif_reference_mem_cap_gb",
                    "massif_reused_across_resource_cases",
                ]
            )
        if "nsys" in execution_tools:
            metadata_keys.extend(
                [
                    "nsys_timeline_semantics",
                    "nsys_repeat",
                    "nsys_version",
                    "nsys_sampling_strategy",
                    "nsys_reference_cpu_cores",
                    "nsys_reference_mem_cap_gb",
                    "nsys_reused_across_resource_cases",
                ]
            )
        nullable_sampling_keys = {
            "massif_reference_cpu_cores",
            "massif_reference_mem_cap_gb",
            "nsys_reference_cpu_cores",
            "nsys_reference_mem_cap_gb",
        }
        for key in metadata_keys:
            if key not in metadata:
                continue
            value = metadata.get(key)
            if key in nullable_sampling_keys or value not in (None, "", "unknown"):
                updated[key] = copy.deepcopy(value)
        updated["execution_profiles_retained"] = True
        updated["execution_profile_provenance"] = _merge_provenance(
            updated.get("execution_profile_provenance"), "posthoc_backfill"
        )

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    record = {
        "completed_at": timestamp,
        "tools": list(selected),
        "result_backup": str(backup_dir.relative_to(context.result_dir)),
        "massif_sampling": massif_sampling if "massif" in selected else None,
        "nsys_sampling": nsys_sampling if "nsys" in selected else None,
    }
    history = updated.get("posthoc_profile_history")
    if not isinstance(history, list):
        history = []
    updated["posthoc_profile_history"] = [*history, record]
    updated["posthoc_profile_last_run"] = record
    return updated


def _timestamp_token() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def create_backup(context: ResultContext) -> Path:
    root = context.result_dir / BACKUP_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    base = _timestamp_token()
    backup = root / base
    suffix = 1
    while backup.exists():
        backup = root / f"{base}-{suffix}"
        suffix += 1
    backup.mkdir()
    shutil.copy2(context.result_csv, backup / RESULT_CSV_NAME)
    shutil.copy2(context.static_meta_path, backup / STATIC_META_NAME)
    return backup


def _write_csv_temporary(
    destination: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    encoding: str,
) -> Path:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(fieldnames),
                quoting=csv.QUOTE_MINIMAL,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        mode = stat.S_IMODE(destination.stat().st_mode)
        temporary_path.chmod(mode)
        return temporary_path
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json_temporary(destination: Path, payload: Mapping[str, Any]) -> Path:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        mode = stat.S_IMODE(destination.stat().st_mode)
        temporary_path.chmod(mode)
        return temporary_path
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _restore_from_backup(destination: Path, backup_file: Path) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.restore.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(backup_file, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def commit_result_files(
    context: ResultContext,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    static_meta: Mapping[str, Any],
    backup_dir: Path,
) -> None:
    csv_temporary = _write_csv_temporary(
        context.result_csv,
        fieldnames=fieldnames,
        rows=rows,
        encoding=context.csv_encoding,
    )
    meta_temporary = _write_json_temporary(context.static_meta_path, static_meta)
    csv_replaced = False
    meta_replaced = False
    try:
        # Validate both complete temporary documents before publishing either.
        with csv_temporary.open(
            "r", encoding=context.csv_encoding, newline=""
        ) as f:
            if sum(1 for _row in csv.DictReader(f)) != len(rows):
                raise PosthocError("temporary result CSV row-count validation failed")
        _load_json_object(meta_temporary, "temporary static metadata")

        os.replace(csv_temporary, context.result_csv)
        csv_replaced = True
        os.replace(meta_temporary, context.static_meta_path)
        meta_replaced = True
    except Exception:
        if csv_replaced:
            _restore_from_backup(
                context.result_csv, backup_dir / RESULT_CSV_NAME
            )
        if meta_replaced:
            _restore_from_backup(
                context.static_meta_path, backup_dir / STATIC_META_NAME
            )
        raise
    finally:
        for temporary in (csv_temporary, meta_temporary):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_process_cmdline(pid_dir: Path) -> List[str]:
    try:
        data = (pid_dir / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return []
    return [
        part.decode("utf-8", errors="replace")
        for part in data.split(b"\0")
        if part
    ]


def _option_value(args: Sequence[str], name: str) -> Optional[str]:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def find_active_processes(
    result_dir: Path,
    *,
    model_id: str = "",
) -> List[Tuple[int, str]]:
    expected = result_dir.resolve()
    matches: List[Tuple[int, str]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == os.getpid():
            continue
        args = _read_process_cmdline(pid_dir)
        if not args:
            continue
        command = " ".join(args)
        executable = Path(args[0]).name.lower()
        profiler_process = executable not in {"bash", "sh", "dash", "zsh"} and any(
            marker in command
            for marker in (
                "run.py",
                "acprof.cli.run",
                "compute_profile_runner",
                " ncu ",
                "/ncu ",
                "nsys",
                "massif",
                "posthoc.py",
                "profile.py",
                "acprof.cli.posthoc",
            )
        )
        direct_match = str(expected) in command and profiler_process
        run_match = False
        if model_id and _option_value(args, "--model") == model_id:
            is_run_command = any(
                Path(arg).name == "run.py" for arg in args
            ) or (
                "-m" in args and "acprof.cli.run" in args
            )
            if is_run_command:
                output_root = _option_value(args, "--output-dir") or "results"
                output_root_path = Path(output_root)
                if not output_root_path.is_absolute():
                    output_root_path = PROJECT_DIR / output_root_path
                candidate = output_root_path / model_id.replace("/", "--")
                run_match = candidate.resolve() == expected
        if direct_match or run_match:
            matches.append((int(pid_dir.name), command[:500]))
    return sorted(matches)


class PosthocLock:
    def __init__(self, result_dir: Path):
        self.path = result_dir / LOCK_FILENAME
        self._owned = False

    def __enter__(self) -> "PosthocLock":
        payload = json.dumps({"pid": os.getpid(), "created_at": _timestamp_token()})
        for _attempt in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    pid = int(existing.get("pid", -1))
                except (OSError, ValueError, TypeError):
                    pid = -1
                if pid > 0 and Path(f"/proc/{pid}").exists():
                    raise PosthocError(
                        f"another profile.py process is active (pid={pid}): {self.path}"
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as f:
                f.write(payload + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._owned = True
            return self
        raise PosthocError(f"cannot acquire profile lock: {self.path}")

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._owned:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._owned = False


def run_posthoc(
    result_dir: str | os.PathLike[str],
    *,
    tools: str | Iterable[str] = SUPPORTED_TOOLS,
    massif_sampling: str = "per-scale",
    massif_reference_cpu: Optional[int] = None,
    massif_reference_mem: Optional[int] = None,
    nsys_sampling: str = "per-cpu-scale",
    nsys_reference_cpu: Optional[int] = None,
    nsys_reference_mem: Optional[int] = None,
    ncu_root: Optional[str] = None,
    nsys_root: Optional[str] = None,
    torch_repeat: int = 1,
    ncu_repeat: int = 1,
    nsys_repeat: int = 1,
    massif_repeat: int = 1,
    compute_profile_cpus: Optional[int] = None,
    compute_profile_mem: Optional[int] = None,
    force_reprofile: bool = False,
    dry_run: bool = False,
) -> PosthocSummary:
    if massif_sampling not in {"per-scale", "full"}:
        raise PosthocError("massif_sampling must be 'per-scale' or 'full'")
    if nsys_sampling not in {"per-cpu-scale", "per-scale", "full"}:
        raise PosthocError(
            "nsys_sampling must be 'per-cpu-scale', 'per-scale', or 'full'"
        )
    for name, value in (
        ("torch_repeat", torch_repeat),
        ("ncu_repeat", ncu_repeat),
        ("nsys_repeat", nsys_repeat),
        ("massif_repeat", massif_repeat),
    ):
        if int(value) <= 0:
            raise PosthocError(f"{name} must be > 0")
    for name, value in (
        ("massif_reference_cpu", massif_reference_cpu),
        ("massif_reference_mem", massif_reference_mem),
        ("nsys_reference_cpu", nsys_reference_cpu),
        ("nsys_reference_mem", nsys_reference_mem),
    ):
        if value is not None and int(value) <= 0:
            raise PosthocError(f"{name} must be > 0")

    selected = parse_tools(tools)
    directory = Path(result_dir).expanduser().resolve()
    early_meta = _read_plan(directory / STATIC_META_NAME) or {}
    model_id = str(early_meta.get("model_name") or "").strip()
    active = find_active_processes(directory, model_id=model_id)
    if active:
        detail = "\n".join(f"  pid={pid}: {command}" for pid, command in active)
        raise PosthocError(
            "run.py/profiler processes are still using this result directory; "
            f"wait for them to finish:\n{detail}"
        )

    context = load_result_context(directory)
    applicable, inapplicable = applicable_tools(context, selected)
    for tool in inapplicable:
        mode = "/".join(TOOL_GPU_MODES[tool])
        print(
            f"[profile][{tool}] Skipped: result_all.csv has no gpu_mode={mode} rows"
        )
    if not applicable:
        print("[profile] No requested profiler applies to this result CSV.")
        return PosthocSummary(
            result_csv=str(context.result_csv),
            static_meta=str(context.static_meta_path),
            backup_dir=None,
            collected_tools=(),
            reused_tools=(),
            skipped_tools=tuple(inapplicable),
            updated_rows_by_tool={},
        )

    already_complete = tuple(
        tool
        for tool in applicable
        if not force_reprofile and csv_tool_complete(context, tool)
    )
    needed = tuple(tool for tool in applicable if tool not in already_complete)
    for tool in already_complete:
        print(f"[profile][{tool}] Skipped: CSV already has successful values")

    print(f"[profile] Result directory: {context.result_dir}")
    print(f"[profile] Model: {context.task_info.model_id}")
    print(
        "[profile] Resource cases: "
        f"CPU-only={len(context.cases_for_mode('off'))}, "
        f"GPU={len(context.cases_for_mode('on'))}"
    )
    print(f"[profile] Tools requiring work: {', '.join(needed) or 'none'}")
    if dry_run or not needed:
        if dry_run:
            print("[profile] Dry run: no profiler was started and no file was changed.")
        return PosthocSummary(
            result_csv=str(context.result_csv),
            static_meta=str(context.static_meta_path),
            backup_dir=None,
            collected_tools=(),
            reused_tools=(),
            skipped_tools=tuple((*inapplicable, *already_complete)),
            updated_rows_by_tool={tool: 0 for tool in needed},
        )

    workspace = context.result_dir / POSTHOC_DIRNAME
    workspace.mkdir(parents=True, exist_ok=True)
    collected: List[str] = []
    reused: List[str] = []
    plans_by_tool: Dict[str, Dict[str, Any]] = {}
    runtime_validated = False

    with PosthocLock(context.result_dir):
        for tool in needed:
            reusable: Optional[Dict[str, Any]] = None
            if not force_reprofile:
                reusable = (
                    _find_reusable_compute_plan(context, workspace, tool)
                    if tool in {"torch", "ncu"}
                    else _find_reusable_execution_plan(context, workspace, tool)
                )
            if reusable is not None:
                plans_by_tool[tool] = reusable
                reused.append(tool)
                continue

            if not runtime_validated:
                _validate_profiler_runtime(context)
                runtime_validated = True
            print(f"[profile][{tool}] Collecting isolated profiler probes...")
            if tool in {"torch", "ncu"}:
                plan = _collect_compute_plan(
                    context,
                    workspace / tool,
                    tool=tool,
                    ncu_root=ncu_root,
                    torch_repeat=torch_repeat,
                    ncu_repeat=ncu_repeat,
                    compute_profile_cpus=compute_profile_cpus,
                    compute_profile_mem=compute_profile_mem,
                )
            else:
                plan = _collect_execution_plan(
                    context,
                    workspace / tool,
                    tool=tool,
                    massif_sampling=massif_sampling,
                    massif_reference_cpu=massif_reference_cpu,
                    massif_reference_mem=massif_reference_mem,
                    nsys_sampling=nsys_sampling,
                    nsys_reference_cpu=nsys_reference_cpu,
                    nsys_reference_mem=nsys_reference_mem,
                    massif_repeat=massif_repeat,
                    nsys_repeat=nsys_repeat,
                    nsys_root=nsys_root,
                )
            plans_by_tool[tool] = plan
            collected.append(tool)

        compute_tool_plans = {
            tool: plan
            for tool, plan in plans_by_tool.items()
            if tool in {"torch", "ncu"}
        }
        compute_plan = (
            merge_compute_plans(context, compute_tool_plans)
            if compute_tool_plans
            else None
        )
        execution_tool_plans = {
            tool: plan
            for tool, plan in plans_by_tool.items()
            if tool in {"massif", "nsys"}
        }
        execution_plan = (
            merge_execution_plans(context, execution_tool_plans)
            if execution_tool_plans
            else None
        )
        if compute_plan is not None:
            _atomic_write_json(workspace / "compute_profile_plan.json", compute_plan)
        if execution_plan is not None:
            _atomic_write_json(
                workspace / "execution_profile_plan.json", execution_plan
            )

        fieldnames, rows, updated_rows = backfill_rows(
            context,
            tools=needed,
            compute_plan=compute_plan,
            execution_plan=execution_plan,
            force=force_reprofile,
        )
        backup_dir = create_backup(context)
        static_meta = update_static_meta(
            context,
            tools=needed,
            compute_plan=compute_plan,
            execution_plan=execution_plan,
            backup_dir=backup_dir,
            massif_sampling=massif_sampling,
            nsys_sampling=nsys_sampling,
        )
        commit_result_files(
            context,
            fieldnames=fieldnames,
            rows=rows,
            static_meta=static_meta,
            backup_dir=backup_dir,
        )

    print(f"[profile] Updated in place: {context.result_csv}")
    print(f"[profile] Updated in place: {context.static_meta_path}")
    print(f"[profile] Original files backed up to: {backup_dir}")
    for tool, count in updated_rows.items():
        print(f"[profile][{tool}] Backfilled rows: {count}")
    return PosthocSummary(
        result_csv=str(context.result_csv),
        static_meta=str(context.static_meta_path),
        backup_dir=str(backup_dir),
        collected_tools=tuple(collected),
        reused_tools=tuple(reused),
        skipped_tools=tuple((*inapplicable, *already_complete)),
        updated_rows_by_tool=updated_rows,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect missing Torch/NCU/Nsys/Massif metrics for an existing AC-Prof "
            "result directory and safely backfill result_all.csv in place."
        )
    )
    parser.add_argument(
        "result_dir",
        help=(
            "Completed model result directory containing result_all.csv, "
            "static_meta.json, and input_scale_plan.json"
        ),
    )
    parser.add_argument(
        "--tools",
        default=",".join(SUPPORTED_TOOLS),
        help="Comma-separated tools (default: torch,ncu,nsys,massif)",
    )
    parser.add_argument(
        "--massif-sampling",
        choices=("per-scale", "full"),
        default="per-scale",
        help=(
            "per-scale profiles one representative CPU/memory case per input "
            "scale and reuses it across CPU rows (default); full profiles the "
            "entire CPU/memory matrix"
        ),
    )
    parser.add_argument("--massif-reference-cpu", type=int, default=None)
    parser.add_argument("--massif-reference-mem", type=int, default=None)
    parser.add_argument(
        "--nsys-sampling",
        choices=("per-cpu-scale", "per-scale", "full"),
        default="per-cpu-scale",
        help=(
            "per-cpu-scale profiles every CPU at one representative memory "
            "per input scale (default); per-scale profiles one representative "
            "CPU/memory case; full profiles the entire CPU/memory matrix"
        ),
    )
    parser.add_argument("--nsys-reference-cpu", type=int, default=None)
    parser.add_argument("--nsys-reference-mem", type=int, default=None)
    parser.add_argument("--ncu-root", default=None)
    parser.add_argument("--nsys-root", default=None)
    parser.add_argument(
        "--torch-profiler-repeat",
        "--torch-repeat",
        dest="torch_repeat",
        type=int,
        default=1,
    )
    parser.add_argument("--ncu-repeat", type=int, default=1)
    parser.add_argument("--nsys-repeat", type=int, default=1)
    parser.add_argument("--massif-repeat", type=int, default=1)
    parser.add_argument("--compute-profile-cpus", type=int, default=None)
    parser.add_argument("--compute-profile-mem", type=int, default=None)
    parser.add_argument(
        "--force-reprofile",
        action="store_true",
        help="Collect again and replace even already-successful profiler fields",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and show which tools need work without collecting",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    bootstrap_project_env(PROJECT_DIR)
    try:
        run_posthoc(
            args.result_dir,
            tools=args.tools,
            massif_sampling=args.massif_sampling,
            massif_reference_cpu=args.massif_reference_cpu,
            massif_reference_mem=args.massif_reference_mem,
            nsys_sampling=args.nsys_sampling,
            nsys_reference_cpu=args.nsys_reference_cpu,
            nsys_reference_mem=args.nsys_reference_mem,
            ncu_root=args.ncu_root,
            nsys_root=args.nsys_root,
            torch_repeat=args.torch_repeat,
            ncu_repeat=args.ncu_repeat,
            nsys_repeat=args.nsys_repeat,
            massif_repeat=args.massif_repeat,
            compute_profile_cpus=args.compute_profile_cpus,
            compute_profile_mem=args.compute_profile_mem,
            force_reprofile=args.force_reprofile,
            dry_run=args.dry_run,
        )
    except (OSError, PosthocError) as exc:
        parser.exit(2, f"[profile][ERROR] {exc}\n")


if __name__ == "__main__":
    main()

"""Result inputs and types for post-hoc profiling."""
from __future__ import annotations

import codecs
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from acprof.metric_registry import tool_fields
from acprof.host.collection_history import COLLECTION_HISTORY_NAME, migrate_legacy_static_meta_history
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
)
from acprof.host.detect import TaskInfo
from acprof.host.execution_profile_plan import (
    MASSIF_ERROR_FIELD,
    MASSIF_METRIC_FIELDS,
    NSYS_ERROR_FIELD,
    NSYS_METRIC_FIELDS,
)


PROJECT_DIR = Path(__file__).resolve().parents[3]


RESULT_CSV_NAME = "result_all.csv"


STATIC_META_NAME = "static_meta.json"


INPUT_SCALE_PLAN_NAME = "input_scale_plan.json"


POSTHOC_DIRNAME = "posthoc_profiles"


BACKUP_DIRNAME = "posthoc_backups"


LOCK_FILENAME = ".posthoc.lock"


SUPPORTED_TOOLS = ("torch", "ncu", "nsys", "massif")


TORCH_FIELDS = tool_fields('torch')


NCU_DERIVED_APP_FIELD = "gpu_executed_mflops_app_ncu"


NCU_DERIVED_PACKET_FIELD = "gpu_executed_mflops_packet_ncu"


NCU_FIELDS = tool_fields('ncu')


MASSIF_FIELDS = tool_fields('massif')


NSYS_FIELDS = tool_fields('nsys')


TOOL_FIELDS = {tool: tool_fields(tool) for tool in SUPPORTED_TOOLS}


TOOL_METRIC_FIELDS = {tool: tool_fields(tool, numeric_only=True) for tool in SUPPORTED_TOOLS}


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
    collection_history_path: Path
    input_scale_plan_path: Path
    fieldnames: List[str]
    rows: List[Dict[str, str]]
    csv_encoding: str
    static_meta: Dict[str, Any]
    collection_history: Dict[str, Any]
    collection_history_existed: bool
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
    collection_history_path = directory / COLLECTION_HISTORY_NAME
    input_scale_plan_path = directory / INPUT_SCALE_PLAN_NAME
    fieldnames, rows, encoding = _load_result_csv(result_csv)
    static_meta = _load_json_object(static_meta_path, STATIC_META_NAME)
    collection_history_existed = collection_history_path.is_file()
    collection_history_payload = (
        _load_json_object(collection_history_path, COLLECTION_HISTORY_NAME)
        if collection_history_existed
        else None
    )
    try:
        static_meta, collection_history = migrate_legacy_static_meta_history(
            static_meta,
            collection_history_payload,
        )
    except ValueError as exc:
        raise PosthocError(f"invalid collection history: {exc}") from exc
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

    image_tag = str(static_meta.get("image_id") or static_meta.get("image_tag") or "").strip()
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
        runtime_profile_id=(static_meta.get("runtime_environment") or {}).get("profile_id", ""),
        model_adapter=(static_meta.get("runtime_environment") or {}).get("adapter", "family-default"),
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
        collection_history_path=collection_history_path,
        input_scale_plan_path=input_scale_plan_path,
        fieldnames=fieldnames,
        rows=rows,
        csv_encoding=encoding,
        static_meta=static_meta,
        collection_history=collection_history,
        collection_history_existed=collection_history_existed,
        input_scale_plan=input_plan,
        task_info=task_info,
        image_tag=image_tag,
        resource_cases=resource_cases,
        scales_by_mode=scales_by_mode,
    )


def _read_plan(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None

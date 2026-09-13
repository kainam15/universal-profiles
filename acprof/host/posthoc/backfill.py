"""Pure row and metadata updates for post-hoc profiling."""
from __future__ import annotations

import copy
import math
from datetime import datetime
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

from acprof.host.collection_history import append_collection_record
from acprof.host.compute_profile_plan import (
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
from acprof.host.execution_profile_plan import (
    MASSIF_METRIC_FIELDS,
    NSYS_METRIC_FIELDS,
    find_execution_profile_entry,
)
from acprof.host.posthoc.context import (
    NCU_DERIVED_APP_FIELD,
    NCU_DERIVED_PACKET_FIELD,
    PosthocError,
    ResultContext,
    TOOL_ERROR_FIELD,
    TOOL_FIELDS,
    TOOL_GPU_MODES,
    TOOL_METRIC_FIELDS,
    _finite_float,
    _fmt_float,
    _integer,
)


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

    return updated


def update_collection_history(
    context: ResultContext,
    *,
    tools: Iterable[str],
    backup_dir: Path,
    massif_sampling: str,
    nsys_sampling: str,
) -> Dict[str, Any]:
    """Record post-hoc collection provenance outside ``static_meta.json``."""
    selected = tuple(tools)
    record = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tools": list(selected),
        "result_backup": str(backup_dir.relative_to(context.result_dir)),
        "massif_sampling": massif_sampling if "massif" in selected else None,
        "nsys_sampling": nsys_sampling if "nsys" in selected else None,
    }
    return append_collection_record(
        context.collection_history,
        "posthoc_profile_history",
        record,
    )

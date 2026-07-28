"""Shared helpers for reading and applying a compute profile plan.

Version 1 plans contain one profile per device class.  Version 2 plans contain
independent ``torch_profiler_eager`` and ``ncu`` profiles so a failure in one
tool never hides results from the other one.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, Tuple


INPUT_SCALE_ABS_TOLERANCE = 1e-6
TORCH_PROFILE_KEY = "torch_profiler_eager"
NCU_PROFILE_KEY = "ncu"

TORCH_LOGICAL_MFLOP_FIELD = (
    "model_logical_mflop_per_request_torch_profiler_eager"
)
TORCH_ERROR_FIELD = "compute_profile_error_torch_profiler_eager"

NCU_TOTAL_MFLOP_FIELD = "gpu_executed_mflop_per_request_ncu"
NCU_TENSOR_MFLOP_FIELD = "gpu_executed_tensor_mflop_per_request_ncu"
NCU_SCALAR_MFLOP_FIELD = "gpu_executed_scalar_mflop_per_request_ncu"
NCU_TENSOR_SHARE_FIELD = "gpu_executed_tensor_share_pct_ncu"
NCU_KERNEL_COUNT_FIELD = "gpu_kernel_launch_count_per_request_ncu"
NCU_KERNEL_TIME_FIELD = "gpu_kernel_time_sum_ms_per_request_ncu"
NCU_ERROR_FIELD = "compute_profile_error_ncu"


def _to_float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _first_finite(entry: Dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        value = _to_float_or_nan(entry.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def load_compute_profile_plan(path: str) -> Dict[str, Any]:
    """Load a profiling plan while preserving the client's diagnostic semantics."""
    if not path:
        return {"profiles": {}, "_load_error": "compute_profile_disabled"}
    if not os.path.exists(path):
        return {"profiles": {}, "_load_error": f"compute_profile_plan_not_found:{path}"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as exc:
        return {"profiles": {}, "_load_error": f"compute_profile_plan_invalid:{exc!r}"}
    if not isinstance(plan, dict):
        return {"profiles": {}, "_load_error": "compute_profile_plan_invalid:not_dict"}
    profiles = plan.get("profiles")
    if not isinstance(profiles, dict):
        return {"profiles": {}, "_load_error": "compute_profile_plan_invalid:missing_profiles"}
    return plan


def _empty_result() -> Dict[str, Any]:
    return {
        # Internal version 1 plan compatibility keys. Result CSV writers emit
        # only the explicit version 2 fields below.
        "tool": "nan",
        "model_mflop_per_request": float("nan"),
        "error": "",
        # Explicit version 2 fields.
        TORCH_LOGICAL_MFLOP_FIELD: float("nan"),
        TORCH_ERROR_FIELD: "",
        NCU_TOTAL_MFLOP_FIELD: float("nan"),
        NCU_TENSOR_MFLOP_FIELD: float("nan"),
        NCU_SCALAR_MFLOP_FIELD: float("nan"),
        NCU_TENSOR_SHARE_FIELD: float("nan"),
        NCU_KERNEL_COUNT_FIELD: float("nan"),
        NCU_KERNEL_TIME_FIELD: float("nan"),
        NCU_ERROR_FIELD: "",
    }


def _plan_schema_version(plan: Dict[str, Any]) -> int:
    for key in (
        "compute_profile_schema_version",
        "profile_schema_version",
        "schema_version",
        "version",
    ):
        try:
            return int(plan.get(key))
        except (TypeError, ValueError):
            continue
    return 1


def _nested_profiles(profile: Dict[str, Any]) -> Dict[str, Any]:
    tools = profile.get("tools")
    return tools if isinstance(tools, dict) else profile


def _is_v2_resource_profile(plan: Dict[str, Any], profile: Any) -> bool:
    if _plan_schema_version(plan) >= 2:
        return True
    if not isinstance(profile, dict):
        return False
    nested = _nested_profiles(profile)
    return any(key in nested for key in (TORCH_PROFILE_KEY, NCU_PROFILE_KEY))


def _resolve_profile_entry(
    profile: Any,
    *,
    profile_key: str,
    input_scale: float,
    default_tool: str,
) -> Tuple[str, Dict[str, Any], str]:
    """Return ``(tool, matching_entry, diagnostic)`` for one tool profile."""
    if not isinstance(profile, dict):
        return (
            default_tool,
            {},
            f"compute_profile_missing_profile:{profile_key}",
        )

    tool = str(profile.get("tool") or default_tool or "nan")
    profile_error = str(profile.get("error") or "")
    entries = profile.get("entries")
    if not isinstance(entries, list):
        return (
            tool,
            {},
            profile_error or f"compute_profile_missing_entries:{profile_key}",
        )

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            entry_scale = float(entry.get("input_scale"))
        except (TypeError, ValueError):
            continue
        if math.isclose(
            entry_scale,
            float(input_scale),
            rel_tol=0.0,
            abs_tol=INPUT_SCALE_ABS_TOLERANCE,
        ):
            entry_error = (
                str(entry.get("error") or "")
                if "error" in entry
                else profile_error
            )
            return (
                str(entry.get("tool") or tool or "nan"),
                entry,
                entry_error,
            )

    return (
        tool,
        {},
        f"compute_profile_missing_scale:{input_scale:g}",
    )


def _apply_torch_profile(
    result: Dict[str, Any],
    *,
    tool: str,
    entry: Dict[str, Any],
    error: str,
) -> None:
    logical_mflop = _first_finite(
        entry,
        (
            TORCH_LOGICAL_MFLOP_FIELD,
            "model_mflop_per_request",
        ),
    )
    result.update(
        {
            TORCH_LOGICAL_MFLOP_FIELD: logical_mflop,
            TORCH_ERROR_FIELD: error,
            # Compatibility aliases for version 2 consumers.
            "tool": tool,
            "model_mflop_per_request": logical_mflop,
            "error": error,
        }
    )


def _apply_ncu_profile(
    result: Dict[str, Any],
    *,
    entry: Dict[str, Any],
    error: str,
) -> None:
    total_mflop = _first_finite(
        entry,
        (
            NCU_TOTAL_MFLOP_FIELD,
            "model_mflop_per_request",
            "total_mflop_per_request",
        ),
    )
    tensor_mflop = _first_finite(
        entry,
        (
            NCU_TENSOR_MFLOP_FIELD,
            "tensor_mflop_per_request",
        ),
    )
    scalar_mflop = _first_finite(
        entry,
        (
            NCU_SCALAR_MFLOP_FIELD,
            "scalar_mflop_per_request",
        ),
    )
    tensor_share_pct = _first_finite(
        entry,
        (
            NCU_TENSOR_SHARE_FIELD,
            "tensor_share_pct",
        ),
    )
    if (
        not math.isfinite(tensor_share_pct)
        and math.isfinite(total_mflop)
        and total_mflop > 0.0
        and math.isfinite(tensor_mflop)
    ):
        tensor_share_pct = tensor_mflop / total_mflop * 100.0

    result.update(
        {
            NCU_TOTAL_MFLOP_FIELD: total_mflop,
            NCU_TENSOR_MFLOP_FIELD: tensor_mflop,
            NCU_SCALAR_MFLOP_FIELD: scalar_mflop,
            NCU_TENSOR_SHARE_FIELD: tensor_share_pct,
            NCU_KERNEL_COUNT_FIELD: _first_finite(
                entry,
                (
                    NCU_KERNEL_COUNT_FIELD,
                    "kernel_launch_count_per_request",
                    "kernel_count_per_request",
                ),
            ),
            NCU_KERNEL_TIME_FIELD: _first_finite(
                entry,
                (
                    NCU_KERNEL_TIME_FIELD,
                    "kernel_time_sum_ms_per_request",
                ),
            ),
            NCU_ERROR_FIELD: error,
        }
    )


def find_compute_profile_entry(
    plan: Dict[str, Any],
    gpu_mode: str,
    input_scale: float,
) -> Dict[str, Any]:
    """Resolve both FLOP profiles for one device/scale.

    The returned dictionary is deliberately flat so the live client, packet
    merger, and offline backfill all use the same field semantics.  Direct
    version 1 profiles keep internal legacy values for input compatibility.
    Result CSV writers emit only the explicit Torch eager and NCU fields.
    """
    profile_key = "gpu" if gpu_mode == "on" else "cpu"
    result = _empty_result()
    load_error = str(plan.get("_load_error", "") or "")
    if load_error:
        result["error"] = load_error
        result[TORCH_ERROR_FIELD] = load_error
        if profile_key == "gpu":
            result[NCU_ERROR_FIELD] = load_error
        return result

    profile = plan.get("profiles", {}).get(profile_key)
    if _is_v2_resource_profile(plan, profile):
        nested = _nested_profiles(profile) if isinstance(profile, dict) else {}
        torch_profile = nested.get(TORCH_PROFILE_KEY)
        if isinstance(torch_profile, dict):
            torch_tool, torch_entry, torch_error = _resolve_profile_entry(
                torch_profile,
                profile_key=profile_key,
                input_scale=input_scale,
                default_tool=TORCH_PROFILE_KEY,
            )
            _apply_torch_profile(
                result,
                tool=torch_tool,
                entry=torch_entry,
                error=torch_error,
            )
        elif profile_key == "cpu" and isinstance(
            nested.get("intel_advisor"),
            dict,
        ):
            # Explicit legacy vendor mode has no Torch profile and no
            # dedicated Advisor result columns. Keep its generic values
            # without mislabelling them as Torch eager logical FLOPs.
            advisor_tool, advisor_entry, advisor_error = _resolve_profile_entry(
                nested["intel_advisor"],
                profile_key=profile_key,
                input_scale=input_scale,
                default_tool="intel_advisor",
            )
            result.update(
                {
                    "tool": advisor_tool,
                    "model_mflop_per_request": _first_finite(
                        advisor_entry,
                        ("model_mflop_per_request",),
                    ),
                    "error": advisor_error,
                }
            )

        # NCU measures GPU execution and is intentionally not applicable to
        # CPU-only result rows.
        ncu_profile = nested.get(NCU_PROFILE_KEY)
        if profile_key == "gpu" and isinstance(ncu_profile, dict):
            _ncu_tool, ncu_entry, ncu_error = _resolve_profile_entry(
                ncu_profile,
                profile_key=profile_key,
                input_scale=input_scale,
                default_tool=NCU_PROFILE_KEY,
            )
            _apply_ncu_profile(
                result,
                entry=ncu_entry,
                error=ncu_error,
            )
        return result

    # Version 1 compatibility: preserve the old generic result exactly.  Map
    # it to an explicit profile only when the legacy tool identity makes that
    # interpretation unambiguous.
    tool, entry, error = _resolve_profile_entry(
        profile,
        profile_key=profile_key,
        input_scale=input_scale,
        default_tool="nan",
    )
    legacy_mflop = _first_finite(entry, ("model_mflop_per_request",))
    result.update(
        {
            "tool": tool,
            "model_mflop_per_request": legacy_mflop,
            "error": error,
        }
    )
    normalized_tool = tool.strip().lower()
    if "torch" in normalized_tool:
        _apply_torch_profile(
            result,
            tool=tool,
            entry=entry,
            error=error,
        )
    elif profile_key == "gpu" and (
        normalized_tool == NCU_PROFILE_KEY or "nsight" in normalized_tool
    ):
        _apply_ncu_profile(
            result,
            entry=entry,
            error=error,
        )
    return result


def compute_mflops(model_mflop_per_request: Any, latency_s: Any) -> float:
    """Calculate MFLOPS for numeric model work and a positive latency."""
    mflop = _to_float_or_nan(model_mflop_per_request)
    latency = _to_float_or_nan(latency_s)
    if mflop == mflop and latency == latency and latency > 0:
        return mflop / latency
    return float("nan")

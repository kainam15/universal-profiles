"""Shared helpers for reading and applying a compute profile plan.

Each device class contains independent profiler entries so a failure in one
tool never hides results from another one.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Tuple


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
    for profile_key, tool_profiles in profiles.items():
        if not isinstance(tool_profiles, dict):
            return {
                "profiles": {},
                "_load_error": (
                    "compute_profile_plan_invalid:"
                    f"invalid_profile:{profile_key}"
                ),
            }
        if tool_profiles and not any(
            isinstance(tool_profiles.get(tool), dict)
            for tool in (TORCH_PROFILE_KEY, NCU_PROFILE_KEY, "intel_advisor")
        ):
            return {
                "profiles": {},
                "_load_error": (
                    "compute_profile_plan_invalid:"
                    f"unsupported_profile_layout:{profile_key}"
                ),
            }
    return plan


def _empty_result() -> Dict[str, Any]:
    return {
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


def _resolve_profile_entry(
    profile: Any,
    *,
    profile_key: str,
    input_scale: float,
) -> Tuple[Dict[str, Any], str]:
    """Return ``(matching_entry, diagnostic)`` for one tool profile."""
    if not isinstance(profile, dict):
        return (
            {},
            f"compute_profile_missing_profile:{profile_key}",
        )

    profile_error = str(profile.get("error") or "")
    entries = profile.get("entries")
    if not isinstance(entries, list):
        return (
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
                entry,
                entry_error,
            )

    return (
        {},
        f"compute_profile_missing_scale:{input_scale:g}",
    )


def _apply_torch_profile(
    result: Dict[str, Any],
    *,
    entry: Dict[str, Any],
    error: str,
) -> None:
    logical_mflop = _to_float_or_nan(entry.get(TORCH_LOGICAL_MFLOP_FIELD))
    result.update(
        {
            TORCH_LOGICAL_MFLOP_FIELD: logical_mflop,
            TORCH_ERROR_FIELD: error,
        }
    )


def _apply_ncu_profile(
    result: Dict[str, Any],
    *,
    entry: Dict[str, Any],
    error: str,
) -> None:
    total_mflop = _to_float_or_nan(entry.get(NCU_TOTAL_MFLOP_FIELD))
    tensor_mflop = _to_float_or_nan(entry.get(NCU_TENSOR_MFLOP_FIELD))
    scalar_mflop = _to_float_or_nan(entry.get(NCU_SCALAR_MFLOP_FIELD))
    tensor_share_pct = _to_float_or_nan(entry.get(NCU_TENSOR_SHARE_FIELD))
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
            NCU_KERNEL_COUNT_FIELD: _to_float_or_nan(
                entry.get(NCU_KERNEL_COUNT_FIELD)
            ),
            NCU_KERNEL_TIME_FIELD: _to_float_or_nan(
                entry.get(NCU_KERNEL_TIME_FIELD)
            ),
            NCU_ERROR_FIELD: error,
        }
    )


def find_compute_profile_entry(
    plan: Dict[str, Any],
    gpu_mode: str,
    input_scale: float,
) -> Dict[str, Any]:
    """Resolve both FLOP profiles for one device/scale."""
    profile_key = "gpu" if gpu_mode == "on" else "cpu"
    result = _empty_result()
    load_error = str(plan.get("_load_error", "") or "")
    if load_error:
        result[TORCH_ERROR_FIELD] = load_error
        if profile_key == "gpu":
            result[NCU_ERROR_FIELD] = load_error
        return result

    profile = plan.get("profiles", {}).get(profile_key)
    if profile is None:
        return result
    if not isinstance(profile, dict) or (
        profile
        and not any(
            isinstance(profile.get(tool), dict)
            for tool in (TORCH_PROFILE_KEY, NCU_PROFILE_KEY, "intel_advisor")
        )
    ):
        layout_error = f"compute_profile_unsupported_profile_layout:{profile_key}"
        result[TORCH_ERROR_FIELD] = layout_error
        if profile_key == "gpu":
            result[NCU_ERROR_FIELD] = layout_error
        return result

    torch_profile = profile.get(TORCH_PROFILE_KEY)
    if isinstance(torch_profile, dict):
        torch_entry, torch_error = _resolve_profile_entry(
            torch_profile,
            profile_key=profile_key,
            input_scale=input_scale,
        )
        _apply_torch_profile(
            result,
            entry=torch_entry,
            error=torch_error,
        )

    # NCU measures GPU execution and is intentionally not applicable to
    # CPU-only result rows.
    ncu_profile = profile.get(NCU_PROFILE_KEY)
    if profile_key == "gpu" and isinstance(ncu_profile, dict):
        ncu_entry, ncu_error = _resolve_profile_entry(
            ncu_profile,
            profile_key=profile_key,
            input_scale=input_scale,
        )
        _apply_ncu_profile(
            result,
            entry=ncu_entry,
            error=ncu_error,
        )
    return result


def compute_mflops(model_mflop_per_request: Any, latency_s: Any) -> float:
    """Calculate MFLOPS for numeric model work and a positive latency."""
    mflop = _to_float_or_nan(model_mflop_per_request)
    latency = _to_float_or_nan(latency_s)
    if mflop == mflop and latency == latency and latency > 0:
        return mflop / latency
    return float("nan")

"""Shared helpers for reading and applying a compute profile plan."""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict


INPUT_SCALE_ABS_TOLERANCE = 1e-6


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
    return plan


def find_compute_profile_entry(
    plan: Dict[str, Any],
    gpu_mode: str,
    input_scale: float,
) -> Dict[str, Any]:
    """Resolve one CPU/GPU plan entry with the same rules as the live client."""
    profile_key = "gpu" if gpu_mode == "on" else "cpu"
    load_error = str(plan.get("_load_error", "") or "")
    if load_error:
        return {
            "tool": "nan",
            "model_mflop_per_request": float("nan"),
            "error": load_error,
        }

    profile = plan.get("profiles", {}).get(profile_key)
    if not isinstance(profile, dict):
        return {
            "tool": "nan",
            "model_mflop_per_request": float("nan"),
            "error": f"compute_profile_missing_profile:{profile_key}",
        }

    tool = str(profile.get("tool") or "nan")
    profile_error = str(profile.get("error") or "")
    entries = profile.get("entries")
    if not isinstance(entries, list):
        return {
            "tool": tool,
            "model_mflop_per_request": float("nan"),
            "error": profile_error or f"compute_profile_missing_entries:{profile_key}",
        }

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
            return {
                "tool": str(entry.get("tool") or tool or "nan"),
                "model_mflop_per_request": _to_float_or_nan(
                    entry.get("model_mflop_per_request")
                ),
                "error": str(entry.get("error") or profile_error),
            }

    return {
        "tool": tool,
        "model_mflop_per_request": float("nan"),
        "error": f"compute_profile_missing_scale:{input_scale:g}",
    }


def compute_mflops(model_mflop_per_request: Any, latency_s: Any) -> float:
    """Calculate MFLOPS for numeric model work and a positive latency."""
    mflop = _to_float_or_nan(model_mflop_per_request)
    latency = _to_float_or_nan(latency_s)
    if mflop == mflop and latency == latency and latency > 0:
        return mflop / latency
    return float("nan")

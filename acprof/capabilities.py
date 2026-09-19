"""Shared execution/measurement declarations and evidence; no runtime imports.

Capability states describe availability, never metric values. A measured zero
stays a number in the existing CSV; missing measurements stay NaN/null.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    VERIFIED = "verified"
    UNSUPPORTED = "unsupported"
    PERMISSION_DENIED = "permission_denied"
    NOT_REQUESTED = "not_requested"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class Capability:
    status: CapabilityStatus | str
    detail: str = ""
    source: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "status", CapabilityStatus(self.status))

    def __bool__(self):
        raise TypeError("Capability requires an explicit status comparison, not bool()")

    def to_dict(self) -> dict:
        return {"status": self.status.value, "detail": self.detail,
                "source": self.source, "evidence": dict(self.evidence)}


def require_profiling_mode(mode: str) -> str:
    if mode not in {"full", "basic"}:
        raise ValueError(f"Unknown profiling mode: {mode!r}; expected full or basic")
    return mode


def measurement_requested(mode: str, name: str, *, gpu: bool = False) -> bool:
    require_profiling_mode(mode)
    if name in {"latency", "throughput", "container_cpu", "container_memory"}:
        return True
    if name == "gpu_power":
        return mode == "full" and gpu
    return mode == "full" and name in {"cpu_energy", "cpu_instructions", "packet_latency"}


@dataclass
class CapabilityReport:
    profiling_mode: str
    execution: dict[str, Capability] = field(default_factory=dict)
    measurement: dict[str, Capability] = field(default_factory=dict)
    requested: set[str] = field(default_factory=set)
    collection_complete: bool = False

    def __post_init__(self):
        require_profiling_mode(self.profiling_mode)

    def to_dict(self) -> dict:
        complete = bool(self.requested) and all(name in self.measurement and self.measurement[name].status in {
            CapabilityStatus.AVAILABLE, CapabilityStatus.VERIFIED,
        } for name in self.requested)
        return {
            "schema_version": 1, "profiling_mode": self.profiling_mode,
            "execution": {name: item.to_dict() for name, item in self.execution.items()},
            "measurement": {name: item.to_dict() for name, item in self.measurement.items()},
            "requested_measurements": sorted(self.requested),
            "requested_measurements_complete": complete,
            "collection_complete": self.collection_complete,
            "full_profile_complete": self.profiling_mode == "full" and complete and self.collection_complete,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CapabilityReport":
        report = cls(payload.get("profiling_mode", "full"))
        for group in ("execution", "measurement"):
            setattr(report, group, {name: Capability(**item) for name, item in payload.get(group, {}).items()})
        report.requested = set(payload.get("requested_measurements", ()))
        report.collection_complete = bool(payload.get("collection_complete", False))
        return report


def capability_from_error(error: str | BaseException, *, source: str = "") -> Capability:
    detail = str(error)
    text = detail.lower()
    if isinstance(error, PermissionError) or any(token in text for token in (
        "permission", "access denied", "operation not permitted", "err_nvgpuctrperm",
        "performance monitoring is limited", "password is required",
    )):
        status = CapabilityStatus.PERMISSION_DENIED
    elif any(token in text for token in ("unsupported", "not supported", "not applicable")):
        status = CapabilityStatus.UNSUPPORTED
    elif isinstance(error, (FileNotFoundError, ModuleNotFoundError)) or any(token in text for token in (
        "not_found", "not found", "unavailable", "no such file", "no module named",
    )):
        status = CapabilityStatus.UNAVAILABLE
    else:
        status = CapabilityStatus.ERROR
    return Capability(status, detail=detail, source=source)


PROFILER_MEASUREMENTS = {
    "torch_profiler_eager": "logical_flops", "ncu": "gpu_flops",
    "intel_advisor": "cpu_flops", "massif": "cpu_heap", "nsys": "gpu_timeline",
}

PROFILER_PRIMARY_FIELDS = {
    "torch_profiler_eager": "model_logical_mflop_per_request_torch_profiler_eager",
    "ncu": "gpu_executed_mflop_per_request_ncu",
    "intel_advisor": "model_mflop_per_request",
    "massif": "cpu_heap_peak_bytes_massif",
    "nsys": "host_inference_wall_time_ms_per_request_nsys",
}


def declared_profiler_error(task_info: Any, tool: str) -> str:
    """Shared by main and post-hoc plans, before launching any probe."""
    from acprof.extensions import select_extension
    extension = select_extension(task_info)
    name = PROFILER_MEASUREMENTS.get(tool, tool)
    if extension.measurement.get(tool, extension.measurement.get(name)) == "unsupported":
        return f"unsupported: profiler {tool} for extension {extension.extension_id}"
    return ""


def measurement_report(mode: str, *, gpu_modes=(), compute_tool="none", execution_tool="none") -> CapabilityReport:
    report = CapabilityReport(mode)
    gpu = "on" in gpu_modes
    cpu = "off" in gpu_modes
    for name in ("latency", "throughput", "container_cpu", "container_memory",
                 "packet_latency", "cpu_energy", "cpu_instructions", "gpu_power"):
        requested = measurement_requested(mode, name, gpu=gpu)
        report.measurement[name] = Capability(
            "available" if requested else "not_requested",
            source="host_preflight" if requested else "profiling_mode",
        )
        if requested:
            report.requested.add(name)
    selection = {
        "logical_flops": compute_tool in {"torch", "both"},
        "gpu_flops": compute_tool in {"ncu", "both", "vendor"},
        "cpu_flops": compute_tool == "vendor",
        "cpu_heap": execution_tool in {"massif", "both"},
        "gpu_timeline": execution_tool in {"nsys", "both"},
    }
    for name, selected in selection.items():
        applicable = gpu if name in {"gpu_flops", "gpu_timeline"} else cpu if name in {"cpu_flops", "cpu_heap"} else bool(gpu_modes)
        report.measurement[name] = Capability(
            "not_requested" if not selected else "unavailable" if applicable else "unsupported",
            detail="awaiting profiler evidence" if selected and applicable else "",
            source="profiler_selection",
        )
        if selected:
            report.requested.add(name)
    return report


def apply_extension(report: CapabilityReport, extension: Any) -> None:
    for name, status in extension.execution.items():
        report.execution[name] = Capability(status, source="extension_manifest",
                                            evidence={"extension_id": extension.extension_id})
    for tool, status in extension.measurement.items():
        name = PROFILER_MEASUREMENTS.get(tool, tool)
        if name in report.requested and status == "unsupported":
            report.measurement[name] = Capability(status, source="extension_manifest")


def apply_runtime_validation(report: CapabilityReport, validation: Mapping, *, environment_id="") -> None:
    for device, result in validation.get("devices", {}).items():
        name = {"off": "cpu", "on": "cuda"}.get(device, device)
        evidence = {"environment_id": environment_id, **result}
        if result.get("status") == "ok":
            capability = Capability("verified", source="runtime_probe", evidence=evidence)
        elif result.get("status") == "resource_limit":
            capability = Capability("unavailable", "validation resource limit", "runtime_probe", evidence)
        else:
            error = capability_from_error(result.get("error") or "runtime validation failed", source="runtime_probe")
            capability = Capability(error.status, error.detail, error.source, evidence)
        report.execution[name] = capability


def apply_profiler_plan(report: CapabilityReport, plan: Mapping, *, source: str) -> None:
    """Read existing compute/execution plans without altering numeric semantics."""
    groups: dict[str, list[Mapping]] = {}
    profiles = plan.get("profiles", {})
    containers = profiles.values() if isinstance(profiles, dict) else (item.get("tools", {}) for item in profiles)
    for container in containers:
        for tool, profile in container.items():
            if tool in PROFILER_MEASUREMENTS and isinstance(profile, dict):
                groups.setdefault(tool, []).append(profile)
    for tool, profiles_for_tool in groups.items():
        errors = []
        count = 0
        missing_metric = False
        for profile in profiles_for_tool:
            if profile.get("error"):
                errors.append(str(profile["error"]))
            entries = profile.get("entries", [])
            count += len(entries)
            errors.extend(str(entry["error"]) for entry in entries if entry.get("error"))
            for entry in entries:
                try:
                    finite = math.isfinite(float(entry.get(PROFILER_PRIMARY_FIELDS[tool], "nan")))
                except (ValueError, TypeError):
                    finite = False
                missing_metric = missing_metric or not finite
        name = PROFILER_MEASUREMENTS[tool]
        if errors:
            report.measurement[name] = capability_from_error("; ".join(dict.fromkeys(errors)), source=source)
        elif count and not missing_metric:
            report.measurement[name] = Capability("verified", source=source, evidence={"entries": count})
        else:
            report.measurement[name] = Capability("unavailable", "profiler produced no complete finite measurements", source)


REQUIRED_MEASUREMENT_FIELDS = {
    "latency": ("latency_app_s",), "throughput": ("throughput_samples_per_s",),
    "container_cpu": ("container_cpu_util_avg_pct",), "container_memory": ("container_mem_usage_avg_bytes",),
    "packet_latency": ("latency_s",), "cpu_energy": ("cpu_energy_total_j", "vcpu_energy_total_j"),
    "cpu_instructions": ("cpu_instructions_per_request",),
    "gpu_power": ("gpu_avg_power_total_w", "gpu_energy_total_j"),
}


def missing_required_measurements(report: CapabilityReport, rows: list[Mapping]) -> list[str]:
    """Require actual evidence for success rows; explicit failed/OOM cases stay failed."""
    return sorted(name for name in REQUIRED_MEASUREMENT_FIELDS if name in report.requested
                  and any(row.get("status") in {"ok", "warn"}
                          and (name != "gpu_power" or row.get("gpu_mode") == "on") for row in rows)
                  and report.measurement[name].status != CapabilityStatus.VERIFIED)


def apply_collection_result(report: CapabilityReport, rows: list[Mapping]) -> None:
    """Attach evidence after collection; this never changes a measured field."""
    report.collection_complete = bool(rows) and all(row.get("status") in {"ok", "warn"} for row in rows)
    for name, metrics in REQUIRED_MEASUREMENT_FIELDS.items():
        if name not in report.requested:
            continue
        relevant = [row for row in rows if row.get("status") in {"ok", "warn"}
                    and (name != "gpu_power" or row.get("gpu_mode") == "on")]
        def finite(row):
            try:
                return all(math.isfinite(float(row.get(metric, "nan"))) for metric in metrics)
            except (ValueError, TypeError):
                return False
        if relevant and all(finite(row) for row in relevant):
            report.measurement[name] = Capability("verified", source="result_all.csv", evidence={"fields": list(metrics), "rows": len(relevant)})
        else:
            errors = "; ".join(str(row.get("error") or "") for row in rows if row.get("status") == "error")
            report.measurement[name] = (
                capability_from_error(errors, source="result_all.csv") if errors.strip("; ")
                else Capability("unavailable", "no complete finite measurement evidence", "result_all.csv", {"fields": list(metrics)})
            )

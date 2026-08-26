"""AC-Prof Universal Orchestrator - Docker lifecycle, resource constraints, monitoring.

Python replacement for run_case.sh / run_matrix.sh.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from acprof.config import (
    CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
    CSV_FIELDS,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DOCKER_IMAGE_PREFIX,
    HF_MIRROR_ENDPOINT,
    IDLE_DIAG_DIRNAME,
    PYPI_MIRROR_INDEX,
    SCALING_DIMENSIONS,
    SERVER_PORT,
    READY_POLL_INTERVAL_S,
    READY_TIMEOUT_S,
    STATIC_META_FIELDS,
    STATIC_META_SCHEMA_VERSION,
)
from acprof.host.detect import TaskInfo
from acprof.host.env_utils import hf_offline_docker_env_args
from acprof.host.compute_profile_plan import (
    NCU_ERROR_FIELD,
    TORCH_ERROR_FIELD,
)
from acprof.monitors.perf_mips import MIPS_EXIT_CODE


@dataclass
class ImageInfo:
    tag: str


@dataclass
class StaticMeta:
    model_name: str
    model_revision: str
    task_family: str
    pipeline_tag: str
    runtime_backend: str
    image_tag: str
    batch_size: int
    input_scale_type: str
    run_command: str
    model_download_url: str
    gpu: str
    gpu_mem_total_bytes: Optional[int]
    model_cache_bytes: int
    docker_image_bytes: int
    environment: str
    cpu_power_source: str
    vcpu_power_method: str
    cpu_governor: str
    cpu_boost: str
    cgroup_version: str = "unknown"
    cgroup_collection_mode: str = "unknown"
    host_mem_total_bytes: Optional[int] = None
    host_swap_total_bytes: Optional[int] = None
    host_swap_used_bytes_at_start: Optional[int] = None
    host_swap_type: str = "unknown"
    host_vm_swappiness: Optional[int] = None
    docker_storage_total_bytes: Optional[int] = None
    docker_storage_available_bytes_at_start: Optional[int] = None
    docker_storage_filesystem: str = "unknown"
    docker_storage_device: str = "unknown"
    docker_storage_type: str = "unknown"
    workload: Dict[str, Any] = field(default_factory=dict)
    input_scale_plan_sha256: str = ""
    schema_version: int = STATIC_META_SCHEMA_VERSION
    parameter_count: Optional[int] = None
    parameter_bytes: Optional[int] = None
    precision_dtype: Optional[str] = None
    parameter_dtype_counts: Dict[str, int] = field(default_factory=dict)
    inference_precision_by_device: Dict[str, str] = field(default_factory=dict)
    static_flops: Optional[Dict[str, Any]] = None
    static_macs: Optional[Dict[str, Any]] = None
    input_format: Dict[str, Any] = field(default_factory=dict)
    output_format: Dict[str, Any] = field(default_factory=dict)
    quantized: Optional[bool] = None
    quantization_method: Optional[str] = None
    quantization_config: Dict[str, Any] = field(default_factory=dict)
    model_license: Optional[str] = None
    model_metadata_source: Optional[str] = None
    compute_profile_tools: List[str] = field(default_factory=list)
    torch_profiler_eager_flop_semantics: str = ""
    torch_profiler_eager_attention_implementation: str = ""
    torch_profiler_eager_repeat_cpu: Optional[int] = None
    torch_profiler_eager_repeat_gpu: Optional[int] = None
    ncu_flop_semantics: str = ""
    ncu_repeat: Optional[int] = None
    ncu_fma_flop_weight: Optional[float] = None
    ncu_metrics: List[str] = field(default_factory=list)
    torch_version: str = ""
    transformers_version: str = ""
    ncu_version: str = ""
    gpu_compute_capability: str = ""
    gpu_sm_count: Any = None
    compute_profiles_retained: bool = False
    compute_profile_provenance: str = ""
    execution_profile_schema_version: Optional[int] = None
    execution_profile_tools: List[str] = field(default_factory=list)
    massif_peak_semantics: str = ""
    massif_repeat: Optional[int] = None
    massif_version: str = ""
    massif_sampling_strategy: str = ""
    massif_reference_cpu_cores: Optional[int] = None
    massif_reference_mem_cap_gb: Optional[int] = None
    massif_reused_across_resource_cases: bool = False
    nsys_timeline_semantics: str = ""
    nsys_repeat: Optional[int] = None
    nsys_version: str = ""
    nsys_sampling_strategy: str = ""
    nsys_reference_cpu_cores: Optional[int] = None
    nsys_reference_mem_cap_gb: Optional[int] = None
    nsys_reused_across_resource_cases: bool = False
    execution_profiles_retained: bool = False
    execution_profile_provenance: str = ""


@dataclass
class RunningContainer:
    name: str
    base_url: str
    host_port: int
    cold_start_s: float
    cold_start_started_at: str = "nan"
    cold_start_ready_at: str = "nan"
    cold_start_container_launch_s: float = float("nan")
    cold_start_server_setup_s: float = float("nan")
    cold_start_cuda_init_s: float = float("nan")
    cold_start_model_load_s: float = float("nan")
    cold_start_ready_wait_s: float = float("nan")


@dataclass
class PlannedInputScales:
    scales: List[float]
    source: str
    plan_file: Optional[str] = None
    workload: Dict[str, Any] = field(default_factory=dict)
    plan_sha256: str = ""


@dataclass(frozen=True)
class MatrixProgress:
    """One safely completed resource case in a matrix sweep."""

    completed_cases: int
    total_cases: int
    cpu: int
    mem: int
    gpu: str
    result_csv: Optional[str]


@dataclass
class PacketLatencyRuntime:
    mode: str
    tcpdump_cmd: List[str]
    parse_cmd: List[str]


class PacketLatencyError(RuntimeError):
    """Raised when required packet-level latency cannot be collected."""


class EnergyProfilingError(RuntimeError):
    """Raised when energy profiling cannot continue reliably."""


class MIPSProfilingError(RuntimeError):
    """Raised when required MIPS profiling cannot continue reliably."""


IDLE_POWER_RELATIVE_RANGE_THRESHOLD = 0.05
CPU_SYSFS_ROOT = "/sys/devices/system/cpu"
AUTO_INPUT_SCALE_COUNT = 6
DEFAULT_NLP_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
CUDA124_NLP_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
DEFAULT_NLP_TORCH_SPEC = "torch>=2.7"
CUDA124_NLP_TORCH_SPEC = "torch>=2.6,<2.7"
TCPDUMP_CAPTURE_CAPABILITY = "cap_net_raw,cap_net_admin=eip"
STARTUP_OOM_PRUNING_PLAN_NAME = "startup_oom_pruning.json"
PACKET_LATENCY_RECOVERY_STEPS = (
    "Recovery steps:\n"
    "  1. Install packet tools: sudo apt-get install -y tcpdump tshark\n"
    "  2. Grant capture capability: sudo setcap "
    f"{TCPDUMP_CAPTURE_CAPABILITY} $(command -v tcpdump)\n"
    "  3. Verify capability: getcap $(command -v tcpdump)\n"
    "  4. Verify Docker bridge: ip link show docker0\n"
    "  5. If your bridge differs, pass --sniff-iface <iface>."
)


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in Docker image tags and file names."""
    return model_id.replace("/", "--").replace(".", "_").lower()


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _format_scale_value(scale: float) -> str:
    value = float(scale)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def serialize_input_scales(scales: List[float]) -> str:
    return ",".join(_format_scale_value(scale) for scale in scales)


def resolve_input_scales(task_family: str, input_scales: Optional[str] = None) -> List[float]:
    if input_scales:
        return sorted(set(_parse_float_list(input_scales)))

    scaling_cfg = SCALING_DIMENSIONS.get(task_family)
    if scaling_cfg:
        return sorted(set(float(v) for v in scaling_cfg.values))
    return [1.0]


def _scale_plan_file_path(output_dir: str) -> str:
    return os.path.join(output_dir, "input_scale_plan.json")


def _clear_scale_plan_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _write_scale_plan_file(
    path: str,
    task_info: TaskInfo,
    entries: List[Dict[str, Any]],
    *,
    workload: Optional[Dict[str, Any]] = None,
    model_constraints: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "schema_version": 2,
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "workload": dict(workload or {}),
        "model_constraints": dict(model_constraints or {}),
        "entries": entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _materialize_scale_plan(
    *,
    task_info: TaskInfo,
    scales: List[float],
    batch_size: int,
    output_dir: str,
    source: str,
    workload_spec_path: Optional[str] = None,
    model_constraints: Optional[Dict[str, Any]] = None,
) -> PlannedInputScales:
    """Generate one reusable payload plan for non-NLP task families."""
    from acprof.workloads import get_generator

    workload_gen = get_generator(
        task_info.task_family,
        task_info.model_id,
        task_info.pipeline_tag,
        batch_size,
        workload_spec_path=workload_spec_path,
    )
    entries: List[Dict[str, Any]] = []
    effective_scales: List[float] = []

    for scale in scales:
        requested_scale = float(scale)
        payload = workload_gen.generate(requested_scale)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"workload generator returned a non-object payload for "
                f"task_family={task_info.task_family}, "
                f"input_scale={_format_scale_value(requested_scale)}"
            )

        effective_scale = workload_gen.effective_input_scale(
            requested_scale,
            payload,
        )
        if effective_scale is None:
            raise RuntimeError(
                f"cannot determine effective input scale for "
                f"task_family={task_info.task_family}, "
                f"input_scale={_format_scale_value(requested_scale)}"
            )

        actual_scale = float(effective_scale)
        if effective_scales and actual_scale <= effective_scales[-1]:
            raise RuntimeError(
                f"input scale plan is not strictly increasing for "
                f"task_family={task_info.task_family}: "
                f"{_format_scale_value(effective_scales[-1])}, "
                f"{_format_scale_value(actual_scale)}"
            )

        effective_scales.append(actual_scale)
        input_metadata: Dict[str, Any] = {}
        metadata_fn = getattr(workload_gen, "input_metadata", None)
        if callable(metadata_fn):
            candidate_metadata = metadata_fn(actual_scale, payload)
            if candidate_metadata is not None:
                if not isinstance(candidate_metadata, dict):
                    raise RuntimeError(
                        "workload generator returned non-object input metadata "
                        f"for input_scale={_format_scale_value(actual_scale)}"
                    )
                input_metadata = candidate_metadata

        entries.append({
            "input_scale": actual_scale,
            "scale_label": workload_gen.scale_label(actual_scale),
            "input_metadata": input_metadata,
            "payload": payload,
        })

    workload_metadata: Dict[str, Any] = {}
    metadata_fn = getattr(workload_gen, "plan_metadata", None)
    if callable(metadata_fn):
        candidate_metadata = metadata_fn()
        if candidate_metadata is not None:
            if not isinstance(candidate_metadata, dict):
                raise RuntimeError("workload generator returned non-object plan metadata")
            workload_metadata = candidate_metadata

    plan_file = _scale_plan_file_path(output_dir)
    constraints = dict(model_constraints or {})
    plan_sha256 = _write_scale_plan_file(
        plan_file,
        task_info,
        entries,
        workload=workload_metadata,
        model_constraints=constraints,
    )
    static_workload = dict(workload_metadata)
    if constraints:
        static_workload["model_constraints"] = constraints
    return PlannedInputScales(
        scales=effective_scales,
        source=source,
        plan_file=plan_file,
        workload=static_workload,
        plan_sha256=plan_sha256,
    )


def _integer_auto_scales(max_value: int, count: int = AUTO_INPUT_SCALE_COUNT) -> List[float]:
    if max_value < count:
        raise RuntimeError(
            f"cannot generate {count} unique integer input scales from max_value={max_value}"
        )

    scales = [max(1, math.floor(max_value * idx / count)) for idx in range(1, count + 1)]
    scales[-1] = int(max_value)
    if len(set(scales)) != count:
        raise RuntimeError(
            f"failed to derive {count} unique integer input scales from max_value={max_value}: {scales}"
        )
    return [float(v) for v in scales]


def _float_auto_scales(max_value: float, count: int = AUTO_INPUT_SCALE_COUNT) -> List[float]:
    if max_value <= 0:
        raise RuntimeError(f"invalid max_value for float scales: {max_value}")
    scales = [round(float(max_value) * idx / count, 6) for idx in range(1, count + 1)]
    scales[-1] = round(float(max_value), 6)
    return scales


def _run(cmd: List[str], check: bool = True, capture: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with error handling."""
    print(f"  [cmd] {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=check,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )


def _inspect_container_state(container_name: str) -> Optional[Dict[str, Any]]:
    """Return Docker's runtime state without flooding readiness logs."""
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                container_name,
                "--format",
                "{{json .State}}",
            ],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        state = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _container_startup_exit_error(
    container_name: str,
    memory_limit_gb: int,
) -> Optional[str]:
    """Describe a container that exited while the server was starting."""
    state = _inspect_container_state(container_name)
    if not state:
        return None

    status = str(state.get("Status") or "").strip().lower()
    running = bool(state.get("Running"))
    restarting = bool(state.get("Restarting"))
    oom_killed = bool(state.get("OOMKilled"))
    if running or restarting:
        return None
    if not oom_killed and status not in {"dead", "exited", "removing"}:
        return None

    try:
        exit_code = int(state.get("ExitCode"))
    except (TypeError, ValueError):
        exit_code = -1
    docker_error = str(state.get("Error") or "").strip()
    detail = (
        f"container={container_name}, memory_limit={memory_limit_gb}g, "
        f"status={status or 'unknown'}, exit_code={exit_code}"
    )
    if docker_error:
        detail += f", docker_error={docker_error}"
    if oom_killed:
        return f"container_oom_killed during startup ({detail})"
    return f"container_exited_before_ready ({detail})"


def _container_runtime_oom_error(
    container_name: str,
    memory_limit_gb: int,
    client_exit_code: int,
) -> Optional[str]:
    """Describe a workload-time cgroup OOM reported by Docker.

    Client-side monitors can observe a dead container before the orchestrator
    does and consequently return a monitor-specific exit code. Docker's
    explicit ``OOMKilled`` state is stronger evidence, so callers must consult
    it before classifying a non-zero client exit as a profiler failure.
    """
    state = _inspect_container_state(container_name)
    if not state or not bool(state.get("OOMKilled")):
        return None

    status = str(state.get("Status") or "").strip().lower()
    try:
        container_exit_code = int(state.get("ExitCode"))
    except (TypeError, ValueError):
        container_exit_code = -1
    docker_error = str(state.get("Error") or "").strip()
    detail = (
        "container_runtime_oom: docker_oom_killed=true; "
        "measurement_row_completed=false; planned_request_attempted=unknown; "
        f"container={container_name}; memory_limit_gb={memory_limit_gb}; "
        f"container_status={status or 'unknown'}; "
        f"container_exit_code={container_exit_code}; "
        f"client_exit_code={client_exit_code}"
    )
    if docker_error:
        detail += f"; docker_error={docker_error}"
    return detail


def _url_host(url: str) -> str:
    """Extract host from a URL for pip trusted-host."""
    parsed = urlparse(url)
    return parsed.netloc or parsed.path


def _build_model_download_url(model_id: str) -> str:
    """Return the canonical Hugging Face model URL."""
    return f"https://huggingface.co/{model_id}"


def _normalize_gpu_mode(gpu: str) -> str:
    return "on" if str(gpu).lower() == "on" else "off"


def _format_watts(values: List[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def _parse_csv_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nonnegative_float_or_nan(value: Any) -> float:
    parsed = _parse_csv_float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else float("nan")


def _iso_from_epoch(epoch_s: float) -> str:
    if not math.isfinite(epoch_s):
        return "nan"
    return datetime.datetime.fromtimestamp(
        epoch_s,
        tz=datetime.timezone.utc,
    ).astimezone().isoformat(timespec="milliseconds")


def _cold_start_breakdown(
    body: Optional[Dict[str, Any]],
    docker_started_at_epoch_s: float,
    ready_received_at_epoch_s: float,
) -> Dict[str, Any]:
    timing = body.get("startup_timing", {}) if isinstance(body, dict) else {}
    if not isinstance(timing, dict):
        timing = {}

    process_started_at = _parse_csv_float(
        timing.get("server_process_started_at_epoch_s")
    )
    model_load_completed_at = _parse_csv_float(
        timing.get("model_load_completed_at_epoch_s")
    )
    container_launch_s = (
        process_started_at - docker_started_at_epoch_s
        if math.isfinite(process_started_at)
        and process_started_at >= docker_started_at_epoch_s
        else float("nan")
    )
    ready_wait_s = (
        ready_received_at_epoch_s - model_load_completed_at
        if math.isfinite(model_load_completed_at)
        and ready_received_at_epoch_s >= model_load_completed_at
        else float("nan")
    )
    model_load_s = _nonnegative_float_or_nan(timing.get("model_load_s"))
    if not math.isfinite(model_load_s) and isinstance(body, dict):
        model_load_s = _nonnegative_float_or_nan(body.get("load_time_s"))

    return {
        "cold_start_started_at": _iso_from_epoch(docker_started_at_epoch_s),
        "cold_start_ready_at": _iso_from_epoch(ready_received_at_epoch_s),
        "cold_start_container_launch_s": container_launch_s,
        "cold_start_server_setup_s": _nonnegative_float_or_nan(
            timing.get("server_setup_s")
        ),
        "cold_start_cuda_init_s": _nonnegative_float_or_nan(
            timing.get("cuda_init_s")
        ),
        "cold_start_model_load_s": model_load_s,
        "cold_start_ready_wait_s": ready_wait_s,
    }


def _cold_start_client_env(session: RunningContainer) -> Dict[str, str]:
    return {
        "COLD_START_STARTED_AT": session.cold_start_started_at,
        "COLD_START_READY_AT": session.cold_start_ready_at,
        "COLD_START_CONTAINER_LAUNCH_S": str(
            session.cold_start_container_launch_s
        ),
        "COLD_START_SERVER_SETUP_S": str(session.cold_start_server_setup_s),
        "COLD_START_CUDA_INIT_S": str(session.cold_start_cuda_init_s),
        "COLD_START_MODEL_LOAD_S": str(session.cold_start_model_load_s),
        "COLD_START_READY_WAIT_S": str(session.cold_start_ready_wait_s),
        "COLD_START_S": f"{session.cold_start_s:.6f}",
    }


def _row_has_error_status(row: Dict[str, Any]) -> bool:
    return str(row.get("status") or "").strip().lower() == "error"


def _check_idle_power_values_stable(
    *,
    csv_path: str,
    metric_name: str,
    idle_values: List[float],
    invalid_rows: int,
    row_count: int,
    threshold: float,
    remediation: str,
    skip_when_no_rows: bool = False,
) -> None:
    if row_count == 0 and skip_when_no_rows:
        return
    if invalid_rows or not idle_values:
        raise EnergyProfilingError(
            f"{metric_name} case validation failed: "
            f"csv={csv_path}, valid_rows={len(idle_values)}, invalid_rows={invalid_rows}. "
            f"This case's energy data is not reliable. {remediation}"
        )

    mean_idle = sum(idle_values) / len(idle_values)
    if mean_idle <= 0.0:
        raise EnergyProfilingError(
            f"{metric_name} case validation failed: idle baseline mean is not positive. "
            f"csv={csv_path}. This case's energy data is not reliable. {remediation}"
        )

    relative_range = (max(idle_values) - min(idle_values)) / mean_idle
    if relative_range >= threshold:
        print(
            f"[energy][WARN] {metric_name} case check warning: "
            f"csv={csv_path}, {metric_name}={_format_watts(idle_values)} W, "
            f"min={min(idle_values):.3f} W, max={max(idle_values):.3f} W, "
            f"mean={mean_idle:.3f} W, relative_range={relative_range * 100.0:.1f}%, "
            f"threshold={threshold * 100.0:.1f}%. This case's energy data may be "
            f"noisy; experiment will continue. {remediation}"
        )
        return

    print(
        f"[energy] {metric_name} case check passed: "
        f"rows={len(idle_values)}, min={min(idle_values):.3f} W, "
        f"max={max(idle_values):.3f} W, mean={mean_idle:.3f} W, "
        f"relative_range={relative_range * 100.0:.1f}%",
    )


def _check_case_gpu_idle_power_stable(
    csv_path: str,
    threshold: float = IDLE_POWER_RELATIVE_RANGE_THRESHOLD,
    ignore_error_rows: bool = False,
) -> None:
    """Validate that GPU idle baseline did not drift across a finished case CSV."""
    if not os.path.exists(csv_path):
        raise EnergyProfilingError(
            f"gpu_idle_power_w case validation failed: result CSV does not exist: {csv_path}"
        )

    gpu_rows = 0
    invalid_rows = 0
    idle_values: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if ignore_error_rows and _row_has_error_status(row):
                continue
            if _normalize_gpu_mode(row.get("gpu_mode", "off")) != "on":
                continue
            gpu_rows += 1
            gpu_idle_power_w = _parse_csv_float(
                row.get("gpu_idle_power_w", row.get("idle_power_w"))
            )
            if not math.isfinite(gpu_idle_power_w) or gpu_idle_power_w <= 0.0:
                invalid_rows += 1
                continue
            idle_values.append(gpu_idle_power_w)

    _check_idle_power_values_stable(
        csv_path=csv_path,
        metric_name="gpu_idle_power_w",
        idle_values=idle_values,
        invalid_rows=invalid_rows,
        row_count=gpu_rows,
        threshold=threshold,
        remediation=(
            "Increase --idle-seconds, close other GPU processes, wait for GPU "
            "clocks/power to stabilize, then rerun."
        ),
        skip_when_no_rows=True,
    )


def _check_case_cpu_idle_power_stable(
    csv_path: str,
    threshold: float = IDLE_POWER_RELATIVE_RANGE_THRESHOLD,
    ignore_error_rows: bool = False,
) -> None:
    """Validate that CPU package idle baseline did not drift across a finished case CSV."""
    if not os.path.exists(csv_path):
        raise EnergyProfilingError(
            f"cpu_idle_power_w case validation failed: result CSV does not exist: {csv_path}"
        )

    rows = 0
    invalid_rows = 0
    idle_values: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if ignore_error_rows and _row_has_error_status(row):
                continue
            rows += 1
            cpu_idle_power_w = _parse_csv_float(row.get("cpu_idle_power_w"))
            if not math.isfinite(cpu_idle_power_w) or cpu_idle_power_w <= 0.0:
                invalid_rows += 1
                continue
            idle_values.append(cpu_idle_power_w)

    _check_idle_power_values_stable(
        csv_path=csv_path,
        metric_name="cpu_idle_power_w",
        idle_values=idle_values,
        invalid_rows=invalid_rows,
        row_count=rows,
        threshold=threshold,
        remediation=(
            "Increase --idle-seconds, close host background processes, wait for "
            "CPU package power to stabilize, then rerun."
        ),
        skip_when_no_rows=ignore_error_rows,
    )


def _host_port(cpu: int, mem: int) -> int:
    return SERVER_PORT + cpu * 100 + mem


def _get_gpu_name(device_index: int = 0) -> str:
    """Detect the host GPU model name for static metadata."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
            gpu_name = pynvml.nvmlDeviceGetName(handle)
        finally:
            pynvml.nvmlShutdown()

        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="ignore")
        gpu_name = str(gpu_name).strip()
        return gpu_name or "unknown"
    except Exception:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "unknown"

    result = _run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        check=False,
    )
    if result.returncode != 0:
        return "unknown"

    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpu_names:
        return "unknown"
    if 0 <= device_index < len(gpu_names):
        return gpu_names[device_index]
    return gpu_names[0]


def _get_gpu_mem_total_bytes(device_index: int = 0) -> Optional[int]:
    """Detect host GPU total VRAM in bytes for static metadata."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        finally:
            pynvml.nvmlShutdown()

        total = int(mem.total)
        return total if total > 0 else None
    except Exception:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    result = _run(
        [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        check=False,
    )
    if result.returncode != 0:
        return None

    memory_mib = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not memory_mib:
        return None

    raw_value = memory_mib[device_index] if 0 <= device_index < len(memory_mib) else memory_mib[0]
    try:
        total = int(float(raw_value) * 1024 ** 2)
    except ValueError:
        return None
    return total if total > 0 else None


def _host_mem_total_bytes() -> Optional[int]:
    """Return total physical host RAM in bytes when the OS exposes it."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * physical_pages
        if total > 0:
            return total
    except (AttributeError, OSError, TypeError, ValueError):
        pass

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("MemTotal:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    total = int(parts[1]) * 1024
                    return total if total > 0 else None
    except (OSError, TypeError, ValueError):
        pass
    return None


def _host_swap_metadata(
    proc_meminfo_path: str = "/proc/meminfo",
    proc_swaps_path: str = "/proc/swaps",
    swappiness_path: str = "/proc/sys/vm/swappiness",
) -> Dict[str, Any]:
    """Snapshot host swap capacity, usage, backing type, and policy."""
    metadata: Dict[str, Any] = {
        "host_swap_total_bytes": None,
        "host_swap_used_bytes_at_start": None,
        "host_swap_type": "unknown",
        "host_vm_swappiness": None,
    }

    try:
        meminfo: Dict[str, int] = {}
        with open(proc_meminfo_path, "r", encoding="utf-8") as f:
            for line in f:
                key, separator, raw_value = line.partition(":")
                if not separator or key not in {"SwapTotal", "SwapFree"}:
                    continue
                parts = raw_value.split()
                if not parts:
                    continue
                value_kib = int(parts[0])
                if value_kib >= 0:
                    meminfo[key] = value_kib * 1024
        total = meminfo.get("SwapTotal")
        free = meminfo.get("SwapFree")
        if total is not None:
            metadata["host_swap_total_bytes"] = total
        if total is not None and free is not None:
            metadata["host_swap_used_bytes_at_start"] = max(0, total - free)
    except (OSError, TypeError, ValueError):
        pass

    swaps_read = False
    swap_types: set[str] = set()
    swap_total_kib = 0
    swap_used_kib = 0
    try:
        with open(proc_swaps_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0].lower() == "filename":
                    continue
                if len(parts) < 4:
                    continue
                filename = parts[0]
                raw_type = parts[1].strip().lower()
                size_kib = int(parts[2])
                used_kib = int(parts[3])
                if size_kib >= 0:
                    swap_total_kib += size_kib
                if used_kib >= 0:
                    swap_used_kib += used_kib
                if os.path.basename(filename).lower().startswith("zram"):
                    swap_types.add("zram")
                elif raw_type == "file":
                    swap_types.add("file")
                elif raw_type == "partition":
                    swap_types.add("partition")
                else:
                    swap_types.add("unknown")
        swaps_read = True
    except (OSError, TypeError, ValueError):
        pass

    if metadata["host_swap_total_bytes"] is None and swaps_read:
        metadata["host_swap_total_bytes"] = swap_total_kib * 1024
    if metadata["host_swap_used_bytes_at_start"] is None and swaps_read:
        metadata["host_swap_used_bytes_at_start"] = swap_used_kib * 1024
    if swaps_read:
        if not swap_types:
            metadata["host_swap_type"] = "none"
        elif len(swap_types) == 1:
            metadata["host_swap_type"] = next(iter(swap_types))
        else:
            metadata["host_swap_type"] = "mixed"

    try:
        with open(swappiness_path, "r", encoding="utf-8") as f:
            swappiness = int(f.read().strip())
        if swappiness >= 0:
            metadata["host_vm_swappiness"] = swappiness
    except (OSError, TypeError, ValueError):
        pass
    return metadata


def _docker_root_dir() -> Optional[str]:
    """Resolve the Docker daemon data directory without assuming a default."""
    try:
        result = _run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    root = str(result.stdout or "").strip()
    return root or None


def _docker_mount_metadata(path: str) -> Tuple[str, str]:
    """Return the source device and filesystem containing ``path``."""
    findmnt = shutil.which("findmnt")
    if not findmnt:
        return "unknown", "unknown"
    try:
        result = _run(
            [
                findmnt,
                "--json",
                "--target",
                path,
                "--output",
                "SOURCE,FSTYPE",
            ],
            check=False,
        )
    except Exception:
        return "unknown", "unknown"
    if result.returncode != 0:
        return "unknown", "unknown"
    try:
        payload = json.loads(result.stdout)
        filesystems = payload.get("filesystems", [])
        filesystem = filesystems[0] if isinstance(filesystems, list) and filesystems else {}
        if not isinstance(filesystem, dict):
            return "unknown", "unknown"
        source = str(filesystem.get("source") or "unknown").strip() or "unknown"
        fs_type = str(filesystem.get("fstype") or "unknown").strip() or "unknown"
        return source, fs_type
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return "unknown", "unknown"


def _block_device_storage_type(device: str) -> str:
    """Classify a block device from kernel-reported transport/rotation data."""
    if not device.startswith("/dev/"):
        return "unknown"
    lsblk = shutil.which("lsblk")
    if not lsblk:
        return "unknown"
    query_device = device.split("[", 1)[0]
    try:
        result = _run(
            [
                lsblk,
                "--json",
                "--output",
                "KNAME,TYPE,PKNAME,ROTA,TRAN",
                query_device,
            ],
            check=False,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    try:
        payload = json.loads(result.stdout)
        devices = payload.get("blockdevices", [])
        block_device = devices[0] if isinstance(devices, list) and devices else {}
        if not isinstance(block_device, dict):
            return "unknown"
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return "unknown"

    transport = str(block_device.get("tran") or "").strip().lower()
    rotational = block_device.get("rota")
    if isinstance(rotational, str):
        normalized = rotational.strip().lower()
        if normalized in {"0", "false", "no"}:
            rotational = False
        elif normalized in {"1", "true", "yes"}:
            rotational = True
        else:
            rotational = None
    elif isinstance(rotational, int) and not isinstance(rotational, bool):
        rotational = bool(rotational) if rotational in {0, 1} else None

    if transport == "nvme":
        return "nvme_ssd"
    if rotational is True:
        return "hdd"
    if rotational is False:
        return "ssd"
    return "unknown"


def _docker_storage_metadata() -> Dict[str, Any]:
    """Snapshot capacity and media metadata for Docker's backing filesystem."""
    metadata: Dict[str, Any] = {
        "docker_storage_total_bytes": None,
        "docker_storage_available_bytes_at_start": None,
        "docker_storage_filesystem": "unknown",
        "docker_storage_device": "unknown",
        "docker_storage_type": "unknown",
    }
    docker_root = _docker_root_dir()
    if not docker_root:
        return metadata

    try:
        usage = shutil.disk_usage(docker_root)
        total = int(usage.total)
        available = int(usage.free)
        metadata["docker_storage_total_bytes"] = total if total > 0 else None
        metadata["docker_storage_available_bytes_at_start"] = (
            available if available >= 0 else None
        )
    except (OSError, TypeError, ValueError):
        pass

    device, fs_type = _docker_mount_metadata(docker_root)
    metadata["docker_storage_device"] = device
    metadata["docker_storage_filesystem"] = fs_type
    if fs_type.lower() in {"tmpfs", "ramfs"}:
        metadata["docker_storage_type"] = "memory"
    else:
        metadata["docker_storage_type"] = _block_device_storage_type(device)
    return metadata


def _parse_cuda_version(raw: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"(\d+)\.(\d+)", str(raw))
    if not match:
        return None

    return int(match.group(1)), int(match.group(2))


def _host_cuda_version() -> Optional[Tuple[int, int]]:
    override = (os.environ.get("ACPROF_HOST_CUDA_VERSION") or "").strip()
    if override:
        return _parse_cuda_version(override)

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    result = _run([nvidia_smi], check=False)
    if result.returncode != 0:
        return None

    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
    if not match:
        return None

    return _parse_cuda_version(match.group(1))


def _select_nlp_torch_index_url() -> str:
    override = (os.environ.get("ACPROF_NLP_TORCH_INDEX_URL") or "").strip()
    if override:
        return override

    cuda_version = _host_cuda_version()
    if cuda_version is None:
        return DEFAULT_NLP_TORCH_INDEX_URL

    if cuda_version >= (12, 8):
        return DEFAULT_NLP_TORCH_INDEX_URL
    if cuda_version >= (12, 4):
        return CUDA124_NLP_TORCH_INDEX_URL
    return DEFAULT_NLP_TORCH_INDEX_URL


def _select_nlp_torch_spec(torch_index_url: Optional[str] = None) -> str:
    override = (os.environ.get("ACPROF_NLP_TORCH_SPEC") or "").strip()
    if override:
        return override

    resolved_index_url = (torch_index_url or _select_nlp_torch_index_url()).rstrip("/")
    if resolved_index_url == CUDA124_NLP_TORCH_INDEX_URL:
        return CUDA124_NLP_TORCH_SPEC
    return DEFAULT_NLP_TORCH_SPEC


def _cpu_power_metadata() -> Tuple[str, str]:
    try:
        from acprof.monitors import energy_cpu

        return (
            energy_cpu.detect_cpu_power_source(),
            energy_cpu.detect_vcpu_power_method(),
        )
    except Exception:
        return "unavailable", "unavailable"


def _read_sysfs_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.readline().strip()
    except OSError:
        return None
    return value or None


def _summarize_cpu_policy_values(values: List[str]) -> str:
    if not values:
        return "unavailable"

    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    if len(counts) == 1:
        return values[0]

    return "mixed:" + ",".join(
        f"{value}={counts[value]}" for value in sorted(counts)
    )


def _detect_cpu_governor() -> str:
    try:
        entries = os.listdir(CPU_SYSFS_ROOT)
    except OSError:
        return "unavailable"

    governors: List[str] = []
    for entry in entries:
        if not re.fullmatch(r"cpu\d+", entry):
            continue
        governor = _read_sysfs_first_line(
            os.path.join(CPU_SYSFS_ROOT, entry, "cpufreq", "scaling_governor")
        )
        if governor:
            governors.append(governor)

    return _summarize_cpu_policy_values(governors)


def _map_boost_flag(value: Optional[str]) -> str:
    if value == "1":
        return "on"
    if value == "0":
        return "off"
    return value or "unavailable"


def _detect_cpu_boost() -> str:
    boost = _read_sysfs_first_line(os.path.join(CPU_SYSFS_ROOT, "cpufreq", "boost"))
    if boost is not None:
        return _map_boost_flag(boost)

    no_turbo = _read_sysfs_first_line(
        os.path.join(CPU_SYSFS_ROOT, "intel_pstate", "no_turbo")
    )
    if no_turbo == "1":
        return "off"
    if no_turbo == "0":
        return "on"
    return "unavailable"


def _cpu_frequency_policy_metadata() -> Tuple[str, str]:
    return _detect_cpu_governor(), _detect_cpu_boost()


def _linux_environment_label() -> str:
    try:
        os_release = platform.freedesktop_os_release()
    except Exception:
        return "linux"

    distro_id = str(os_release.get("ID", "")).strip().lower()
    version_id = str(os_release.get("VERSION_ID", "")).strip().strip('"')
    if not distro_id:
        return "linux"
    if version_id:
        return f"{distro_id}{version_id}"
    return distro_id


def _windows_environment_label() -> str:
    release = str(platform.release()).strip().lower()
    if release in {"10", "11"}:
        return f"windows{release}"
    return "windows"


def _macos_environment_label() -> str:
    release = platform.mac_ver()[0].strip()
    if not release:
        return "macos"
    major = release.split(".", 1)[0]
    if major.isdigit():
        return f"macos{major}"
    return "macos"


def _process_is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True

    try:
        return platform.system() == "Linux" and "microsoft" in platform.release().lower()
    except Exception:
        return False


def _docker_kernel_indicates_wsl() -> bool:
    try:
        result = _run(
            ["docker", "info", "--format", "{{.KernelVersion}}"],
            check=False,
        )
    except Exception:
        return False

    if result.returncode != 0:
        return False
    return "microsoft-standard-wsl" in result.stdout.strip().lower()


def _detect_environment() -> str:
    try:
        system = platform.system()
    except Exception:
        return "unknown"

    if system == "Windows":
        label = _windows_environment_label()
    elif system == "Linux":
        label = _linux_environment_label()
    elif system == "Darwin":
        label = _macos_environment_label()
    else:
        label = str(system).strip().lower() or "unknown"

    if label != "unknown" and (_process_is_wsl() or _docker_kernel_indicates_wsl()):
        return f"{label}+wsl"
    return label


def _tcpdump_can_capture_without_sudo(tcpdump_path: str) -> bool:
    if os.geteuid() == 0:
        return True

    result = _run(["getcap", tcpdump_path], check=False)
    if result.returncode != 0:
        return False

    caps = result.stdout.lower()
    return "cap_net_raw" in caps and "cap_net_admin" in caps


def _packet_latency_error(reason: str, detail: str = "") -> PacketLatencyError:
    parts = [f"packet latency is required but unavailable: {reason}."]
    if detail:
        parts.append(f"Details: {detail}")
    parts.append(PACKET_LATENCY_RECOVERY_STEPS)
    return PacketLatencyError("\n".join(parts))


def _sniff_interface_exists(sniff_iface: str) -> bool:
    if not sniff_iface:
        return False

    sysfs_path = os.path.join("/sys/class/net", sniff_iface)
    if os.path.exists(sysfs_path):
        return True

    ip_cmd = shutil.which("ip")
    if not ip_cmd:
        return False

    result = _run([ip_cmd, "link", "show", sniff_iface], check=False)
    return result.returncode == 0


def require_packet_latency_prerequisites(project_dir: str, sniff_iface: str) -> None:
    """Fail early when native-Linux packet latency cannot be collected."""
    del project_dir  # Kept for compatibility with existing callers.

    missing_tools = [
        name for name in ("tcpdump", "tshark")
        if shutil.which(name) is None
    ]
    if missing_tools:
        raise _packet_latency_error(
            f"missing required command(s): {', '.join(missing_tools)}"
        )

    if not _sniff_interface_exists(sniff_iface):
        raise _packet_latency_error(
            f"network interface {sniff_iface!r} was not found"
        )

    tcpdump_path = shutil.which("tcpdump")
    if not tcpdump_path:
        raise _packet_latency_error("tcpdump was not found")

    if not _ensure_tcpdump_capture_capability(tcpdump_path):
        raise _packet_latency_error(
            f"tcpdump lacks capture capability: {tcpdump_path}"
        )


def _try_set_tcpdump_capture_capability(tcpdump_path: str) -> bool:
    setcap_cmd = ["setcap", TCPDUMP_CAPTURE_CAPABILITY, tcpdump_path]

    if os.geteuid() == 0:
        result = _run(setcap_cmd, check=False)
        return result.returncode == 0

    result = _run(["sudo", "-n", *setcap_cmd], check=False)
    if result.returncode == 0:
        return True

    sudo_password = os.environ.get("ACPROF_SUDO_PASSWORD", "").strip()
    if not sudo_password:
        print(
            "[sniff][WARN] tcpdump lacks capture capability and sudo needs a password. "
            "Set ACPROF_SUDO_PASSWORD in .env.local or run "
            f"`sudo setcap {TCPDUMP_CAPTURE_CAPABILITY} {tcpdump_path}` once.",
            file=sys.stderr,
        )
        return False

    result = _run(
        ["sudo", "-S", "-p", "", *setcap_cmd],
        check=False,
        input=f"{sudo_password}\n",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        print(
            "[sniff][WARN] Failed to grant tcpdump capture capability via sudo. "
            f"Packet latency may remain nan. Details: {detail}",
            file=sys.stderr,
        )
        return False

    return True


def _ensure_tcpdump_capture_capability(tcpdump_path: str) -> bool:
    if _tcpdump_can_capture_without_sudo(tcpdump_path):
        return True

    print(f"[sniff] tcpdump capture capability missing; trying to grant it on {tcpdump_path}")
    if not _try_set_tcpdump_capture_capability(tcpdump_path):
        return False

    if _tcpdump_can_capture_without_sudo(tcpdump_path):
        print("[sniff] tcpdump capture capability is ready")
        return True

    print(
        "[sniff][WARN] setcap finished but tcpdump capability is still unavailable. "
        "Packet latency may remain nan.",
        file=sys.stderr,
    )
    return False


def _resolve_packet_latency_runtime(
    project_dir: str,
    pcap_file: str,
    sniff_iface: str,
) -> Optional[PacketLatencyRuntime]:
    del project_dir  # Packet capture and parsing now always run on the local Linux host.
    local_tcpdump = shutil.which("tcpdump")
    local_tshark = shutil.which("tshark")
    if local_tcpdump and local_tshark:
        tcpdump_cmd = (
            [local_tcpdump]
            if _ensure_tcpdump_capture_capability(local_tcpdump)
            else ["sudo", "-n", "tcpdump"]
        )
        return PacketLatencyRuntime(
            mode="local",
            tcpdump_cmd=tcpdump_cmd + [
                "-i",
                sniff_iface,
                "-s",
                "0",
                "-B",
                "4096",
                "-w",
                pcap_file,
                "tcp",
                "port",
                str(SERVER_PORT),
            ],
            parse_cmd=[
                sys.executable,
                "-m",
                "acprof.packet.sniff_parse_pcap",
                pcap_file,
                str(SERVER_PORT),
            ],
        )

    return None


def _assert_packet_latency_csv_complete(
    csv_path: str,
    *,
    ignore_error_rows: bool = False,
) -> None:
    import csv

    if not os.path.exists(csv_path):
        raise _packet_latency_error(f"result CSV does not exist: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise _packet_latency_error(f"result CSV is empty: {csv_path}")
        if "latency_s" not in reader.fieldnames:
            raise _packet_latency_error(f"latency_s column is missing in {csv_path}")

        total = 0
        missing = 0
        for row in reader:
            if ignore_error_rows and _row_has_error_status(row):
                continue
            total += 1
            raw_value = (row.get("latency_s") or "").strip()
            try:
                value = float(raw_value)
            except Exception:
                missing += 1
                continue

            if not math.isfinite(value) or value <= 0:
                missing += 1

    if total == 0 and not ignore_error_rows:
        raise _packet_latency_error(f"result CSV has no rows: {csv_path}")

    if missing:
        raise _packet_latency_error(
            f"latency_s is missing for {missing}/{total} row(s) in {csv_path}"
        )


def _docker_image_size_bytes(image_tag: str) -> int:
    """Get the local Docker image size in bytes."""
    result = _run(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Size}}"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to inspect image size for {image_tag}: {result.stderr.strip()}")

    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"invalid docker image size for {image_tag}: {result.stdout.strip()!r}"
        ) from exc


def _docker_model_cache_bytes(image_tag: str, cache_root: str = "/models/hf") -> int:
    """Measure unique regular-file logical bytes beneath the model cache root."""
    script = (
        "import os, stat\n"
        f"root = {cache_root!r}\n"
        "if not os.path.isdir(root):\n"
        "    raise SystemExit(f'model cache directory not found: {root}')\n"
        "total = 0\n"
        "seen = set()\n"
        "for dirpath, _, filenames in os.walk(root):\n"
        "    for name in filenames:\n"
        "        path = os.path.join(dirpath, name)\n"
        "        st = os.lstat(path)\n"
        "        if stat.S_ISLNK(st.st_mode):\n"
        "            continue\n"
        "        if not stat.S_ISREG(st.st_mode):\n"
        "            continue\n"
        "        key = (st.st_dev, st.st_ino)\n"
        "        if key in seen:\n"
        "            continue\n"
        "        seen.add(key)\n"
        "        total += st.st_size\n"
        "print(total)\n"
    )
    result = _run(
        ["docker", "run", "--rm", "--entrypoint", "python", image_tag, "-c", script],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to inspect model cache size for {image_tag}: {result.stderr.strip()}"
        )

    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"invalid model cache size for {image_tag}: {result.stdout.strip()!r}"
        ) from exc


def _json_object_schema(
    properties: Dict[str, Any],
    required: List[str],
) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
    }


def _model_io_formats(task_info: TaskInfo) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Describe the actual /predict JSON contract used by the selected handler."""
    string_schema = {"type": "string"}
    params_schema = {"type": "object"}
    input_properties: Dict[str, Any]
    input_required: List[str]
    output_properties: Dict[str, Any]
    output_required = ["task"]

    if task_info.task_family == "nlp":
        if task_info.pipeline_tag == "question-answering":
            input_properties = {
                "question": string_schema,
                "context": string_schema,
                "params": params_schema,
            }
            input_required = ["question", "context"]
        else:
            input_properties = {
                "text": string_schema,
                "params": params_schema,
            }
            input_required = ["text"]
        output_properties = {
            "task": string_schema,
            "output_type": {
                "type": "string",
                "enum": ["text", "label"],
            },
            "n_results": {"type": "integer"},
            "effective_input_scale": {"type": "number"},
        }
        output_required.extend(["output_type", "n_results"])
    elif task_info.task_family == "cv":
        input_properties = {
            "image_base64": {
                "type": "string",
                "contentEncoding": "base64",
                "contentMediaType": "image/png",
            },
            "params": params_schema,
        }
        input_required = ["image_base64"]
        output_properties = {
            "task": string_schema,
            "output_type": {
                "type": "string",
                "enum": ["classification", "detection"],
            },
            "n_results": {"type": "integer"},
        }
        output_required.extend(["output_type", "n_results"])
    elif task_info.task_family == "audio":
        input_properties = {
            "audio_base64": {
                "type": "string",
                "contentEncoding": "base64",
                "contentMediaType": "audio/wav",
            },
            "audio_format": {"type": "string", "enum": ["wav"]},
            "sample_rate": {"type": "integer", "unit": "Hz"},
            "params": params_schema,
        }
        input_required = ["audio_base64", "audio_format", "sample_rate"]
        output_properties = {
            "task": string_schema,
            "output_type": {
                "type": "string",
                "enum": ["transcription", "classification", "unknown"],
            },
            "text": string_schema,
            "output_length": {"type": "integer"},
            "output_token_count": {"type": ["integer", "null"]},
            "n_results": {"type": "integer"},
            "effective_input_scale": {"type": "number"},
        }
        output_required.append("output_type")
    elif task_info.task_family == "timeseries":
        input_properties = {
            "context": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number", "format": "float32"},
                },
            },
            "prediction_length": {"type": "integer"},
        }
        input_required = ["context", "prediction_length"]
        output_properties = {
            "task": string_schema,
            "forecast_shape": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "output_type": {
                "type": "string",
                "enum": ["forecast"],
            },
        }
    elif task_info.task_family == "diffusion":
        input_properties = {
            "prompt": {
                "oneOf": [
                    string_schema,
                    {
                        "type": "array",
                        "items": string_schema,
                        "minItems": 1,
                    },
                ],
            },
            "resolution": {
                "type": "integer",
                "minimum": 64,
                "multipleOf": 8,
                "unit": "px",
            },
            "params": params_schema,
        }
        input_required = ["prompt", "resolution"]
        output_properties = {
            "task": string_schema,
            "output_type": {
                "type": "string",
                "enum": ["image"],
            },
            "n_results": {"type": "integer"},
            "output_length": {"type": "integer"},
            "image_width": {"type": ["integer", "null"], "unit": "px"},
            "image_height": {"type": ["integer", "null"], "unit": "px"},
            "effective_input_scale": {"type": "number"},
        }
        output_required.extend(["output_type", "n_results"])
    else:
        input_properties = {}
        input_required = []
        output_properties = {}
        output_required = []

    common = {
        "transport": "HTTP",
        "media_type": "application/json",
    }
    input_format = {
        **common,
        "method": "POST",
        "endpoint": "/predict",
        "json_schema": _json_object_schema(input_properties, input_required),
    }
    output_format = {
        **common,
        "status": 200,
        "json_schema": _json_object_schema(
            output_properties,
            output_required,
        ),
    }
    return input_format, output_format


def _inference_precision_by_device(task_info: TaskInfo) -> Dict[str, str]:
    if (
        task_info.runtime_backend == "diffusers"
        and task_info.task_family == "diffusion"
    ):
        return {"cpu": "FP32", "gpu": "FP16"}
    if (
        task_info.runtime_backend in {"transformers_pipeline", "transformers_model"}
        and task_info.task_family in {"nlp", "cv", "audio"}
    ):
        return {"cpu": "FP32", "gpu": "FP16"}
    if task_info.precision_dtype:
        return {
            "cpu": task_info.precision_dtype,
            "gpu": task_info.precision_dtype,
        }
    return {}


def collect_static_meta(
    task_info: TaskInfo,
    image_info: ImageInfo,
    batch_size: int,
    input_scale_type: str,
    run_command: str = "",
    device_index: int = 0,
    cgroup_version: str = "unknown",
    cgroup_collection_mode: str = "unknown",
    compute_profile_enabled: bool = True,
    execution_profile_enabled: bool = False,
) -> StaticMeta:
    """Collect static metadata for the current model/image pair."""
    cpu_power_source, vcpu_power_method = _cpu_power_metadata()
    cpu_governor, cpu_boost = _cpu_frequency_policy_metadata()
    host_swap = _host_swap_metadata()
    docker_storage = _docker_storage_metadata()
    input_format, output_format = _model_io_formats(task_info)
    static_meta = StaticMeta(
        model_name=task_info.model_id,
        model_revision=task_info.model_revision,
        parameter_count=task_info.parameter_count,
        parameter_bytes=task_info.parameter_bytes,
        precision_dtype=task_info.precision_dtype,
        parameter_dtype_counts=dict(task_info.parameter_dtype_counts),
        inference_precision_by_device=_inference_precision_by_device(task_info),
        input_format=input_format,
        output_format=output_format,
        quantized=task_info.quantized,
        quantization_method=task_info.quantization_method,
        quantization_config=dict(task_info.quantization_config),
        model_license=task_info.model_license,
        model_metadata_source=task_info.model_metadata_source,
        task_family=task_info.task_family,
        pipeline_tag=task_info.pipeline_tag,
        runtime_backend=task_info.runtime_backend,
        image_tag=image_info.tag,
        batch_size=batch_size,
        input_scale_type=input_scale_type,
        run_command=run_command,
        model_download_url=_build_model_download_url(task_info.model_id),
        gpu=_get_gpu_name(device_index=device_index),
        gpu_mem_total_bytes=_get_gpu_mem_total_bytes(device_index=device_index),
        host_mem_total_bytes=_host_mem_total_bytes(),
        host_swap_total_bytes=host_swap["host_swap_total_bytes"],
        host_swap_used_bytes_at_start=host_swap[
            "host_swap_used_bytes_at_start"
        ],
        host_swap_type=host_swap["host_swap_type"],
        host_vm_swappiness=host_swap["host_vm_swappiness"],
        model_cache_bytes=_docker_model_cache_bytes(image_info.tag),
        docker_image_bytes=_docker_image_size_bytes(image_info.tag),
        docker_storage_total_bytes=docker_storage[
            "docker_storage_total_bytes"
        ],
        docker_storage_available_bytes_at_start=docker_storage[
            "docker_storage_available_bytes_at_start"
        ],
        docker_storage_filesystem=docker_storage[
            "docker_storage_filesystem"
        ],
        docker_storage_device=docker_storage["docker_storage_device"],
        docker_storage_type=docker_storage["docker_storage_type"],
        environment=_detect_environment(),
        cgroup_version=cgroup_version,
        cgroup_collection_mode=cgroup_collection_mode,
        cpu_power_source=cpu_power_source,
        vcpu_power_method=vcpu_power_method,
        cpu_governor=cpu_governor,
        cpu_boost=cpu_boost,
    )
    disabled_metadata: Dict[str, Any] = {}
    if not compute_profile_enabled:
        disabled_metadata.update({
            "compute_profile_tools": [],
            "torch_profiler_eager_flop_semantics": (
                "logical_operator_shape_flops"
            ),
            "torch_profiler_eager_attention_implementation": "eager",
            "ncu_flop_semantics": (
                "gpu_executed_floating_point_operations"
            ),
            "ncu_fma_flop_weight": 2,
            "ncu_metrics": [],
            "torch_version": "unknown",
            "transformers_version": "unknown",
            "ncu_version": "unknown",
            "gpu_compute_capability": "unknown",
            "gpu_sm_count": "unknown",
            "compute_profiles_retained": False,
            "compute_profile_provenance": "disabled",
        })
    if not execution_profile_enabled:
        disabled_metadata.update({
            "execution_profile_schema_version": 1,
            "execution_profile_tools": [],
            "massif_peak_semantics": (
                "process_lifetime_heap_peak_including_model_load_and_warmup"
            ),
            "massif_version": "unknown",
            "nsys_timeline_semantics": "nvtx_acprof_compute_range",
            "nsys_version": "unknown",
            "execution_profiles_retained": False,
            "execution_profile_provenance": "disabled",
        })
    return (
        enrich_static_meta(static_meta, disabled_metadata)
        if disabled_metadata
        else static_meta
    )


def enrich_static_meta(
    static_meta: StaticMeta,
    metadata: Dict[str, Any],
) -> StaticMeta:
    """Return static metadata enriched with recognized profiling fields."""
    updates: Dict[str, Any] = {}
    for field in STATIC_META_FIELDS:
        if field not in metadata:
            continue
        value = metadata[field]
        if isinstance(value, tuple):
            value = list(value)
        updates[field] = value
    return replace(static_meta, **updates) if updates else static_meta


def enrich_static_meta_from_input_plan(
    static_meta: StaticMeta,
    planned: PlannedInputScales,
) -> StaticMeta:
    """Attach the exact workload provenance used to build the payload plan."""
    # CLI orchestration tests and third-party integrations may supply a metadata
    # stand-in while mocking collection. Preserve that compatibility boundary.
    if not isinstance(static_meta, StaticMeta):
        return static_meta
    return replace(
        static_meta,
        workload=dict(planned.workload),
        input_scale_plan_sha256=str(planned.plan_sha256 or ""),
    )


def _static_flops_from_compute_plan(
    plan: Dict[str, Any],
    static_meta: StaticMeta,
) -> Optional[Dict[str, Any]]:
    profiles = plan.get("profiles", {})
    if not isinstance(profiles, dict):
        return None

    for profile_name in ("gpu", "cpu"):
        profile_group = profiles.get(profile_name, {})
        if not isinstance(profile_group, dict):
            continue
        torch_profile = profile_group.get("torch_profiler_eager", {})
        if not isinstance(torch_profile, dict):
            continue
        entries = torch_profile.get("entries", [])
        if not isinstance(entries, list):
            continue

        values: List[Dict[str, Any]] = []
        seen_scales = set()
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("error"):
                continue
            try:
                input_scale = float(entry["input_scale"])
                mflop_per_request = float(
                    entry[
                        "model_logical_mflop_per_request_torch_profiler_eager"
                    ]
                )
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(input_scale)
                or not math.isfinite(mflop_per_request)
                or mflop_per_request < 0
            ):
                continue
            normalized_scale: Any = (
                int(input_scale) if input_scale.is_integer() else input_scale
            )
            if normalized_scale in seen_scales:
                continue
            seen_scales.add(normalized_scale)
            values.append({
                "input_scale": normalized_scale,
                "flops_per_request": int(round(mflop_per_request * 1_000_000)),
            })

        if values:
            values.sort(key=lambda item: float(item["input_scale"]))
            static_metadata = plan.get("static_metadata", {})
            semantics = torch_profile.get("flop_semantics")
            if not semantics and isinstance(static_metadata, dict):
                semantics = static_metadata.get(
                    "torch_profiler_eager_flop_semantics"
                )
            return {
                "source": "torch_profiler_eager",
                "profile": profile_name,
                "semantics": semantics or "logical_operator_shape_flops",
                "unit": "FLOP/request",
                "input_scale_type": static_meta.input_scale_type,
                "batch_size": static_meta.batch_size,
                "values": values,
            }
    return None


def enrich_static_meta_from_compute_plan(
    static_meta: StaticMeta,
    plan_path: str,
) -> StaticMeta:
    """Read compute-profile metadata without making a failed probe fatal."""
    if not plan_path or not os.path.exists(plan_path):
        return static_meta
    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except (OSError, ValueError, TypeError) as exc:
        print(f"[meta][WARN] Cannot read compute profile metadata: {exc}")
        return static_meta
    metadata = plan.get("static_metadata", {})
    if not isinstance(metadata, dict):
        print("[meta][WARN] compute_profile_plan static_metadata is not an object")
        return static_meta
    enriched = enrich_static_meta(static_meta, metadata)
    static_flops = _static_flops_from_compute_plan(plan, enriched)
    if static_flops is not None:
        enriched = replace(enriched, static_flops=static_flops)
    return enriched


def enrich_static_meta_from_execution_plan(
    static_meta: StaticMeta,
    plan_path: str,
) -> StaticMeta:
    """Read execution-profile metadata without making a failed probe fatal."""
    if not plan_path or not os.path.exists(plan_path):
        return static_meta
    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except (OSError, ValueError, TypeError) as exc:
        print(f"[meta][WARN] Cannot read execution profile metadata: {exc}")
        return static_meta
    metadata = plan.get("static_metadata", {})
    if not isinstance(metadata, dict):
        print("[meta][WARN] execution_profile_plan static_metadata is not an object")
        return static_meta
    return enrich_static_meta(static_meta, metadata)


def write_static_meta_json(static_meta: StaticMeta, output_path: str) -> None:
    """Atomically write static metadata as one JSON object."""
    payload = {
        field: getattr(static_meta, field)
        for field in STATIC_META_FIELDS
    }
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        output_mode = (
            stat.S_IMODE(os.stat(output_path).st_mode)
            if os.path.exists(output_path)
            else 0o644
        )
        os.chmod(temporary_path, output_mode)
        os.replace(temporary_path, output_path)
        temporary_path = ""
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    print(f"[meta] Static meta JSON: {output_path}")


# ─────────────────────────────────────────────
# Docker Image Building
# ─────────────────────────────────────────────

def build_image(task_info: TaskInfo, project_dir: str) -> ImageInfo:
    """Build the Docker image for this model's task family.

    Two-stage build:
    1. Build base image (if not exists)
    2. Build task-family image with model weights baked in
    """
    dockerfiles_dir = os.path.join(project_dir, "dockerfiles")
    model_tag = _sanitize_model_id(task_info.model_id)
    base_tag = f"{DOCKER_IMAGE_PREFIX}-base:latest"
    family_tag = f"{DOCKER_IMAGE_PREFIX}-{task_info.task_family}-{model_tag}:latest"

    # Stage 1: Build base image
    print(f"\n[build] Stage 1: Building base image {base_tag} ...")
    base_dockerfile = os.path.join(dockerfiles_dir, "base.Dockerfile")

    result = _run([
        "docker", "build",
        "-f", base_dockerfile,
        "--build-arg", f"HF_ENDPOINT={HF_MIRROR_ENDPOINT}",
        "--build-arg", "HF_FALLBACK_ENDPOINTS=https://huggingface.co",
        "--build-arg", f"PYPI_INDEX_URL={PYPI_MIRROR_INDEX}",
        "--build-arg", f"PYPI_TRUSTED_HOST={_url_host(PYPI_MIRROR_INDEX)}",
        "-t", base_tag,
        project_dir,
    ], check=False)
    if result.returncode != 0:
        print(f"[build] Base image build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Stage 2: Build family-specific image with model
    print(f"\n[build] Stage 2: Building {task_info.task_family} image {family_tag} ...")
    family_dockerfile = os.path.join(dockerfiles_dir, f"{task_info.task_family}.Dockerfile")

    if not os.path.exists(family_dockerfile):
        print(f"[build] Dockerfile not found: {family_dockerfile}", file=sys.stderr)
        sys.exit(1)

    family_build_args = [
        "--build-arg", f"BASE_IMAGE={base_tag}",
        "--build-arg", f"MODEL_ID={task_info.model_id}",
        "--build-arg", f"MODEL_REVISION={task_info.model_revision or 'main'}",
    ]
    if (os.environ.get("HF_TOKEN") or "").strip():
        family_build_args.extend([
            "--secret",
            "id=hf_token,env=HF_TOKEN",
        ])
    if task_info.task_family in {"nlp", "diffusion"}:
        torch_index_url = _select_nlp_torch_index_url()
        torch_spec = _select_nlp_torch_spec(torch_index_url)
        family_build_args.extend([
            "--build-arg",
            f"TORCH_INDEX_URL={torch_index_url}",
            "--build-arg",
            f"TORCH_PACKAGE_SPEC={torch_spec}",
        ])
        family_label = task_info.task_family.upper()
        print(f"[build] {family_label} torch index: {torch_index_url}")
        print(f"[build] {family_label} torch spec:  {torch_spec}")

    result = _run([
        "docker", "build",
        "-f", family_dockerfile,
        *family_build_args,
        "-t", family_tag,
        project_dir,
    ], check=False)
    if result.returncode != 0:
        print(f"[build] Family image build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"[build] Image ready: {family_tag}")
    return ImageInfo(tag=family_tag)


def _start_container_session(
    task_info: TaskInfo,
    cpu: int,
    mem: int,
    gpu: str,
    image_info: ImageInfo,
    container_name: str,
    log_prefix: str,
) -> RunningContainer:
    import requests

    gpu = _normalize_gpu_mode(gpu)
    host_port = _host_port(cpu, mem)

    _run(["docker", "rm", "-f", container_name], check=False)

    gpu_flag = []
    use_gpu = 0
    if gpu == "on":
        gpu_flag = ["--gpus", "all"]
        use_gpu = 1

    docker_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        f"--cpus={cpu}",
        f"--memory={mem}g",
        *gpu_flag,
        "-e", f"MODEL_ID={task_info.model_id}",
        "-e", f"MODEL_REVISION={task_info.model_revision or 'main'}",
        "-e", f"TASK_FAMILY={task_info.task_family}",
        "-e", f"TASK_TYPE={task_info.pipeline_tag}",
        "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
        "-e", f"USE_GPU={use_gpu}",
        *hf_offline_docker_env_args(),
        "-p", f"{host_port}:{SERVER_PORT}",
        image_info.tag,
    ]

    t0_wall = time.time()
    t0 = time.perf_counter()
    result = _run(docker_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

    base_url = f"http://127.0.0.1:{host_port}"
    deadline = time.perf_counter() + READY_TIMEOUT_S

    def fail_startup(reason: str) -> None:
        logs = _run(["docker", "logs", container_name, "--tail", "200"], check=False)
        if logs.stdout:
            print(logs.stdout[-500:])
        _run(["docker", "rm", "-f", container_name], check=False)
        raise RuntimeError(reason)

    while time.perf_counter() < deadline:
        try:
            response = requests.get(
                f"{base_url}/ready",
                timeout=2,
                headers={"Connection": "close"},
            )
            if response.status_code == 200:
                ready_received_at = time.time()
                try:
                    body = response.json()
                except Exception:
                    body = None

                if isinstance(body, dict) and body.get("status") == "ok":
                    cold_start_s = time.perf_counter() - t0
                    breakdown = _cold_start_breakdown(
                        body,
                        t0_wall,
                        ready_received_at,
                    )
                    print(
                        f"{log_prefix} Model: {body.get('model_id')}, "
                        f"device: {body.get('device')}, load: {body.get('load_time_s')}s"
                    )
                    print(f"{log_prefix} Server ready. cold_start={cold_start_s:.3f}s")
                    return RunningContainer(
                        name=container_name,
                        base_url=base_url,
                        host_port=host_port,
                        cold_start_s=cold_start_s,
                        **breakdown,
                    )

                if response.text.strip() == "ok":
                    cold_start_s = time.perf_counter() - t0
                    breakdown = _cold_start_breakdown(
                        None,
                        t0_wall,
                        ready_received_at,
                    )
                    print(f"{log_prefix} Server ready. cold_start={cold_start_s:.3f}s")
                    return RunningContainer(
                        name=container_name,
                        base_url=base_url,
                        host_port=host_port,
                        cold_start_s=cold_start_s,
                        **breakdown,
                    )
        except Exception:
            pass

        startup_exit_error = _container_startup_exit_error(container_name, mem)
        if startup_exit_error:
            print(f"{log_prefix} Container exited before server became ready: {startup_exit_error}")
            fail_startup(startup_exit_error)
        time.sleep(READY_POLL_INTERVAL_S)

    cold_start_s = time.perf_counter() - t0
    print(f"{log_prefix} Server not ready after {READY_TIMEOUT_S}s. cold_start={cold_start_s:.3f}s")
    startup_exit_error = _container_startup_exit_error(container_name, mem)
    fail_startup(
        startup_exit_error
        or f"server not ready after {READY_TIMEOUT_S}s for container {container_name}"
    )


def _stop_container_session(container_name: str, log_prefix: Optional[str] = None) -> None:
    if log_prefix:
        print(f"{log_prefix} Stopping container...")
    _run(["docker", "stop", container_name], check=False)
    _run(["docker", "rm", container_name], check=False)


def _start_probe_session(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
) -> RunningContainer:
    probe_cpu = max(cpu_list)
    probe_mem = max(mem_list)
    normalized_gpu = [_normalize_gpu_mode(gpu) for gpu in gpu_list]
    probe_gpu = "on" if "on" in normalized_gpu else "off"
    model_tag = _sanitize_model_id(task_info.model_id)
    container_name = f"probe_{model_tag}_{probe_cpu}c_{probe_mem}g_{probe_gpu}"

    print(
        f"[scale] Starting probe container with CPU={probe_cpu}, "
        f"MEM={probe_mem}GB, GPU={probe_gpu}"
    )
    return _start_container_session(
        task_info=task_info,
        cpu=probe_cpu,
        mem=probe_mem,
        gpu=probe_gpu,
        image_info=image_info,
        container_name=container_name,
        log_prefix="[probe]",
    )


def _parse_probe_response(data: Dict[str, Any], desc: str) -> Dict[str, Any]:
    required = {"effective_input_scale", "truncated_by_limit", "reason"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise RuntimeError(f"/probe response missing fields for {desc}: {missing}")

    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError(f"/probe returned empty reason for {desc}")

    if data.get("effective_input_scale") is None or data.get("truncated_by_limit") is None:
        raise RuntimeError(f"cannot reliably determine effective input scale for {desc}: {reason}")

    try:
        effective_scale = float(data["effective_input_scale"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"/probe returned invalid effective_input_scale for {desc}: "
            f"{data.get('effective_input_scale')!r}"
        ) from exc

    truncated_by_limit = data.get("truncated_by_limit")
    if not isinstance(truncated_by_limit, bool):
        raise RuntimeError(
            f"/probe returned non-boolean truncated_by_limit for {desc}: {truncated_by_limit!r}"
        )

    return {
        "effective_input_scale": effective_scale,
        "truncated_by_limit": truncated_by_limit,
        "reason": reason,
    }


def _post_probe_payload(
    session: RunningContainer,
    payload: Dict[str, Any],
    desc: str,
) -> Dict[str, Any]:
    import requests

    response = requests.post(
        session.base_url + "/probe",
        json=payload,
        timeout=300,
        headers={"Connection": "close"},
    )
    if response.status_code >= 400:
        stale_image_hint = ""
        if response.status_code == 404:
            stale_image_hint = " /probe endpoint is missing; rebuild the image without --skip-build."
        raise RuntimeError(
            f"/probe HTTP {response.status_code} for {desc}: {response.text[:300]}{stale_image_hint}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"/probe returned non-JSON response for {desc}") from exc

    parsed = _parse_probe_response(data, desc)
    parsed["payload"] = payload
    return parsed


def _request_scale_meta(
    session: RunningContainer,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    import requests

    response = requests.post(
        session.base_url + "/scale_meta",
        json=payload,
        timeout=300,
        headers={"Connection": "close"},
    )
    if response.status_code >= 400:
        stale_image_hint = ""
        if response.status_code == 404:
            stale_image_hint = " /scale_meta endpoint is missing; rebuild the image without --skip-build."
        raise RuntimeError(
            f"/scale_meta HTTP {response.status_code}: {response.text[:300]}{stale_image_hint}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError("/scale_meta returned non-JSON response") from exc

    if not isinstance(data, dict):
        raise RuntimeError("/scale_meta returned a non-object response")
    return data


def _request_nlp_scale_meta(
    session: RunningContainer,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    data = _request_scale_meta(session, payload)
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError("/scale_meta returned empty reason")

    raw_max = data.get("max_effective_input_scale")
    if raw_max is None:
        raise RuntimeError(f"cannot reliably determine NLP max input scale: {reason}")

    try:
        max_effective_input_scale = int(float(raw_max))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"/scale_meta returned invalid max_effective_input_scale: {raw_max!r}"
        ) from exc

    if max_effective_input_scale <= 0:
        raise RuntimeError(
            f"/scale_meta returned non-positive max_effective_input_scale={max_effective_input_scale}: {reason}"
        )

    return {
        "input_scale_type": str(data.get("input_scale_type", "")).strip() or "input_scale",
        "max_effective_input_scale": max_effective_input_scale,
        "reason": reason,
    }


def _request_audio_scale_meta(
    session: RunningContainer,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    data = _request_scale_meta(session, payload)
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError(
            "cannot determine audio input constraints from /scale_meta; "
            "rebuild the audio image without --skip-build"
        )

    result: Dict[str, Any] = {
        "input_scale_type": str(data.get("input_scale_type") or "duration_s"),
        "reason": reason,
    }
    numeric_fields = (
        "required_sampling_rate",
        "max_short_form_duration_s",
        "model_input_num_samples",
        "model_input_frames",
        "fixed_frontend_num_samples",
        "fixed_frontend_num_frames",
        "frontend_feature_bins",
        "encoder_positions",
        "decoder_output_token_limit",
    )
    for field_name in numeric_fields:
        raw_value = data.get(field_name)
        if raw_value is None:
            result[field_name] = None
            continue
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"/scale_meta returned invalid {field_name}: {raw_value!r}"
            ) from exc
        if not math.isfinite(number) or number <= 0:
            raise RuntimeError(
                f"/scale_meta returned non-positive/non-finite "
                f"{field_name}: {raw_value!r}"
            )
        result[field_name] = int(number) if number.is_integer() else number

    result["model_type"] = str(data.get("model_type") or "unknown")
    raw_fixed_padding = data.get("short_form_fixed_padding")
    if not isinstance(raw_fixed_padding, bool):
        raise RuntimeError(
            "/scale_meta returned invalid short_form_fixed_padding: "
            f"{raw_fixed_padding!r}"
        )
    result["short_form_fixed_padding"] = raw_fixed_padding
    return result


def _assert_manual_timeseries_scales_legal(
    task_info: TaskInfo,
    scales: List[float],
    batch_size: int,
) -> List[float]:
    from acprof.workloads import get_generator

    workload_gen = get_generator(task_info.task_family, task_info.model_id, task_info.pipeline_tag, batch_size)
    invalid: Dict[float, str] = {}

    for scale in scales:
        payload = workload_gen.generate(scale)
        effective = workload_gen.effective_input_scale(scale, payload)
        if effective is None:
            raise RuntimeError(
                f"cannot determine effective context length for requested scale {_format_scale_value(scale)}"
            )
        if float(effective) + 1e-9 < float(scale):
            invalid[scale] = f"context_length_clamped_to_{_format_scale_value(float(effective))}"

    if invalid:
        details = ", ".join(
            f"{_format_scale_value(scale)} ({reason})" for scale, reason in invalid.items()
        )
        raise RuntimeError(f"manual input scales exceed the usable timeseries context length: {details}")

    print(f"[scale] Using manual input scales: {serialize_input_scales(scales)}")
    return scales


def _assert_manual_nlp_scales_legal(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    scales: List[float],
    batch_size: int,
) -> List[float]:
    from acprof.workloads import get_generator

    session: Optional[RunningContainer] = None
    try:
        session = _start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list)
        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
        )
        invalid: Dict[float, str] = {}
        for scale in scales:
            payload = workload_gen.generate(scale)
            result = _post_probe_payload(
                session,
                payload,
                f"manual scale {_format_scale_value(scale)}",
            )
            if result["truncated_by_limit"]:
                invalid[scale] = (
                    f"{result['reason']}; effective_input_scale="
                    f"{_format_scale_value(result['effective_input_scale'])}"
                )

        if invalid:
            details = ", ".join(
                f"{_format_scale_value(scale)} ({reason})" for scale, reason in invalid.items()
            )
            raise RuntimeError(f"manual NLP input scales exceed the usable tokenizer limit: {details}")
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")

    print(f"[scale] Using manual input scales: {serialize_input_scales(scales)}")
    return scales


def _plan_manual_nlp_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    scales: List[float],
    batch_size: int,
    output_dir: str,
) -> PlannedInputScales:
    from acprof.workloads import get_generator

    session: Optional[RunningContainer] = None
    try:
        session = _start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list)
        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
        )
        invalid: Dict[float, str] = {}
        entries: List[Dict[str, Any]] = []
        actual_scales: List[float] = []

        for scale in scales:
            payload = workload_gen.generate(scale)
            result = _post_probe_payload(
                session,
                payload,
                f"manual scale {_format_scale_value(scale)}",
            )
            if result["truncated_by_limit"]:
                invalid[scale] = (
                    f"{result['reason']}; effective_input_scale="
                    f"{_format_scale_value(result['effective_input_scale'])}"
                )
                continue

            actual_scale = float(result["effective_input_scale"])
            actual_scales.append(actual_scale)
            entries.append({
                "input_scale": actual_scale,
                "scale_label": workload_gen.scale_label(actual_scale),
                "payload": result["payload"],
            })

        if invalid:
            details = ", ".join(
                f"{_format_scale_value(scale)} ({reason})" for scale, reason in invalid.items()
            )
            raise RuntimeError(f"manual NLP input scales exceed the usable tokenizer limit: {details}")
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")

    plan_file = _scale_plan_file_path(output_dir)
    plan_sha256 = _write_scale_plan_file(plan_file, task_info, entries)

    print(
        "[scale] Using manual input scales: "
        f"{serialize_input_scales(scales)}; effective scales: "
        f"{serialize_input_scales(actual_scales)}"
    )
    return PlannedInputScales(
        scales=actual_scales,
        source="manual",
        plan_file=plan_file,
        plan_sha256=plan_sha256,
    )


def _plan_nlp_auto_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    batch_size: int,
    output_dir: str,
) -> PlannedInputScales:
    from acprof.workloads import get_generator

    session: Optional[RunningContainer] = None
    try:
        session = _start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list)
        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
        )
        metadata_payload = workload_gen.generate(1.0)
        scale_meta = _request_nlp_scale_meta(session, metadata_payload)
        max_effective = int(scale_meta["max_effective_input_scale"])
        target_scales = _integer_auto_scales(max_effective, count=AUTO_INPUT_SCALE_COUNT)

        probe_cache: Dict[int, Dict[str, Any]] = {}

        def probe_word_count(word_count: int) -> Dict[str, Any]:
            word_count = max(1, int(word_count))
            cached = probe_cache.get(word_count)
            if cached is not None:
                return cached

            if hasattr(workload_gen, "generate_for_word_count"):
                payload = workload_gen.generate_for_word_count(word_count)
            else:
                payload = workload_gen.generate(float(word_count))
            result = _post_probe_payload(
                session,
                payload,
                f"word_count={word_count}",
            )
            probe_cache[word_count] = result
            return result

        def find_best_candidate(target_scale: int) -> Dict[str, Any]:
            left = 1
            right = max(1, int(target_scale))
            best_below_wc: Optional[int] = None
            best_below_result: Optional[Dict[str, Any]] = None

            while left <= right:
                mid = (left + right) // 2
                result = probe_word_count(mid)
                condition = (not result["truncated_by_limit"]) and (
                    result["effective_input_scale"] <= float(target_scale)
                )
                if condition:
                    best_below_wc = mid
                    best_below_result = result
                    left = mid + 1
                else:
                    right = mid - 1

            candidates: List[Dict[str, Any]] = []
            if best_below_result is not None:
                candidates.append(best_below_result)

            if best_below_wc is None:
                upper_wc = 1
            else:
                upper_wc = min(max(1, int(target_scale)), best_below_wc + 1)

            if upper_wc >= 1:
                upper_result = probe_word_count(upper_wc)
                if not upper_result["truncated_by_limit"]:
                    candidates.append(upper_result)

            if not candidates:
                raise RuntimeError(
                    f"failed to build a non-truncated NLP payload near target scale {target_scale}"
                )

            best = min(
                candidates,
                key=lambda item: (
                    abs(item["effective_input_scale"] - float(target_scale)),
                    -item["effective_input_scale"],
                ),
            )
            return best

        entries: List[Dict[str, Any]] = []
        actual_scales: List[float] = []
        previous_scale = float("-inf")
        for target_scale in target_scales:
            candidate = find_best_candidate(int(target_scale))
            actual_scale = float(candidate["effective_input_scale"])
            if actual_scale <= previous_scale:
                raise RuntimeError(
                    "failed to derive 6 strictly increasing NLP input scales: "
                    f"target={_format_scale_value(target_scale)} produced "
                    f"{_format_scale_value(actual_scale)} after "
                    f"{_format_scale_value(previous_scale)}"
                )
            previous_scale = actual_scale
            actual_scales.append(actual_scale)
            entries.append({
                "input_scale": actual_scale,
                "scale_label": workload_gen.scale_label(actual_scale),
                "payload": candidate["payload"],
            })

        plan_file = _scale_plan_file_path(output_dir)
        plan_sha256 = _write_scale_plan_file(plan_file, task_info, entries)

        print(
            "[scale] Auto-planned NLP scales from tokenizer limit "
            f"{max_effective}: {serialize_input_scales(actual_scales)}"
        )
        print(f"[scale] Scale metadata: {scale_meta['reason']}")
        return PlannedInputScales(
            scales=actual_scales,
            source="auto",
            plan_file=plan_file,
            plan_sha256=plan_sha256,
        )
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")


def _default_family_max_scale(task_info: TaskInfo, batch_size: int) -> float:
    if task_info.task_family == "timeseries":
        from acprof.workloads import get_generator

        workload_gen = get_generator(task_info.task_family, task_info.model_id, task_info.pipeline_tag, batch_size)
        if hasattr(workload_gen, "max_input_scale"):
            max_scale = workload_gen.max_input_scale()
            if max_scale is not None:
                return float(max_scale)

    scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
    if scaling_cfg and scaling_cfg.values:
        return float(max(scaling_cfg.values))

    raise RuntimeError(f"cannot determine default max input scale for task_family={task_info.task_family}")


def _plan_audio_scales(
    *,
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    scales: List[float],
    batch_size: int,
    output_dir: str,
    source: str,
    workload_spec_path: Optional[str],
) -> PlannedInputScales:
    """Validate real audio payloads against the loaded model before a sweep."""
    from acprof.workloads import get_generator

    workload_gen = get_generator(
        task_info.task_family,
        task_info.model_id,
        task_info.pipeline_tag,
        batch_size,
        workload_spec_path=workload_spec_path,
    )
    normalized_scales = sorted(set(float(scale) for scale in scales))
    if not normalized_scales or normalized_scales[0] <= 0:
        raise RuntimeError("audio input scales must be finite positive durations")
    if any(not math.isfinite(scale) for scale in normalized_scales):
        raise RuntimeError("audio input scales must be finite positive durations")

    # Validate manifest/asset limits before paying the cost of loading a large
    # model in the probe container (for example, reject Whisper 31s directly).
    materialized_payloads = [
        (scale, workload_gen.generate(scale)) for scale in normalized_scales
    ]

    session: Optional[RunningContainer] = None
    model_constraints: Dict[str, Any] = {}
    try:
        session = _start_probe_session(
            task_info,
            image_info,
            cpu_list,
            mem_list,
            gpu_list,
        )
        first_payload = materialized_payloads[0][1]
        model_constraints = _request_audio_scale_meta(session, first_payload)

        expected_rate = model_constraints.get("required_sampling_rate")
        payload_rate = first_payload.get("sample_rate")
        if expected_rate is not None and payload_rate is not None:
            if int(expected_rate) != int(payload_rate):
                raise RuntimeError(
                    "audio workload sampling rate does not match the model: "
                    f"payload={payload_rate}Hz, model={expected_rate}Hz"
                )

        max_short = model_constraints.get("max_short_form_duration_s")
        if max_short is not None:
            invalid = [scale for scale in normalized_scales if scale > float(max_short) + 1e-9]
            if invalid:
                raise RuntimeError(
                    "audio input scales exceed the model short-form limit "
                    f"of {_format_scale_value(float(max_short))} seconds: "
                    f"{serialize_input_scales(invalid)}. Use a separate long-form workload."
                )

        for scale, payload in materialized_payloads:
            result = _post_probe_payload(
                session,
                payload,
                f"audio scale {_format_scale_value(scale)}s",
            )
            if result["truncated_by_limit"]:
                raise RuntimeError(
                    "audio input scale is not valid for short-form inference: "
                    f"{_format_scale_value(scale)}s ({result['reason']})"
                )
            if not math.isclose(
                float(result["effective_input_scale"]),
                float(scale),
                rel_tol=0.0,
                abs_tol=(0.5 / max(1, int(payload.get("sample_rate", 16000)))),
            ):
                raise RuntimeError(
                    "audio effective duration differs from the requested scale: "
                    f"requested={scale}, effective={result['effective_input_scale']}"
                )
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")

    print(
        f"[scale] Using {source} audio scales: "
        f"{serialize_input_scales(normalized_scales)}"
    )
    return _materialize_scale_plan(
        task_info=task_info,
        scales=normalized_scales,
        batch_size=batch_size,
        output_dir=output_dir,
        source=source,
        workload_spec_path=workload_spec_path,
        model_constraints=model_constraints,
    )


def plan_input_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    batch_size: int,
    output_dir: str,
    input_scales: Optional[str] = None,
    workload_spec_path: Optional[str] = None,
) -> PlannedInputScales:
    if workload_spec_path and task_info.task_family != "audio":
        raise ValueError(
            "--workload-spec is currently implemented only for task_family='audio'"
        )
    plan_file = _scale_plan_file_path(output_dir)
    _clear_scale_plan_file(plan_file)

    if input_scales:
        manual_scales = resolve_input_scales(task_info.task_family, input_scales=input_scales)
        if task_info.task_family == "audio":
            return _plan_audio_scales(
                task_info=task_info,
                image_info=image_info,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                scales=manual_scales,
                batch_size=batch_size,
                output_dir=output_dir,
                source="manual",
                workload_spec_path=workload_spec_path,
            )
        if task_info.task_family == "nlp":
            return _plan_manual_nlp_scales(
                task_info=task_info,
                image_info=image_info,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                scales=manual_scales,
                batch_size=batch_size,
                output_dir=output_dir,
            )
        if task_info.task_family == "timeseries":
            manual_scales = _assert_manual_timeseries_scales_legal(
                task_info=task_info,
                scales=manual_scales,
                batch_size=batch_size,
            )
        else:
            print(f"[scale] Using manual input scales: {serialize_input_scales(manual_scales)}")

        return _materialize_scale_plan(
            task_info=task_info,
            scales=manual_scales,
            batch_size=batch_size,
            output_dir=output_dir,
            source="manual",
            workload_spec_path=workload_spec_path,
        )

    if task_info.task_family == "nlp":
        return _plan_nlp_auto_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=batch_size,
            output_dir=output_dir,
        )

    if task_info.task_family == "audio":
        from acprof.workloads import get_generator

        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
            workload_spec_path=workload_spec_path,
        )
        default_scales = workload_gen.default_input_scales()
        if not default_scales:
            raise RuntimeError(
                "audio workload manifest did not provide default input scales"
            )
        return _plan_audio_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            scales=[float(scale) for scale in default_scales],
            batch_size=batch_size,
            output_dir=output_dir,
            source="workload_spec",
            workload_spec_path=workload_spec_path,
        )

    if task_info.task_family == "diffusion":
        from acprof.workloads import get_generator

        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
        )
        default_scales = workload_gen.default_input_scales()
        if not default_scales:
            raise RuntimeError(
                "diffusion workload did not provide default output resolutions"
            )
        scales = [float(scale) for scale in default_scales]
        print(
            "[scale] Using diffusion output resolutions: "
            f"{serialize_input_scales(scales)}"
        )
        return _materialize_scale_plan(
            task_info=task_info,
            scales=scales,
            batch_size=batch_size,
            output_dir=output_dir,
            source="workload_default",
        )

    max_scale = _default_family_max_scale(task_info, batch_size)
    if task_info.task_family == "cv":
        scales = _float_auto_scales(max_scale, count=AUTO_INPUT_SCALE_COUNT)
    else:
        scales = _integer_auto_scales(int(max_scale), count=AUTO_INPUT_SCALE_COUNT)

    print(
        f"[scale] Auto-planned {task_info.task_family} scales: "
        f"{serialize_input_scales(scales)}"
    )
    return _materialize_scale_plan(
        task_info=task_info,
        scales=scales,
        batch_size=batch_size,
        output_dir=output_dir,
        source="auto",
        workload_spec_path=workload_spec_path,
    )


# ─────────────────────────────────────────────
# Single Case Execution
# ─────────────────────────────────────────────

def _run_single_case_legacy(
    task_info: TaskInfo,
    cpu: int,
    mem: int,
    gpu: str,
    image_info: ImageInfo,
    output_dir: str,
    project_dir: str,
    batch_size: int = 1,
    warmup: int = 2,
    repeat: int = 5,
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW,
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS,
    sample_hz: float = 20.0,
    idle_seconds: float = 3.0,
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS,
    idle_debug: bool = False,
    sniff_iface: str = "docker0",
    input_scales: Optional[str] = None,
) -> str:
    """Run one profiling case and return result CSV path."""
    model_tag = _sanitize_model_id(task_info.model_id)
    case_name = f"case_{model_tag}_{cpu}c_{mem}g_{gpu}"
    container_name = case_name

    host_port = _host_port(cpu, mem)
    out_csv = os.path.join(output_dir, f"result_{case_name}.csv")
    pcap_file = os.path.join(output_dir, f"sniff_{case_name}.pcap")
    lat_json = os.path.join(output_dir, f"lat_{case_name}.json")

    print(f"\n{'='*60}")
    print(f"[case] {case_name}")
    print(f"  CPU={cpu}, MEM={mem}GB, GPU={gpu}")
    print(f"  Port: {host_port}")
    print(f"{'='*60}")

    # Clean up any existing container
    _run(["docker", "rm", "-f", container_name], check=False)

    # ── Docker run ──
    gpu_flag = []
    use_gpu = 0
    if gpu == "on":
        gpu_flag = ["--gpus", "all"]
        use_gpu = 1

    docker_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        f"--cpus={cpu}",
        f"--memory={mem}g",
        *gpu_flag,
        "-e", f"MODEL_ID={task_info.model_id}",
        "-e", f"MODEL_REVISION={task_info.model_revision or 'main'}",
        "-e", f"TASK_FAMILY={task_info.task_family}",
        "-e", f"TASK_TYPE={task_info.pipeline_tag}",
        "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
        "-e", f"USE_GPU={use_gpu}",
        *hf_offline_docker_env_args(),
        "-p", f"{host_port}:{SERVER_PORT}",
        image_info.tag,
    ]

    t0 = time.perf_counter()
    result = _run(docker_cmd, check=False)
    if result.returncode != 0:
        print(f"[case] Docker run failed: {result.stderr}", file=sys.stderr)
        return ""

    # ── Wait for /ready ──
    base_url = f"http://127.0.0.1:{host_port}"
    ready_ok = False
    deadline = time.perf_counter() + READY_TIMEOUT_S

    while time.perf_counter() < deadline:
        try:
            import requests
            r = requests.get(f"{base_url}/ready", timeout=2, headers={"Connection": "close"})
            if r.status_code == 200:
                try:
                    body = r.json()
                    if body.get("status") == "ok":
                        print(f"[case] Model: {body.get('model_id')}, device: {body.get('device')}, load: {body.get('load_time_s')}s")
                        ready_ok = True
                        break
                except Exception:
                    # Legacy: accept plain "ok" text
                    if r.text.strip() == "ok":
                        ready_ok = True
                        break
        except Exception:
            pass
        time.sleep(READY_POLL_INTERVAL_S)

    t1 = time.perf_counter()
    cold_start_s = t1 - t0

    if not ready_ok:
        print(f"[case] Server not ready after {READY_TIMEOUT_S}s. cold_start={cold_start_s:.3f}s")
        # Dump container logs for debugging
        logs = _run(["docker", "logs", container_name, "--tail", "200"], check=False)
        if logs.stdout:
            print(logs.stdout[-500:])
        _run(["docker", "rm", "-f", container_name], check=False)
        return ""

    print(f"[case] Server ready. cold_start={cold_start_s:.3f}s")

    # ── Start tcpdump ──
    tcpdump_proc = None
    sniff_runtime = _resolve_packet_latency_runtime(
        project_dir=project_dir,
        pcap_file=pcap_file,
        sniff_iface=sniff_iface,
    )

    if sniff_runtime is not None:
        print(f"[sniff] Starting tcpdump on {sniff_iface} via {sniff_runtime.mode}")
        tcpdump_proc = subprocess.Popen(
            sniff_runtime.tcpdump_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)  # Wait for tcpdump to start capturing
    else:
        print("[sniff] tcpdump/tshark unavailable, skipping packet-level latency")

    # ── Run client workload ──
    print("[case] Running workload...")

    # Determine input scales
    scales_str = input_scales
    if not scales_str:
        scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
        if scaling_cfg:
            scales_str = ",".join(str(v) for v in scaling_cfg.values)
        else:
            scales_str = "1.0"

    client_env = {
        **os.environ,
        "MODEL_ID": task_info.model_id,
        "MODEL_REVISION": task_info.model_revision,
        "TASK_FAMILY": task_info.task_family,
        "PIPELINE_TAG": task_info.pipeline_tag,
        "RUNTIME_BACKEND": task_info.runtime_backend,
        "IMAGE_TAG": image_info.tag,
        "CPU_CORES": str(cpu),
        "MEM_CAP_GB": str(mem),
        "GPU_MODE": gpu,
        "BASE_URL": base_url,
        "ENDPOINT": "/predict",
        "BATCH_SIZE": str(batch_size),
        "WARMUP": str(warmup),
        "REPEAT": str(repeat),
        "REPEAT_IN_WINDOW": str(repeat_in_window),
        "COLD_START_S": f"{cold_start_s:.3f}",
        "OUT_CSV": out_csv,
        "CASE_NAME": case_name,
        "CONTAINER_NAME": container_name,
        "USE_MIPS": "1",
        "SAMPLE_HZ": str(sample_hz),
        "IDLE_SECONDS": str(idle_seconds),
        "IDLE_COOLDOWN_SECONDS": str(idle_cooldown_seconds),
        "DEVICE_INDEX": "0",
        "INPUT_SCALES": scales_str,
    }

    client_result = _run(
        [sys.executable, "-m", "acprof.host.client"],
        check=False,
        capture=False,
        env=client_env,
    )

    if client_result.returncode != 0:
        if client_result.returncode == MIPS_EXIT_CODE:
            raise MIPSProfilingError(
                "client.py exited because MIPS profiling failed; review the "
                "[mips][ERROR] output above for the perf remediation steps."
            )
        print(f"[case] Client exited with code {client_result.returncode}")

    # ── Stop tcpdump and parse ──
    if tcpdump_proc is not None:
        time.sleep(1.0)  # Wait for last packets
        tcpdump_proc.terminate()
        try:
            tcpdump_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tcpdump_proc.kill()
        time.sleep(0.2)  # Buffer flush

        # Parse PCAP
        if os.path.exists(pcap_file) and os.path.getsize(pcap_file) > 0:
            print("[sniff] Parsing pcap -> packet latencies...")
            parse_result = _run(
                sniff_runtime.parse_cmd,
                check=False,
            )
            if parse_result.returncode == 0 and parse_result.stdout.strip():
                with open(lat_json, "w", encoding="utf-8") as lf:
                    lf.write(parse_result.stdout)
            else:
                with open(lat_json, "w", encoding="utf-8") as lf:
                    lf.write("{}")

            # Merge packet latency into CSV
            if os.path.exists(lat_json) and os.path.exists(out_csv):
                print("[sniff] Merging packet latency into CSV...")
                merged_csv = out_csv + ".merged"
                _run(
                    [
                        sys.executable,
                        "-m",
                        "acprof.packet.merge_packet_latency",
                        out_csv,
                        lat_json,
                        merged_csv,
                    ],
                    check=False,
                )
                if os.path.exists(merged_csv):
                    os.replace(merged_csv, out_csv)

    # ── Cleanup container ──
    print("[case] Stopping container...")
    _run(["docker", "stop", container_name], check=False)
    _run(["docker", "rm", container_name], check=False)

    print(f"[case] Done. Output: {out_csv}")
    return out_csv


# ─────────────────────────────────────────────
# Matrix Sweep
# ─────────────────────────────────────────────

def run_single_case(
    task_info: TaskInfo,
    cpu: int,
    mem: int,
    gpu: str,
    image_info: ImageInfo,
    output_dir: str,
    project_dir: str,
    batch_size: int = 1,
    warmup: int = 2,
    repeat: int = 5,
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW,
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS,
    sample_hz: float = 20.0,
    idle_seconds: float = 3.0,
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS,
    idle_debug: bool = False,
    sniff_iface: str = "docker0",
    input_scales: Optional[str] = None,
    input_scale_plan_file: Optional[str] = None,
    compute_profile_plan_file: Optional[str] = None,
    execution_profile_plan_file: Optional[str] = None,
    require_packet_latency: bool = True,
) -> str:
    """Run one profiling case and return result CSV path."""
    model_tag = _sanitize_model_id(task_info.model_id)
    case_name = f"case_{model_tag}_{cpu}c_{mem}g_{gpu}"
    container_name = case_name

    host_port = _host_port(cpu, mem)
    out_csv = os.path.join(output_dir, f"result_{case_name}.csv")
    idle_diag_path = os.path.join(
        output_dir,
        IDLE_DIAG_DIRNAME,
        f"{os.path.basename(out_csv)}.idle_diag.jsonl",
    )
    pcap_file = os.path.join(output_dir, f"sniff_{case_name}.pcap")
    lat_json = os.path.join(output_dir, f"lat_{case_name}.json")
    client_error_path = f"{out_csv}.client_error.json"

    print(f"\n{'='*60}")
    print(f"[case] {case_name}")
    print(f"  CPU={cpu}, MEM={mem}GB, GPU={gpu}")
    print(f"  Port: {host_port}")
    print(f"{'='*60}")

    try:
        os.remove(client_error_path)
    except FileNotFoundError:
        pass

    try:
        session = _start_container_session(
            task_info=task_info,
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            image_info=image_info,
            container_name=container_name,
            log_prefix="[case]",
        )
    except RuntimeError as exc:
        error = f"container_start_failed: {exc}"
        print(f"[case] {error}", file=sys.stderr)
        _write_case_error_csv(
            task_info=task_info,
            out_csv=out_csv,
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            warmup=warmup,
            repeat=repeat,
            repeat_in_window=repeat_in_window,
            input_scales=input_scales,
            error=error,
        )
        return out_csv

    base_url = session.base_url
    cold_start_s = session.cold_start_s
    tcpdump_proc = None
    case_incomplete = False
    completed_rows_before_failure = 0
    incomplete_case_reason = ""
    sniff_runtime = _resolve_packet_latency_runtime(
        project_dir=project_dir,
        pcap_file=pcap_file,
        sniff_iface=sniff_iface,
    )

    try:
        if sniff_runtime is not None:
            print(f"[sniff] Starting tcpdump on {sniff_iface} via {sniff_runtime.mode}")
            try:
                tcpdump_proc = subprocess.Popen(
                    sniff_runtime.tcpdump_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                raise _packet_latency_error("failed to start tcpdump", repr(exc)) from exc
            time.sleep(0.3)
            if require_packet_latency and tcpdump_proc.poll() is not None:
                raise _packet_latency_error(
                    "tcpdump exited before workload started",
                    f"command={' '.join(sniff_runtime.tcpdump_cmd)}",
                )
        else:
            if require_packet_latency:
                raise _packet_latency_error(
                    "tcpdump/tshark runtime could not be resolved"
                )
            print("[sniff] tcpdump/tshark unavailable, skipping packet-level latency")

        print("[case] Running workload...")
        scales_str = input_scales or serialize_input_scales(
            resolve_input_scales(task_info.task_family, input_scales=None)
        )

        client_env = {
            **os.environ,
            "MODEL_ID": task_info.model_id,
            "MODEL_REVISION": task_info.model_revision,
            "TASK_FAMILY": task_info.task_family,
            "PIPELINE_TAG": task_info.pipeline_tag,
            "RUNTIME_BACKEND": task_info.runtime_backend,
            "IMAGE_TAG": image_info.tag,
            "CPU_CORES": str(cpu),
            "MEM_CAP_GB": str(mem),
            "GPU_MODE": gpu,
            "BASE_URL": base_url,
            "ENDPOINT": "/predict",
            "BATCH_SIZE": str(batch_size),
            "WARMUP": str(warmup),
            "REPEAT": str(repeat),
            "REPEAT_IN_WINDOW": str(repeat_in_window),
            "REPEAT_WINDOW_SECONDS": str(repeat_window_seconds),
            **_cold_start_client_env(session),
            "OUT_CSV": out_csv,
            "CASE_NAME": case_name,
            "CONTAINER_NAME": container_name,
            "USE_MIPS": "1",
            "SAMPLE_HZ": str(sample_hz),
            "IDLE_SECONDS": str(idle_seconds),
            "IDLE_COOLDOWN_SECONDS": str(idle_cooldown_seconds),
            "IDLE_DEBUG": "1" if idle_debug else "0",
            "IDLE_DIAG_PATH": idle_diag_path if idle_debug else "",
            "CLIENT_ERROR_PATH": client_error_path,
            "DEVICE_INDEX": "0",
            "INPUT_SCALES": scales_str,
            "INPUT_SCALE_PLAN_FILE": input_scale_plan_file or "",
            "COMPUTE_PROFILE_PLAN_FILE": compute_profile_plan_file or "",
            "EXECUTION_PROFILE_PLAN_FILE": execution_profile_plan_file or "",
        }
        # The host-side client never sends notifications.  Keep the webhook
        # credential out of the measured subprocess environment entirely.
        client_env.pop("ACPROF_WECOM_WEBHOOK_URL", None)

        client_result = _run(
            [sys.executable, "-m", "acprof.host.client"],
            check=False,
            capture=False,
            env=client_env,
        )

        if client_result.returncode != 0:
            runtime_oom_error = _container_runtime_oom_error(
                container_name,
                mem,
                client_result.returncode,
            )
            if runtime_oom_error is not None:
                case_incomplete = True
                incomplete_case_reason = "runtime OOM"
                print(f"[case][WARN] {runtime_oom_error}", file=sys.stderr)
                completed_rows_before_failure, _ = _write_case_error_csv(
                    task_info=task_info,
                    out_csv=out_csv,
                    cpu=cpu,
                    mem=mem,
                    gpu=gpu,
                    warmup=warmup,
                    repeat=repeat,
                    repeat_in_window=repeat_in_window,
                    input_scales=input_scales,
                    error=runtime_oom_error,
                    preserve_existing=True,
                    annotate_existing_error_rows=True,
                )
            elif client_result.returncode == CLIENT_REQUEST_TIMEOUT_EXIT_CODE:
                case_incomplete = True
                incomplete_case_reason = "request timeout"
                timeout_context = _load_client_error_context(client_error_path)
                timeout_s = _timeout_context_float(
                    timeout_context.get("request_timeout_s"),
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
                error = (
                    "client_request_timeout: triggering request exceeded "
                    f"{timeout_s:g}s; incomplete and later rows will be marked "
                    "individually"
                )
                print(f"[case][WARN] {error}", file=sys.stderr)
                completed_rows_before_failure, _ = _write_case_error_csv(
                    task_info=task_info,
                    out_csv=out_csv,
                    cpu=cpu,
                    mem=mem,
                    gpu=gpu,
                    warmup=warmup,
                    repeat=repeat,
                    repeat_in_window=repeat_in_window,
                    input_scales=input_scales,
                    error=error,
                    preserve_existing=True,
                    timeout_context=timeout_context,
                )
            elif client_result.returncode == MIPS_EXIT_CODE:
                raise MIPSProfilingError(
                    "client.py exited because MIPS profiling failed; review the "
                    "[mips][ERROR] output above for the perf remediation steps."
                )
            else:
                raise EnergyProfilingError(
                    "client.py exited with code "
                    f"{client_result.returncode}; aborting profiling matrix. "
                    "Review the client output above for the energy stability diagnostic."
                )

        if tcpdump_proc is not None:
            time.sleep(1.0)
            tcpdump_proc.terminate()
            try:
                tcpdump_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tcpdump_proc.kill()
            time.sleep(0.2)

            if case_incomplete and completed_rows_before_failure == 0:
                print(
                    "[sniff] No completed measurements before "
                    f"{incomplete_case_reason}; skipping packet-latency merge"
                )
            elif not os.path.exists(pcap_file) or os.path.getsize(pcap_file) <= 0:
                if require_packet_latency:
                    raise _packet_latency_error(
                        f"pcap file is missing or empty: {pcap_file}"
                    )
            else:
                print("[sniff] Parsing pcap -> packet latencies...")
                assert sniff_runtime is not None
                parse_result = _run(sniff_runtime.parse_cmd, check=False)
                parse_output = parse_result.stdout.strip()
                if parse_result.returncode != 0:
                    raise _packet_latency_error(
                        "pcap parsing failed",
                        (parse_result.stderr or parse_result.stdout or "").strip(),
                    )
                if not parse_output:
                    raise _packet_latency_error("pcap parser produced no output")
                try:
                    latency_payload = json.loads(parse_output)
                except json.JSONDecodeError as exc:
                    raise _packet_latency_error(
                        "pcap parser produced invalid JSON",
                        repr(exc),
                    ) from exc
                latency_records = (
                    latency_payload.get("requests")
                    if isinstance(latency_payload, dict)
                    and "requests" in latency_payload
                    else latency_payload
                )
                if not isinstance(latency_records, dict) or not latency_records:
                    raise _packet_latency_error(
                        "pcap parser did not find matching request latency records"
                    )
                with open(lat_json, "w", encoding="utf-8") as lf:
                    json.dump(latency_payload, lf, ensure_ascii=True, indent=2)

                if not os.path.exists(out_csv):
                    raise _packet_latency_error(
                        f"client did not produce result CSV: {out_csv}"
                    )

                print("[sniff] Merging packet latency into CSV...")
                merged_csv = out_csv + ".merged"
                merge_result = _run(
                    [
                        sys.executable,
                        "-m",
                        "acprof.packet.merge_packet_latency",
                        out_csv,
                        lat_json,
                        merged_csv,
                    ],
                    check=False,
                )
                if merge_result.returncode != 0:
                    raise _packet_latency_error(
                        "packet latency merge failed",
                        (merge_result.stderr or merge_result.stdout or "").strip(),
                    )
                if not os.path.exists(merged_csv):
                    raise _packet_latency_error(
                        f"packet latency merge did not produce {merged_csv}"
                    )
                os.replace(merged_csv, out_csv)
                if require_packet_latency:
                    _assert_packet_latency_csv_complete(
                        out_csv,
                        ignore_error_rows=case_incomplete,
                    )

        if not case_incomplete or completed_rows_before_failure > 0:
            _check_case_cpu_idle_power_stable(
                out_csv,
                ignore_error_rows=case_incomplete,
            )
            if _normalize_gpu_mode(gpu) == "on":
                _check_case_gpu_idle_power_stable(
                    out_csv,
                    ignore_error_rows=case_incomplete,
                )
    finally:
        if tcpdump_proc is not None and tcpdump_proc.poll() is None:
            tcpdump_proc.terminate()
            try:
                tcpdump_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tcpdump_proc.kill()
        _stop_container_session(container_name, log_prefix="[case]")

    print(f"[case] Done. Output: {out_csv}")
    return out_csv


def _write_case_error_csv(
    *,
    task_info: TaskInfo,
    out_csv: str,
    cpu: int,
    mem: int,
    gpu: str,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    input_scales: Optional[str],
    error: str,
    preserve_existing: bool = False,
    annotate_existing_error_rows: bool = False,
    timeout_context: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    """Write missing error rows and return successful-preserved/added counts."""
    if not str(error or "").strip():
        raise ValueError("error rows require a non-empty diagnostic")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    scales = resolve_input_scales(task_info.task_family, input_scales)

    def make_row(scale: float, repeat_idx: int, is_warmup: bool) -> Dict[str, Any]:
        row = {field: "nan" for field in CSV_FIELDS}
        row.update({
            "cpu_cores": str(cpu),
            "mem_cap_gb": str(mem),
            "gpu_mode": gpu,
            "input_scale": _format_scale_value(scale),
            "task_param": "",
            "repeat_idx": str(repeat_idx),
            "warmup": "1" if is_warmup else "0",
            "repeat_in_window": str(repeat_in_window),
            TORCH_ERROR_FIELD: "not_run",
            "status": "error",
            "error": error,
        })
        if gpu == "on":
            row[NCU_ERROR_FIELD] = "not_run"
            row["compute_profile_error_nsys"] = "not_run"
        else:
            row["compute_profile_error_massif"] = "not_run"
        return row

    planned_rows: List[Dict[str, Any]] = []
    for scale in scales:
        for idx in range(warmup):
            planned_rows.append(make_row(scale, idx, True))
        for idx in range(repeat):
            planned_rows.append(make_row(scale, idx, False))

    def measurement_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
        try:
            scale = _format_scale_value(float(row.get("input_scale", "nan")))
            repeat_idx = int(row.get("repeat_idx", ""))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid measurement key in partial case CSV: {row!r}"
            ) from exc
        warmup_flag = "1" if str(row.get("warmup") or "0").strip() == "1" else "0"
        return scale, warmup_flag, repeat_idx

    planned_by_key = {
        measurement_key(row): row
        for row in planned_rows
    }
    existing_rows: List[Dict[str, Any]] = []
    existing_keys = set()
    if preserve_existing and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing_fields = set(CSV_FIELDS) - set(reader.fieldnames or [])
            if missing_fields:
                raise RuntimeError(
                    "partial case CSV is missing required fields: "
                    + ", ".join(sorted(missing_fields))
                )
            for row in reader:
                key = measurement_key(row)
                if key not in planned_by_key:
                    raise RuntimeError(
                        f"partial case CSV contains an unplanned measurement: {key!r}"
                    )
                if key in existing_keys:
                    raise RuntimeError(
                        f"partial case CSV contains a duplicate measurement: {key!r}"
                    )
                existing_keys.add(key)
                normalized_row = {
                    field: row.get(field, "")
                    for field in CSV_FIELDS
                }
                if (
                    annotate_existing_error_rows
                    and _row_has_error_status(normalized_row)
                ):
                    existing_error = str(normalized_row.get("error") or "").strip()
                    if error not in existing_error:
                        normalized_row["error"] = (
                            f"{existing_error}; {error}"
                            if existing_error
                            else error
                        )
                existing_rows.append(normalized_row)

    missing_rows = [
        row
        for row in planned_rows
        if measurement_key(row) not in existing_keys
    ]
    if timeout_context is not None and missing_rows:
        _annotate_timeout_placeholder_rows(
            missing_rows,
            measurement_key=measurement_key,
            timeout_context=timeout_context,
        )
    rows = [*existing_rows, *missing_rows]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    preserved_success_rows = sum(
        not _row_has_error_status(row)
        for row in existing_rows
    )
    if preserve_existing:
        print(
            f"[case] Preserved existing rows: {len(existing_rows)} "
            f"(successful: {preserved_success_rows}); "
            f"wrote error rows: {len(missing_rows)}; total: {len(rows)}"
        )
    else:
        print(f"[case] Wrote error rows: {out_csv} ({len(rows)} rows)")
    return preserved_success_rows, len(missing_rows)


def _timeout_context_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else float(fallback)


def _load_client_error_context(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[case][WARN] could not read structured client error context {path}: {exc}",
            file=sys.stderr,
        )
        return {}
    if not isinstance(payload, dict):
        print(
            f"[case][WARN] ignored non-object client error context: {path}",
            file=sys.stderr,
        )
        return {}
    return payload


def _annotate_timeout_placeholder_rows(
    rows: List[Dict[str, Any]],
    *,
    measurement_key,
    timeout_context: Dict[str, Any],
) -> None:
    first_missing_scale = measurement_key(rows[0])[0]
    try:
        triggering_scale = _format_scale_value(
            float(timeout_context.get("input_scale", first_missing_scale))
        )
    except (TypeError, ValueError):
        triggering_scale = first_missing_scale

    timeout_s = _timeout_context_float(
        timeout_context.get("request_timeout_s"),
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    request_id = str(timeout_context.get("request_id") or "unknown")
    request_phase = str(timeout_context.get("request_phase") or "unknown")
    repeat_idx = timeout_context.get("measurement_repeat_idx")
    attempted_key = None
    if request_phase == "measurement_warmup" and repeat_idx is not None:
        attempted_key = (triggering_scale, "1", int(repeat_idx))
    elif request_phase == "measurement_repeat" and repeat_idx is not None:
        attempted_key = (triggering_scale, "0", int(repeat_idx))

    context_source = (
        "client_timeout_sidecar"
        if timeout_context.get("input_scale") is not None
        else "inferred_from_first_missing_scale"
    )
    trigger_fields = (
        f"trigger_error=client_request_timeout; triggering_input_scale={triggering_scale}; "
        f"request_timeout_s={timeout_s:g}; triggering_request_latency_s>{timeout_s:g}; "
        f"trigger_phase={request_phase}; trigger_request_id={request_id}; "
        f"context_source={context_source}"
    )

    for row in rows:
        key = measurement_key(row)
        planned_scale = key[0]
        if attempted_key is not None and key == attempted_key:
            row["error"] = (
                "client_request_timeout: planned_request_attempted=true; "
                "measurement_row_completed=false; "
                f"planned_input_scale={planned_scale}; {trigger_fields}"
            )
            continue

        reason = (
            "triggering_scale_probe_timed_out"
            if planned_scale == triggering_scale
            else "skipped_after_prior_scale_timeout"
        )
        row["error"] = (
            "not_measured_after_timeout: planned_request_attempted=false; "
            "measurement_row_completed=false; "
            f"planned_input_scale={planned_scale}; reason={reason}; {trigger_fields}"
        )


def _case_startup_outcome(csv_path: str) -> Tuple[str, str]:
    """Classify whether a completed case reached the server-ready state.

    Only Docker's explicit ``OOMKilled`` startup diagnostic is strong enough
    for cross-case pruning. Runtime OOMs, request timeouts, and generic startup
    failures deliberately remain outside this classification.
    """
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error) as exc:
        return "unknown", f"cannot_read_case_csv:{type(exc).__name__}"

    if not rows:
        return "unknown", "empty_case_csv"

    errors = [str(row.get("error") or "").strip() for row in rows]
    all_error_rows = all(_row_has_error_status(row) for row in rows)
    startup_oom_marker = "container_oom_killed during startup"
    startup_failure_marker = "container_start_failed:"

    if all_error_rows and all(
        startup_oom_marker in error.lower()
        for error in errors
    ):
        return "startup_oom", errors[0]

    if all_error_rows and all(
        startup_failure_marker in error.lower()
        for error in errors
    ):
        return "startup_failure", errors[0]

    # The case may subsequently fail during workload collection, but reaching
    # this branch proves that model startup itself was feasible at this cap.
    return "startup_feasible", ""


def _write_json_payload_atomic(payload: Dict[str, Any], output_path: str) -> None:
    """Atomically persist a JSON provenance payload without changing schemas."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        output_mode = (
            stat.S_IMODE(os.stat(output_path).st_mode)
            if os.path.exists(output_path)
            else 0o644
        )
        os.chmod(temporary_path, output_mode)
        os.replace(temporary_path, output_path)
        temporary_path = ""
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _new_startup_oom_pruning_plan(
    *,
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    reference_cpu: int,
) -> Dict[str, Any]:
    gpu_modes = []
    for gpu in gpu_list:
        normalized = _normalize_gpu_mode(gpu)
        if normalized not in gpu_modes:
            gpu_modes.append(normalized)

    return {
        "schema_version": 1,
        "status": "running",
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "strategy": "minimum_cpu_contiguous_startup_oom_prefix",
        "scope": "container_startup_oom_only",
        "model_id": task_info.model_id,
        "model_revision": task_info.model_revision,
        "image_tag": image_info.tag,
        "reference_cpu_cores": reference_cpu,
        "selected_cpu_cores": list(cpu_list),
        "selected_mem_caps_gb": list(mem_list),
        "selected_gpu_modes": gpu_modes,
        "execution_cpu_order": [
            reference_cpu,
            *[cpu for cpu in cpu_list if cpu != reference_cpu],
        ],
        "execution_mem_order": sorted(mem_list),
        "assumption": (
            "A memory cap that Docker explicitly OOM-kills during model startup "
            "at the minimum selected CPU count is treated as startup-infeasible "
            "for larger selected CPU counts in the same GPU mode. Pruned cases "
            "are inferred, never represented as measured performance rows."
        ),
        "exclusions": [
            "runtime_oom",
            "cuda_oom",
            "request_timeout",
            "generic_startup_failure",
        ],
        "gpu_mode_results": {
            gpu: {
                "confirmed_startup_oom_prefix_gb": [],
                "minimum_startup_feasible_mem_cap_gb": None,
                "first_non_oom_mem_cap_gb": None,
                "first_non_oom_outcome": None,
                "reference_cases": [],
            }
            for gpu in gpu_modes
        },
        "pruned_cases": [],
        "planned_case_count": len(cpu_list) * len(mem_list) * len(gpu_list),
        "pruned_case_count": 0,
    }


def _validate_startup_oom_pruning_matrix(
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
) -> None:
    if not cpu_list or not mem_list or not gpu_list:
        raise ValueError("startup OOM pruning requires non-empty resource lists")
    if len(set(cpu_list)) != len(cpu_list):
        raise ValueError("startup OOM pruning requires unique CPU values")
    if len(set(mem_list)) != len(mem_list):
        raise ValueError("startup OOM pruning requires unique memory values")
    normalized_gpu = [_normalize_gpu_mode(gpu) for gpu in gpu_list]
    if len(set(normalized_gpu)) != len(normalized_gpu):
        raise ValueError("startup OOM pruning requires unique GPU modes")
    if any(cpu <= 0 for cpu in cpu_list):
        raise ValueError("startup OOM pruning requires positive CPU values")
    if any(mem <= 0 for mem in mem_list):
        raise ValueError("startup OOM pruning requires positive memory values")


def _write_startup_oom_pruned_case_csv(
    *,
    task_info: TaskInfo,
    output_dir: str,
    cpu: int,
    mem: int,
    gpu: str,
    reference_cpu: int,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    input_scales: Optional[str],
) -> str:
    model_tag = _sanitize_model_id(task_info.model_id)
    case_name = f"case_{model_tag}_{cpu}c_{mem}g_{gpu}"
    out_csv = os.path.join(output_dir, f"result_{case_name}.csv")
    error = (
        "not_measured_after_startup_oom_pruning: "
        "planned_request_attempted=false; measurement_row_completed=false; "
        "reason=confirmed_startup_oom_at_reference_cpu; "
        f"reference_cpu_cores={reference_cpu}; reference_mem_cap_gb={mem}; "
        f"gpu_mode={_normalize_gpu_mode(gpu)}; "
        "pruning_scope=container_startup_only; "
        "result_origin=inferred_not_measured"
    )
    _write_case_error_csv(
        task_info=task_info,
        out_csv=out_csv,
        cpu=cpu,
        mem=mem,
        gpu=gpu,
        warmup=warmup,
        repeat=repeat,
        repeat_in_window=repeat_in_window,
        input_scales=input_scales,
        error=error,
    )
    return out_csv


def run_matrix(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    output_dir: str,
    project_dir: str,
    batch_size: int = 1,
    warmup: int = 2,
    repeat: int = 5,
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW,
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS,
    sample_hz: float = 20.0,
    idle_seconds: float = 3.0,
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS,
    idle_debug: bool = False,
    sniff_iface: str = "docker0",
    input_scales: Optional[str] = None,
    input_scale_plan_file: Optional[str] = None,
    compute_profile_plan_file: Optional[str] = None,
    execution_profile_plan_file: Optional[str] = None,
    progress_callback: Optional[Callable[[MatrixProgress], None]] = None,
    prune_startup_oom: bool = False,
) -> List[str]:
    """Sweep all resource combinations, optionally pruning proven startup OOMs.

    Pruning is intentionally limited to a contiguous low-memory prefix that
    Docker explicitly OOM-killed at the minimum selected CPU count. Every
    skipped cell still receives planned error rows, while feasible cells retain
    the exact same collection protocol as an unpruned run.
    """
    os.makedirs(output_dir, exist_ok=True)
    result_csvs = []

    total = len(cpu_list) * len(mem_list) * len(gpu_list)
    current = 0
    execution_cpu_list = list(cpu_list)
    execution_mem_list = list(mem_list)
    reference_cpu: Optional[int] = None
    pruning_plan: Optional[Dict[str, Any]] = None
    pruning_plan_path: Optional[str] = None

    if prune_startup_oom:
        _validate_startup_oom_pruning_matrix(cpu_list, mem_list, gpu_list)
        reference_cpu = min(cpu_list)
        execution_cpu_list = [
            reference_cpu,
            *[cpu for cpu in cpu_list if cpu != reference_cpu],
        ]
        execution_mem_list = sorted(mem_list)
        pruning_plan = _new_startup_oom_pruning_plan(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            reference_cpu=reference_cpu,
        )
        pruning_plan_path = os.path.join(
            output_dir,
            STARTUP_OOM_PRUNING_PLAN_NAME,
        )
        _write_json_payload_atomic(pruning_plan, pruning_plan_path)
        print(
            "[oom-prune] Enabled: reference CPU="
            f"{reference_cpu}, memory order={execution_mem_list}; "
            "only confirmed startup OOM prefixes may be inferred"
        )

    def persist_pruning_plan() -> None:
        if pruning_plan is None or pruning_plan_path is None:
            return
        pruning_plan["updated_at"] = _utc_now_iso()
        pruning_plan["pruned_case_count"] = len(pruning_plan["pruned_cases"])
        _write_json_payload_atomic(pruning_plan, pruning_plan_path)

    for cpu in execution_cpu_list:
        for mem in execution_mem_list:
            for gpu in gpu_list:
                current += 1
                print(f"\n{'#'*60}")
                print(f"# Case {current}/{total}: CPU={cpu}, MEM={mem}GB, GPU={gpu}")
                print(f"{'#'*60}")

                normalized_gpu = _normalize_gpu_mode(gpu)
                gpu_pruning_result = (
                    pruning_plan["gpu_mode_results"][normalized_gpu]
                    if pruning_plan is not None
                    else None
                )
                prunable_mem_caps = (
                    gpu_pruning_result["confirmed_startup_oom_prefix_gb"]
                    if gpu_pruning_result is not None
                    else []
                )
                should_prune = bool(
                    pruning_plan is not None
                    and reference_cpu is not None
                    and cpu != reference_cpu
                    and mem in prunable_mem_caps
                )

                if should_prune:
                    print(
                        "[oom-prune] Skipping inferred startup-infeasible case: "
                        f"CPU={cpu}, MEM={mem}GB, GPU={normalized_gpu}; "
                        f"evidence CPU={reference_cpu}, MEM={mem}GB"
                    )
                    csv_path = _write_startup_oom_pruned_case_csv(
                        task_info=task_info,
                        output_dir=output_dir,
                        cpu=cpu,
                        mem=mem,
                        gpu=gpu,
                        reference_cpu=reference_cpu,
                        warmup=warmup,
                        repeat=repeat,
                        repeat_in_window=repeat_in_window,
                        input_scales=input_scales,
                    )
                    pruning_plan["pruned_cases"].append({
                        "cpu_cores": cpu,
                        "mem_cap_gb": mem,
                        "gpu_mode": normalized_gpu,
                        "reason": "confirmed_startup_oom_at_reference_cpu",
                        "reference_cpu_cores": reference_cpu,
                        "reference_mem_cap_gb": mem,
                        "result_origin": "inferred_not_measured",
                    })
                    persist_pruning_plan()
                else:
                    csv_path = run_single_case(
                        task_info=task_info,
                        cpu=cpu,
                        mem=mem,
                        gpu=gpu,
                        image_info=image_info,
                        output_dir=output_dir,
                        project_dir=project_dir,
                        batch_size=batch_size,
                        warmup=warmup,
                        repeat=repeat,
                        repeat_in_window=repeat_in_window,
                        repeat_window_seconds=repeat_window_seconds,
                        sample_hz=sample_hz,
                        idle_seconds=idle_seconds,
                        idle_cooldown_seconds=idle_cooldown_seconds,
                        idle_debug=idle_debug,
                        sniff_iface=sniff_iface,
                        input_scales=input_scales,
                        input_scale_plan_file=input_scale_plan_file,
                        compute_profile_plan_file=compute_profile_plan_file,
                        execution_profile_plan_file=execution_profile_plan_file,
                    )

                    if (
                        pruning_plan is not None
                        and reference_cpu is not None
                        and cpu == reference_cpu
                        and csv_path
                    ):
                        outcome, diagnostic = _case_startup_outcome(csv_path)
                        gpu_pruning_result["reference_cases"].append({
                            "cpu_cores": cpu,
                            "mem_cap_gb": mem,
                            "gpu_mode": normalized_gpu,
                            "outcome": outcome,
                            "diagnostic": diagnostic,
                        })
                        if gpu_pruning_result["first_non_oom_outcome"] is None:
                            if outcome == "startup_oom":
                                gpu_pruning_result[
                                    "confirmed_startup_oom_prefix_gb"
                                ].append(mem)
                            else:
                                gpu_pruning_result["first_non_oom_mem_cap_gb"] = mem
                                gpu_pruning_result["first_non_oom_outcome"] = outcome
                                if outcome == "startup_feasible":
                                    gpu_pruning_result[
                                        "minimum_startup_feasible_mem_cap_gb"
                                    ] = mem
                        persist_pruning_plan()

                if csv_path:
                    result_csvs.append(csv_path)
                if progress_callback is not None:
                    try:
                        progress_callback(
                            MatrixProgress(
                                completed_cases=current,
                                total_cases=total,
                                cpu=cpu,
                                mem=mem,
                                gpu=gpu,
                                result_csv=csv_path or None,
                            )
                        )
                    except Exception as exc:
                        # Progress reporting is ancillary and must never abort
                        # or change the result of an experiment matrix.
                        print(
                            "[progress][WARN] Progress callback failed: "
                            f"{type(exc).__name__}",
                            file=sys.stderr,
                        )

    if pruning_plan is not None:
        pruning_plan["status"] = "complete"
        pruning_plan["completed_at"] = _utc_now_iso()
        pruning_plan["attempted_case_count"] = (
            total - len(pruning_plan["pruned_cases"])
        )
        persist_pruning_plan()
        print(
            f"[oom-prune] Plan: {pruning_plan_path}; "
            f"pruned={len(pruning_plan['pruned_cases'])}/{total} cases"
        )

    return result_csvs


def merge_all_csvs(csv_paths: List[str], output_path: str) -> None:
    """Merge all per-case CSVs into one final CSV."""
    import csv
    from acprof.config import CSV_FIELDS

    if not csv_paths:
        print("[merge] No CSV files to merge.")
        return

    all_rows = []
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_number, row in enumerate(reader, start=2):
                if _row_has_error_status(row) and not str(row.get("error") or "").strip():
                    raise RuntimeError(
                        "refusing to merge status=error without an error diagnostic: "
                        f"{path}:{row_number}"
                    )
                all_rows.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[merge] Final CSV: {output_path} ({len(all_rows)} rows)")

"""Host-side high-overhead execution profiling for AC-Prof.

Massif and Nsight Systems are deliberately collected outside the normal
benchmark path: both tools materially perturb latency.  Every tool and input
scale is isolated so a missing profiler or an invalid report remains a
diagnostic entry rather than aborting the other profiler.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from acprof.host.detect import TaskInfo
from acprof.host.profiler_common import (
    _base_docker_cmd,
    _format_scale_value,
    _load_input_scale_plan_entries,
    _parse_last_json_line,
    _run,
    _runner_args,
    _write_json_atomic,
)
from acprof.host.profiler_progress import (
    ProfilerProgressCallback,
    report_profiler_completion,
)


from acprof.host.profilers.execution_parsers import (
    NSYS_REPORTS,
    _finite_float,
    parse_massif_output,
    parse_massif_snapshots,
    _header_base,
    _duration_factor_to_ms,
    _memory_factor_to_bytes,
    _csv_table,
    _field_for_base,
    _sum_numeric_column,
    _nsys_memory_report_has_no_data,
    parse_nsys_stats_csv,
    parse_nsys_stats_reports,
)

from acprof.host.profilers.tool_discovery import (
    NSYS_DEFAULT_SEARCH_ROOTS,
    _candidate_nsys_paths,
    _nsys_path_rank,
    _find_nsys_executable,
    _nsys_mount_root,
    _find_nsys_importer,
)

from acprof.host.profilers.execution_environment import (
    EXECUTION_RUNTIME_LABEL_PREFIX,
    EXECUTION_RUNTIME_VERSION,
    EXECUTION_BASE_IMAGE_LABEL,
    EXECUTION_DOCKERFILE_LABEL,
    _command_detail,
    _inspect_execution_image,
    _ensure_execution_image,
    _build_massif_image,
    _build_nsys_image,
    _massif_version,
    _nsys_version,
    _validate_nsys_container_runtime,
)


EXECUTION_PROFILE_PLAN_NAME = "execution_profile_plan.json"
EXECUTION_PROFILE_DIRNAME = "execution_profiles"
EXECUTION_PROFILE_SCHEMA_VERSION = 1
MASSIF_CHECKPOINT_SCHEMA_VERSION = 1
EXECUTION_PROFILE_TOOL_MODES = {"none", "both", "massif", "nsys"}
MASSIF_TOOL = "massif"
NSYS_TOOL = "nsys"
MASSIF_SAMPLING_MODES = {"per-scale", "full"}
NSYS_SAMPLING_MODES = {"per-cpu-scale", "per-scale", "full"}
SAMPLING_STRATEGY_METADATA = {
    "full": "full_resource_matrix",
    "per-scale": "representative_per_scale",
    "per-cpu-scale": "representative_per_cpu_scale",
}
NSYS_NVTX_RANGE = "acprof_compute"
NSYS_TRACE_DOMAINS = "cuda,nvtx"
NSYS_RAW_STREAM_SUFFIX = ".qdstrm"
COMPUTE_THREAD_ENV_NAMES = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
}

MASSIF_FIELDS = (
    "cpu_heap_peak_bytes_massif",
    "cpu_heap_extra_peak_bytes_massif",
    "cpu_stack_peak_bytes_massif",
    "cpu_heap_peak_total_bytes_massif",
    "cpu_heap_peak_at_ms_massif",
)
NSYS_FIELDS = (
    "host_inference_wall_time_ms_per_request_nsys",
    "cuda_api_time_sum_ms_per_request_nsys",
    "cuda_api_call_count_per_request_nsys",
    "gpu_kernel_time_sum_ms_per_request_nsys",
    "gpu_kernel_launch_count_per_request_nsys",
    "gpu_memcpy_time_sum_ms_per_request_nsys",
    "gpu_memcpy_count_per_request_nsys",
    "gpu_memcpy_bytes_per_request_nsys",
)


def _safe_filename_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return token or "unknown"


def _relative_artifact(path: str, output_dir: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(output_dir))


def _normalize_gpu_modes(gpu_list: Iterable[str]) -> List[str]:
    modes: List[str] = []
    for gpu in gpu_list:
        mode = "on" if str(gpu).strip().lower() == "on" else "off"
        if mode not in modes:
            modes.append(mode)
    return modes


def _normalize_resources(values: Iterable[int], name: str) -> List[int]:
    normalized: List[int] = []
    for value in values:
        integer = int(value)
        if integer <= 0:
            raise ValueError(f"{name} values must be positive, got {value!r}")
        if integer not in normalized:
            normalized.append(integer)
    return normalized


def _normalize_sampling_mode(
    value: str,
    *,
    name: str,
    allowed: Iterable[str],
) -> str:
    normalized = str(value or "").strip().lower()
    allowed_values = set(allowed)
    if normalized not in allowed_values:
        raise ValueError(
            f"{name} must be one of {', '.join(sorted(allowed_values))}, "
            f"got {value!r}"
        )
    return normalized


def _resolve_reference_resource(
    values: Sequence[int],
    requested: Optional[int],
    *,
    name: str,
) -> int:
    reference = max(values) if requested is None else int(requested)
    if reference not in values:
        raise ValueError(
            f"{name}={reference} is not present in the requested resource "
            f"matrix {list(values)}"
        )
    return reference


def _sampled_resource_cases(
    *,
    tool: str,
    cpus: Sequence[int],
    memories: Sequence[int],
    sampling: str,
    reference_cpu: Optional[int],
    reference_mem: Optional[int],
) -> Tuple[List[Tuple[int, int]], Optional[int], Optional[int]]:
    """Return actual profiler resources and resolved representative values."""
    if sampling == "full":
        return (
            [(cpu, mem) for cpu in cpus for mem in memories],
            None,
            None,
        )

    resolved_mem = _resolve_reference_resource(
        memories,
        reference_mem,
        name=f"{tool}_reference_mem",
    )
    if tool == NSYS_TOOL and sampling == "per-cpu-scale":
        return ([(cpu, resolved_mem) for cpu in cpus], None, resolved_mem)

    resolved_cpu = _resolve_reference_resource(
        cpus,
        reference_cpu,
        name=f"{tool}_reference_cpu",
    )
    return ([(resolved_cpu, resolved_mem)], resolved_cpu, resolved_mem)


def _profile_source_resource(
    *,
    cpu: int,
    mem: int,
    sampling: str,
    reference_cpu: Optional[int],
    reference_mem: Optional[int],
) -> Tuple[int, int]:
    if sampling == "full":
        return cpu, mem
    if sampling == "per-cpu-scale":
        if reference_mem is None:
            raise ValueError("per-cpu-scale sampling requires reference memory")
        return cpu, reference_mem
    if reference_cpu is None or reference_mem is None:
        raise ValueError("per-scale sampling requires reference CPU and memory")
    return reference_cpu, reference_mem


def _copy_profile_with_provenance(
    profile: Mapping[str, Any],
    *,
    source_cpu: int,
    source_mem: int,
    sampling: str,
) -> Dict[str, Any]:
    copied = copy.deepcopy(dict(profile))
    strategy = SAMPLING_STRATEGY_METADATA[sampling]
    entries = copied.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry["profile_source_cpu_cores"] = source_cpu
            entry["profile_source_mem_cap_gb"] = source_mem
            entry["profile_sampling_strategy"] = strategy
    return copied


def _massif_error_entry(
    entry: Mapping[str, Any],
    error: str,
    *,
    report: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "input_scale": float(entry["input_scale"]),
        "tool": MASSIF_TOOL,
        **{field: None for field in MASSIF_FIELDS},
        "compute_profile_error_massif": error,
        "error": error,
    }
    if report is not None:
        result["report"] = report
    return result


def _nsys_error_entry(
    entry: Mapping[str, Any],
    error: str,
    *,
    report: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "input_scale": float(entry["input_scale"]),
        "tool": NSYS_TOOL,
        **{field: None for field in NSYS_FIELDS},
        "compute_profile_error_nsys": error,
        "error": error,
    }
    if report is not None:
        result["report"] = report
    return result


def _docker_env(cmd: Sequence[str], name: str, value: str) -> List[str]:
    if not cmd:
        return []
    return [*cmd[:-1], "-e", f"{name}={value}", cmd[-1]]


def _without_compute_thread_env(cmd: Sequence[str]) -> List[str]:
    """Keep execution probes aligned with the normal matrix runtime config."""
    filtered: List[str] = []
    index = 0
    while index < len(cmd):
        value = str(cmd[index])
        if value == "-e" and index + 1 < len(cmd):
            assignment = str(cmd[index + 1])
            if assignment.split("=", 1)[0] in COMPUTE_THREAD_ENV_NAMES:
                index += 2
                continue
        filtered.append(value)
        index += 1
    return filtered


def _massif_artifact_paths(
    *,
    profile_root: str,
    cpu: int,
    mem: int,
    input_scale: float,
) -> Tuple[str, str]:
    scale_label = _safe_filename_token(
        _format_scale_value(input_scale)
    )
    filename = f"massif_cpu_{cpu}_mem_{mem}_scale_{scale_label}.out"
    host_report = os.path.join(profile_root, filename)
    checkpoint = os.path.join(
        profile_root,
        f"massif_cpu_{cpu}_mem_{mem}_scale_{scale_label}.checkpoint.json",
    )
    return host_report, checkpoint


def _massif_entry_complete(entry: Mapping[str, Any]) -> bool:
    return not str(entry.get("error") or "").strip() and all(
        _finite_float(entry.get(field)) is not None
        for field in MASSIF_FIELDS
    )


def _massif_entry_from_report(
    *,
    entry: Mapping[str, Any],
    host_report: str,
    output_dir: str,
) -> Dict[str, Any]:
    relative_report = _relative_artifact(host_report, output_dir)
    try:
        parsed = parse_massif_output(host_report)
    except Exception as exc:
        detail = str(exc)
        error = (
            detail
            if detail.startswith("massif_parse_failed:")
            else f"massif_parse_failed:{detail}"
        )
        return _massif_error_entry(
            entry,
            error,
            report=relative_report if os.path.isfile(host_report) else None,
        )
    return {
        "input_scale": float(entry["input_scale"]),
        "tool": MASSIF_TOOL,
        **parsed,
        "compute_profile_error_massif": "",
        "error": "",
        "report": relative_report,
    }


def _write_massif_checkpoint(
    *,
    checkpoint_path: str,
    task_info: TaskInfo,
    derived_image: str,
    cpu: int,
    mem: int,
    input_scale: float,
    repeat: int,
    host_report: str,
    entry: Mapping[str, Any],
) -> None:
    _write_json_atomic(
        checkpoint_path,
        {
            "schema_version": MASSIF_CHECKPOINT_SCHEMA_VERSION,
            "model_id": task_info.model_id,
            "model_revision": task_info.model_revision or "main",
            "derived_image": derived_image,
            "cpu_cores": int(cpu),
            "mem_cap_gb": int(mem),
            "input_scale": float(input_scale),
            "repeat": max(1, int(repeat)),
            "report_size_bytes": os.path.getsize(host_report),
            "entry": dict(entry),
        },
    )


def _read_massif_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as checkpoint_file:
            payload = json.load(checkpoint_file)
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _massif_checkpoint_matches(
    checkpoint: Mapping[str, Any],
    *,
    task_info: TaskInfo,
    derived_image: str,
    cpu: int,
    mem: int,
    input_scale: float,
    repeat: int,
) -> bool:
    try:
        schema_version = int(checkpoint.get("schema_version"))
        checkpoint_cpu = int(checkpoint.get("cpu_cores"))
        checkpoint_mem = int(checkpoint.get("mem_cap_gb"))
        checkpoint_scale = float(checkpoint.get("input_scale"))
        checkpoint_repeat = int(checkpoint.get("repeat"))
    except (TypeError, ValueError):
        return False
    return (
        schema_version == MASSIF_CHECKPOINT_SCHEMA_VERSION
        and str(checkpoint.get("model_id") or "") == task_info.model_id
        and str(checkpoint.get("model_revision") or "main")
        == str(task_info.model_revision or "main")
        and str(checkpoint.get("derived_image") or "") == derived_image
        and checkpoint_cpu == int(cpu)
        and checkpoint_mem == int(mem)
        and math.isclose(checkpoint_scale, input_scale, abs_tol=1e-9)
        and checkpoint_repeat == max(1, int(repeat))
    )


def _resume_massif_entry(
    *,
    task_info: TaskInfo,
    derived_image: str,
    cpu: int,
    mem: int,
    profile_root: str,
    output_dir: str,
    entry: Mapping[str, Any],
    repeat: int,
) -> Optional[Dict[str, Any]]:
    input_scale = float(entry["input_scale"])
    scale_label = _format_scale_value(input_scale)
    host_report, checkpoint_path = _massif_artifact_paths(
        profile_root=profile_root,
        cpu=cpu,
        mem=mem,
        input_scale=input_scale,
    )
    checkpoint_exists = os.path.isfile(checkpoint_path)
    checkpoint = (
        _read_massif_checkpoint(checkpoint_path)
        if checkpoint_exists
        else None
    )
    if checkpoint_exists and (
        checkpoint is None
        or not _massif_checkpoint_matches(
            checkpoint,
            task_info=task_info,
            derived_image=derived_image,
            cpu=cpu,
            mem=mem,
            input_scale=input_scale,
            repeat=repeat,
        )
    ):
        print(
            f"[execution-profile][massif][resume] scale={scale_label}: "
            "checkpoint does not match this run; recollecting"
        )
        return None

    if checkpoint is not None and os.path.isfile(host_report):
        checkpoint_entry = checkpoint.get("entry")
        expected_size = _finite_float(checkpoint.get("report_size_bytes"))
        if (
            isinstance(checkpoint_entry, Mapping)
            and _massif_entry_complete(checkpoint_entry)
            and expected_size is not None
            and os.path.getsize(host_report) == int(expected_size)
        ):
            resumed = dict(checkpoint_entry)
            resumed["report"] = _relative_artifact(host_report, output_dir)
            print(
                f"[execution-profile][massif][resume] scale={scale_label}: "
                f"reusing checkpoint {checkpoint_path}"
            )
            return resumed

    if checkpoint is None and os.path.isfile(host_report):
        resumed = _massif_entry_from_report(
            entry=entry,
            host_report=host_report,
            output_dir=output_dir,
        )
        if _massif_entry_complete(resumed):
            _write_massif_checkpoint(
                checkpoint_path=checkpoint_path,
                task_info=task_info,
                derived_image=derived_image,
                cpu=cpu,
                mem=mem,
                input_scale=input_scale,
                repeat=repeat,
                host_report=host_report,
                entry=resumed,
            )
            print(
                f"[execution-profile][massif][resume] scale={scale_label}: "
                f"reusing valid report {host_report}"
            )
            return resumed
        print(
            f"[execution-profile][massif][resume] scale={scale_label}: "
            "existing report is incomplete; recollecting"
        )
    return None


def _collect_massif_entry(
    *,
    task_info: TaskInfo,
    derived_image: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    output_dir: str,
    entry: Mapping[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    input_scale = float(entry["input_scale"])
    scale_label = _safe_filename_token(
        _format_scale_value(input_scale)
    )
    host_report, checkpoint_path = _massif_artifact_paths(
        profile_root=profile_root,
        cpu=cpu,
        mem=mem,
        input_scale=input_scale,
    )
    filename = os.path.basename(host_report)
    relative_report = _relative_artifact(host_report, output_dir)
    base_cmd = _base_docker_cmd(
        task_info=task_info,
        image_tag=derived_image,
        cpu=cpu,
        mem=mem,
        use_gpu=False,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=(),
    )
    base_cmd = _without_compute_thread_env(base_cmd)
    command = [
        *base_cmd,
        "valgrind",
        "--tool=massif",
        "--time-unit=ms",
        "--stacks=yes",
        f"--massif-out-file=/profiles/{filename}",
        *_runner_args(dict(entry), repeat, "cpu"),
    ]
    result = _run(command, check=False)
    if result.returncode != 0:
        return _massif_error_entry(
            entry,
            f"massif_failed:{_command_detail(result)}",
            report=relative_report if os.path.isfile(host_report) else None,
        )
    profiled = _massif_entry_from_report(
        entry=entry,
        host_report=host_report,
        output_dir=output_dir,
    )
    if _massif_entry_complete(profiled):
        _write_massif_checkpoint(
            checkpoint_path=checkpoint_path,
            task_info=task_info,
            derived_image=derived_image,
            cpu=cpu,
            mem=mem,
            input_scale=input_scale,
            repeat=repeat,
            host_report=host_report,
            entry=profiled,
        )
    return profiled


def _nsys_sqlite_path(report_path: str) -> str:
    """Return the SQLite cache path generated by ``nsys stats``."""
    return f"{os.path.splitext(os.fspath(report_path))[0]}.sqlite"


def _discard_nsys_sqlite(report_path: str) -> None:
    """Remove the derived SQLite cache while preserving the raw report."""
    try:
        os.remove(_nsys_sqlite_path(report_path))
    except FileNotFoundError:
        pass


def _run_nsys_stats(nsys_bin: str, report_path: str) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    sqlite_path = _nsys_sqlite_path(report_path)
    try:
        for index, report_name in enumerate(NSYS_REPORTS):
            refresh_export = ["--force-export=true"] if index == 0 else []
            # Nsys 2026.1 can report a freshly exported SQLite cache as older
            # than its .nsys-rep when both files were written in the same
            # second.  Export once from the raw report, then pass the SQLite
            # file directly so later reports do not repeat that freshness
            # check.
            stats_input = report_path if index == 0 else sqlite_path
            result = _run(
                [
                    nsys_bin,
                    "stats",
                    "--report",
                    report_name,
                    "--format",
                    "csv",
                    "--timeunit",
                    "nsec",
                    "--output",
                    "-",
                    *refresh_export,
                    stats_input,
                ],
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"nsys_stats_failed:{report_name}:"
                    f"{_command_detail(result)}"
                )
            stdout = str(result.stdout or "")
            stderr = str(result.stderr or "")
            outputs[report_name] = "\n".join(
                part for part in (stdout, stderr) if part.strip()
            )
        return outputs
    finally:
        _discard_nsys_sqlite(report_path)


def _discard_nsys_raw_stream(path: str) -> int:
    """Remove an intermediate QDSTRM and return its former byte size."""
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    return size


def _collect_nsys_entry(
    *,
    task_info: TaskInfo,
    image_tag: str,
    nsys_bin: str,
    nsys_mount_root: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    output_dir: str,
    entry: Mapping[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    scale_label = _safe_filename_token(
        _format_scale_value(float(entry["input_scale"]))
    )
    stem = f"nsys_cpu_{cpu}_mem_{mem}_scale_{scale_label}"
    filename = f"{stem}.nsys-rep"
    host_report = os.path.join(profile_root, filename)
    host_raw_stream = os.path.join(
        profile_root,
        f"{stem}{NSYS_RAW_STREAM_SUFFIX}",
    )
    relative_report = _relative_artifact(host_report, output_dir)
    # A failed importer may have left a huge stream from an earlier attempt.
    _discard_nsys_raw_stream(host_raw_stream)
    base_cmd = _base_docker_cmd(
        task_info=task_info,
        image_tag=image_tag,
        cpu=cpu,
        mem=mem,
        use_gpu=True,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=(nsys_mount_root,),
    )
    base_cmd = _without_compute_thread_env(base_cmd)
    base_cmd = _docker_env(
        base_cmd,
        "NSYS_NVTX_PROFILER_REGISTER_ONLY",
        "0",
    )
    command = [
        *base_cmd,
        nsys_bin,
        "profile",
        f"--trace={NSYS_TRACE_DOMAINS}",
        "--capture-range=nvtx",
        f"--nvtx-capture={NSYS_NVTX_RANGE}",
        "--capture-range-end=stop",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        f"--output=/profiles/{stem}",
        *_runner_args(dict(entry), repeat, "gpu"),
    ]
    try:
        result = _run(command, check=False)
    except Exception:
        _discard_nsys_raw_stream(host_raw_stream)
        raise
    if result.returncode != 0:
        discarded_bytes = _discard_nsys_raw_stream(host_raw_stream)
        discarded = (
            f":discarded_qdstrm_bytes={discarded_bytes}"
            if discarded_bytes
            else ""
        )
        return _nsys_error_entry(
            entry,
            f"nsys_failed:{_command_detail(result)}{discarded}",
            report=relative_report if os.path.isfile(host_report) else None,
        )
    if not os.path.isfile(host_report):
        discarded_bytes = _discard_nsys_raw_stream(host_raw_stream)
        discarded = (
            f":discarded_qdstrm_bytes={discarded_bytes}"
            if discarded_bytes
            else ""
        )
        return _nsys_error_entry(
            entry,
            f"nsys_import_failed:report_not_found{discarded}",
        )
    _discard_nsys_raw_stream(host_raw_stream)

    runner_payload = _parse_last_json_line(str(result.stdout or ""))
    wall_time = _finite_float(
        runner_payload.get("profile_window_wall_time_ms_per_request")
    )
    if wall_time is None:
        total_wall_time = _finite_float(
            runner_payload.get("profile_window_wall_time_ms")
        )
        if total_wall_time is not None:
            wall_time = total_wall_time / max(1, int(repeat))
    if wall_time is None or wall_time < 0:
        return _nsys_error_entry(
            entry,
            "nsys_parse_failed:profile_window_wall_time_missing",
            report=relative_report,
        )

    try:
        stats = parse_nsys_stats_reports(
            _run_nsys_stats(nsys_bin, host_report),
            repeat=repeat,
        )
    except Exception as exc:
        detail = str(exc)
        error = (
            detail
            if detail.startswith(("nsys_parse_failed:", "nsys_stats_failed:"))
            else f"nsys_parse_failed:{detail}"
        )
        return _nsys_error_entry(
            entry,
            error,
            report=relative_report,
        )

    return {
        "input_scale": float(entry["input_scale"]),
        "tool": NSYS_TOOL,
        "host_inference_wall_time_ms_per_request_nsys": wall_time,
        **stats,
        "compute_profile_error_nsys": "",
        "error": "",
        "report": relative_report,
    }


def _profile_massif_tool(
    *,
    entries: List[Dict[str, Any]],
    global_error: str,
    task_info: TaskInfo,
    derived_image: Optional[str],
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    output_dir: str,
    repeat: int,
    resume_existing: bool = False,
) -> Dict[str, Any]:
    if global_error or not derived_image:
        error = global_error or "massif_not_found"
        profiled_entries = [
            _massif_error_entry(entry, error)
            for entry in entries
        ]
    else:
        profiled_entries = []
        for entry in entries:
            try:
                profiled = None
                if resume_existing:
                    profiled = _resume_massif_entry(
                        task_info=task_info,
                        derived_image=derived_image,
                        cpu=cpu,
                        mem=mem,
                        profile_root=profile_root,
                        output_dir=output_dir,
                        entry=entry,
                        repeat=repeat,
                    )
                if profiled is None:
                    profiled = _collect_massif_entry(
                        task_info=task_info,
                        derived_image=derived_image,
                        cpu=cpu,
                        mem=mem,
                        payload_file=payload_file,
                        profile_root=profile_root,
                        output_dir=output_dir,
                        entry=entry,
                        repeat=repeat,
                    )
                profiled_entries.append(profiled)
            except Exception as exc:
                profiled_entries.append(
                    _massif_error_entry(
                        entry,
                        f"massif_failed:{exc!r}",
                    )
                )
    return {
        "tool": MASSIF_TOOL,
        "repeat": repeat,
        "error": global_error,
        "entries": profiled_entries,
    }


def _profile_nsys_tool(
    *,
    entries: List[Dict[str, Any]],
    global_error: str,
    task_info: TaskInfo,
    image_tag: str,
    nsys_bin: Optional[str],
    nsys_mount_root: Optional[str],
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    output_dir: str,
    repeat: int,
) -> Dict[str, Any]:
    if global_error or not nsys_bin or not nsys_mount_root:
        error = global_error or "nsys_not_found"
        profiled_entries = [
            _nsys_error_entry(entry, error)
            for entry in entries
        ]
    else:
        profiled_entries = []
        for entry in entries:
            try:
                profiled_entries.append(
                    _collect_nsys_entry(
                        task_info=task_info,
                        image_tag=image_tag,
                        nsys_bin=nsys_bin,
                        nsys_mount_root=nsys_mount_root,
                        cpu=cpu,
                        mem=mem,
                        payload_file=payload_file,
                        profile_root=profile_root,
                        output_dir=output_dir,
                        entry=entry,
                        repeat=repeat,
                    )
                )
            except Exception as exc:
                profiled_entries.append(
                    _nsys_error_entry(
                        entry,
                        f"nsys_failed:{exc!r}",
                    )
                )
    return {
        "tool": NSYS_TOOL,
        "repeat": repeat,
        "error": global_error,
        "entries": profiled_entries,
    }


def _strip_artifact_paths(profiles: Sequence[Mapping[str, Any]]) -> None:
    for profile in profiles:
        tools = profile.get("tools")
        if not isinstance(tools, Mapping):
            continue
        for tool_profile in tools.values():
            if not isinstance(tool_profile, Mapping):
                continue
            entries = tool_profile.get("entries")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and "report" in entry:
                    entry["report"] = None


def collect_execution_profile_plan(
    task_info: TaskInfo,
    image_tag: str,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    output_dir: str,
    input_scale_plan_file: str,
    project_dir: str,
    tool_mode: str = "both",
    massif_repeat: int = 1,
    nsys_repeat: int = 1,
    nsys_root: Optional[str] = None,
    keep_profiles: bool = True,
    massif_sampling: str = "per-scale",
    massif_reference_cpu: Optional[int] = None,
    massif_reference_mem: Optional[int] = None,
    nsys_sampling: str = "per-cpu-scale",
    nsys_reference_cpu: Optional[int] = None,
    nsys_reference_mem: Optional[int] = None,
    resume_existing_profiles: bool = False,
    progress_callback: Optional[ProfilerProgressCallback] = None,
) -> str:
    """Collect sampled probes and expand them to a full resource-grid plan."""
    normalized_tool_mode = (tool_mode or "both").strip().lower()
    if normalized_tool_mode not in EXECUTION_PROFILE_TOOL_MODES:
        raise ValueError(
            "tool_mode must be one of "
            f"{', '.join(sorted(EXECUTION_PROFILE_TOOL_MODES))}, "
            f"got {tool_mode!r}"
        )

    cpus = _normalize_resources(cpu_list, "cpu_list")
    memories = _normalize_resources(mem_list, "mem_list")
    gpu_modes = _normalize_gpu_modes(gpu_list)
    normalized_massif_sampling = _normalize_sampling_mode(
        massif_sampling,
        name="massif_sampling",
        allowed=MASSIF_SAMPLING_MODES,
    )
    normalized_nsys_sampling = _normalize_sampling_mode(
        nsys_sampling,
        name="nsys_sampling",
        allowed=NSYS_SAMPLING_MODES,
    )
    entries = _load_input_scale_plan_entries(input_scale_plan_file)
    normalized_massif_repeat = max(1, int(massif_repeat))
    normalized_nsys_repeat = max(1, int(nsys_repeat))
    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    profile_root = os.path.join(output_dir, EXECUTION_PROFILE_DIRNAME)

    collect_massif = (
        normalized_tool_mode in {"both", MASSIF_TOOL}
        and "off" in gpu_modes
    )
    collect_nsys = (
        normalized_tool_mode in {"both", NSYS_TOOL}
        and "on" in gpu_modes
    )
    massif_sources: List[Tuple[int, int]] = []
    massif_reference: Tuple[Optional[int], Optional[int]] = (None, None)
    if collect_massif:
        (
            massif_sources,
            massif_reference_cpu_resolved,
            massif_reference_mem_resolved,
        ) = _sampled_resource_cases(
            tool=MASSIF_TOOL,
            cpus=cpus,
            memories=memories,
            sampling=normalized_massif_sampling,
            reference_cpu=massif_reference_cpu,
            reference_mem=massif_reference_mem,
        )
        massif_reference = (
            massif_reference_cpu_resolved,
            massif_reference_mem_resolved,
        )

    nsys_sources: List[Tuple[int, int]] = []
    nsys_reference: Tuple[Optional[int], Optional[int]] = (None, None)
    if collect_nsys:
        (
            nsys_sources,
            nsys_reference_cpu_resolved,
            nsys_reference_mem_resolved,
        ) = _sampled_resource_cases(
            tool=NSYS_TOOL,
            cpus=cpus,
            memories=memories,
            sampling=normalized_nsys_sampling,
            reference_cpu=nsys_reference_cpu,
            reference_mem=nsys_reference_mem,
        )
        nsys_reference = (
            nsys_reference_cpu_resolved,
            nsys_reference_mem_resolved,
        )

    if collect_massif or collect_nsys:
        os.makedirs(profile_root, exist_ok=True)
        probe_count = len(entries) * (
            len(massif_sources) + len(nsys_sources)
        )
        print(
            "[execution-profile] Collecting "
            f"{probe_count} sampled isolated probe(s) before the normal CSV "
            "sweep; "
            "result CSV files appear after this stage completes."
        )
        if collect_massif:
            print(
                "[execution-profile][massif] Sampling="
                f"{normalized_massif_sampling}, resources={massif_sources}"
            )
        if collect_nsys:
            print(
                "[execution-profile][nsys] Sampling="
                f"{normalized_nsys_sampling}, resources={nsys_sources}"
            )

    derived_image: Optional[str] = None
    massif_error = ""
    massif_version = "unknown"
    if collect_massif:
        try:
            derived_image = _build_massif_image(image_tag, project_dir)
        except Exception as exc:
            massif_error = str(exc)
            if not massif_error.startswith("massif_"):
                massif_error = f"massif_image_build_failed:{exc!r}"
        if derived_image:
            massif_version = _massif_version(derived_image)

    nsys_bin: Optional[str] = None
    nsys_mount_root: Optional[str] = None
    nsys_profile_image = image_tag
    nsys_error = ""
    nsys_version = "unknown"
    if collect_nsys:
        try:
            nsys_bin = _find_nsys_executable(nsys_root)
        except Exception as exc:
            nsys_error = f"nsys_discovery_failed:{exc!r}"
        if nsys_bin:
            try:
                nsys_mount_root = _nsys_mount_root(nsys_bin)
            except Exception as exc:
                nsys_error = f"nsys_mount_failed:{exc!r}"
            nsys_version = _nsys_version(nsys_bin)
            if nsys_mount_root and not nsys_error:
                try:
                    nsys_profile_image = _build_nsys_image(
                        image_tag,
                        project_dir,
                    )
                    _validate_nsys_container_runtime(
                        nsys_profile_image,
                        nsys_mount_root,
                    )
                    print(
                        "[execution-profile][nsys] QdstrmImporter preflight "
                        f"passed in {nsys_profile_image}"
                    )
                except Exception as exc:
                    detail = str(exc)
                    nsys_error = (
                        detail
                        if detail.startswith("nsys_")
                        else f"nsys_runtime_preflight_failed:{exc!r}"
                    )
        elif not nsys_error:
            nsys_error = "nsys_not_found"

    source_profiles: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    massif_started = perf_counter()
    for cpu, mem in massif_sources:
        source_profiles[(MASSIF_TOOL, cpu, mem)] = _profile_massif_tool(
            entries=entries,
            global_error=massif_error,
            task_info=task_info,
            derived_image=derived_image,
            cpu=cpu,
            mem=mem,
            payload_file=input_scale_plan_file,
            profile_root=profile_root,
            output_dir=output_dir,
            repeat=normalized_massif_repeat,
            resume_existing=resume_existing_profiles,
        )
    if collect_massif:
        report_profiler_completion(
            progress_callback,
            profiler="Massif",
            profiles=(
                source_profiles[(MASSIF_TOOL, cpu, mem)]
                for cpu, mem in massif_sources
            ),
            elapsed_seconds=perf_counter() - massif_started,
        )
    nsys_started = perf_counter()
    for cpu, mem in nsys_sources:
        source_profiles[(NSYS_TOOL, cpu, mem)] = _profile_nsys_tool(
            entries=entries,
            global_error=nsys_error,
            task_info=task_info,
            image_tag=nsys_profile_image,
            nsys_bin=nsys_bin,
            nsys_mount_root=nsys_mount_root,
            cpu=cpu,
            mem=mem,
            payload_file=input_scale_plan_file,
            profile_root=profile_root,
            output_dir=output_dir,
            repeat=normalized_nsys_repeat,
        )
    if collect_nsys:
        report_profiler_completion(
            progress_callback,
            profiler="Nsys",
            profiles=(
                source_profiles[(NSYS_TOOL, cpu, mem)]
                for cpu, mem in nsys_sources
            ),
            elapsed_seconds=perf_counter() - nsys_started,
        )

    profiles: List[Dict[str, Any]] = []
    for cpu in cpus:
        for mem in memories:
            for gpu_mode in gpu_modes:
                tools: Dict[str, Any] = {}
                if gpu_mode == "off" and collect_massif:
                    source_cpu, source_mem = _profile_source_resource(
                        cpu=cpu,
                        mem=mem,
                        sampling=normalized_massif_sampling,
                        reference_cpu=massif_reference[0],
                        reference_mem=massif_reference[1],
                    )
                    tools[MASSIF_TOOL] = _copy_profile_with_provenance(
                        source_profiles[(MASSIF_TOOL, source_cpu, source_mem)],
                        source_cpu=source_cpu,
                        source_mem=source_mem,
                        sampling=normalized_massif_sampling,
                    )
                if gpu_mode == "on" and collect_nsys:
                    source_cpu, source_mem = _profile_source_resource(
                        cpu=cpu,
                        mem=mem,
                        sampling=normalized_nsys_sampling,
                        reference_cpu=nsys_reference[0],
                        reference_mem=nsys_reference[1],
                    )
                    tools[NSYS_TOOL] = _copy_profile_with_provenance(
                        source_profiles[(NSYS_TOOL, source_cpu, source_mem)],
                        source_cpu=source_cpu,
                        source_mem=source_mem,
                        sampling=normalized_nsys_sampling,
                    )
                if tools:
                    profiles.append(
                        {
                            "cpu_cores": cpu,
                            "mem_cap_gb": mem,
                            "gpu_mode": gpu_mode,
                            "tools": tools,
                        }
                    )

    enabled_tools = [
        tool
        for tool, enabled in (
            (MASSIF_TOOL, collect_massif),
            (NSYS_TOOL, collect_nsys),
        )
        if enabled
    ]
    static_metadata = {
        "execution_profile_schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "execution_profile_tools": enabled_tools,
        "massif_peak_semantics": (
            "process_lifetime_including_model_load_and_warmup; "
            "independent_component_maxima; total=max_snapshot("
            "heap+heap_extra+stack), time=that_snapshot"
        ),
        "massif_repeat": (
            normalized_massif_repeat if collect_massif else None
        ),
        "massif_version": massif_version,
        "massif_sampling_strategy": (
            SAMPLING_STRATEGY_METADATA[normalized_massif_sampling]
            if collect_massif
            else None
        ),
        "massif_reference_cpu_cores": (
            massif_reference[0] if collect_massif else None
        ),
        "massif_reference_mem_cap_gb": (
            massif_reference[1] if collect_massif else None
        ),
        "massif_reused_across_resource_cases": bool(
            collect_massif and normalized_massif_sampling != "full"
        ),
        "nsys_timeline_semantics": (
            "NVTX acprof_compute capture; CUDA API, kernel, and memcpy "
            "sums/counts normalized per request"
        ),
        "nsys_repeat": normalized_nsys_repeat if collect_nsys else None,
        "nsys_version": nsys_version,
        "nsys_sampling_strategy": (
            SAMPLING_STRATEGY_METADATA[normalized_nsys_sampling]
            if collect_nsys
            else None
        ),
        "nsys_reference_cpu_cores": (
            nsys_reference[0] if collect_nsys else None
        ),
        "nsys_reference_mem_cap_gb": (
            nsys_reference[1] if collect_nsys else None
        ),
        "nsys_reused_across_resource_cases": bool(
            collect_nsys and normalized_nsys_sampling != "full"
        ),
        "execution_profiles_retained": bool(keep_profiles and enabled_tools),
        "execution_profile_provenance": (
            "collected" if enabled_tools else "disabled"
        ),
    }
    plan = {
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "model_id": task_info.model_id,
        "model_revision": task_info.model_revision or "main",
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "runtime_backend": task_info.runtime_backend,
        "execution_profile_tool_mode": normalized_tool_mode,
        "massif_sampling": (
            normalized_massif_sampling if collect_massif else None
        ),
        "nsys_sampling": normalized_nsys_sampling if collect_nsys else None,
        "static_metadata": static_metadata,
        "profiles": profiles,
    }
    plan_path = os.path.join(output_dir, EXECUTION_PROFILE_PLAN_NAME)

    if not keep_profiles and enabled_tools:
        _strip_artifact_paths(profiles)
        shutil.rmtree(profile_root, ignore_errors=True)

    _write_json_atomic(plan_path, plan)
    print(f"[execution] Execution profile plan: {plan_path}")
    return plan_path

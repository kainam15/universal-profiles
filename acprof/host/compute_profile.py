"""Host-side FLOP profiling plan generation for AC-Prof."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from acprof.config import DEFAULT_COMPUTE_PROFILE_TOOL
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


from acprof.host.profilers.compute_parsers import (
    NCU_TENSOR_METRIC_RE,
    NCU_SASS_FLOP_WEIGHTS,
    NCU_DURATION_METRIC,
    _to_float,
    _finite_or_none,
    parse_advisor_self_gflop_csv,
    _is_ncu_flop_metric,
    parse_ncu_profile_csv,
)

from acprof.host.profilers.tool_discovery import _find_executable, _tool_mount_roots


COMPUTE_PROFILE_PLAN_NAME = "compute_profile_plan.json"
TORCH_PROFILER_TOOL = "torch_profiler_eager"
NCU_TOOL = "ncu"
COMPUTE_PROFILE_TOOL_MODES = {"none", "both", "ncu", "torch", "vendor"}
NCU_FMA_FLOP_WEIGHT = 2.0
NCU_CHECKPOINT_SCHEMA_VERSION = 1
NCU_COMPLETE_ENTRY_FIELDS = (
    "gpu_executed_mflop_per_request_ncu",
    "gpu_executed_tensor_mflop_per_request_ncu",
    "gpu_executed_scalar_mflop_per_request_ncu",
    "gpu_executed_tensor_share_pct_ncu",
    "gpu_kernel_launch_count_per_request_ncu",
    "gpu_kernel_time_sum_ms_per_request_ncu",
)


def _normal_gpu_mode(gpu: str) -> str:
    return "on" if str(gpu).lower() == "on" else "off"


def _host_logical_cpus() -> int:
    return max(1, int(os.cpu_count() or 1))


def _host_memory_gb_fraction(fraction: float = 0.75) -> int:
    total_bytes = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_bytes = int(line.split()[1]) * 1024
                    break
    except OSError:
        total_bytes = 0

    if total_bytes <= 0:
        return 1
    return max(1, int((total_bytes * fraction) // (1024 ** 3)))


def _default_compute_profile_resources(
    compute_profile_cpus: Optional[int],
    compute_profile_mem: Optional[int],
) -> Tuple[int, int]:
    cpu = max(1, int(compute_profile_cpus or _host_logical_cpus()))
    mem = max(1, int(compute_profile_mem or _host_memory_gb_fraction(0.75)))
    return cpu, mem


def _tool_error_entries(
    entries: List[Dict[str, Any]],
    error: str,
    tool: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "input_scale": float(entry["input_scale"]),
            "tool": tool,
            "model_mflop_per_request": None,
            "error": error,
        }
        for entry in entries
    ]


def _ncu_error_entries(
    entries: List[Dict[str, Any]],
    error: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "input_scale": float(entry["input_scale"]),
            "tool": NCU_TOOL,
            "gpu_executed_mflop_per_request_ncu": None,
            "gpu_executed_tensor_mflop_per_request_ncu": None,
            "gpu_executed_scalar_mflop_per_request_ncu": None,
            "gpu_executed_tensor_share_pct_ncu": None,
            "gpu_kernel_launch_count_per_request_ncu": None,
            "gpu_kernel_time_sum_ms_per_request_ncu": None,
            "error": error,
        }
        for entry in entries
    ]


def _torch_error_entries(
    entries: List[Dict[str, Any]],
    error: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "input_scale": float(entry["input_scale"]),
            "tool": TORCH_PROFILER_TOOL,
            "model_logical_mflop_per_request_torch_profiler_eager": None,
            "error": error,
        }
        for entry in entries
    ]


def _run_advisor_for_entry(
    *,
    advisor_bin: str,
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    tool_mount_roots: Sequence[str],
    entry: Dict[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    scale_label = _format_scale_value(float(entry["input_scale"]))
    project_dir = f"/profiles/advisor_scale_{scale_label}"
    report_path = f"/profiles/advisor_scale_{scale_label}.csv"
    host_report_path = os.path.join(profile_root, f"advisor_scale_{scale_label}.csv")
    base_cmd = _base_docker_cmd(
        task_info=task_info,
        image_tag=image_tag,
        cpu=cpu,
        mem=mem,
        use_gpu=False,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=tool_mount_roots,
    )
    runner_args = _runner_args(entry, repeat, "cpu")
    commands = [
        [
            advisor_bin,
            "--collect=survey",
            "--profile-python=off",
            "--start-paused",
            "--project-dir", project_dir,
            "--",
            *runner_args,
        ],
        [
            advisor_bin,
            "--collect=tripcounts",
            "--flop",
            "--profile-jit",
            "--start-paused",
            "--project-dir", project_dir,
            "--",
            *runner_args,
        ],
        [
            advisor_bin,
            "--report=survey",
            "--format=csv",
            "--show-all-columns",
            "--project-dir", project_dir,
            "--report-output", report_path,
        ],
    ]
    for command in commands:
        result = _run([*base_cmd, *command], check=False)
        if result.returncode != 0:
            return {
                "input_scale": float(entry["input_scale"]),
                "model_mflop_per_request": None,
                "error": f"advisor_failed:{result.stderr.strip() or result.stdout.strip()}",
            }
    gflop = parse_advisor_self_gflop_csv(host_report_path)
    if gflop != gflop:
        return {
            "input_scale": float(entry["input_scale"]),
            "model_mflop_per_request": None,
            "error": "advisor_parse_failed:self_gflop_missing",
        }
    return {
        "input_scale": float(entry["input_scale"]),
        "tool": "intel_advisor",
        "model_mflop_per_request": (gflop * 1000.0) / float(max(1, int(repeat))),
        "error": "",
        "report": host_report_path,
    }


def _parse_ncu_metric_names(query_output: str) -> List[str]:
    names = set()
    for line in query_output.splitlines():
        for token in re.split(r"[\s,]+", line.strip()):
            token = token.strip()
            if not token:
                continue
            if _is_ncu_flop_metric(token):
                names.add(token)
    return sorted(names)


def _select_ncu_flop_metrics(available_metrics: Iterable[str]) -> List[str]:
    available = set(available_metrics)
    sass_metrics = []
    for metric in NCU_SASS_FLOP_WEIGHTS:
        rollup_metric = f"{metric}.sum"
        if rollup_metric in available:
            sass_metrics.append(rollup_metric)
        elif metric in available:
            sass_metrics.append(metric)

    tensor_metrics = sorted(
        metric for metric in available
        if NCU_TENSOR_METRIC_RE.match(metric)
    )
    return [*sass_metrics, *tensor_metrics]


def _resolve_ncu_metrics(
    ncu_bin: str,
    *,
    container_base_cmd: Optional[Sequence[str]] = None,
) -> Tuple[List[str], str]:
    query_errors = []
    command_prefixes: List[Sequence[str]] = [()]
    if container_base_cmd:
        command_prefixes.append(container_base_cmd)

    for command_prefix in command_prefixes:
        for query_args in (
            ["--query-metrics", "--query-metrics-mode", "all"],
            ["--query-metrics"],
        ):
            result = _run(
                [*command_prefix, ncu_bin, *query_args],
                check=False,
            )
            if result.returncode != 0:
                query_errors.append(result.stderr.strip() or result.stdout.strip())
                continue
            metrics = _select_ncu_flop_metrics(
                _parse_ncu_metric_names(result.stdout)
            )
            if metrics:
                return metrics, ""
            detail = result.stderr.strip() or result.stdout.strip()
            if detail:
                query_errors.append(detail)
    error = "; ".join(error for error in query_errors if error)
    if error:
        return [], f"ncu_no_flop_metrics_found:{error}"
    return [], "ncu_no_flop_metrics_found"


def _ncu_supports_nvtx_filter(ncu_bin: str) -> bool:
    result = _run([ncu_bin, "--help"], check=False)
    return result.returncode == 0 and "--nvtx-include" in result.stdout


def _ncu_collect_filter_args(ncu_bin: str) -> List[str]:
    if _ncu_supports_nvtx_filter(ncu_bin):
        return ["--nvtx", "--nvtx-include", "acprof_compute/"]
    return ["--nvtx", "--nvtx-include", "acprof_compute"]


def _ncu_section_args(ncu_bin: str) -> List[str]:
    section_dir = os.path.join(os.path.dirname(os.path.realpath(ncu_bin)), "sections")
    if os.path.isdir(section_dir):
        return ["--section-folder", section_dir, "--apply-rules", "no"]
    return []


def _write_text_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _ncu_artifact_paths(
    profile_root: str,
    input_scale: float,
) -> Tuple[str, str, str, str]:
    scale_label = _format_scale_value(input_scale)
    report_base = f"/profiles/ncu_scale_{scale_label}"
    host_csv = os.path.join(profile_root, f"ncu_scale_{scale_label}.csv")
    host_report = os.path.join(profile_root, f"ncu_scale_{scale_label}.ncu-rep")
    checkpoint = os.path.join(
        profile_root,
        f"ncu_scale_{scale_label}.checkpoint.json",
    )
    return report_base, host_csv, host_report, checkpoint


def _ncu_report_reference(profile_root: str, host_csv: str) -> str:
    return os.path.relpath(
        host_csv,
        start=os.path.dirname(os.path.abspath(profile_root)),
    )


def _ncu_entry_complete(entry: Dict[str, Any]) -> bool:
    return not str(entry.get("error") or "").strip() and all(
        math.isfinite(_to_float(entry.get(field)))
        for field in NCU_COMPLETE_ENTRY_FIELDS
    )


def _ncu_entry_from_csv(
    *,
    entry: Dict[str, Any],
    host_csv: str,
    profile_root: str,
    repeat: int,
    runner_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_path = _ncu_report_reference(profile_root, host_csv)
    try:
        parsed = parse_ncu_profile_csv(host_csv, repeat=repeat)
    except Exception as exc:
        error_entry = _ncu_error_entries(
            [entry],
            f"ncu_parse_failed:{exc!r}",
        )[0]
        error_entry["report"] = report_path
        return error_entry

    payload = runner_payload or {}
    total_flops = _to_float(parsed.get("total_flops_per_request"))
    if not math.isfinite(total_flops):
        error_entry = _ncu_error_entries(
            [entry],
            "ncu_parse_failed:flop_metrics_missing",
        )[0]
        error_entry["report"] = report_path
        return error_entry

    tensor_flops = _finite_or_none(parsed.get("tensor_flops_per_request"))
    scalar_flops = _finite_or_none(parsed.get("scalar_flops_per_request"))
    result = {
        "input_scale": float(entry["input_scale"]),
        "tool": NCU_TOOL,
        "gpu_executed_mflop_per_request_ncu": total_flops / 1_000_000.0,
        "gpu_executed_tensor_mflop_per_request_ncu": (
            tensor_flops / 1_000_000.0 if tensor_flops is not None else None
        ),
        "gpu_executed_scalar_mflop_per_request_ncu": (
            scalar_flops / 1_000_000.0 if scalar_flops is not None else None
        ),
        "gpu_executed_tensor_share_pct_ncu": _finite_or_none(
            parsed.get("tensor_share_pct")
        ),
        "gpu_kernel_launch_count_per_request_ncu": _finite_or_none(
            parsed.get("kernel_launch_count_per_request")
        ),
        "gpu_kernel_time_sum_ms_per_request_ncu": _finite_or_none(
            parsed.get("kernel_time_sum_ms_per_request")
        ),
        "gpu_compute_capability": str(
            payload.get("gpu_compute_capability")
            or parsed.get("gpu_compute_capability")
            or ""
        ),
        "gpu_sm_count": _finite_or_none(
            payload.get("gpu_sm_count", parsed.get("gpu_sm_count"))
        ),
        "error": "",
        "report": report_path,
    }
    missing = [
        field
        for field in NCU_COMPLETE_ENTRY_FIELDS
        if not math.isfinite(_to_float(result.get(field)))
    ]
    if missing:
        error_entry = _ncu_error_entries(
            [entry],
            "ncu_parse_failed:required_metrics_missing:" + ",".join(missing),
        )[0]
        error_entry["report"] = report_path
        return error_entry
    return result


def _export_ncu_report(
    *,
    ncu_bin: str,
    base_cmd: Sequence[str],
    report_base: str,
    host_csv: str,
) -> Tuple[bool, str]:
    import_cmd = [
        ncu_bin,
        "--import", f"{report_base}.ncu-rep",
        "--page", "raw",
        "--csv",
    ]
    result = _run([*base_cmd, *import_cmd], check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "ncu import failed").strip()
        return False, detail
    if not result.stdout:
        return False, "ncu import produced an empty CSV"
    _write_text_atomic(host_csv, result.stdout)
    return True, ""


def _read_ncu_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _ncu_checkpoint_matches(
    checkpoint: Dict[str, Any],
    *,
    task_info: TaskInfo,
    image_tag: str,
    input_scale: float,
    repeat: int,
    metrics: Sequence[str],
) -> bool:
    try:
        checkpoint_scale = float(checkpoint.get("input_scale"))
        checkpoint_repeat = int(checkpoint.get("repeat"))
        schema_version = int(checkpoint.get("schema_version"))
    except (TypeError, ValueError):
        return False
    return (
        schema_version == NCU_CHECKPOINT_SCHEMA_VERSION
        and str(checkpoint.get("model_id") or "") == task_info.model_id
        and str(checkpoint.get("model_revision") or "main")
        == str(task_info.model_revision or "main")
        and str(checkpoint.get("image_tag") or "") == image_tag
        and math.isclose(checkpoint_scale, input_scale, abs_tol=1e-9)
        and checkpoint_repeat == max(1, int(repeat))
        and list(checkpoint.get("metrics") or []) == list(metrics)
    )


def _write_ncu_checkpoint(
    *,
    checkpoint_path: str,
    task_info: TaskInfo,
    image_tag: str,
    input_scale: float,
    repeat: int,
    metrics: Sequence[str],
    host_csv: str,
    entry: Dict[str, Any],
) -> None:
    _write_json_atomic(
        checkpoint_path,
        {
            "schema_version": NCU_CHECKPOINT_SCHEMA_VERSION,
            "model_id": task_info.model_id,
            "model_revision": task_info.model_revision or "main",
            "image_tag": image_tag,
            "input_scale": float(input_scale),
            "repeat": max(1, int(repeat)),
            "metrics": list(metrics),
            "csv_size_bytes": os.path.getsize(host_csv),
            "entry": entry,
        },
    )


def _resume_ncu_for_entry(
    *,
    ncu_bin: str,
    ncu_metrics: Sequence[str],
    task_info: TaskInfo,
    image_tag: str,
    base_cmd: Sequence[str],
    profile_root: str,
    entry: Dict[str, Any],
    repeat: int,
) -> Optional[Dict[str, Any]]:
    input_scale = float(entry["input_scale"])
    scale_label = _format_scale_value(input_scale)
    report_base, host_csv, host_report, checkpoint_path = _ncu_artifact_paths(
        profile_root,
        input_scale,
    )
    checkpoint_exists = os.path.isfile(checkpoint_path)
    checkpoint = _read_ncu_checkpoint(checkpoint_path) if checkpoint_exists else None
    if checkpoint_exists and (
        checkpoint is None
        or not _ncu_checkpoint_matches(
            checkpoint,
            task_info=task_info,
            image_tag=image_tag,
            input_scale=input_scale,
            repeat=repeat,
            metrics=ncu_metrics,
        )
    ):
        print(
            f"[compute][ncu][resume] scale={scale_label}: "
            "checkpoint does not match this run; recollecting"
        )
        return None

    if checkpoint is not None and os.path.isfile(host_csv):
        checkpoint_entry = checkpoint.get("entry")
        expected_size = _to_float(checkpoint.get("csv_size_bytes"))
        if (
            isinstance(checkpoint_entry, dict)
            and _ncu_entry_complete(checkpoint_entry)
            and math.isfinite(expected_size)
            and os.path.getsize(host_csv) == int(expected_size)
        ):
            resumed = dict(checkpoint_entry)
            resumed["report"] = _ncu_report_reference(profile_root, host_csv)
            print(
                f"[compute][ncu][resume] scale={scale_label}: "
                f"reusing checkpoint {checkpoint_path}"
            )
            return resumed

    if checkpoint is None and os.path.isfile(host_csv):
        resumed = _ncu_entry_from_csv(
            entry=entry,
            host_csv=host_csv,
            profile_root=profile_root,
            repeat=repeat,
        )
        if _ncu_entry_complete(resumed):
            _write_ncu_checkpoint(
                checkpoint_path=checkpoint_path,
                task_info=task_info,
                image_tag=image_tag,
                input_scale=input_scale,
                repeat=repeat,
                metrics=ncu_metrics,
                host_csv=host_csv,
                entry=resumed,
            )
            print(
                f"[compute][ncu][resume] scale={scale_label}: "
                f"reusing valid CSV {host_csv}"
            )
            return resumed
        print(
            f"[compute][ncu][resume] scale={scale_label}: "
            "existing CSV is incomplete"
        )

    if os.path.isfile(host_report):
        print(
            f"[compute][ncu][resume] scale={scale_label}: "
            f"exporting existing report {host_report}"
        )
        exported, detail = _export_ncu_report(
            ncu_bin=ncu_bin,
            base_cmd=base_cmd,
            report_base=report_base,
            host_csv=host_csv,
        )
        if exported:
            resumed = _ncu_entry_from_csv(
                entry=entry,
                host_csv=host_csv,
                profile_root=profile_root,
                repeat=repeat,
            )
            if _ncu_entry_complete(resumed):
                _write_ncu_checkpoint(
                    checkpoint_path=checkpoint_path,
                    task_info=task_info,
                    image_tag=image_tag,
                    input_scale=input_scale,
                    repeat=repeat,
                    metrics=ncu_metrics,
                    host_csv=host_csv,
                    entry=resumed,
                )
                print(
                    f"[compute][ncu][resume] scale={scale_label}: "
                    "recovered from existing .ncu-rep"
                )
                return resumed
            detail = str(resumed.get("error") or "exported CSV is incomplete")
        print(
            f"[compute][ncu][resume] scale={scale_label}: "
            f"report recovery failed ({detail[:300]}); recollecting"
        )
    return None


def _run_ncu_for_entry(
    *,
    ncu_bin: str,
    ncu_metrics: List[str],
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    tool_mount_roots: Sequence[str],
    entry: Dict[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    report_base, host_csv, _host_report, _checkpoint = _ncu_artifact_paths(
        profile_root,
        float(entry["input_scale"]),
    )
    base_cmd = _base_docker_cmd(
        task_info=task_info,
        image_tag=image_tag,
        cpu=cpu,
        mem=mem,
        use_gpu=True,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=tool_mount_roots,
    )
    collect_cmd = [
        ncu_bin,
        "--target-processes", "all",
        *_ncu_collect_filter_args(ncu_bin),
        *_ncu_section_args(ncu_bin),
        "--page", "raw",
        "--csv",
        "--metrics", ",".join(ncu_metrics),
        "-f",
        "-o", report_base,
        *_runner_args(entry, repeat, "gpu"),
    ]
    result = _run([*base_cmd, *collect_cmd], check=False)
    if result.returncode != 0:
        return _ncu_error_entries(
            [entry],
            f"ncu_failed:{result.stderr.strip() or result.stdout.strip()}",
        )[0]

    exported, detail = _export_ncu_report(
        ncu_bin=ncu_bin,
        base_cmd=base_cmd,
        report_base=report_base,
        host_csv=host_csv,
    )
    if not exported:
        error_entry = _ncu_error_entries(
            [entry],
            f"ncu_import_failed:{detail}",
        )[0]
        error_entry["report"] = _ncu_report_reference(profile_root, host_csv)
        return error_entry
    return _ncu_entry_from_csv(
        entry=entry,
        host_csv=host_csv,
        profile_root=profile_root,
        repeat=repeat,
        runner_payload=_parse_last_json_line(result.stdout),
    )


def _run_torch_profiler_for_entry(
    *,
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    use_gpu: bool,
    payload_file: str,
    profile_root: str,
    entry: Dict[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    base_cmd = _base_docker_cmd(
        task_info=task_info,
        image_tag=image_tag,
        cpu=cpu,
        mem=mem,
        use_gpu=use_gpu,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=(),
    )
    runner_mode = "torch_eager_gpu" if use_gpu else "torch_eager_cpu"
    result = _run([*base_cmd, *_runner_args(entry, repeat, runner_mode)], check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return _torch_error_entries(
            [entry],
            f"torch_profiler_eager_failed:{detail}",
        )[0]

    payload = _parse_last_json_line(result.stdout)
    attention_implementation = str(
        payload.get("attention_implementation") or ""
    )
    attention_verified = payload.get("attention_implementation_verified") is True
    if attention_implementation != "eager" or not attention_verified:
        return _torch_error_entries(
            [entry],
            "torch_profiler_eager_parse_failed:"
            "attention_implementation_not_verified",
        )[0]
    mflop = _to_float(
        payload.get("model_logical_mflop_per_request_torch_profiler_eager")
    )
    total_flops = _to_float(payload.get("total_flops"))
    if mflop != mflop and total_flops == total_flops:
        mflop = (total_flops / 1_000_000.0) / float(max(1, int(repeat)))
    if mflop != mflop or mflop <= 0:
        return _torch_error_entries(
            [entry],
            "torch_profiler_eager_parse_failed:model_logical_mflop_missing",
        )[0]

    return {
        "input_scale": float(entry["input_scale"]),
        "tool": TORCH_PROFILER_TOOL,
        "model_logical_mflop_per_request_torch_profiler_eager": mflop,
        "error": "",
        "total_flops": total_flops if total_flops == total_flops else None,
        "attention_implementation": attention_implementation,
        "attention_implementation_verified": attention_verified,
        "torch_version": str(payload.get("torch_version") or "unknown"),
        "transformers_version": str(
            payload.get("transformers_version") or "unknown"
        ),
    }


def _profile_torch_entries(
    *,
    entries: List[Dict[str, Any]],
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    use_gpu: bool,
    profile_key: str,
    payload_file: str,
    profile_root: str,
    repeat: int,
) -> Dict[str, Any]:
    profile_entries = []
    for entry in entries:
        try:
            profile_entry = _run_torch_profiler_for_entry(
                task_info=task_info,
                image_tag=image_tag,
                cpu=cpu,
                mem=mem,
                use_gpu=use_gpu,
                payload_file=payload_file,
                profile_root=profile_root,
                entry=entry,
                repeat=repeat,
            )
        except Exception as exc:
            profile_entry = _torch_error_entries(
                [entry],
                f"torch_profiler_eager_failed:{exc!r}",
            )[0]
        profile_entries.append(profile_entry)
    errors = [entry["error"] for entry in profile_entries if entry.get("error")]
    successful_entry = next(
        (entry for entry in profile_entries if not entry.get("error")),
        {},
    )
    return {
        "tool": TORCH_PROFILER_TOOL,
        "repeat": max(1, int(repeat)),
        "profile": profile_key,
        "flop_semantics": "logical_operator_shape_flops",
        "attention_implementation": "eager",
        "torch_version": successful_entry.get("torch_version", "unknown"),
        "transformers_version": successful_entry.get(
            "transformers_version",
            "unknown",
        ),
        "error": "; ".join(errors),
        "entries": profile_entries,
    }


def _profile_cpu_entries(
    *,
    entries: List[Dict[str, Any]],
    advisor_bin: Optional[str],
    advisor_root: Optional[str],
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    repeat: int,
) -> Dict[str, Any]:
    if advisor_bin is None:
        return {
            "tool": "intel_advisor",
            "repeat": max(1, int(repeat)),
            "error": "advisor_not_found",
            "entries": _tool_error_entries(
                entries,
                "advisor_not_found",
                "intel_advisor",
            ),
        }
    mount_roots = _tool_mount_roots(advisor_bin, advisor_root)
    profile_entries = [
        _run_advisor_for_entry(
            advisor_bin=advisor_bin,
            task_info=task_info,
            image_tag=image_tag,
            cpu=cpu,
            mem=mem,
            payload_file=payload_file,
            profile_root=profile_root,
            tool_mount_roots=mount_roots,
            entry=entry,
            repeat=repeat,
        )
        for entry in entries
    ]
    errors = [entry["error"] for entry in profile_entries if entry.get("error")]
    return {
        "tool": "intel_advisor",
        "repeat": max(1, int(repeat)),
        "error": "; ".join(errors),
        "entries": profile_entries,
    }


def _profile_gpu_entries(
    *,
    entries: List[Dict[str, Any]],
    ncu_bin: Optional[str],
    ncu_root: Optional[str],
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    repeat: int,
    resume_existing: bool = False,
) -> Dict[str, Any]:
    if ncu_bin is None:
        return {
            "tool": NCU_TOOL,
            "repeat": max(1, int(repeat)),
            "error": "ncu_not_found",
            "entries": _ncu_error_entries(entries, "ncu_not_found"),
        }
    mount_roots = _tool_mount_roots(ncu_bin, ncu_root)
    metric_query_base_cmd = _base_docker_cmd(
        task_info=task_info,
        image_tag=image_tag,
        cpu=cpu,
        mem=mem,
        use_gpu=True,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=mount_roots,
    )
    ncu_metrics, metric_error = _resolve_ncu_metrics(
        ncu_bin,
        container_base_cmd=metric_query_base_cmd,
    )
    if metric_error:
        return {
            "tool": NCU_TOOL,
            "repeat": max(1, int(repeat)),
            "error": metric_error,
            "entries": _ncu_error_entries(entries, metric_error),
        }
    collection_metrics = list(dict.fromkeys([*ncu_metrics, NCU_DURATION_METRIC]))
    profile_entries = []
    for entry in entries:
        try:
            profile_entry = None
            if resume_existing:
                profile_entry = _resume_ncu_for_entry(
                    ncu_bin=ncu_bin,
                    ncu_metrics=collection_metrics,
                    task_info=task_info,
                    image_tag=image_tag,
                    base_cmd=metric_query_base_cmd,
                    profile_root=profile_root,
                    entry=entry,
                    repeat=repeat,
                )
            if profile_entry is None:
                scale_label = _format_scale_value(float(entry["input_scale"]))
                print(f"[compute][ncu] scale={scale_label}: collecting")
                profile_entry = _run_ncu_for_entry(
                    ncu_bin=ncu_bin,
                    ncu_metrics=collection_metrics,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=cpu,
                    mem=mem,
                    payload_file=payload_file,
                    profile_root=profile_root,
                    tool_mount_roots=mount_roots,
                    entry=entry,
                    repeat=repeat,
                )
                if _ncu_entry_complete(profile_entry):
                    _report_base, host_csv, _host_report, checkpoint_path = (
                        _ncu_artifact_paths(
                            profile_root,
                            float(entry["input_scale"]),
                        )
                    )
                    _write_ncu_checkpoint(
                        checkpoint_path=checkpoint_path,
                        task_info=task_info,
                        image_tag=image_tag,
                        input_scale=float(entry["input_scale"]),
                        repeat=repeat,
                        metrics=collection_metrics,
                        host_csv=host_csv,
                        entry=profile_entry,
                    )
        except Exception as exc:
            profile_entry = _ncu_error_entries(
                [entry],
                f"ncu_failed:{exc!r}",
            )[0]
        profile_entries.append(profile_entry)
    errors = [entry["error"] for entry in profile_entries if entry.get("error")]
    return {
        "tool": NCU_TOOL,
        "repeat": max(1, int(repeat)),
        "flop_semantics": "gpu_executed_floating_point_operations",
        "fma_flop_weight": NCU_FMA_FLOP_WEIGHT,
        "metrics": collection_metrics,
        "error": "; ".join(errors),
        "entries": profile_entries,
    }


def _failed_tool_profile(
    *,
    tool: str,
    entries: List[Dict[str, Any]],
    repeat: int,
    error: str,
) -> Dict[str, Any]:
    error_entries = (
        _ncu_error_entries(entries, error)
        if tool == NCU_TOOL
        else _torch_error_entries(entries, error)
        if tool == TORCH_PROFILER_TOOL
        else _tool_error_entries(entries, error, tool)
    )
    return {
        "tool": tool,
        "repeat": max(1, int(repeat)),
        "error": error,
        "entries": error_entries,
    }


def _safe_profile_tool(
    *,
    tool: str,
    entries: List[Dict[str, Any]],
    repeat: int,
    callback: Any,
) -> Dict[str, Any]:
    try:
        return callback()
    except Exception as exc:
        return _failed_tool_profile(
            tool=tool,
            entries=entries,
            repeat=repeat,
            error=f"{tool}_failed:{exc!r}",
        )


def _executable_version(executable: Optional[str]) -> str:
    if not executable:
        return "unknown"
    try:
        result = _run([executable, "--version"], check=False)
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    text = (result.stdout or result.stderr or "").strip()
    if not text:
        return "unknown"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def _profile_tool_maps(profiles: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for device_profile in profiles.values():
        if not isinstance(device_profile, dict):
            continue
        for tool in (TORCH_PROFILER_TOOL, NCU_TOOL, "intel_advisor"):
            profile = device_profile.get(tool)
            if isinstance(profile, dict):
                yield profile


def _first_profile_value(
    profiles: Dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    for profile in _profile_tool_maps(profiles):
        value = profile.get(key)
        if value not in (None, "", "unknown"):
            return value
        for entry in profile.get("entries", []):
            if not isinstance(entry, dict):
                continue
            value = entry.get(key)
            if value not in (None, "", "unknown"):
                if not isinstance(value, float) or value == value:
                    return value
    return default


def _strip_discarded_profile_paths(profiles: Dict[str, Any]) -> None:
    for profile in _profile_tool_maps(profiles):
        for entry in profile.get("entries", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("tool") == NCU_TOOL:
                entry["report"] = None


def collect_compute_profile_plan(
    *,
    task_info: TaskInfo,
    image_tag: str,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    output_dir: str,
    input_scale_plan_file: str,
    advisor_root: Optional[str],
    ncu_root: Optional[str],
    advisor_repeat: int,
    ncu_repeat: int,
    keep_profiles: bool,
    compute_profile_cpus: Optional[int] = None,
    compute_profile_mem: Optional[int] = None,
    compute_profile_tool: str = DEFAULT_COMPUTE_PROFILE_TOOL,
    torch_profiler_repeat: int = 1,
    resume_existing_ncu_profiles: bool = False,
    progress_callback: Optional[ProfilerProgressCallback] = None,
) -> str:
    """Collect or synthesize compute profiles and write a plan file."""
    os.makedirs(output_dir, exist_ok=True)
    tool_mode = (
        compute_profile_tool or DEFAULT_COMPUTE_PROFILE_TOOL
    ).strip().lower()
    if tool_mode not in COMPUTE_PROFILE_TOOL_MODES:
        raise ValueError(
            "compute_profile_tool must be one of "
            f"{', '.join(sorted(COMPUTE_PROFILE_TOOL_MODES))}, got {compute_profile_tool!r}"
        )

    profile_root = os.path.join(output_dir, "compute_profiles")
    entries = _load_input_scale_plan_entries(input_scale_plan_file)
    payload_file = input_scale_plan_file

    normalized_gpus = {_normal_gpu_mode(gpu) for gpu in gpu_list}
    collect_torch_cpu = (
        "off" in normalized_gpus
        and tool_mode in {"both", "torch"}
    )
    collect_torch_gpu = (
        "on" in normalized_gpus
        and tool_mode in {"both", "torch"}
    )
    collect_advisor_cpu = (
        "off" in normalized_gpus
        and tool_mode == "vendor"
    )
    collect_ncu_gpu = (
        "on" in normalized_gpus
        and tool_mode in {"both", "ncu", "vendor"}
    )
    if (
        collect_torch_cpu
        or collect_torch_gpu
        or collect_advisor_cpu
        or collect_ncu_gpu
    ):
        os.makedirs(profile_root, exist_ok=True)
    advisor_bin = (
        _find_executable(advisor_root, ("advisor", "advixe-cl"))
        if collect_advisor_cpu
        else None
    )
    ncu_bin = (
        _find_executable(ncu_root, ("ncu", "nv-nsight-cu-cli"))
        if collect_ncu_gpu
        else None
    )
    max_cpu, max_mem = _default_compute_profile_resources(
        compute_profile_cpus,
        compute_profile_mem,
    )

    profiles: Dict[str, Any] = {}
    if "off" in normalized_gpus:
        cpu_tools: Dict[str, Dict[str, Any]] = {}
        if collect_advisor_cpu:
            started_at = time.perf_counter()
            cpu_tools["intel_advisor"] = _safe_profile_tool(
                tool="intel_advisor",
                entries=entries,
                repeat=advisor_repeat,
                callback=lambda: _profile_cpu_entries(
                    entries=entries,
                    advisor_bin=advisor_bin,
                    advisor_root=advisor_root,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=max_cpu,
                    mem=max_mem,
                    payload_file=payload_file,
                    profile_root=profile_root,
                    repeat=advisor_repeat,
                ),
            )
            report_profiler_completion(
                progress_callback,
                profiler="CPU Advisor",
                profiles=[cpu_tools["intel_advisor"]],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if collect_torch_cpu:
            started_at = time.perf_counter()
            cpu_tools[TORCH_PROFILER_TOOL] = _safe_profile_tool(
                tool=TORCH_PROFILER_TOOL,
                entries=entries,
                repeat=torch_profiler_repeat,
                callback=lambda: _profile_torch_entries(
                    entries=entries,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=max_cpu,
                    mem=max_mem,
                    use_gpu=False,
                    profile_key="cpu",
                    payload_file=payload_file,
                    profile_root=profile_root,
                    repeat=torch_profiler_repeat,
                ),
            )
            report_profiler_completion(
                progress_callback,
                profiler="CPU Torch",
                profiles=[cpu_tools[TORCH_PROFILER_TOOL]],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if cpu_tools:
            profiles["cpu"] = cpu_tools

    if "on" in normalized_gpus:
        gpu_tools: Dict[str, Dict[str, Any]] = {}
        if collect_torch_gpu:
            started_at = time.perf_counter()
            gpu_tools[TORCH_PROFILER_TOOL] = _safe_profile_tool(
                tool=TORCH_PROFILER_TOOL,
                entries=entries,
                repeat=torch_profiler_repeat,
                callback=lambda: _profile_torch_entries(
                    entries=entries,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=max_cpu,
                    mem=max_mem,
                    use_gpu=True,
                    profile_key="gpu",
                    payload_file=payload_file,
                    profile_root=profile_root,
                    repeat=torch_profiler_repeat,
                ),
            )
            report_profiler_completion(
                progress_callback,
                profiler="GPU Torch",
                profiles=[gpu_tools[TORCH_PROFILER_TOOL]],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if collect_ncu_gpu:
            started_at = time.perf_counter()
            gpu_tools[NCU_TOOL] = _safe_profile_tool(
                tool=NCU_TOOL,
                entries=entries,
                repeat=ncu_repeat,
                callback=lambda: _profile_gpu_entries(
                    entries=entries,
                    ncu_bin=ncu_bin,
                    ncu_root=ncu_root,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=max_cpu,
                    mem=max_mem,
                    payload_file=payload_file,
                    profile_root=profile_root,
                    repeat=ncu_repeat,
                    resume_existing=resume_existing_ncu_profiles,
                ),
            )
            report_profiler_completion(
                progress_callback,
                profiler="NCU",
                profiles=[gpu_tools[NCU_TOOL]],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if gpu_tools:
            profiles["gpu"] = gpu_tools

    enabled_tools = [
        tool
        for tool, enabled in (
            (TORCH_PROFILER_TOOL, collect_torch_cpu or collect_torch_gpu),
            (NCU_TOOL, collect_ncu_gpu),
            ("intel_advisor", collect_advisor_cpu),
        )
        if enabled
    ]
    ncu_metrics: List[str] = []
    gpu_profile = profiles.get("gpu", {})
    if isinstance(gpu_profile, dict):
        ncu_profile = gpu_profile.get(NCU_TOOL, {})
        if isinstance(ncu_profile, dict):
            ncu_metrics = list(ncu_profile.get("metrics") or [])

    static_metadata = {
        "compute_profile_tools": enabled_tools,
        "torch_profiler_eager_flop_semantics": "logical_operator_shape_flops",
        "torch_profiler_eager_attention_implementation": "eager",
        "torch_profiler_eager_repeat_cpu": (
            max(1, int(torch_profiler_repeat)) if collect_torch_cpu else None
        ),
        "torch_profiler_eager_repeat_gpu": (
            max(1, int(torch_profiler_repeat)) if collect_torch_gpu else None
        ),
        "ncu_flop_semantics": "gpu_executed_floating_point_operations",
        "ncu_repeat": max(1, int(ncu_repeat)) if collect_ncu_gpu else None,
        "ncu_fma_flop_weight": NCU_FMA_FLOP_WEIGHT,
        "ncu_metrics": ncu_metrics,
        "torch_version": _first_profile_value(
            profiles,
            "torch_version",
            "unknown",
        ),
        "transformers_version": _first_profile_value(
            profiles,
            "transformers_version",
            "unknown",
        ),
        "ncu_version": (
            _executable_version(ncu_bin) if collect_ncu_gpu else "unknown"
        ),
        "gpu_compute_capability": _first_profile_value(
            profiles,
            "gpu_compute_capability",
            "unknown",
        ),
        "gpu_sm_count": _first_profile_value(
            profiles,
            "gpu_sm_count",
            "unknown",
        ),
        "compute_profiles_retained": bool(keep_profiles and enabled_tools),
        "compute_profile_provenance": (
            "collected" if enabled_tools else "disabled"
        ),
    }

    plan = {
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "runtime_backend": task_info.runtime_backend,
        "compute_profile_tool_mode": tool_mode,
        "static_metadata": static_metadata,
        "profiles": profiles,
    }
    plan_path = os.path.join(output_dir, COMPUTE_PROFILE_PLAN_NAME)

    if not keep_profiles:
        _strip_discarded_profile_paths(profiles)
        shutil.rmtree(profile_root, ignore_errors=True)

    _write_json_atomic(plan_path, plan)
    print(f"[compute] Compute profile plan: {plan_path}")
    return plan_path

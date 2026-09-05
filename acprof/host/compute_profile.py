"""Host-side FLOP profiling plan generation for AC-Prof."""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from acprof.config import DEFAULT_COMPUTE_PROFILE_TOOL
from acprof.host.detect import TaskInfo
from acprof.host.env_utils import hf_offline_docker_env_args
from acprof.host.profiler_progress import (
    ProfilerProgressCallback,
    report_profiler_completion,
)


COMPUTE_PROFILE_PLAN_NAME = "compute_profile_plan.json"
CONTAINER_INPUT_SCALE_PLAN_FILE = "/payloads/input_scale_plan.json"
TORCH_PROFILER_TOOL = "torch_profiler_eager"
NCU_TOOL = "ncu"
COMPUTE_PROFILE_TOOL_MODES = {"none", "auto", "both", "ncu", "torch", "vendor"}
NCU_FLOAT_TYPE_PATTERN = r"(?:bf\d+|fp\d+|tf\d+)"
NCU_TENSOR_METRIC_RE = re.compile(
    rf"^sm__ops_path_tensor_src_{NCU_FLOAT_TYPE_PATTERN}"
    rf"(?:_{NCU_FLOAT_TYPE_PATTERN})*_dst_{NCU_FLOAT_TYPE_PATTERN}\.sum$"
)
NCU_SCALAR_FLOP_METRICS = ("flop_count_hp", "flop_count_sp", "flop_count_dp")
NCU_SASS_FLOP_WEIGHTS = {
    "smsp__sass_thread_inst_executed_op_dadd_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_dmul_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_dfma_pred_on": 2.0,
    "smsp__sass_thread_inst_executed_op_fadd_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_ffma_pred_on": 2.0,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on": 1.0,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on": 2.0,
}
NCU_DURATION_METRIC = "gpu__time_duration.sum"
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
DEFAULT_TOOL_SEARCH_ROOTS = (
    "/opt/intel/oneapi/advisor",
    "/opt/intel/oneapi",
    "/opt/nvidia/nsight-compute",
    "/usr/local/NVIDIA-Nsight-Compute",
    "/usr/local/cuda",
    "/usr/local/cuda-*",
    "/usr/lib/nsight-compute",
)


def _run(cmd: Sequence[str], check: bool = False, **kwargs) -> subprocess.CompletedProcess:
    print(f"  [cmd] {' '.join(str(part) for part in cmd)}")
    return subprocess.run(
        list(cmd),
        capture_output=kwargs.pop("capture_output", True),
        text=True,
        check=check,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )


def _format_scale_value(scale: float) -> str:
    value = float(scale)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


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


def _clean_numeric_text(value: str) -> str:
    value = value.strip().replace(",", "")
    if not value:
        return value
    suffix = value[-1].lower()
    factors = {
        "k": 1_000.0,
        "m": 1_000_000.0,
        "g": 1_000_000_000.0,
        "t": 1_000_000_000_000.0,
    }
    if suffix in factors:
        return str(float(value[:-1]) * factors[suffix])
    return value


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        if isinstance(value, str):
            value = _clean_numeric_text(value)
        return float(value)
    except Exception:
        return float("nan")


def _finite_or_none(value: Any) -> Optional[float]:
    number = _to_float(value)
    return number if math.isfinite(number) else None


def _parse_last_json_line(text: str) -> Dict[str, Any]:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_advisor_self_gflop_csv(report_path: str) -> float:
    """Return summed Intel Advisor Self GFLOP from a CSV report."""
    total_gflop = 0.0
    with open(report_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        fieldnames = None
        for row in reader:
            field_map = {field.strip().lower(): field for field in row}
            if "self gflop" in field_map:
                fieldnames = row
                break
        if fieldnames is None:
            return float("nan")
        field_map = {field.strip().lower(): field for field in fieldnames}
        gflop_field = field_map.get("self gflop")
        if gflop_field is None:
            return float("nan")
        found = False
        for row in csv.DictReader(f, fieldnames=fieldnames):
            value = _to_float(row.get(gflop_field))
            if value == value:
                total_gflop += value
                found = True
    return total_gflop if found else float("nan")


def _metric_name_from_row(row: Dict[str, str]) -> str:
    for key in ("Metric Name", "Metric", "Name"):
        value = row.get(key)
        if value:
            return value.strip()
    for key, value in row.items():
        if key and key.strip().lower() == "metric name" and value:
            return value.strip()
    return ""


def _metric_value_from_row(row: Dict[str, str]) -> float:
    for key in ("Metric Value", "Value"):
        if key in row:
            return _to_float(row.get(key))
    for key, value in row.items():
        if key and key.strip().lower() in {"metric value", "value"}:
            return _to_float(value)
    return float("nan")


def _metric_unit_from_row(row: Dict[str, str]) -> str:
    for key in ("Metric Unit", "Unit"):
        if key in row and row.get(key):
            return str(row[key]).strip()
    for key, value in row.items():
        if key and key.strip().lower() in {"metric unit", "unit"} and value:
            return str(value).strip()
    return ""


def _ncu_metric_flop_weight(metric_name: str) -> Optional[float]:
    if metric_name in NCU_SCALAR_FLOP_METRICS or NCU_TENSOR_METRIC_RE.match(metric_name):
        return 1.0
    if metric_name in NCU_SASS_FLOP_WEIGHTS:
        return NCU_SASS_FLOP_WEIGHTS[metric_name]
    if metric_name.endswith(".sum"):
        base_name = metric_name[:-len(".sum")]
        if base_name in NCU_SASS_FLOP_WEIGHTS:
            return NCU_SASS_FLOP_WEIGHTS[base_name]
    return None


def _is_ncu_flop_metric(metric_name: str) -> bool:
    return _ncu_metric_flop_weight(metric_name) is not None


def _is_ncu_tensor_metric(metric_name: str) -> bool:
    return NCU_TENSOR_METRIC_RE.match(metric_name) is not None


def _is_ncu_sass_metric(metric_name: str) -> bool:
    base_name = (
        metric_name[:-len(".sum")]
        if metric_name.endswith(".sum")
        else metric_name
    )
    return base_name in NCU_SASS_FLOP_WEIGHTS


def _duration_to_ms(value: float, unit: str) -> float:
    normalized = str(unit or "").strip().lower().replace(" ", "")
    factors = {
        "s": 1_000.0,
        "sec": 1_000.0,
        "second": 1_000.0,
        "seconds": 1_000.0,
        "ms": 1.0,
        "msec": 1.0,
        "msecond": 1.0,
        "millisecond": 1.0,
        "milliseconds": 1.0,
        "us": 0.001,
        "usec": 0.001,
        "usecond": 0.001,
        "µs": 0.001,
        "microsecond": 0.001,
        "microseconds": 0.001,
        "ns": 0.000001,
        "nsec": 0.000001,
        "nsecond": 0.000001,
        "nanosecond": 0.000001,
        "nanoseconds": 0.000001,
    }
    return value * factors.get(normalized, 0.000001)


def _ncu_csv_rows(report_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read an NCU CSV while skipping profiler diagnostics before the header."""
    with open(report_path, "r", encoding="utf-8-sig", newline="") as f:
        filtered_lines = [
            line
            for line in f
            if line.strip() and not line.lstrip().startswith("==PROF==")
        ]
    parsed_rows = list(csv.reader(filtered_lines))

    header_index = None
    for idx, row in enumerate(parsed_rows):
        normalized = {field.strip().lower() for field in row if field}
        if "metric name" in normalized or "kernel name" in normalized:
            header_index = idx
            break
        if any(
            _is_ncu_flop_metric(field.strip())
            or field.strip() == NCU_DURATION_METRIC
            for field in row
        ):
            header_index = idx
            break
    if header_index is None:
        return [], []

    fieldnames = parsed_rows[header_index]
    rows = [
        {
            fieldnames[column]: value
            for column, value in enumerate(row[:len(fieldnames)])
        }
        for row in parsed_rows[header_index + 1:]
    ]
    return fieldnames, rows


def _first_finite(rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> float:
    for row in rows:
        for field in fields:
            value = _to_float(row.get(field))
            if value == value:
                return value
    return float("nan")


def _first_text(rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> str:
    for row in rows:
        for field in fields:
            value = str(row.get(field) or "").strip()
            if value:
                return value
    return ""


def _ncu_launch_count_wide(rows: Sequence[Dict[str, str]]) -> int:
    count = 0
    for row in rows:
        launch_id = str(row.get("ID") or "").strip()
        kernel_name = str(
            row.get("Kernel Name")
            or row.get("Kernel")
            or row.get("launch__kernel_name")
            or ""
        ).strip()
        if launch_id or kernel_name:
            count += 1
    return count


def _ncu_launch_count_long(rows: Sequence[Dict[str, str]]) -> int:
    launch_ids = set()
    for row in rows:
        launch_id = str(row.get("ID") or row.get("Launch ID") or "").strip()
        if launch_id:
            process_id = str(row.get("Process ID") or "").strip()
            launch_ids.add((process_id, launch_id))
    if launch_ids:
        return len(launch_ids)

    duration_rows = sum(
        1 for row in rows if _metric_name_from_row(row) == NCU_DURATION_METRIC
    )
    if duration_rows:
        return duration_rows

    metric_counts: Dict[str, int] = {}
    for row in rows:
        metric_name = _metric_name_from_row(row)
        if _is_ncu_flop_metric(metric_name):
            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
    return max(metric_counts.values(), default=0)


def parse_ncu_profile_csv(
    report_path: str,
    repeat: int = 1,
) -> Dict[str, Any]:
    """Parse wide or long NCU CSV into per-request hardware execution metrics."""
    fieldnames, rows = _ncu_csv_rows(report_path)
    normalized_repeat = max(1, int(repeat))
    if not fieldnames:
        return {
            "total_flops_per_request": float("nan"),
            "tensor_flops_per_request": float("nan"),
            "scalar_flops_per_request": float("nan"),
            "tensor_share_pct": float("nan"),
            "kernel_launch_count_per_request": float("nan"),
            "kernel_time_sum_ms_per_request": float("nan"),
            "gpu_compute_capability": "",
            "gpu_sm_count": float("nan"),
        }

    is_long = any(
        field.strip().lower() == "metric name"
        for field in fieldnames
    )
    metric_values: List[Tuple[str, float, str]] = []
    kernel_time_ms = 0.0
    duration_found = False

    if is_long:
        for row in rows:
            metric_name = _metric_name_from_row(row)
            value = _metric_value_from_row(row)
            if value != value:
                continue
            if _is_ncu_flop_metric(metric_name):
                metric_values.append((metric_name, value, _metric_unit_from_row(row)))
            elif metric_name == NCU_DURATION_METRIC:
                kernel_time_ms += _duration_to_ms(value, _metric_unit_from_row(row))
                duration_found = True
        launch_count = _ncu_launch_count_long(rows)
    else:
        metric_fields = [
            field
            for field in fieldnames
            if _is_ncu_flop_metric(field)
        ]
        duration_fields = [
            field for field in fieldnames if field == NCU_DURATION_METRIC
        ]
        unit_by_field: Dict[str, str] = {}
        for row in rows:
            for field in [*metric_fields, *duration_fields]:
                raw_value = str(row.get(field) or "").strip()
                if raw_value and _to_float(raw_value) != _to_float(raw_value):
                    unit_by_field.setdefault(field, raw_value)
        for row in rows:
            for field in metric_fields:
                value = _to_float(row.get(field))
                if value == value:
                    metric_values.append((field, value, unit_by_field.get(field, "")))
            for field in duration_fields:
                value = _to_float(row.get(field))
                if value == value:
                    kernel_time_ms += _duration_to_ms(
                        value,
                        unit_by_field.get(field, ""),
                    )
                    duration_found = True
        launch_count = _ncu_launch_count_wide(rows)

    # Legacy flop_count_* and SASS counters describe the same scalar work.
    # Prefer SASS when both families appear to avoid counting it twice.
    has_sass = any(_is_ncu_sass_metric(name) for name, _, _ in metric_values)
    tensor_flops = 0.0
    scalar_flops = 0.0
    tensor_found = False
    scalar_found = False
    for metric_name, value, _unit in metric_values:
        weight = _ncu_metric_flop_weight(metric_name)
        if weight is None:
            continue
        if _is_ncu_tensor_metric(metric_name):
            tensor_flops += value * weight
            tensor_found = True
        elif not (has_sass and metric_name in NCU_SCALAR_FLOP_METRICS):
            scalar_flops += value * weight
            scalar_found = True

    any_flops = tensor_found or scalar_found
    total_flops = tensor_flops + scalar_flops
    total_per_request = (
        total_flops / normalized_repeat if any_flops else float("nan")
    )
    tensor_per_request = (
        tensor_flops / normalized_repeat if any_flops else float("nan")
    )
    scalar_per_request = (
        scalar_flops / normalized_repeat if any_flops else float("nan")
    )
    tensor_share_pct = (
        tensor_flops / total_flops * 100.0
        if any_flops and total_flops > 0
        else float("nan")
    )

    compute_capability = _first_text(rows, ("CC", "Compute Capability"))
    if not compute_capability:
        major = _first_finite(
            rows,
            ("device__attribute_compute_capability_major",),
        )
        minor = _first_finite(
            rows,
            ("device__attribute_compute_capability_minor",),
        )
        if major == major and minor == minor:
            compute_capability = f"{int(major)}.{int(minor)}"
    sm_count = _first_finite(
        rows,
        ("launch__sm_count", "device__attribute_multiprocessor_count"),
    )

    return {
        "total_flops_per_request": total_per_request,
        "tensor_flops_per_request": tensor_per_request,
        "scalar_flops_per_request": scalar_per_request,
        "tensor_share_pct": tensor_share_pct,
        "kernel_launch_count_per_request": (
            float(launch_count) / normalized_repeat
            if launch_count > 0
            else float("nan")
        ),
        "kernel_time_sum_ms_per_request": (
            kernel_time_ms / normalized_repeat
            if duration_found
            else float("nan")
        ),
        "gpu_compute_capability": compute_capability,
        "gpu_sm_count": sm_count,
    }


def parse_ncu_flop_csv(report_path: str) -> float:
    """Backward-compatible total FLOP parser for an NCU raw CSV export."""
    return float(parse_ncu_profile_csv(report_path)["total_flops_per_request"])


def _candidate_executable_paths(root: str, names: Sequence[str]) -> Iterable[str]:
    for expanded_root in glob.glob(os.path.abspath(root)):
        if os.path.isfile(expanded_root) and os.path.basename(expanded_root) in names:
            yield expanded_root
            continue
        if not os.path.isdir(expanded_root):
            continue
        for name in names:
            for suffix in (
                name,
                os.path.join("bin", name),
                os.path.join("bin64", name),
                os.path.join("*", name),
                os.path.join("*", "bin", name),
                os.path.join("*", "bin64", name),
                os.path.join("latest", "bin", name),
                os.path.join("latest", "bin64", name),
                os.path.join("advisor", "latest", "bin64", name),
                os.path.join("target", "linux-desktop-glibc_2_11_3-x64", name),
            ):
                for candidate in glob.glob(os.path.join(expanded_root, suffix)):
                    yield candidate


def _executable_version_key(path: str, names: Sequence[str]) -> Tuple[Tuple[int, ...], int, str]:
    basename = os.path.basename(path)
    try:
        name_rank = len(names) - names.index(basename)
    except ValueError:
        name_rank = 0
    return tuple(int(part) for part in re.findall(r"\d+", path)), name_rank, path


def _best_existing_executable(roots: Sequence[str], names: Sequence[str]) -> Optional[str]:
    candidates = [
        os.path.realpath(candidate)
        for root in roots
        for candidate in _candidate_executable_paths(root, names)
        if os.path.isfile(candidate)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: _executable_version_key(path, names))


def _find_executable(root: Optional[str], names: Sequence[str]) -> Optional[str]:
    if root:
        best = _best_existing_executable([root], names)
        if best:
            return best
        for name in names:
            for candidate in glob.glob(os.path.join(root, "**", name), recursive=True):
                if os.path.isfile(candidate):
                    return os.path.realpath(candidate)
    best_default = _best_existing_executable(DEFAULT_TOOL_SEARCH_ROOTS, names)
    if best_default:
        return best_default
    for name in names:
        found = shutil.which(name)
        if found:
            return os.path.realpath(found)
    return None


def _tool_mount_root(tool_path: str, requested_root: Optional[str]) -> str:
    if requested_root:
        return os.path.abspath(requested_root)
    path = os.path.realpath(tool_path)
    parts = path.split(os.sep)

    if "advisor" in parts:
        idx = parts.index("advisor")
        if idx + 1 < len(parts):
            return os.sep + os.path.join(*parts[1:idx + 2])
    if "nsight-compute" in parts:
        idx = parts.index("nsight-compute")
        if idx + 1 < len(parts) and parts[idx + 1] not in {"ncu", "nv-nsight-cu-cli"}:
            return os.sep + os.path.join(*parts[1:idx + 2])
        return os.sep + os.path.join(*parts[1:idx + 1])
    if os.path.basename(path) in {"ncu", "nv-nsight-cu-cli"}:
        cuda_marker = next((part for part in parts if part.startswith("cuda")), None)
        if cuda_marker and "bin" in parts:
            idx = parts.index(cuda_marker)
            return os.sep + os.path.join(*parts[1:idx + 1])
    if os.path.isfile(path):
        return os.path.dirname(path)
    return path


def _tool_mount_roots(tool_path: str, requested_root: Optional[str]) -> List[str]:
    root = _tool_mount_root(tool_path, requested_root)
    roots = [root]
    for ncu_target in glob.glob(os.path.join(root, "target", "*")):
        real_target = os.path.realpath(ncu_target)
        if real_target == os.path.abspath(ncu_target):
            continue
        parts = real_target.split(os.sep)
        if "nsight-compute" not in parts:
            continue
        idx = parts.index("nsight-compute")
        real_root = os.sep + os.path.join(*parts[1:idx + 1])
        if real_root not in roots:
            roots.append(real_root)
    return roots


def _load_input_scale_plan_entries(
    input_scale_plan_file: str,
) -> List[Dict[str, Any]]:
    if not input_scale_plan_file:
        raise ValueError("input_scale_plan_file is required for compute profiling")
    if not os.path.isfile(input_scale_plan_file):
        raise FileNotFoundError(
            f"input scale plan not found: {input_scale_plan_file}"
        )

    with open(input_scale_plan_file, "r", encoding="utf-8") as f:
        plan = json.load(f)
    if not isinstance(plan, dict):
        raise ValueError(
            f"invalid input scale plan (expected object): {input_scale_plan_file}"
        )

    raw_entries = plan.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(
            f"invalid input scale plan (missing entries): {input_scale_plan_file}"
        )

    entries: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"invalid input scale plan entry at index {idx}: {entry!r}"
            )
        raw_scale = entry.get("input_scale")
        payload = entry.get("payload")
        if raw_scale is None or not isinstance(payload, dict):
            raise ValueError(
                f"input scale plan entry missing input_scale/payload "
                f"at index {idx}"
            )
        scale = float(raw_scale)
        entries.append({
            "input_scale": scale,
            "scale_label": str(
                entry.get("scale_label") or _format_scale_value(scale)
            ),
            "payload": payload,
        })
    return entries


def _base_docker_cmd(
    *,
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    use_gpu: bool,
    payload_file: str,
    profile_root: str,
    tool_mount_roots: Sequence[str],
) -> List[str]:
    package_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )
    cmd = [
        "docker", "run", "--rm",
        f"--cpus={cpu}",
        f"--memory={mem}g",
        "-v", f"{os.path.abspath(payload_file)}:{CONTAINER_INPUT_SCALE_PLAN_FILE}:ro",
        "-v", f"{os.path.abspath(profile_root)}:/profiles",
        "-e", f"MODEL_ID={task_info.model_id}",
        "-e", f"MODEL_REVISION={task_info.model_revision or 'main'}",
        "-e", f"TASK_FAMILY={task_info.task_family}",
        "-e", f"TASK_TYPE={task_info.pipeline_tag}",
        "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
        "-e", f"USE_GPU={1 if use_gpu else 0}",
        *hf_offline_docker_env_args(),
        "-e", "HOME=/tmp",
        "-e", f"OMP_NUM_THREADS={max(1, int(cpu))}",
        "-e", f"MKL_NUM_THREADS={max(1, int(cpu))}",
        "-e", f"OPENBLAS_NUM_THREADS={max(1, int(cpu))}",
        "-e", f"NUMEXPR_NUM_THREADS={max(1, int(cpu))}",
        "-e", f"TORCH_NUM_THREADS={max(1, int(cpu))}",
    ]
    if os.path.isdir(package_root):
        cmd.extend(["-v", f"{package_root}:/app/acprof:ro"])
    for tool_mount_root in tool_mount_roots:
        abs_root = os.path.abspath(tool_mount_root)
        cmd.extend(["-v", f"{abs_root}:{abs_root}:ro"])
    if use_gpu:
        cmd.extend([
            "--gpus", "all",
            "--cap-add=SYS_ADMIN",
            "--cap-add=SYS_PTRACE",
            "--security-opt=seccomp=unconfined",
        ])
    cmd.append(image_tag)
    return cmd


def _runner_args(entry: Dict[str, Any], repeat: int, mode: str) -> List[str]:
    return [
        "python", "-m", "acprof.container.compute_profile_runner",
        "--payload-file", CONTAINER_INPUT_SCALE_PLAN_FILE,
        "--input-scale", _format_scale_value(float(entry["input_scale"])),
        "--repeat", str(max(1, int(repeat))),
        "--profile-mode", mode,
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
    # Legacy flop_count_* metrics overlap with the SASS metrics. Use them only
    # when the current Nsight Compute version does not expose the SASS counters.
    scalar_metrics = (
        []
        if sass_metrics
        else [
            metric
            for metric in NCU_SCALAR_FLOP_METRICS
            if metric in available
        ]
    )
    return [*sass_metrics, *scalar_metrics, *tensor_metrics]


def _fallback_ncu_flop_metrics() -> List[str]:
    return list(NCU_SASS_FLOP_WEIGHTS)


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
    fallback_metrics = _fallback_ncu_flop_metrics()
    if fallback_metrics:
        return fallback_metrics, ""
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


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


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
        and tool_mode in {"auto", "both", "torch"}
    )
    collect_torch_gpu = (
        "on" in normalized_gpus
        and tool_mode in {"auto", "both", "torch"}
    )
    collect_advisor_cpu = (
        "off" in normalized_gpus
        and tool_mode == "vendor"
    )
    collect_ncu_gpu = (
        "on" in normalized_gpus
        and tool_mode in {"auto", "both", "ncu", "vendor"}
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

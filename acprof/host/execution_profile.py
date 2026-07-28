"""Host-side high-overhead execution profiling for AC-Prof.

Massif and Nsight Systems are deliberately collected outside the normal
benchmark path: both tools materially perturb latency.  Every tool and input
scale is isolated so a missing profiler or an invalid report remains a
diagnostic entry rather than aborting the other profiler.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import io
import math
import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from acprof.host import compute_profile
from acprof.host.detect import TaskInfo


EXECUTION_PROFILE_PLAN_NAME = "execution_profile_plan.json"
EXECUTION_PROFILE_DIRNAME = "execution_profiles"
EXECUTION_PROFILE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_TOOL_MODES = {"none", "both", "massif", "nsys"}
MASSIF_TOOL = "massif"
NSYS_TOOL = "nsys"
NSYS_NVTX_RANGE = "acprof_compute"
NSYS_REPORTS = (
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
)
COMPUTE_THREAD_ENV_NAMES = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
}
NSYS_DEFAULT_SEARCH_ROOTS = (
    "/opt/nvidia/nsight-systems",
    "/usr/local/NVIDIA-Nsight-Systems",
    "/usr/local/cuda",
    "/usr/local/cuda-*",
    "/usr/lib/nsight-systems",
)

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

# Local aliases make the collector easy to test without changing the shared
# compute profiler implementation.
_run = compute_profile._run
_base_docker_cmd = compute_profile._base_docker_cmd
_runner_args = compute_profile._runner_args
_load_input_scale_plan_entries = compute_profile._load_input_scale_plan_entries
_parse_last_json_line = compute_profile._parse_last_json_line
_write_json_atomic = compute_profile._write_json_atomic


def _finite_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if not value:
                return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _command_detail(result: Any, limit: int = 2000) -> str:
    detail = str(
        getattr(result, "stderr", "")
        or getattr(result, "stdout", "")
        or f"exit_code={getattr(result, 'returncode', 'unknown')}"
    ).strip()
    return detail[:limit]


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


def parse_massif_output(report_path: str) -> Dict[str, Any]:
    """Parse native Massif snapshots and return the execution-plan fields.

    Component peaks are independent maxima.  The total peak and its timestamp
    come from the single snapshot maximizing heap + heap-extra + stack.
    """
    snapshots: List[Dict[str, float]] = []
    current: Optional[Dict[str, float]] = None
    time_unit = ""

    with open(report_path, "r", encoding="utf-8", errors="replace") as report:
        for raw_line in report:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("time_unit:"):
                time_unit = line.split(":", 1)[1].strip().lower()
                continue
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if key == "snapshot":
                if current is not None:
                    snapshots.append(current)
                current = {}
                continue
            if current is None or key not in {
                "time",
                "mem_heap_B",
                "mem_heap_extra_B",
                "mem_stacks_B",
            }:
                continue
            value = _finite_float(raw_value)
            if value is not None:
                current[key] = value

    if current is not None:
        snapshots.append(current)
    if time_unit != "ms":
        raise ValueError(
            f"massif_parse_failed:expected_time_unit_ms,got={time_unit or 'missing'}"
        )

    required = {"time", "mem_heap_B", "mem_heap_extra_B", "mem_stacks_B"}
    complete = [snapshot for snapshot in snapshots if required <= snapshot.keys()]
    if not complete:
        raise ValueError("massif_parse_failed:no_complete_snapshots")

    heap_peak = max(snapshot["mem_heap_B"] for snapshot in complete)
    extra_peak = max(snapshot["mem_heap_extra_B"] for snapshot in complete)
    stack_peak = max(snapshot["mem_stacks_B"] for snapshot in complete)
    total_snapshot = max(
        complete,
        key=lambda snapshot: (
            snapshot["mem_heap_B"]
            + snapshot["mem_heap_extra_B"]
            + snapshot["mem_stacks_B"]
        ),
    )
    total_peak = (
        total_snapshot["mem_heap_B"]
        + total_snapshot["mem_heap_extra_B"]
        + total_snapshot["mem_stacks_B"]
    )

    def _integer_if_exact(number: float) -> Any:
        return int(number) if float(number).is_integer() else number

    return {
        "cpu_heap_peak_bytes_massif": _integer_if_exact(heap_peak),
        "cpu_heap_extra_peak_bytes_massif": _integer_if_exact(extra_peak),
        "cpu_stack_peak_bytes_massif": _integer_if_exact(stack_peak),
        "cpu_heap_peak_total_bytes_massif": _integer_if_exact(total_peak),
        "cpu_heap_peak_at_ms_massif": _integer_if_exact(total_snapshot["time"]),
    }


def parse_massif_snapshots(report_path: str) -> Dict[str, Any]:
    """Compatibility alias with an explicit parser-oriented name."""
    return parse_massif_output(report_path)


def _header_base(header: str) -> str:
    value = re.sub(r"\([^)]*\)|\[[^]]*\]", "", str(header))
    value = value.split(":", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _duration_factor_to_ms(header: str) -> float:
    value = str(header).lower().replace("μ", "u").replace("µ", "u")
    tokens = re.findall(r"[a-z]+", value)
    for token in reversed(tokens):
        if token in {"ns", "nsec", "nsecs", "nanosecond", "nanoseconds"}:
            return 1e-6
        if token in {"us", "usec", "usecs", "microsecond", "microseconds"}:
            return 1e-3
        if token in {"ms", "msec", "msecs", "millisecond", "milliseconds"}:
            return 1.0
        if token in {"s", "sec", "secs", "second", "seconds"}:
            return 1000.0
    # Nsys report scripts use native nanoseconds when no formatter unit is
    # visible in the header.
    return 1e-6


def _memory_factor_to_bytes(header: str) -> float:
    value = str(header)
    candidates = re.findall(
        r"(?<![A-Za-z])(KiB|MiB|GiB|KB|MB|GB|B)(?![A-Za-z])",
        value,
        flags=re.IGNORECASE,
    )
    if not candidates:
        return 1.0
    unit = candidates[-1]
    binary = unit.lower().endswith("ib")
    prefix = unit[0].upper() if len(unit) > 1 else ""
    exponent = {"": 0, "K": 1, "M": 2, "G": 3}.get(prefix)
    if exponent is None:
        raise ValueError(f"nsys_parse_failed:unsupported_memory_unit:{unit}")
    return float((1024 if binary else 1000) ** exponent)


def _csv_table(
    csv_text: str,
    *,
    required_header_bases: Sequence[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    lines = (csv_text or "").splitlines()
    for index, line in enumerate(lines):
        try:
            candidate = next(csv.reader([line]))
        except (csv.Error, StopIteration):
            continue
        bases = {_header_base(header) for header in candidate}
        if not all(required in bases for required in required_header_bases):
            continue
        reader = csv.DictReader(io.StringIO("\n".join(lines[index:])))
        fieldnames = [str(field) for field in (reader.fieldnames or [])]
        return fieldnames, [
            {str(key): str(value or "") for key, value in row.items() if key is not None}
            for row in reader
        ]
    raise ValueError(
        "nsys_parse_failed:csv_header_missing:"
        + ",".join(required_header_bases)
    )


def _field_for_base(fieldnames: Sequence[str], bases: Sequence[str]) -> str:
    for expected in bases:
        for field in fieldnames:
            if _header_base(field) == expected:
                return field
    raise ValueError(
        "nsys_parse_failed:field_missing:" + ",".join(bases)
    )


def _sum_numeric_column(
    rows: Sequence[Mapping[str, str]],
    field: str,
    *,
    include_row: Optional[Any] = None,
) -> float:
    total = 0.0
    found = False
    eligible = False
    for row in rows:
        if include_row is not None and not include_row(row):
            continue
        eligible = True
        value = _finite_float(row.get(field))
        if value is None:
            continue
        total += value
        found = True
    if eligible and not found:
        raise ValueError(f"nsys_parse_failed:no_numeric_values:{field}")
    return total


def _nsys_memory_report_has_no_data(csv_text: str) -> bool:
    """Recognize Nsys' successful no-MemOps report without hiding real errors."""
    normalized = " ".join(str(csv_text or "").lower().split())
    return (
        "skipped" in normalized
        and "does not contain" in normalized
        and "memory data" in normalized
    )


def parse_nsys_stats_csv(
    csv_text: str,
    report_name: str,
    *,
    repeat: int = 1,
) -> Dict[str, float]:
    """Parse one Nsys stats CSV report, normalizing units and repetitions."""
    normalized_repeat = max(1, int(repeat))
    report_name = str(report_name).strip()
    if report_name not in NSYS_REPORTS:
        raise ValueError(f"unsupported nsys report: {report_name}")

    if (
        report_name.startswith("cuda_gpu_mem_")
        and _nsys_memory_report_has_no_data(csv_text)
    ):
        if report_name == "cuda_gpu_mem_size_sum":
            return {
                "total_bytes_per_request": 0.0,
                "count_per_request": 0.0,
            }
        return {
            "total_time_ms_per_request": 0.0,
            "count_per_request": 0.0,
        }

    if report_name == "cuda_api_sum":
        count_bases = ("num calls", "calls", "count")
        required = ("total time",)
    elif report_name == "cuda_gpu_kern_sum":
        count_bases = ("instances", "count", "num calls")
        required = ("total time",)
    elif report_name == "cuda_gpu_mem_time_sum":
        count_bases = ("operations", "count", "instances", "num calls")
        required = ("total time",)
    else:
        count_bases = ("operations", "count", "instances", "num calls")
        required = ("total",)

    fieldnames, rows = _csv_table(
        csv_text,
        required_header_bases=required,
    )
    count_field = _field_for_base(fieldnames, count_bases)

    include_row = None
    if report_name.startswith("cuda_gpu_mem_"):
        operation_fields = [
            field
            for field in fieldnames
            if _header_base(field) in {"operation", "name"}
        ]
        if operation_fields:
            operation_field = operation_fields[0]

            def include_row(row: Mapping[str, str]) -> bool:
                return "memcpy" in str(row.get(operation_field, "")).lower()

    count = _sum_numeric_column(
        rows,
        count_field,
        include_row=include_row,
    ) / normalized_repeat

    if report_name == "cuda_gpu_mem_size_sum":
        total_field = _field_for_base(fieldnames, ("total",))
        total = _sum_numeric_column(
            rows,
            total_field,
            include_row=include_row,
        )
        return {
            "total_bytes_per_request": (
                total * _memory_factor_to_bytes(total_field) / normalized_repeat
            ),
            "count_per_request": count,
        }

    total_field = _field_for_base(fieldnames, ("total time",))
    total = _sum_numeric_column(
        rows,
        total_field,
        include_row=include_row,
    )
    return {
        "total_time_ms_per_request": (
            total * _duration_factor_to_ms(total_field) / normalized_repeat
        ),
        "count_per_request": count,
    }


def parse_nsys_stats_reports(
    report_outputs: Mapping[str, str],
    *,
    repeat: int = 1,
) -> Dict[str, float]:
    """Combine the four Nsys summary reports into execution-plan fields."""
    missing = [report for report in NSYS_REPORTS if report not in report_outputs]
    if missing:
        raise ValueError("nsys_parse_failed:missing_reports:" + ",".join(missing))

    api = parse_nsys_stats_csv(
        report_outputs["cuda_api_sum"],
        "cuda_api_sum",
        repeat=repeat,
    )
    kernel = parse_nsys_stats_csv(
        report_outputs["cuda_gpu_kern_sum"],
        "cuda_gpu_kern_sum",
        repeat=repeat,
    )
    memcpy_time = parse_nsys_stats_csv(
        report_outputs["cuda_gpu_mem_time_sum"],
        "cuda_gpu_mem_time_sum",
        repeat=repeat,
    )
    memcpy_size = parse_nsys_stats_csv(
        report_outputs["cuda_gpu_mem_size_sum"],
        "cuda_gpu_mem_size_sum",
        repeat=repeat,
    )
    return {
        "cuda_api_time_sum_ms_per_request_nsys": api[
            "total_time_ms_per_request"
        ],
        "cuda_api_call_count_per_request_nsys": api["count_per_request"],
        "gpu_kernel_time_sum_ms_per_request_nsys": kernel[
            "total_time_ms_per_request"
        ],
        "gpu_kernel_launch_count_per_request_nsys": kernel[
            "count_per_request"
        ],
        "gpu_memcpy_time_sum_ms_per_request_nsys": memcpy_time[
            "total_time_ms_per_request"
        ],
        "gpu_memcpy_count_per_request_nsys": memcpy_time["count_per_request"],
        "gpu_memcpy_bytes_per_request_nsys": memcpy_size[
            "total_bytes_per_request"
        ],
    }


def _build_massif_image(image_tag: str, project_dir: str) -> str:
    dockerfile = os.path.join(
        os.path.abspath(os.fspath(project_dir)),
        "dockerfiles",
        "massif.Dockerfile",
    )
    if not os.path.isfile(dockerfile):
        raise FileNotFoundError(
            f"massif_image_build_failed:dockerfile_not_found:{dockerfile}"
        )
    digest = hashlib.sha256(str(image_tag).encode("utf-8")).hexdigest()[:12]
    derived_tag = f"acprof-massif-{digest}:latest"
    result = _run(
        [
            "docker",
            "build",
            "--file",
            dockerfile,
            "--build-arg",
            f"BASE_IMAGE={image_tag}",
            "--tag",
            derived_tag,
            os.path.abspath(os.fspath(project_dir)),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"massif_image_build_failed:{_command_detail(result)}"
        )
    return derived_tag


def _massif_version(derived_image: Optional[str]) -> str:
    if not derived_image:
        return "unknown"
    try:
        result = _run(
            ["docker", "run", "--rm", derived_image, "valgrind", "--version"],
            check=False,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    output = str(result.stdout or result.stderr or "").strip()
    return output.splitlines()[-1].strip() if output else "unknown"


def _candidate_nsys_paths(root: str) -> Iterable[str]:
    for expanded in glob.glob(os.path.abspath(os.fspath(root))):
        if os.path.isfile(expanded) and os.path.basename(expanded) == "nsys":
            yield expanded
            continue
        if not os.path.isdir(expanded):
            continue
        for suffix in (
            "nsys",
            os.path.join("bin", "nsys"),
            os.path.join("bin64", "nsys"),
            os.path.join("target-linux-x64", "nsys"),
            os.path.join("*", "target-linux-x64", "nsys"),
            os.path.join("*", "host", "target-linux-x64", "nsys"),
        ):
            yield from glob.glob(os.path.join(expanded, suffix))
        yield from glob.iglob(
            os.path.join(expanded, "**", "nsys"),
            recursive=True,
        )


def _nsys_path_rank(path: str) -> Tuple[Tuple[int, ...], str]:
    return tuple(int(part) for part in re.findall(r"\d+", path)), path


def _find_nsys_executable(nsys_root: Optional[str]) -> Optional[str]:
    roots = [nsys_root] if nsys_root else list(NSYS_DEFAULT_SEARCH_ROOTS)
    candidates: List[str] = []
    seen = set()
    for root in roots:
        if not root:
            continue
        for candidate in _candidate_nsys_paths(root):
            real = os.path.realpath(candidate)
            if real in seen or not os.path.isfile(real):
                continue
            if not os.access(real, os.X_OK):
                continue
            seen.add(real)
            candidates.append(real)
    if candidates:
        return max(candidates, key=_nsys_path_rank)
    found = shutil.which("nsys")
    return os.path.realpath(found) if found else None


def _nsys_mount_root(nsys_bin: str) -> str:
    """Return an install root containing Nsys reports, Python, and libraries."""
    path = os.path.realpath(nsys_bin)
    parts = path.split(os.sep)
    target_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part.startswith("target-")
        ),
        None,
    )
    if target_index is not None and target_index > 0:
        root = os.sep + os.path.join(*parts[1:target_index])
        if os.path.basename(root) == "host":
            root = os.path.dirname(root)
        return root

    parent = os.path.dirname(path)
    if os.path.basename(parent) in {"bin", "bin64"} and any(
        marker in parent.lower()
        for marker in ("nsight", "nvidia", "cuda")
    ):
        return os.path.dirname(parent)
    return parent


def _nsys_version(nsys_bin: Optional[str]) -> str:
    if not nsys_bin:
        return "unknown"
    try:
        result = _run([nsys_bin, "--version"], check=False)
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    output = str(result.stdout or result.stderr or "").strip()
    return output.splitlines()[-1].strip() if output else "unknown"


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
    scale_label = _safe_filename_token(
        compute_profile._format_scale_value(float(entry["input_scale"]))
    )
    filename = f"massif_cpu_{cpu}_mem_{mem}_scale_{scale_label}.out"
    host_report = os.path.join(profile_root, filename)
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


def _run_nsys_stats(nsys_bin: str, report_path: str) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for index, report_name in enumerate(NSYS_REPORTS):
        refresh_export = ["--force-export=true"] if index == 0 else []
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
                report_path,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"nsys_stats_failed:{report_name}:{_command_detail(result)}"
            )
        stdout = str(result.stdout or "")
        stderr = str(result.stderr or "")
        outputs[report_name] = "\n".join(
            part for part in (stdout, stderr) if part.strip()
        )
    return outputs


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
        compute_profile._format_scale_value(float(entry["input_scale"]))
    )
    stem = f"nsys_cpu_{cpu}_mem_{mem}_scale_{scale_label}"
    filename = f"{stem}.nsys-rep"
    host_report = os.path.join(profile_root, filename)
    relative_report = _relative_artifact(host_report, output_dir)
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
        "--trace=cuda,nvtx,osrt",
        "--capture-range=nvtx",
        f"--nvtx-capture={NSYS_NVTX_RANGE}",
        "--capture-range-end=stop",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        f"--output=/profiles/{stem}",
        *_runner_args(dict(entry), repeat, "gpu"),
    ]
    result = _run(command, check=False)
    if result.returncode != 0:
        return _nsys_error_entry(
            entry,
            f"nsys_failed:{_command_detail(result)}",
            report=relative_report if os.path.isfile(host_report) else None,
        )
    if not os.path.isfile(host_report):
        return _nsys_error_entry(
            entry,
            "nsys_parse_failed:report_not_found",
        )

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
                profiled_entries.append(
                    _collect_massif_entry(
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
                )
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
) -> str:
    """Collect a full resource-grid execution profile plan atomically."""
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
    if collect_massif or collect_nsys:
        os.makedirs(profile_root, exist_ok=True)

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
        elif not nsys_error:
            nsys_error = "nsys_not_found"

    profiles: List[Dict[str, Any]] = []
    for cpu in cpus:
        for mem in memories:
            for gpu_mode in gpu_modes:
                tools: Dict[str, Any] = {}
                if gpu_mode == "off" and collect_massif:
                    tools[MASSIF_TOOL] = _profile_massif_tool(
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
                    )
                if gpu_mode == "on" and collect_nsys:
                    tools[NSYS_TOOL] = _profile_nsys_tool(
                        entries=entries,
                        global_error=nsys_error,
                        task_info=task_info,
                        image_tag=image_tag,
                        nsys_bin=nsys_bin,
                        nsys_mount_root=nsys_mount_root,
                        cpu=cpu,
                        mem=mem,
                        payload_file=input_scale_plan_file,
                        profile_root=profile_root,
                        output_dir=output_dir,
                        repeat=normalized_nsys_repeat,
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
        "nsys_timeline_semantics": (
            "NVTX acprof_compute capture; CUDA API, kernel, and memcpy "
            "sums/counts normalized per request"
        ),
        "nsys_repeat": normalized_nsys_repeat if collect_nsys else None,
        "nsys_version": nsys_version,
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

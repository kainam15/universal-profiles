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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from acprof.config import SCALING_DIMENSIONS
from acprof.host.detect import TaskInfo
from acprof.host.env_utils import hf_offline_docker_env_args


COMPUTE_PROFILE_PLAN_NAME = "compute_profile_plan.json"
COMPUTE_PROFILE_PAYLOADS_NAME = "compute_profile_payloads.json"
TORCH_PROFILER_TOOL = "torch_profiler"
COMPUTE_PROFILE_TOOL_MODES = {"auto", "torch", "vendor"}
NCU_TENSOR_METRIC_RE = re.compile(r"^sm__ops_path_tensor_src_.*_dst_.*\.sum$")
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


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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


def _tool_error_entries(entries: List[Dict[str, Any]], error: str) -> List[Dict[str, Any]]:
    return [
        {
            "input_scale": float(entry["input_scale"]),
            "model_mflop_per_request": None,
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


def parse_ncu_flop_csv(report_path: str) -> float:
    """Return summed FLOP-like metric values from an ncu raw CSV export."""
    total_flop = 0.0
    found = False
    with open(report_path, "r", encoding="utf-8-sig", newline="") as f:
        filtered_lines = [
            line for line in f
            if line.strip() and not line.startswith("==PROF==")
        ]
    reader = csv.DictReader(filtered_lines)
    if reader.fieldnames is None:
        return float("nan")

    wide_metric_fields = [
        (field, weight)
        for field in reader.fieldnames
        for weight in [_ncu_metric_flop_weight(field)]
        if weight is not None
    ]
    if wide_metric_fields and "Metric Name" not in reader.fieldnames:
        for row in reader:
            for field, weight in wide_metric_fields:
                value = _to_float(row.get(field))
                if value == value:
                    total_flop += value * weight
                    found = True
        return total_flop if found else float("nan")

    for row in reader:
        metric_name = _metric_name_from_row(row)
        weight = _ncu_metric_flop_weight(metric_name)
        if weight is None:
            continue
        value = _metric_value_from_row(row)
        if value == value:
            total_flop += value * weight
            found = True
    return total_flop if found else float("nan")


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


def _default_scales(task_info: TaskInfo) -> List[float]:
    scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
    if scaling_cfg and scaling_cfg.values:
        return [float(value) for value in scaling_cfg.values]
    return [1.0]


def _load_payload_entries(
    task_info: TaskInfo,
    batch_size: int,
    input_scale_plan_file: Optional[str],
    input_scales: Optional[str],
) -> List[Dict[str, Any]]:
    if input_scale_plan_file and os.path.exists(input_scale_plan_file):
        with open(input_scale_plan_file, "r", encoding="utf-8") as f:
            plan = json.load(f)
        entries = plan.get("entries")
        if isinstance(entries, list) and entries:
            return [
                {
                    "input_scale": float(entry["input_scale"]),
                    "scale_label": str(entry.get("scale_label") or _format_scale_value(float(entry["input_scale"]))),
                    "payload": entry["payload"],
                }
                for entry in entries
                if isinstance(entry, dict) and isinstance(entry.get("payload"), dict)
            ]

    from acprof.workloads import get_generator

    scales = _parse_float_list(input_scales) if input_scales else _default_scales(task_info)
    workload_gen = get_generator(
        task_info.task_family,
        task_info.model_id,
        task_info.pipeline_tag,
        batch_size,
    )
    return [
        {
            "input_scale": float(scale),
            "scale_label": workload_gen.scale_label(float(scale)),
            "payload": workload_gen.generate(float(scale)),
        }
        for scale in scales
    ]


def _write_payload_file(output_dir: str, task_info: TaskInfo, entries: List[Dict[str, Any]]) -> str:
    payload_file = os.path.join(output_dir, COMPUTE_PROFILE_PAYLOADS_NAME)
    payload = {
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "entries": entries,
    }
    with open(payload_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return payload_file


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
        "-v", f"{os.path.abspath(payload_file)}:/payloads/{COMPUTE_PROFILE_PAYLOADS_NAME}:ro",
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
        "--payload-file", f"/payloads/{COMPUTE_PROFILE_PAYLOADS_NAME}",
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
    modern_metrics = sorted(
        metric for metric in available
        if metric in NCU_SCALAR_FLOP_METRICS or NCU_TENSOR_METRIC_RE.match(metric)
    )
    if modern_metrics:
        return modern_metrics
    return [
        metric for metric in NCU_SASS_FLOP_WEIGHTS
        if metric in available
    ]


def _fallback_ncu_flop_metrics() -> List[str]:
    return list(NCU_SASS_FLOP_WEIGHTS)


def _resolve_ncu_metrics(ncu_bin: str) -> Tuple[List[str], str]:
    query_errors = []
    for command in (
        [ncu_bin, "--query-metrics", "--query-metrics-mode", "all"],
        [ncu_bin, "--query-metrics"],
    ):
        result = _run(command, check=False)
        if result.returncode != 0:
            query_errors.append(result.stderr.strip() or result.stdout.strip())
            continue
        metrics = _select_ncu_flop_metrics(_parse_ncu_metric_names(result.stdout))
        if metrics:
            return metrics, ""
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
    scale_label = _format_scale_value(float(entry["input_scale"]))
    report_base = f"/profiles/ncu_scale_{scale_label}"
    host_csv = os.path.join(profile_root, f"ncu_scale_{scale_label}.csv")
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
        return {
            "input_scale": float(entry["input_scale"]),
            "model_mflop_per_request": None,
            "error": f"ncu_failed:{result.stderr.strip() or result.stdout.strip()}",
        }

    import_cmd = [
        ncu_bin,
        "--import", f"{report_base}.ncu-rep",
        "--page", "raw",
        "--csv",
    ]
    import_result = _run([*base_cmd, *import_cmd], check=False)
    csv_text = import_result.stdout if import_result.returncode == 0 and import_result.stdout else result.stdout
    with open(host_csv, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)

    flop = parse_ncu_flop_csv(host_csv)
    if flop != flop:
        return {
            "input_scale": float(entry["input_scale"]),
            "model_mflop_per_request": None,
            "error": "ncu_parse_failed:flop_metrics_missing",
        }
    return {
        "input_scale": float(entry["input_scale"]),
        "tool": "ncu",
        "model_mflop_per_request": (flop / 1_000_000.0) / float(max(1, int(repeat))),
        "error": "",
        "report": host_csv,
    }


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
    runner_mode = "torch_gpu" if use_gpu else "torch_cpu"
    result = _run([*base_cmd, *_runner_args(entry, repeat, runner_mode)], check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return {
            "input_scale": float(entry["input_scale"]),
            "tool": TORCH_PROFILER_TOOL,
            "model_mflop_per_request": None,
            "error": f"torch_profiler_failed:{detail}",
        }

    payload = _parse_last_json_line(result.stdout)
    mflop = _to_float(payload.get("model_mflop_per_request"))
    total_flops = _to_float(payload.get("total_flops"))
    if mflop != mflop and total_flops == total_flops:
        mflop = (total_flops / 1_000_000.0) / float(max(1, int(repeat)))
    if mflop != mflop or mflop <= 0:
        return {
            "input_scale": float(entry["input_scale"]),
            "tool": TORCH_PROFILER_TOOL,
            "model_mflop_per_request": None,
            "error": "torch_profiler_parse_failed:model_mflop_per_request_missing",
        }

    return {
        "input_scale": float(entry["input_scale"]),
        "tool": TORCH_PROFILER_TOOL,
        "model_mflop_per_request": mflop,
        "error": "",
        "total_flops": total_flops if total_flops == total_flops else None,
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
    profile_entries = [
        _run_torch_profiler_for_entry(
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
        for entry in entries
    ]
    errors = [entry["error"] for entry in profile_entries if entry.get("error")]
    return {
        "tool": TORCH_PROFILER_TOOL,
        "repeat": max(1, int(repeat)),
        "profile": profile_key,
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
            "entries": _tool_error_entries(entries, "advisor_not_found"),
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
) -> Dict[str, Any]:
    if ncu_bin is None:
        return {
            "tool": "ncu",
            "repeat": max(1, int(repeat)),
            "error": "ncu_not_found",
            "entries": _tool_error_entries(entries, "ncu_not_found"),
        }
    ncu_metrics, metric_error = _resolve_ncu_metrics(ncu_bin)
    if metric_error:
        return {
            "tool": "ncu",
            "repeat": max(1, int(repeat)),
            "error": metric_error,
            "entries": _tool_error_entries(entries, metric_error),
        }
    mount_roots = _tool_mount_roots(ncu_bin, ncu_root)
    profile_entries = [
        _run_ncu_for_entry(
            ncu_bin=ncu_bin,
            ncu_metrics=ncu_metrics,
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
        "tool": "ncu",
        "repeat": max(1, int(repeat)),
        "metrics": ncu_metrics,
        "error": "; ".join(errors),
        "entries": profile_entries,
    }


def collect_compute_profile_plan(
    *,
    task_info: TaskInfo,
    image_tag: str,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    batch_size: int,
    output_dir: str,
    input_scale_plan_file: Optional[str],
    input_scales: Optional[str],
    advisor_root: Optional[str],
    ncu_root: Optional[str],
    advisor_repeat: int,
    ncu_repeat: int,
    keep_profiles: bool,
    compute_profile_cpus: Optional[int] = None,
    compute_profile_mem: Optional[int] = None,
    compute_profile_tool: str = "auto",
) -> str:
    """Collect or synthesize compute profiles and write a plan file."""
    os.makedirs(output_dir, exist_ok=True)
    tool_mode = (compute_profile_tool or "auto").strip().lower()
    if tool_mode not in COMPUTE_PROFILE_TOOL_MODES:
        raise ValueError(
            "compute_profile_tool must be one of "
            f"{', '.join(sorted(COMPUTE_PROFILE_TOOL_MODES))}, got {compute_profile_tool!r}"
        )

    profile_root = os.path.join(output_dir, "compute_profiles")
    os.makedirs(profile_root, exist_ok=True)
    entries = _load_payload_entries(task_info, batch_size, input_scale_plan_file, input_scales)
    payload_file = _write_payload_file(output_dir, task_info, entries)

    normalized_gpus = {_normal_gpu_mode(gpu) for gpu in gpu_list}
    advisor_bin = (
        _find_executable(advisor_root, ("advisor", "advixe-cl"))
        if tool_mode == "vendor" and "off" in normalized_gpus
        else None
    )
    ncu_bin = (
        _find_executable(ncu_root, ("ncu", "nv-nsight-cu-cli"))
        if tool_mode in {"auto", "vendor"} and "on" in normalized_gpus
        else None
    )
    max_cpu, max_mem = _default_compute_profile_resources(
        compute_profile_cpus,
        compute_profile_mem,
    )

    profiles: Dict[str, Any] = {}
    if "off" in normalized_gpus:
        if tool_mode == "vendor":
            profiles["cpu"] = _profile_cpu_entries(
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
            )
        else:
            profiles["cpu"] = _profile_torch_entries(
                entries=entries,
                task_info=task_info,
                image_tag=image_tag,
                cpu=max_cpu,
                mem=max_mem,
                use_gpu=False,
                profile_key="cpu",
                payload_file=payload_file,
                profile_root=profile_root,
                repeat=advisor_repeat,
            )
    if "on" in normalized_gpus:
        if tool_mode in {"auto", "vendor"}:
            profiles["gpu"] = _profile_gpu_entries(
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
            )
        else:
            profiles["gpu"] = _profile_torch_entries(
                entries=entries,
                task_info=task_info,
                image_tag=image_tag,
                cpu=max_cpu,
                mem=max_mem,
                use_gpu=True,
                profile_key="gpu",
                payload_file=payload_file,
                profile_root=profile_root,
                repeat=ncu_repeat,
            )

    plan = {
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "runtime_backend": task_info.runtime_backend,
        "compute_profile_tool_mode": tool_mode,
        "profiles": profiles,
    }
    plan_path = os.path.join(output_dir, COMPUTE_PROFILE_PLAN_NAME)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=True, indent=2)

    if not keep_profiles:
        shutil.rmtree(profile_root, ignore_errors=True)

    print(f"[compute] Compute profile plan: {plan_path}")
    return plan_path

"""计算与执行 profiler 共用的命令、负载计划及产物写入工具。"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Sequence

from acprof.host.detect import TaskInfo
from acprof.host.env_utils import hf_offline_docker_env_args


CONTAINER_INPUT_SCALE_PLAN_FILE = "/payloads/input_scale_plan.json"


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
    if not task_info.runtime_profile_id and os.path.isdir(package_root):
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

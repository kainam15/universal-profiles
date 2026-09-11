"""Docker 镜像、容器生命周期及启动状态。"""
from __future__ import annotations

import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from acprof.config import (
    DOCKER_IMAGE_PREFIX,
    HF_MIRROR_ENDPOINT,
    PYPI_MIRROR_INDEX,
    SERVER_PORT,
    READY_POLL_INTERVAL_S,
    READY_TIMEOUT_S,
)
from acprof.host.detect import TaskInfo
from acprof.host.env_utils import hf_offline_docker_env_args


@dataclass
class ImageInfo:
    tag: str


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


DEFAULT_NLP_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
CUDA124_NLP_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
DEFAULT_NLP_TORCH_SPEC = "torch>=2.7"
CUDA124_NLP_TORCH_SPEC = "torch>=2.6,<2.7"


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in Docker image tags and file names."""
    return model_id.replace("/", "--").replace(".", "_").lower()


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


def _normalize_gpu_mode(gpu: str) -> str:
    return "on" if str(gpu).lower() == "on" else "off"


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


def _host_port(cpu: int, mem: int) -> int:
    return SERVER_PORT + cpu * 100 + mem


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


def _model_image_tag(task_info: TaskInfo) -> str:
    model_tag = _sanitize_model_id(task_info.model_id)
    return f"{DOCKER_IMAGE_PREFIX}-{task_info.task_family}-{model_tag}:latest"


def prepare_image(
    task_info: TaskInfo,
    project_dir: str,
    *,
    reuse_existing: bool = False,
) -> ImageInfo:
    """Check requested local-image reuse before falling back to a normal build."""
    if reuse_existing:
        tag = _model_image_tag(task_info)
        print(f"\n[build] 检查本地模型镜像：{tag}", flush=True)
        # A successful empty listing means the image is missing; a failed
        # Docker query must not be mistaken for a missing image.
        result = _run(
            ["docker", "image", "ls", "--quiet", "--filter", f"reference={tag}"],
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"无法检查本地模型镜像 {tag}（Docker 退出码 {result.returncode}）：{detail}"
            )
        if result.stdout.strip():
            print(f"[build] 已找到本地模型镜像，跳过构建并复用：{tag}", flush=True)
            return ImageInfo(tag=tag)
        print(
            f"[build][WARN] 未找到本地模型镜像：{tag}；将自动构建镜像，完成后继续任务。",
            flush=True,
        )

    return build_image(task_info, project_dir)


def build_image(task_info: TaskInfo, project_dir: str) -> ImageInfo:
    """Build the Docker image for this model's task family.

    Two-stage build:
    1. Build base image (if not exists)
    2. Build task-family image with model weights baked in
    """
    dockerfiles_dir = os.path.join(project_dir, "dockerfiles")
    base_tag = f"{DOCKER_IMAGE_PREFIX}-base:latest"
    family_tag = _model_image_tag(task_info)

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
    if task_info.task_family in {"nlp", "diffusion", "multimodal", "structured"}:
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

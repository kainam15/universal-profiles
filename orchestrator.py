"""AC-Prof Universal Orchestrator - Docker lifecycle, resource constraints, monitoring.

Python replacement for run_case.sh / run_matrix.sh.
"""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from config import (
    DEFAULT_REPEAT_IN_WINDOW,
    DOCKER_IMAGE_PREFIX,
    HF_MIRROR_ENDPOINT,
    PYPI_MIRROR_INDEX,
    SCALING_DIMENSIONS,
    SERVER_PORT,
    READY_POLL_INTERVAL_S,
    READY_TIMEOUT_S,
    STATIC_META_FIELDS,
)
from detect import TaskInfo


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
    model_download_url: str
    gpu: str
    model_weight_bytes: int
    docker_image_bytes: int


@dataclass
class RunningContainer:
    name: str
    base_url: str
    host_port: int
    cold_start_s: float


@dataclass
class PlannedInputScales:
    scales: List[float]
    source: str
    plan_file: Optional[str] = None


AUTO_INPUT_SCALE_COUNT = 6


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
) -> None:
    payload = {
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "entries": entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


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


def _url_host(url: str) -> str:
    """Extract host from a URL for pip trusted-host."""
    parsed = urlparse(url)
    return parsed.netloc or parsed.path


def _build_model_download_url(model_id: str) -> str:
    """Return the canonical Hugging Face model URL."""
    return f"https://huggingface.co/{model_id}"


def _normalize_gpu_mode(gpu: str) -> str:
    return "on" if str(gpu).lower() == "on" else "off"


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


def _docker_model_weight_bytes(image_tag: str, cache_root: str = "/models/hf") -> int:
    """Measure total bytes of downloaded model artifacts stored in the image."""
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


def collect_static_meta(
    task_info: TaskInfo,
    image_info: ImageInfo,
    batch_size: int,
    input_scale_type: str,
    device_index: int = 0,
) -> StaticMeta:
    """Collect static metadata for the current model/image pair."""
    return StaticMeta(
        model_name=task_info.model_id,
        model_revision=task_info.model_revision,
        task_family=task_info.task_family,
        pipeline_tag=task_info.pipeline_tag,
        runtime_backend=task_info.runtime_backend,
        image_tag=image_info.tag,
        batch_size=batch_size,
        input_scale_type=input_scale_type,
        model_download_url=_build_model_download_url(task_info.model_id),
        gpu=_get_gpu_name(device_index=device_index),
        model_weight_bytes=_docker_model_weight_bytes(image_info.tag),
        docker_image_bytes=_docker_image_size_bytes(image_info.tag),
    )


def write_static_meta_csv(static_meta: StaticMeta, output_path: str) -> None:
    """Write static metadata to a single-row CSV."""
    import csv

    row = {field: getattr(static_meta, field) for field in STATIC_META_FIELDS}

    def _serialize_csv_value(value: Any, *, force_quote: bool = False) -> str:
        text = "" if value is None else str(value)
        escaped = text.replace('"', '""')
        needs_quote = force_quote or any(ch in text for ch in [",", '"', "\n", "\r"])
        return f'"{escaped}"' if needs_quote else escaped

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        header = ",".join(_serialize_csv_value(field) for field in STATIC_META_FIELDS)
        values = ",".join(
            _serialize_csv_value(
                row[field],
                force_quote=field == "model_download_url",
            )
            for field in STATIC_META_FIELDS
        )
        f.write(header + "\n")
        f.write(values + "\n")

    print(f"[meta] Static meta CSV: {output_path}")


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

    result = _run([
        "docker", "build",
        "-f", family_dockerfile,
        "--build-arg", f"BASE_IMAGE={base_tag}",
        "--build-arg", f"MODEL_ID={task_info.model_id}",
        "--build-arg", "HF_TOKEN",
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
        "-e", f"TASK_FAMILY={task_info.task_family}",
        "-e", f"TASK_TYPE={task_info.pipeline_tag}",
        "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
        "-e", f"USE_GPU={use_gpu}",
        "-e", "HF_TOKEN",
        "-e", "HUGGING_FACE_HUB_TOKEN",
        "-e", "HF_HUB_DISABLE_TELEMETRY=1",
        "-e", "HF_HOME=/models/hf",
        "-e", "HF_HUB_CACHE=/models/hf",
        "-e", "TRANSFORMERS_CACHE=/models/hf",
        "-p", f"{host_port}:{SERVER_PORT}",
        image_info.tag,
    ]

    t0 = time.perf_counter()
    result = _run(docker_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

    base_url = f"http://127.0.0.1:{host_port}"
    deadline = time.perf_counter() + READY_TIMEOUT_S

    while time.perf_counter() < deadline:
        try:
            response = requests.get(
                f"{base_url}/ready",
                timeout=2,
                headers={"Connection": "close"},
            )
            if response.status_code == 200:
                try:
                    body = response.json()
                except Exception:
                    body = None

                if isinstance(body, dict) and body.get("status") == "ok":
                    cold_start_s = time.perf_counter() - t0
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
                    )

                if response.text.strip() == "ok":
                    cold_start_s = time.perf_counter() - t0
                    print(f"{log_prefix} Server ready. cold_start={cold_start_s:.3f}s")
                    return RunningContainer(
                        name=container_name,
                        base_url=base_url,
                        host_port=host_port,
                        cold_start_s=cold_start_s,
                    )
        except Exception:
            pass
        time.sleep(READY_POLL_INTERVAL_S)

    cold_start_s = time.perf_counter() - t0
    print(f"{log_prefix} Server not ready after {READY_TIMEOUT_S}s. cold_start={cold_start_s:.3f}s")
    logs = _run(["docker", "logs", container_name, "--tail", "200"], check=False)
    if logs.stdout:
        print(logs.stdout[-500:])
    _run(["docker", "rm", "-f", container_name], check=False)
    raise RuntimeError(f"server not ready after {READY_TIMEOUT_S}s for container {container_name}")


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


def _request_nlp_scale_meta(
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


def _assert_manual_timeseries_scales_legal(
    task_info: TaskInfo,
    scales: List[float],
    batch_size: int,
) -> List[float]:
    from workloads import get_generator

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
    from workloads import get_generator

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


def _plan_nlp_auto_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    batch_size: int,
    output_dir: str,
) -> PlannedInputScales:
    from workloads import get_generator

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
        _write_scale_plan_file(plan_file, task_info, entries)

        print(
            "[scale] Auto-planned NLP scales from tokenizer limit "
            f"{max_effective}: {serialize_input_scales(actual_scales)}"
        )
        print(f"[scale] Scale metadata: {scale_meta['reason']}")
        return PlannedInputScales(
            scales=actual_scales,
            source="auto",
            plan_file=plan_file,
        )
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")


def _default_family_max_scale(task_info: TaskInfo, batch_size: int) -> float:
    if task_info.task_family == "timeseries":
        from workloads import get_generator

        workload_gen = get_generator(task_info.task_family, task_info.model_id, task_info.pipeline_tag, batch_size)
        if hasattr(workload_gen, "max_input_scale"):
            max_scale = workload_gen.max_input_scale()
            if max_scale is not None:
                return float(max_scale)

    scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
    if scaling_cfg and scaling_cfg.values:
        return float(max(scaling_cfg.values))

    raise RuntimeError(f"cannot determine default max input scale for task_family={task_info.task_family}")


def plan_input_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    batch_size: int,
    output_dir: str,
    input_scales: Optional[str] = None,
) -> PlannedInputScales:
    plan_file = _scale_plan_file_path(output_dir)
    _clear_scale_plan_file(plan_file)

    if input_scales:
        manual_scales = resolve_input_scales(task_info.task_family, input_scales=input_scales)
        if task_info.task_family == "nlp":
            manual_scales = _assert_manual_nlp_scales_legal(
                task_info=task_info,
                image_info=image_info,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                scales=manual_scales,
                batch_size=batch_size,
            )
        elif task_info.task_family == "timeseries":
            manual_scales = _assert_manual_timeseries_scales_legal(
                task_info=task_info,
                scales=manual_scales,
                batch_size=batch_size,
            )
        else:
            print(f"[scale] Using manual input scales: {serialize_input_scales(manual_scales)}")

        return PlannedInputScales(scales=manual_scales, source="manual")

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

    max_scale = _default_family_max_scale(task_info, batch_size)
    if task_info.task_family == "cv":
        scales = _float_auto_scales(max_scale, count=AUTO_INPUT_SCALE_COUNT)
    else:
        scales = _integer_auto_scales(int(max_scale), count=AUTO_INPUT_SCALE_COUNT)

    print(
        f"[scale] Auto-planned {task_info.task_family} scales: "
        f"{serialize_input_scales(scales)}"
    )
    return PlannedInputScales(scales=scales, source="auto")


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
    sample_hz: float = 20.0,
    idle_seconds: float = 3.0,
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
        "-e", f"TASK_FAMILY={task_info.task_family}",
        "-e", f"TASK_TYPE={task_info.pipeline_tag}",
        "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
        "-e", f"USE_GPU={use_gpu}",
        "-e", "HF_TOKEN",
        "-e", "HUGGING_FACE_HUB_TOKEN",
        "-e", "HF_HUB_DISABLE_TELEMETRY=1",
        "-e", "HF_HOME=/models/hf",
        "-e", "HF_HUB_CACHE=/models/hf",
        "-e", "TRANSFORMERS_CACHE=/models/hf",
        # Note: removed HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE.
        # Newer huggingface_hub (v1.x) has a bug where OFFLINE=1 prevents
        # resolving cached files. Since model weights are baked into the image,
        # no network access is needed at runtime.
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
    can_sniff = shutil.which("tcpdump") is not None

    if can_sniff:
        print(f"[sniff] Starting tcpdump on {sniff_iface}")
        tcpdump_cmd = [
            "sudo", "tcpdump",
            "-i", sniff_iface,
            "-s", "0",
            "-B", "4096",
            "-w", pcap_file,
            "tcp", "port", str(SERVER_PORT),
        ]
        tcpdump_proc = subprocess.Popen(
            tcpdump_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)  # Wait for tcpdump to start capturing
    else:
        print("[sniff] tcpdump not available, skipping packet-level latency")

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
        "SAMPLE_HZ": str(sample_hz),
        "IDLE_SECONDS": str(idle_seconds),
        "DEVICE_INDEX": "0",
        "INPUT_SCALES": scales_str,
    }

    client_result = _run(
        [sys.executable, os.path.join(project_dir, "client.py")],
        check=False,
        capture=False,
        env=client_env,
    )

    if client_result.returncode != 0:
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
            sniff_script = os.path.join(project_dir, "sniff_parse_pcap.py")
            parse_result = _run(
                [sys.executable, sniff_script, pcap_file, str(SERVER_PORT)],
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
                merge_script = os.path.join(project_dir, "merge_packet_latency.py")
                merged_csv = out_csv + ".merged"
                _run(
                    [sys.executable, merge_script, out_csv, lat_json, merged_csv],
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
    sample_hz: float = 20.0,
    idle_seconds: float = 3.0,
    sniff_iface: str = "docker0",
    input_scales: Optional[str] = None,
    input_scale_plan_file: Optional[str] = None,
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
        print(f"[case] {exc}", file=sys.stderr)
        return ""

    base_url = session.base_url
    cold_start_s = session.cold_start_s
    tcpdump_proc = None
    can_sniff = shutil.which("tcpdump") is not None

    try:
        if can_sniff:
            print(f"[sniff] Starting tcpdump on {sniff_iface}")
            tcpdump_cmd = [
                "sudo", "tcpdump",
                "-i", sniff_iface,
                "-s", "0",
                "-B", "4096",
                "-w", pcap_file,
                "tcp", "port", str(SERVER_PORT),
            ]
            tcpdump_proc = subprocess.Popen(
                tcpdump_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.3)
        else:
            print("[sniff] tcpdump not available, skipping packet-level latency")

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
            "COLD_START_S": f"{cold_start_s:.3f}",
            "OUT_CSV": out_csv,
            "CASE_NAME": case_name,
            "SAMPLE_HZ": str(sample_hz),
            "IDLE_SECONDS": str(idle_seconds),
            "DEVICE_INDEX": "0",
            "INPUT_SCALES": scales_str,
            "INPUT_SCALE_PLAN_FILE": input_scale_plan_file or "",
        }

        client_result = _run(
            [sys.executable, os.path.join(project_dir, "client.py")],
            check=False,
            capture=False,
            env=client_env,
        )

        if client_result.returncode != 0:
            print(f"[case] Client exited with code {client_result.returncode}")

        if tcpdump_proc is not None:
            time.sleep(1.0)
            tcpdump_proc.terminate()
            try:
                tcpdump_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tcpdump_proc.kill()
            time.sleep(0.2)

            if os.path.exists(pcap_file) and os.path.getsize(pcap_file) > 0:
                print("[sniff] Parsing pcap -> packet latencies...")
                sniff_script = os.path.join(project_dir, "sniff_parse_pcap.py")
                parse_result = _run(
                    [sys.executable, sniff_script, pcap_file, str(SERVER_PORT)],
                    check=False,
                )
                if parse_result.returncode == 0 and parse_result.stdout.strip():
                    with open(lat_json, "w", encoding="utf-8") as lf:
                        lf.write(parse_result.stdout)
                else:
                    with open(lat_json, "w", encoding="utf-8") as lf:
                        lf.write("{}")

                if os.path.exists(lat_json) and os.path.exists(out_csv):
                    print("[sniff] Merging packet latency into CSV...")
                    merge_script = os.path.join(project_dir, "merge_packet_latency.py")
                    merged_csv = out_csv + ".merged"
                    _run(
                        [sys.executable, merge_script, out_csv, lat_json, merged_csv],
                        check=False,
                    )
                    if os.path.exists(merged_csv):
                        os.replace(merged_csv, out_csv)
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
    sample_hz: float = 20.0,
    idle_seconds: float = 3.0,
    sniff_iface: str = "docker0",
    input_scales: Optional[str] = None,
    input_scale_plan_file: Optional[str] = None,
) -> List[str]:
    """Sweep all resource combinations."""
    os.makedirs(output_dir, exist_ok=True)
    result_csvs = []

    total = len(cpu_list) * len(mem_list) * len(gpu_list)
    current = 0

    for cpu in cpu_list:
        for mem in mem_list:
            for gpu in gpu_list:
                current += 1
                print(f"\n{'#'*60}")
                print(f"# Case {current}/{total}: CPU={cpu}, MEM={mem}GB, GPU={gpu}")
                print(f"{'#'*60}")

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
                    sample_hz=sample_hz,
                    idle_seconds=idle_seconds,
                    sniff_iface=sniff_iface,
                    input_scales=input_scales,
                    input_scale_plan_file=input_scale_plan_file,
                )
                if csv_path:
                    result_csvs.append(csv_path)

    return result_csvs


def merge_all_csvs(csv_paths: List[str], output_path: str) -> None:
    """Merge all per-case CSVs into one final CSV."""
    import csv
    from config import CSV_FIELDS

    if not csv_paths:
        print("[merge] No CSV files to merge.")
        return

    all_rows = []
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[merge] Final CSV: {output_path} ({len(all_rows)} rows)")

"""AC-Prof Universal Orchestrator - Docker lifecycle, resource constraints, monitoring.

Python replacement for run_case.sh / run_matrix.sh.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
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
    model_download_url: str
    gpu: str
    model_weight_bytes: int
    docker_image_bytes: int


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


def _url_host(url: str) -> str:
    """Extract host from a URL for pip trusted-host."""
    parsed = urlparse(url)
    return parsed.netloc or parsed.path


def _build_model_download_url(model_id: str) -> str:
    """Return the canonical Hugging Face model URL."""
    return f"https://huggingface.co/{model_id}"


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
    device_index: int = 0,
) -> StaticMeta:
    """Collect static metadata for the current model/image pair."""
    return StaticMeta(
        model_name=task_info.model_id,
        model_download_url=_build_model_download_url(task_info.model_id),
        gpu=_get_gpu_name(device_index=device_index),
        model_weight_bytes=_docker_model_weight_bytes(image_info.tag),
        docker_image_bytes=_docker_image_size_bytes(image_info.tag),
    )


def write_static_meta_csv(static_meta: StaticMeta, output_path: str) -> None:
    """Write static metadata to a single-row CSV."""
    import csv

    row = {field: getattr(static_meta, field) for field in STATIC_META_FIELDS}
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATIC_META_FIELDS)
        writer.writeheader()
        writer.writerow(row)

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
        "-t", family_tag,
        project_dir,
    ], check=False)
    if result.returncode != 0:
        print(f"[build] Family image build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"[build] Image ready: {family_tag}")
    return ImageInfo(tag=family_tag)


# ─────────────────────────────────────────────
# Single Case Execution
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
) -> str:
    """Run one profiling case and return result CSV path."""
    model_tag = _sanitize_model_id(task_info.model_id)
    case_name = f"case_{model_tag}_{cpu}c_{mem}g_{gpu}"
    container_name = case_name

    host_port = SERVER_PORT + cpu * 100 + mem
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
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[merge] Final CSV: {output_path} ({len(all_rows)} rows)")

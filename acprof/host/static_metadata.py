"""主机、模型与 profiler 计划的静态元数据。"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from acprof.config import STATIC_META_FIELDS, STATIC_META_SCHEMA_VERSION
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import (
    ImageInfo,
    _run,
)
from acprof.host.input_plan import (
    PlannedInputScales,
)
from acprof.host.model_schema import (
    _model_io_formats,
    _inference_precision_by_device,
)


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
    image_id: str = ""
    image_name: str = ""
    runtime_environment: Dict[str, Any] = field(default_factory=dict)
    runtime_validation: Dict[str, Any] = field(default_factory=dict)
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


CPU_SYSFS_ROOT = "/sys/devices/system/cpu"


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
        image_id=image_info.tag if image_info.tag.startswith("sha256:") else "",
        image_name=getattr(image_info, "name", ""),
        runtime_environment=dict(getattr(image_info, "runtime_environment", {})),
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
    if getattr(image_info, "runtime_environment", {}):
        packages = image_info.runtime_environment.get("packages", {})
        disabled_metadata.update({
            "torch_version": packages.get("torch", "unknown"),
            "transformers_version": packages.get("transformers", "unknown"),
        })
        if image_info.runtime_environment.get("adapter") == "moss-transcribe-diarize":
            static_meta = replace(static_meta, inference_precision_by_device={"cpu": "FP32", "gpu": "BF16"})
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
    if not isinstance(static_meta, StaticMeta):
        raise TypeError("static_meta must be a StaticMeta instance")
    return replace(
        static_meta,
        workload=dict(planned.workload),
        input_scale_type=str(planned.workload.get("input_scale_type") or static_meta.input_scale_type),
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

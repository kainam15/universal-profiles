"""AC-Prof Universal Profiler - Configuration & Constants."""

from dataclasses import dataclass, field
from typing import List, Dict, Any

# 保留公共常量名称；任务、后端与架构声明统一由 extension manifests 提供。
from acprof.extensions import CATALOG

PIPELINE_TAG_TO_FAMILY: Dict[str, str] = dict(CATALOG.task_families)
LIBRARY_TO_BACKEND: Dict[str, str] = dict(CATALOG.library_backends)
ARCHITECTURE_TO_TASK: Dict[str, str] = dict(CATALOG.architecture_tasks)
DEFAULT_BACKEND = "transformers_pipeline"

# ─────────────────────────────────────────────
# 各任务族的输入缩放维度
# ─────────────────────────────────────────────
@dataclass
class ScalingConfig:
    param_name: str
    values: list
    csv_field: str = "input_scale"
    description: str = ""

SCALING_DIMENSIONS: Dict[str, ScalingConfig] = {
    "nlp": ScalingConfig(
        param_name="seq_length",
        values=[64, 128, 256, 512, 1024, 2048],
        description="sequence length (tokens)",
    ),
    "cv": ScalingConfig(
        param_name="resolution_scale",
        values=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        description="resolution multiplier of base size",
    ),
    "audio": ScalingConfig(
        param_name="duration_s",
        values=[1, 2, 5, 10, 20, 30],
        description="audio duration (seconds)",
    ),
    "timeseries": ScalingConfig(
        param_name="context_length",
        values=[64, 128, 256, 512, 1024, 2048],
        description="context length (time steps)",
    ),
    "structured": ScalingConfig(
        param_name="structured_scale",
        values=[1, 8, 32, 128],
        description="rows, observations or graph nodes; unit is recorded in the workload plan",
    ),
    "diffusion": ScalingConfig(
        param_name="resolution_px",
        values=[128, 192, 256, 320, 384, 512],
        description="square output image side length (pixels)",
    ),
    "multimodal": ScalingConfig(
        param_name="media_scale",
        values=[224, 336, 448],
        description="task-specific media scale; authoritative unit is in the workload plan",
    ),
}

# ─────────────────────────────────────────────
# 各任务族的默认 task_param（二级参数）
# ─────────────────────────────────────────────
DEFAULT_TASK_PARAMS: Dict[str, Dict[str, Any]] = {
    "nlp": {"max_new_tokens": 64},
    "cv": {},
    "audio": {},
    "timeseries": {"prediction_length": 64},
    "structured": {},
    "diffusion": {"num_inference_steps": 20, "guidance_scale": 7.5},
    "multimodal": {"max_new_tokens": 64},
}

# ─────────────────────────────────────────────
# 默认资源矩阵
# ─────────────────────────────────────────────
DEFAULT_CPU_LIST = [1, 2, 4, 8]
DEFAULT_MEM_LIST = [2, 4, 8, 16]
DEFAULT_GPU_LIST = ["off", "on"]

# ─────────────────────────────────────────────
# 实验参数默认值
# ─────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMUP = 2
DEFAULT_REPEAT = 5
DEFAULT_REPEAT_IN_WINDOW = 0
DEFAULT_REPEAT_WINDOW_SECONDS = 10.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
CLIENT_REQUEST_TIMEOUT_EXIT_CODE = 9
DEFAULT_SAMPLE_HZ = 20.0
DEFAULT_IDLE_SECONDS = 20.0
DEFAULT_IDLE_COOLDOWN_SECONDS = 5.0
DEFAULT_COOLDOWN_SECONDS = DEFAULT_IDLE_COOLDOWN_SECONDS
DEFAULT_COMPUTE_PROFILE_TOOL = "none"
IDLE_DIAG_DIRNAME = "debug_idle_diag"
SERVER_PORT = 8002
READY_TIMEOUT_S = 180
READY_POLL_INTERVAL_S = 0.1

# ─────────────────────────────────────────────
# CSV 输出字段
# ─────────────────────────────────────────────
from acprof.metric_registry import CSV_FIELDS, GPU_RUNTIME_STATE_FIELDS

STATIC_META_FIELDS = [
    "profiling_mode",
    "capability_report",
    "schema_version",
    "model_name",
    "model_revision",
    "parameter_count",
    "parameter_bytes",
    "precision_dtype",
    "parameter_dtype_counts",
    "inference_precision_by_device",
    "static_flops",
    "static_macs",
    "input_format",
    "output_format",
    "quantized",
    "quantization_method",
    "quantization_config",
    "model_license",
    "model_metadata_source",
    "task_family",
    "pipeline_tag",
    "runtime_backend",
    "image_tag",
    "image_id",
    "image_name",
    "runtime_environment",
    "runtime_validation",
    "batch_size",
    "input_scale_type",
    "workload",
    "input_scale_plan_sha256",
    "run_command",
    "model_download_url",
    "gpu",
    "gpu_mem_total_bytes",
    "host_mem_total_bytes",
    "host_swap_total_bytes",
    "host_swap_used_bytes_at_start",
    "host_swap_type",
    "host_vm_swappiness",
    "model_cache_bytes",
    "docker_image_bytes",
    "docker_storage_total_bytes",
    "docker_storage_available_bytes_at_start",
    "docker_storage_filesystem",
    "docker_storage_device",
    "docker_storage_type",
    "environment",
    "cgroup_version",
    "cgroup_collection_mode",
    "cpu_power_source",
    "vcpu_power_method",
    "cpu_governor",
    "cpu_boost",
    "compute_profile_tools",
    "torch_profiler_eager_flop_semantics",
    "torch_profiler_eager_attention_implementation",
    "torch_profiler_eager_repeat_cpu",
    "torch_profiler_eager_repeat_gpu",
    "ncu_flop_semantics",
    "ncu_repeat",
    "ncu_fma_flop_weight",
    "ncu_metrics",
    "torch_version",
    "transformers_version",
    "ncu_version",
    "gpu_compute_capability",
    "gpu_sm_count",
    "compute_profiles_retained",
    "compute_profile_provenance",
    "execution_profile_schema_version",
    "execution_profile_tools",
    "massif_peak_semantics",
    "massif_repeat",
    "massif_version",
    "massif_sampling_strategy",
    "massif_reference_cpu_cores",
    "massif_reference_mem_cap_gb",
    "massif_reused_across_resource_cases",
    "nsys_timeline_semantics",
    "nsys_repeat",
    "nsys_version",
    "nsys_sampling_strategy",
    "nsys_reference_cpu_cores",
    "nsys_reference_mem_cap_gb",
    "nsys_reused_across_resource_cases",
    "execution_profiles_retained",
    "execution_profile_provenance",
]
STATIC_META_SCHEMA_VERSION = 7

# ─────────────────────────────────────────────
# Docker 镜像命名
# ─────────────────────────────────────────────
DOCKER_IMAGE_PREFIX = "acprof"

# ─────────────────────────────────────────────
# HF 环境变量（中国网络可选镜像）
# ─────────────────────────────────────────────
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
PYPI_MIRROR_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
CONTAINER_HF_HOME = "/models/hf"
CONTAINER_MODEL_LOCAL_PATH = "/models/model-snapshot"

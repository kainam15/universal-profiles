"""AC-Prof Universal Profiler - Configuration & Constants."""

from dataclasses import dataclass, field
from typing import List, Dict, Any

# ─────────────────────────────────────────────
# Pipeline Tag → Task Family 映射
# ─────────────────────────────────────────────
PIPELINE_TAG_TO_FAMILY: Dict[str, str] = {
    # NLP
    "text-generation": "nlp",
    "text2text-generation": "nlp",
    "text-classification": "nlp",
    "token-classification": "nlp",
    "question-answering": "nlp",
    "summarization": "nlp",
    "translation": "nlp",
    "fill-mask": "nlp",
    "feature-extraction": "nlp",
    "zero-shot-classification": "nlp",
    "sentence-similarity": "nlp",
    "conversational": "nlp",
    # CV
    "image-classification": "cv",
    "object-detection": "cv",
    "image-segmentation": "cv",
    "depth-estimation": "cv",
    "image-to-text": "cv",
    "zero-shot-image-classification": "cv",
    "image-feature-extraction": "cv",
    # Audio
    "automatic-speech-recognition": "audio",
    "audio-classification": "audio",
    "text-to-speech": "audio",
    "audio-to-audio": "audio",
    # Time-series
    "time-series-forecasting": "timeseries",
}

# ─────────────────────────────────────────────
# Library → Runtime Backend 映射
# ─────────────────────────────────────────────
LIBRARY_TO_BACKEND: Dict[str, str] = {
    "transformers": "transformers_pipeline",
    "sentence-transformers": "transformers_pipeline",
    "chronos": "chronos",
    "diffusers": "diffusers",
    "timm": "transformers_model",
}

DEFAULT_BACKEND = "transformers_pipeline"

# ─────────────────────────────────────────────
# Architecture → Pipeline Tag 推断（Level 2 检测兜底）
# ─────────────────────────────────────────────
ARCHITECTURE_TO_TASK: Dict[str, str] = {
    "ForCausalLM": "text-generation",
    "ForMaskedLM": "fill-mask",
    "ForSequenceClassification": "text-classification",
    "ForTokenClassification": "token-classification",
    "ForQuestionAnswering": "question-answering",
    "ForSeq2SeqLM": "text2text-generation",
    "ForConditionalGeneration": "text2text-generation",
    "ForImageClassification": "image-classification",
    "ForObjectDetection": "object-detection",
    "ForSemanticSegmentation": "image-segmentation",
    "ForDepthEstimation": "depth-estimation",
    "ForAudioClassification": "audio-classification",
    "ForCTC": "automatic-speech-recognition",
    "ForSpeechSeq2Seq": "automatic-speech-recognition",
}

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
}

# ─────────────────────────────────────────────
# 各任务族的默认 task_param（二级参数）
# ─────────────────────────────────────────────
DEFAULT_TASK_PARAMS: Dict[str, Dict[str, Any]] = {
    "nlp": {"max_new_tokens": 64},
    "cv": {},
    "audio": {},
    "timeseries": {"prediction_length": 64},
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
DEFAULT_SAMPLE_HZ = 20.0
DEFAULT_IDLE_SECONDS = 3.0
DEFAULT_IDLE_COOLDOWN_SECONDS = 3.0
DEFAULT_COOLDOWN_SECONDS = DEFAULT_IDLE_COOLDOWN_SECONDS
IDLE_DIAG_DIRNAME = "debug_idle_diag"
SERVER_PORT = 8002
READY_TIMEOUT_S = 180
READY_POLL_INTERVAL_S = 0.1

# ─────────────────────────────────────────────
# CSV 输出字段
# ─────────────────────────────────────────────
CSV_FIELDS = [
    "cpu_cores",
    "mem_cap_gb",
    "gpu_mode",
    "input_scale",
    "task_param",
    "repeat_idx",
    "warmup",
    "repeat_in_window",
    "latency_s",
    "latency_p50_s",
    "latency_p90_s",
    "latency_p95_s",
    "latency_slow_ratio",
    "latency_app_s",
    "latency_app_p50_s",
    "latency_app_p90_s",
    "latency_app_p95_s",
    "latency_app_slow_ratio",
    "throughput_samples_per_s",
    "model_logical_mflop_per_request_torch_profiler_eager",
    "model_logical_mflops_app_torch_profiler_eager",
    "model_logical_mflops_packet_torch_profiler_eager",
    "compute_profile_error_torch_profiler_eager",
    "gpu_executed_mflop_per_request_ncu",
    "gpu_executed_tensor_mflop_per_request_ncu",
    "gpu_executed_scalar_mflop_per_request_ncu",
    "gpu_executed_tensor_share_pct_ncu",
    "gpu_executed_mflops_app_ncu",
    "gpu_executed_mflops_packet_ncu",
    "gpu_kernel_launch_count_per_request_ncu",
    "gpu_kernel_time_sum_ms_per_request_ncu",
    "compute_profile_error_ncu",
    "cpu_heap_peak_bytes_massif",
    "cpu_heap_extra_peak_bytes_massif",
    "cpu_stack_peak_bytes_massif",
    "cpu_heap_peak_total_bytes_massif",
    "cpu_heap_peak_at_ms_massif",
    "compute_profile_error_massif",
    "host_inference_wall_time_ms_per_request_nsys",
    "cuda_api_time_sum_ms_per_request_nsys",
    "cuda_api_call_count_per_request_nsys",
    "gpu_kernel_time_sum_ms_per_request_nsys",
    "gpu_kernel_launch_count_per_request_nsys",
    "gpu_memcpy_time_sum_ms_per_request_nsys",
    "gpu_memcpy_count_per_request_nsys",
    "gpu_memcpy_bytes_per_request_nsys",
    "compute_profile_error_nsys",
    "gpu_idle_power_w",
    "gpu_idle_measured_at",
    "gpu_idle_rel_range_so_far",
    "gpu_energy_iters",
    "gpu_avg_power_total_w",
    "gpu_peak_power_total_w",
    "gpu_energy_total_j",
    "gpu_avg_power_eff_w",
    "gpu_peak_power_eff_w",
    "gpu_energy_eff_j",
    "cpu_idle_power_w",
    "cpu_idle_measured_at",
    "cpu_idle_rel_range_so_far",
    "cpu_energy_iters",
    "cpu_avg_power_total_w",
    "cpu_peak_power_total_w",
    "cpu_energy_total_j",
    "cpu_avg_power_eff_w",
    "cpu_peak_power_eff_w",
    "cpu_energy_eff_j",
    "vcpu_cpu_share",
    "vcpu_cpu_time_s",
    "vcpu_avg_power_total_w",
    "vcpu_peak_power_total_w",
    "vcpu_energy_total_j",
    "vcpu_avg_power_eff_w",
    "vcpu_peak_power_eff_w",
    "vcpu_energy_eff_j",
    "resource_usage_iters",
    "container_cpu_util_avg_pct",
    "container_cpu_util_peak_pct",
    "cpu_freq_avg_hz",
    "cpu_freq_peak_hz",
    "cpu_cycles_est_app",
    "cpu_cycles_est_packet",
    "cpu_instructions_per_request",
    "cpu_mips_app",
    "cpu_mips_packet",
    "cpu_perf_elapsed_s",
    "cpu_cache_references_per_request",
    "cpu_cache_misses_per_request",
    "cpu_cache_miss_rate_pct",
    "cpu_dtlb_loads_per_request",
    "cpu_dtlb_load_misses_per_request",
    "cpu_dtlb_load_miss_rate_pct",
    "container_mem_usage_avg_bytes",
    "container_mem_usage_peak_bytes",
    "container_mem_util_avg_pct",
    "container_mem_util_peak_pct",
    "gpu_util_avg_pct",
    "gpu_util_peak_pct",
    "gpu_mem_used_avg_bytes",
    "gpu_mem_used_peak_bytes",
    "gpu_mem_util_avg_pct",
    "gpu_mem_util_peak_pct",
    "cold_start_s",
    "status",
    "error",
]

STATIC_META_FIELDS = [
    "schema_version",
    "model_name",
    "model_revision",
    "parameter_count",
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
    "batch_size",
    "input_scale_type",
    "run_command",
    "model_download_url",
    "gpu",
    "gpu_mem_total_bytes",
    "model_weight_bytes",
    "docker_image_bytes",
    "environment",
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
    "nsys_timeline_semantics",
    "nsys_repeat",
    "nsys_version",
    "execution_profiles_retained",
    "execution_profile_provenance",
]
STATIC_META_SCHEMA_VERSION = 1

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

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
DEFAULT_REPEAT_IN_WINDOW = 20
DEFAULT_SAMPLE_HZ = 20.0
DEFAULT_IDLE_SECONDS = 3.0
DEFAULT_COOLDOWN_SECONDS = 3
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
    "sniff_group_id",
    "repeat_in_window",
    "latency_s",
    "latency_app_s",
    "throughput_samples_per_s",
    "idle_power_w",
    "energy_iters",
    "avg_power_total_w",
    "peak_power_total_w",
    "energy_total_j",
    "avg_power_eff_w",
    "peak_power_eff_w",
    "energy_eff_j",
    "cpu_idle_power_w",
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
    "cold_start_s",
    "status",
    "error",
]

STATIC_META_FIELDS = [
    "model_name",
    "model_revision",
    "task_family",
    "pipeline_tag",
    "runtime_backend",
    "image_tag",
    "batch_size",
    "input_scale_type",
    "model_download_url",
    "gpu",
    "model_weight_bytes",
    "docker_image_bytes",
    "environment",
    "cpu_power_source",
    "vcpu_power_method",
]

# ─────────────────────────────────────────────
# Docker 镜像命名
# ─────────────────────────────────────────────
DOCKER_IMAGE_PREFIX = "acprof"

# ─────────────────────────────────────────────
# HF 环境变量（中国网络可选镜像）
# ─────────────────────────────────────────────
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
PYPI_MIRROR_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

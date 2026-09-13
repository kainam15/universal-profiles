"""声明模型运行环境及适配器；主机端只读取元数据，不导入推理框架。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MOSS_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MOSS_ADAPTER = "moss-transcribe-diarize"
PYTHON_BASE_IMAGE = (
    "docker.m.daocloud.io/library/python:3.10-slim@sha256:"
    "fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2"
)
# OpenMOSS official inference_utils.DEFAULT_PROMPT (Apache-2.0).
MOSS_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    family: str
    adapter: str = "family-default"
    requirements_lock: str = ""
    python_base_image: str = PYTHON_BASE_IMAGE
    torch_index_url: str = "https://download.pytorch.org/whl/cu128"
    gpu_dtype: str = "FP16"
    trust_remote_code: bool = False
    task_types: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()
    backends: tuple[str, ...] = ("transformers_model", "transformers_pipeline")

    def __post_init__(self) -> None:
        if not self.requirements_lock:
            raise ValueError("运行环境必须登记完整 requirements_lock；不支持未锁定环境")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def common_requirements_lock(self) -> str:
        variant = self.torch_index_url.rstrip("/").rsplit("/", 1)[-1]
        return f"dockerfiles/locks/common-{variant}.txt"


PROFILES = {
    "multimodal-transformers4576": RuntimeProfile(
        "multimodal-transformers4576", "multimodal",
        requirements_lock="dockerfiles/locks/multimodal-transformers4576.txt",
    ),
    "moss-transformers560": RuntimeProfile(
        "moss-transformers560", "multimodal", MOSS_ADAPTER,
        "dockerfiles/locks/moss-transformers560.txt",
        gpu_dtype="BF16", trust_remote_code=True,
        task_types=("audio-text-to-text",), model_types=("moss_transcribe_diarize",),
    ),
}
for _family in ("nlp", "cv", "audio", "diffusion", "structured", "timeseries", "multimodal"):
    for _variant in ("cu128", "cu124", "cpu"):
        _stem = "multimodal-transformers4576" if _family == "multimodal" else _family
        _name = _stem if _family == "multimodal" and _variant == "cu128" else f"{_stem}-{_variant}"
        if _name not in PROFILES:
            PROFILES[_name] = RuntimeProfile(
                _name, _family, requirements_lock=f"dockerfiles/locks/{_name}.txt",
                torch_index_url=f"https://download.pytorch.org/whl/{_variant}",
            )
# 同架构 checkpoint 可复用适配器；任务标签本身不授予架构兼容性。
ARCHITECTURE_PROFILES = {"moss_transcribe_diarize": "moss-transformers560"}
MODEL_PROFILES = {MOSS_MODEL_ID.lower(): "moss-transformers560"}


def select_runtime_profile(task_info: Any) -> RuntimeProfile:
    config = getattr(task_info, "model_config", {}) or {}
    model_type = str(config.get("model_type") or "")
    profile_id = (
        ARCHITECTURE_PROFILES.get(model_type)
        or MODEL_PROFILES.get(task_info.model_id.lower())
    )
    if profile_id:
        profile = PROFILES[profile_id]
        if (
            task_info.task_family != profile.family
            or (profile.task_types and task_info.pipeline_tag not in profile.task_types)
            or task_info.runtime_backend not in profile.backends
        ):
            raise ValueError(
                f"Runtime {profile.profile_id!r} does not support "
                f"{task_info.task_family}/{task_info.pipeline_tag}/{task_info.runtime_backend}"
            )
        if model_type and profile.model_types and model_type not in profile.model_types:
            raise ValueError("Model metadata does not match its registered runtime architecture")
        return profile
    if task_info.task_family == "multimodal":
        supported = {
            "audio-text-to-text": {"qwen2_audio", "qwen2_5_omni"},
            "any-to-any": {"qwen2_5_omni"},
            "visual-document-retrieval": {"colpali", "colqwen2"},
        }.get(task_info.pipeline_tag)
        if model_type and supported is not None and model_type not in supported:
            raise ValueError(
                f"No registered runtime/adapter for {task_info.pipeline_tag} architecture {model_type!r}"
            )
        if (config.get("auto_map") or {}).get("AutoConfig") and not supported:
            raise ValueError("Custom multimodal architecture requires a registered runtime/adapter")
    family = task_info.task_family
    default = "multimodal-transformers4576" if family == "multimodal" else f"{family}-cu128"
    selected = getattr(task_info, "runtime_profile_id", "") or default
    profile = PROFILES.get(selected)
    if profile is None or profile.family != family or profile.adapter != "family-default":
        raise ValueError(f"No registered runtime for {family}/{selected}")
    return profile


def default_model_adapter(model_id: str) -> str:
    profile_id = MODEL_PROFILES.get(model_id.lower())
    return PROFILES[profile_id].adapter if profile_id else "family-default"

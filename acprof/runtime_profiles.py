"""声明模型运行环境及适配器；主机端只读取元数据，不导入推理框架。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from acprof.dependency_locks import (
    content_digest, package_versions, read_python_lock, read_system_lock,
    require_parent_subset, system_lock_identity,
)


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
class PlatformSpec:
    platform_id: str
    torch_version: str
    torch_index_url: str
    requirements_lock: str
    python_base_image: str = PYTHON_BASE_IMAGE
    python_version: str = "3.10.21"
    architecture: str = "linux/amd64"
    python_target: str = "x86_64-manylinux_2_28"
    system_lock: str = "dockerfiles/locks/system-trixie-amd64.json"


@dataclass(frozen=True)
class DependencyEnvironment:
    environment_key: str
    platform: PlatformSpec
    requirements_lock: str
    requirements_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    family: str
    environment: DependencyEnvironment | None = None
    adapter: str = "family-default"
    gpu_dtype: str = "FP16"
    trust_remote_code: bool = False
    task_types: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()
    backends: tuple[str, ...] = ("transformers_model", "transformers_pipeline")

    def __post_init__(self) -> None:
        if not isinstance(self.environment, DependencyEnvironment) or not self.environment.requirements_lock:
            raise ValueError("逻辑 profile 必须引用完整 dependency environment lock；不支持未锁定环境")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

PLATFORMS = {
    key: PlatformSpec(key, version, f"https://download.pytorch.org/whl/{key}",
                      f"dockerfiles/locks/platform-{key}.txt")
    for key, version in (("cpu", "2.11.0+cpu"), ("cu124", "2.6.0+cu124"), ("cu128", "2.11.0+cu128"))
}

# 名称只是环境声明的引用键；缓存身份取决于完整锁内容。
ENVIRONMENTS = {}
for _name, _platform, _inputs in (
    ("nlp-cpu", "cpu", ("nlp",)),
    ("nlp-cu124", "cu124", ("nlp",)),
    ("nlp-cu128", "cu128", ("nlp",)),
    ("cv-cpu", "cpu", ("cv",)),
    ("cv-cu124", "cu124", ("cv",)),
    ("cv-cu128", "cu128", ("cv",)),
    ("audio-cpu", "cpu", ("audio", "multimodal-transformers4576")),
    ("audio-cu124", "cu124", ("audio", "multimodal-transformers4576")),
    ("audio-cu128", "cu128", ("audio",)),
    ("diffusion-cpu", "cpu", ("diffusion",)),
    ("diffusion-cu124", "cu124", ("diffusion",)),
    ("diffusion-cu128", "cu128", ("diffusion",)),
    ("structured-cpu", "cpu", ("structured",)),
    ("structured-cu124", "cu124", ("structured",)),
    ("structured-cu128", "cu128", ("structured",)),
    ("timeseries-cpu", "cpu", ("timeseries",)),
    ("timeseries-cu124", "cu124", ("timeseries",)),
    ("timeseries-cu128", "cu128", ("timeseries",)),
    ("multimodal-transformers4576", "cu128", ("multimodal-transformers4576",)),
    ("moss-transformers560", "cu128", ("moss-transformers560",)),
):
    ENVIRONMENTS[_name] = DependencyEnvironment(
        _name, PLATFORMS[_platform], f"dockerfiles/locks/{_name}.txt",
        tuple(f"dockerfiles/requirements/{item}.in" for item in _inputs),
    )

PROFILES = {}
for _name, _family, _environment in (
    ("nlp-cpu", "nlp", "nlp-cpu"), ("nlp-cu124", "nlp", "nlp-cu124"), ("nlp-cu128", "nlp", "nlp-cu128"),
    ("cv-cpu", "cv", "cv-cpu"), ("cv-cu124", "cv", "cv-cu124"), ("cv-cu128", "cv", "cv-cu128"),
    ("audio-cpu", "audio", "audio-cpu"), ("audio-cu124", "audio", "audio-cu124"), ("audio-cu128", "audio", "audio-cu128"),
    ("diffusion-cpu", "diffusion", "diffusion-cpu"), ("diffusion-cu124", "diffusion", "diffusion-cu124"), ("diffusion-cu128", "diffusion", "diffusion-cu128"),
    ("structured-cpu", "structured", "structured-cpu"), ("structured-cu124", "structured", "structured-cu124"), ("structured-cu128", "structured", "structured-cu128"),
    ("timeseries-cpu", "timeseries", "timeseries-cpu"), ("timeseries-cu124", "timeseries", "timeseries-cu124"), ("timeseries-cu128", "timeseries", "timeseries-cu128"),
    ("multimodal-transformers4576-cpu", "multimodal", "audio-cpu"),
    ("multimodal-transformers4576-cu124", "multimodal", "audio-cu124"),
    ("multimodal-transformers4576", "multimodal", "multimodal-transformers4576"),
):
    PROFILES[_name] = RuntimeProfile(_name, _family, ENVIRONMENTS[_environment])
PROFILES["moss-transformers560"] = RuntimeProfile(
    "moss-transformers560", "multimodal", ENVIRONMENTS["moss-transformers560"], MOSS_ADAPTER,
    gpu_dtype="BF16", trust_remote_code=True,
    task_types=("audio-text-to-text",), model_types=("moss_transcribe_diarize",),
)
DEFAULT_PROFILES = {
    (profile.family, profile.environment.platform.platform_id): profile.profile_id
    for profile in PROFILES.values() if profile.adapter == "family-default"
}


def platform_identity(platform: PlatformSpec, project_dir) -> dict:
    from pathlib import Path
    root = Path(project_dir)
    system = read_system_lock(root / platform.system_lock)
    if system["base_image"] != platform.python_base_image or platform.architecture != "linux/" + system["architecture"]:
        raise ValueError("平台与 system lock 的基础镜像或架构不符")
    packages = read_python_lock(root / platform.requirements_lock)
    if package_versions(packages).get("torch") != platform.torch_version:
        raise ValueError("平台 Torch 版本与依赖 lock 不符")
    return {"schema_version": 1, "python_base_image": platform.python_base_image,
            "python_version": platform.python_version, "architecture": platform.architecture,
            "python_target": platform.python_target, "torch_index_url": platform.torch_index_url,
            "system": system_lock_identity(system), "packages": packages}


def environment_identity(environment: DependencyEnvironment, project_dir) -> dict:
    from pathlib import Path
    platform = platform_identity(environment.platform, project_dir)
    packages = read_python_lock(Path(project_dir) / environment.requirements_lock)
    require_parent_subset(platform["packages"], packages)
    return {"schema_version": 1, "platform": platform, "packages": packages}


def environment_id(environment: DependencyEnvironment, project_dir) -> str:
    return content_digest(environment_identity(environment, project_dir))
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
    default = DEFAULT_PROFILES.get((family, "cu128"), "")
    selected = getattr(task_info, "runtime_profile_id", "") or default
    profile = PROFILES.get(selected)
    if profile is None or profile.family != family or profile.adapter != "family-default":
        raise ValueError(f"No registered runtime for {family}/{selected}")
    return profile


def default_model_adapter(model_id: str) -> str:
    profile_id = MODEL_PROFILES.get(model_id.lower())
    return PROFILES[profile_id].adapter if profile_id else "family-default"

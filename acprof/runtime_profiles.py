"""声明模型运行环境及适配器；主机端只读取元数据，不导入推理框架。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from acprof.dependency_locks import (
    content_digest, package_versions, read_python_lock, read_system_lock,
    require_parent_subset, system_lock_identity,
)
from acprof.extensions import CATALOG, select_extension


PYTHON_BASE_IMAGE = (
    "docker.m.daocloud.io/library/python:3.10-slim@sha256:"
    "fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2"
)
@dataclass(frozen=True)
class PlatformSpec:
    platform_id: str
    # 保留旧位置参数与身份；新平台只需声明 Python、系统与基础安装工具。
    torch_version: str | None = None
    torch_index_url: str | None = None
    requirements_lock: str = ""
    python_base_image: str = PYTHON_BASE_IMAGE
    python_version: str = "3.10.21"
    architecture: str = "linux/amd64"
    python_target: str = "x86_64-manylinux_2_28"
    system_lock: str = "dockerfiles/locks/system-trixie-amd64.json"

    def __post_init__(self) -> None:
        if not self.requirements_lock:
            raise ValueError("平台必须声明完整 requirements lock")
        if bool(self.torch_version) != bool(self.torch_index_url):
            raise ValueError("旧 Torch 平台必须同时声明 torch_version 和 torch_index_url")


@dataclass(frozen=True)
class RuntimeSpec:
    """运行时及其分发包约束；不是另一个执行 adapter。"""
    type: str
    version: str
    package: str = ""

    def __post_init__(self) -> None:
        if not self.type or not self.version:
            raise ValueError("runtime 必须声明 type 和精确 version")


@dataclass(frozen=True)
class DependencyEnvironment:
    environment_key: str
    platform: PlatformSpec
    requirements_lock: str
    requirements_inputs: tuple[str, ...] = ()
    runtime: RuntimeSpec | None = None

    @property
    def runtime_type(self) -> str:
        return self.runtime.type if self.runtime else ("torch" if self.platform.torch_version else "")


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
    runtime_line: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.environment, DependencyEnvironment) or not self.environment.requirements_lock:
            raise ValueError("逻辑 profile 必须引用完整 dependency environment lock；不支持未锁定环境")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def runtime_declarations(catalog) -> tuple[dict, dict, dict]:
    """Materialize declared references through the existing locked runtime types."""
    platforms = {key: PlatformSpec(**values) for key, values in catalog.platforms.items()}
    raw_environments = dict(catalog.dependency_environments)
    for extension in catalog.extensions.values():
        if extension.dependency_environment:
            previous = raw_environments.get(extension.environment)
            if previous is not None and previous != extension.dependency_environment:
                raise ValueError(f"conflicting dependency environment: {extension.environment}")
            raw_environments[extension.environment] = extension.dependency_environment
    environments = {}
    for key, raw in raw_environments.items():
        values = dict(raw)
        values["platform"] = platforms[values["platform"]]
        values["requirements_inputs"] = tuple(values.get("requirements_inputs", ()))
        if values.get("runtime"):
            values["runtime"] = RuntimeSpec(**values["runtime"])
        environments[key] = DependencyEnvironment(environment_key=key, **values)
    profiles = {}
    for key, raw in catalog.runtime_profiles.items():
        values = dict(raw)
        values["environment"] = environments[values["environment"]]
        for field_name in ("task_types", "model_types", "backends"):
            if field_name in values:
                values[field_name] = tuple(values[field_name])
        profiles[key] = RuntimeProfile(**values)
    for extension in catalog.extensions.values():
        if not extension.profile:
            continue
        profile = RuntimeProfile(
            extension.profile, extension.family, environments[extension.environment], extension.adapter,
            task_types=extension.tasks, model_types=extension.model_types, backends=extension.backends,
            **{"gpu_dtype": extension.dtypes[0], **extension.profile_options},
        )
        if extension.profile in profiles and profiles[extension.profile] != profile:
            raise ValueError(f"conflicting runtime profile: {extension.profile}")
        profiles[extension.profile] = profile
    return platforms, environments, profiles


PLATFORMS, ENVIRONMENTS, PROFILES = runtime_declarations(CATALOG)
DEFAULT_PROFILES = {
    (profile.family, profile.environment.platform.platform_id): profile.profile_id
    for profile in PROFILES.values() if profile.adapter == "family-default" and profile.runtime_line == "default"
}


@lru_cache(maxsize=None)
def _transformers_version(environment: DependencyEnvironment) -> str | None:
    from acprof.installation import resource_root
    return package_versions(read_python_lock(resource_root() / environment.requirements_lock)).get("transformers")


def _native_compatible(task_info: Any, profile: RuntimeProfile) -> bool | None:
    from acprof.model_resolution import supports_transformers_task
    if task_info.runtime_backend not in {"transformers_model", "transformers_pipeline"}:
        return None
    config = getattr(task_info, "model_config", {}) or {}
    # Custom Auto classes are verified by the selected extension/container.
    if config.get("auto_map") or (getattr(task_info, "model_resolution", {}) or {}).get("interface_kind") == "custom_pipeline":
        return None
    version = _transformers_version(profile.environment)
    if not version:
        return None
    return supports_transformers_task(version, task_info.pipeline_tag, str(config.get("model_type") or ""), config)


def profile_for_platform(profile: RuntimeProfile, platform: str) -> RuntimeProfile:
    if profile.runtime_line == "default":
        return PROFILES[DEFAULT_PROFILES[(profile.family, platform)]]
    matches = [item for item in PROFILES.values() if item.family == profile.family
               and item.adapter == profile.adapter and item.runtime_line == profile.runtime_line
               and item.environment.platform.platform_id == platform]
    if len(matches) != 1:
        raise ValueError(f"No unique {platform} environment for {profile.profile_id}")
    return matches[0]


def platform_identity(platform: PlatformSpec, project_dir) -> dict:
    from pathlib import Path
    root = Path(project_dir)
    system = read_system_lock(root / platform.system_lock)
    if system["base_image"] != platform.python_base_image or platform.architecture != "linux/" + system["architecture"]:
        raise ValueError("平台与 system lock 的基础镜像或架构不符")
    packages = read_python_lock(root / platform.requirements_lock)
    if platform.torch_version and package_versions(packages).get("torch") != platform.torch_version:
        raise ValueError("平台 Torch 版本与依赖 lock 不符")
    identity = {"schema_version": 1, "python_base_image": platform.python_base_image,
            "python_version": platform.python_version, "architecture": platform.architecture,
            "python_target": platform.python_target,
            "system": system_lock_identity(system), "packages": packages}
    if platform.torch_version:
        identity["torch_index_url"] = platform.torch_index_url
    return identity


def environment_identity(environment: DependencyEnvironment, project_dir) -> dict:
    from pathlib import Path
    platform = platform_identity(environment.platform, project_dir)
    packages = read_python_lock(Path(project_dir) / environment.requirements_lock)
    require_parent_subset(platform["packages"], packages)
    if environment.runtime is not None:
        from acprof.dependency_locks import normalized_name
        runtime = environment.runtime
        package = normalized_name(runtime.package or runtime.type)
        if package_versions(packages).get(package) != runtime.version:
            raise ValueError(f"Runtime {runtime.type} 要求 {package}=={runtime.version}，与依赖 lock 不符")
    # 运行时角色不改变已锁定制品集合；同一完整环境可供不同执行器复用。
    return {"schema_version": 1, "platform": platform, "packages": packages}


def environment_id(environment: DependencyEnvironment, project_dir) -> str:
    return content_digest(environment_identity(environment, project_dir))


def select_runtime_profile(task_info: Any) -> RuntimeProfile:
    extension = select_extension(task_info)
    if extension.profile:
        profile = PROFILES.get(extension.profile)
        if profile is None:
            raise ValueError(f"No registered runtime for extension {extension.extension_id!r}: {extension.profile}")
        if (
            task_info.task_family != profile.family
            or (profile.task_types and task_info.pipeline_tag not in profile.task_types)
            or task_info.runtime_backend not in profile.backends
            or extension.adapter != profile.adapter
        ):
            raise ValueError(
                f"Runtime {profile.profile_id!r} does not support "
                f"{task_info.task_family}/{task_info.pipeline_tag}/{task_info.runtime_backend}"
            )
        explicit = getattr(task_info, "runtime_profile_id", "")
        if explicit and explicit != profile.profile_id:
            raise ValueError(f"Runtime {explicit!r} does not match extension {extension.extension_id!r}")
        return profile
    family = task_info.task_family
    default = DEFAULT_PROFILES.get((family, "cu128"), "")
    from acprof.model_spec import declared_multimodal_pipeline
    custom_pipeline = declared_multimodal_pipeline(task_info)
    if custom_pipeline:
        default = "custom-multimodal-cu128"
    selected = getattr(task_info, "runtime_profile_id", "") or default
    profile = PROFILES.get(selected)
    if profile is None or profile.family != family or profile.adapter != "family-default":
        raise ValueError(f"No registered runtime for {family}/{selected}")
    if custom_pipeline and profile.runtime_line != "custom-multimodal":
        raise ValueError("Declared multimodal pipelines require a locked custom-multimodal runtime")
    if _native_compatible(task_info, profile) is False:
        if getattr(task_info, "runtime_profile_id", ""):
            raise ValueError(f"Runtime {selected} does not register {task_info.pipeline_tag}/{task_info.model_config.get('model_type')}")
        candidates = [item for item in PROFILES.values() if item.family == family
                      and item.runtime_line != "default" and item.adapter == "family-default"
                      and item.environment.platform.platform_id == profile.environment.platform.platform_id
                      and _native_compatible(task_info, item) is True]
        if not candidates:
            raise ValueError(f"No registered Transformers runtime for {task_info.pipeline_tag}/{task_info.model_config.get('model_type')}")
        profile = candidates[0]
    return profile


def default_model_adapter(model_id: str) -> str:
    adapters = {extension.adapter for extension in CATALOG.extensions.values()
                if model_id.lower() in {value.lower() for value in extension.model_ids}}
    if len(adapters) > 1:
        raise ValueError("Model has multiple backend adapters; select the runtime first and pass model_adapter explicitly")
    return next(iter(adapters), "family-default")

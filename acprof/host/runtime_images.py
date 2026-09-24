"""按运行环境、模型 revision 和代码内容构建并核验不可变镜像。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from acprof.dependency_locks import content_digest
from acprof.host.dependency_images import (
    platform_fingerprint, prepare_environment_image, require_image_source, runtime_fingerprint,
    verify_labels, verify_manifest,
)
from acprof.runtime_profiles import RuntimeProfile, environment_id, environment_identity, select_runtime_profile
from acprof.model_spec import encode_model_spec, task_model_spec


from acprof.installation import resource_root

PROJECT_ROOT = resource_root()
FINGERPRINT_LABEL = "org.acprof.build-fingerprint"
REQUEST_LABEL = "org.acprof.request-fingerprint"
MODEL_KEY_LABEL = "org.acprof.model-files-key"


def configure_runtime_profile(task_info: Any) -> RuntimeProfile:
    """只在主机构建预检选择驱动分支；静态路由模块不探测硬件。"""
    from acprof.host.docker_runtime import _select_nlp_torch_index_url
    from acprof.runtime_profiles import PLATFORMS, profile_for_platform

    profile = select_runtime_profile(task_info)
    if profile.adapter == "family-default" and profile.environment.platform.torch_version:
        index = _select_nlp_torch_index_url().rstrip("/")
        variant = index.rsplit("/", 1)[-1]
        if index != f"https://download.pytorch.org/whl/{variant}" or variant not in {"cu128", "cu124", "cpu"}:
            raise ValueError("自定义 Torch 索引需要注册完整依赖锁；支持 cu128、cu124、cpu")
        override = os.environ.get("ACPROF_NLP_TORCH_SPEC", "").strip()
        version = PLATFORMS[variant].torch_version.split("+", 1)[0]
        if override and override not in {f"torch=={version}", f"torch=={version}+{variant}"}:
            raise ValueError(f"Torch 版本与依赖锁不符；{variant} 要求 torch=={version}+{variant}")
        profile = profile_for_platform(profile, variant)
    task_info.runtime_profile_id, task_info.model_adapter = profile.profile_id, profile.adapter
    if getattr(task_info, "model_resolution", None):
        task_info.model_resolution["runtime_profile"] = profile.profile_id
    return profile


def download_policy(task_info: Any) -> str:
    policy = getattr(task_info, "model_download_policy", "auto")
    if policy not in {"auto", "full"}:
        raise ValueError("model download policy must be auto or full")
    return policy


def request_fingerprint(task_info: Any, project_dir: str | Path = PROJECT_ROOT) -> str:
    root = Path(project_dir)
    profile = select_runtime_profile(task_info)
    overrides = {
        key: os.environ.get(key, "").strip()
        for key in ("ACPROF_NLP_TORCH_INDEX_URL", "ACPROF_NLP_TORCH_SPEC")
    } if profile.adapter == "family-default" and profile.environment.platform.torch_version else {}
    logical_profile = profile.to_dict()
    logical_profile["environment"] = environment_id(profile.environment, root)
    digest = hashlib.sha256(json.dumps({
        "schema_version": 1, "model_id": task_info.model_id,
        "model_revision": task_info.model_revision,
        "task": task_info.pipeline_tag, "backend": task_info.runtime_backend,
        "profile": logical_profile,
        "dependency_build": runtime_fingerprint(profile.environment, root),
        "build_overrides": overrides,
        "model_download_policy": download_policy(task_info),
        "model_spec": task_model_spec(task_info),
    }, sort_keys=True).encode())
    paths = sorted((root / "acprof").rglob("*.py"))
    paths += sorted((root / "acprof" / "extensions").rglob("*.json"))
    paths += [root / "dockerfiles" / name for name in ("runtime-model.Dockerfile", "runtime-final.Dockerfile")]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_fingerprint(request_id: str, model_image_id: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", model_image_id):
        raise ValueError("服务构建必须绑定不可变的模型父镜像 ID")
    return content_digest({"schema_version": 2, "request_fingerprint": request_id,
                           "model_image_id": model_image_id})


def model_fingerprint(task_info: Any, runtime_id: str, project_dir: str | Path = PROJECT_ROOT) -> str:
    root = Path(project_dir)
    digest = hashlib.sha256(json.dumps({
        "runtime_id": runtime_id, "model_id": task_info.model_id, "revision": task_info.model_revision,
        "family": task_info.task_family, "backend": task_info.runtime_backend,
        "adapter": select_runtime_profile(task_info).adapter, "policy": download_policy(task_info),
    }, sort_keys=True).encode())
    for relative in ("acprof/container/download_model.py", "acprof/container/model_files.py", "dockerfiles/runtime-model.Dockerfile"):
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def inspect_identity(image: str) -> dict | None:
    from acprof.host.docker_runtime import _run

    result = _run([
        "docker", "image", "inspect", image, "--format",
        '{"image_id":{{json .Id}},"labels":{{json .Config.Labels}}}',
    ], check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        if "No such image" in detail or "No such object" in detail:
            return None
        raise RuntimeError(f"无法检查 Docker 镜像 {image}: {detail}")
    try:
        identity = json.loads(result.stdout)
        if not str(identity["image_id"]).startswith("sha256:"):
            raise ValueError("invalid image ID")
        return identity
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Docker 返回了无效的镜像信息: {image}") from exc


def verified_image(task_info: Any, name: str, fingerprint: str, project_dir=PROJECT_ROOT):
    from acprof.host.docker_runtime import ImageInfo, _run

    identity = inspect_identity(name)
    if identity is None:
        return None
    labels = identity.get("labels") or {}
    if labels.get(REQUEST_LABEL) != fingerprint:
        raise RuntimeError(f"镜像 {name} 的构建指纹不匹配；请重新构建")
    result = _run([
        "docker", "run", "--rm", "--network", "none", "--entrypoint", "cat",
        identity["image_id"], "/app/runtime_environment.json",
    ], check=False)
    if result.returncode:
        raise RuntimeError(f"镜像 {name} 缺少运行环境清单；请重新构建")
    try:
        manifest = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError("镜像运行环境清单无效") from exc
    profile = select_runtime_profile(task_info)
    expected = {
        "request_fingerprint": fingerprint, "profile_id": profile.profile_id,
        "adapter": profile.adapter, "model_id": task_info.model_id,
        "model_revision": task_info.model_revision,
        "environment_id": environment_id(profile.environment, project_dir),
        "platform_id": profile.environment.platform.platform_id,
        "python_version": profile.environment.platform.python_version,
    }
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("镜像环境、适配器或模型 revision 与本次任务不匹配")
    if manifest.get("model_spec", {}) != task_model_spec(task_info):
        raise RuntimeError("镜像模型声明与本次 --model-spec 不匹配")
    for key in ("platform_image_id", "environment_image_id", "model_image_id"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(manifest.get(key, ""))):
            raise RuntimeError(f"镜像缺少不可变的 {key}")
    environment = environment_identity(profile.environment, project_dir)
    expected.update({
        "python_base_image": profile.environment.platform.python_base_image,
        "architecture": profile.environment.platform.architecture,
        "platform_definition_id": content_digest(environment["platform"]),
        "platform_build_fingerprint": platform_fingerprint(profile.environment.platform, project_dir),
        "environment_build_fingerprint": content_digest({
            "definition": runtime_fingerprint(profile.environment, project_dir),
            "platform_image_id": manifest["platform_image_id"],
        }),
    })
    actual_build = build_fingerprint(fingerprint, manifest["model_image_id"])
    if labels.get(FINGERPRINT_LABEL) != actual_build or manifest.get("build_fingerprint") != actual_build:
        raise RuntimeError("服务镜像与不可变父镜像构建指纹不匹配")
    verify_labels(identity, expected, "model")
    if labels.get("org.acprof.runtime-profile") != profile.profile_id or labels.get("org.acprof.model-adapter") != profile.adapter:
        raise RuntimeError("服务镜像 profile 或 adapter 标签不匹配")
    verify_manifest(manifest, expected, environment)
    if manifest.get("model_snapshot_revision") != task_info.model_revision:
        raise RuntimeError("镜像内实际 snapshot revision 与本次任务不匹配")
    from acprof.container.model_files import ModelFilesError, validate_plan

    plan = manifest.get("model_download")
    try:
        validate_plan(plan)
        if plan.get("verification") != "sha256":
            raise ModelFilesError("model download plan has not been verified")
    except ModelFilesError as exc:
        raise RuntimeError(f"镜像模型文件清单缺失或无效；请重新构建：{exc}") from exc
    if any(plan.get(key) != value for key, value in {
        "model_id": task_info.model_id, "model_revision": task_info.model_revision,
        "requested_policy": download_policy(task_info), "backend": task_info.runtime_backend,
        "task_family": task_info.task_family, "adapter": profile.adapter,
    }.items()):
        raise RuntimeError("镜像模型文件清单与本次下载策略/模型/加载器不匹配")
    return ImageInfo(tag=identity["image_id"], name=name, runtime_environment=manifest)


def prepare_runtime_image(task_info: Any, project_dir: str, *, reuse_existing: bool = False):
    from acprof.host.docker_runtime import _model_image_tag, build_image

    profile = configure_runtime_profile(task_info)
    task_info.runtime_profile_id, task_info.model_adapter = profile.profile_id, profile.adapter
    if not re.fullmatch(r"[0-9a-f]{40}", task_info.model_revision or ""):
        from huggingface_hub import model_info

        try:
            revision = model_info(task_info.model_id, revision=task_info.model_revision or "main").sha
        except Exception as exc:
            raise RuntimeError(f"无法固定模型 revision，尚未构建或复用镜像：{exc}") from exc
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError("模型 revision 未解析为完整 commit SHA，尚未构建或复用镜像")
        task_info.model_revision = revision
    fingerprint = request_fingerprint(task_info, project_dir)
    name = _model_image_tag(task_info, project_dir)
    print(f"[runtime] 配置: {profile.profile_id}；平台: {profile.environment.platform.platform_id}；"
          f"依赖环境: {environment_id(profile.environment, project_dir)[:20]}；适配器: {profile.adapter}", flush=True)
    if reuse_existing:
        image = verified_image(task_info, name, fingerprint, project_dir)
        if image is not None:
            print(f"[build] 指纹核验通过，跳过构建并复用：{name}", flush=True)
            return image
        print(f"[build] 未找到本地模型镜像 {name}；将自动构建，旧版 :latest 不用于本次采集。", flush=True)
    return build_image(task_info, project_dir)


def build_runtime_image(task_info: Any, project_dir: str):
    from acprof.host.docker_runtime import ImageInfo, _model_image_tag, _run, _sanitize_model_id
    from acprof.config import HF_MIRROR_ENDPOINT

    profile = select_runtime_profile(task_info)
    task_info.runtime_profile_id, task_info.model_adapter = profile.profile_id, profile.adapter
    if not re.fullmatch(r"[0-9a-f]{40}", task_info.model_revision or ""):
        raise RuntimeError("镜像构建要求固定 model revision；请通过 prepare_image 准备镜像")
    root = Path(project_dir)
    fingerprint = request_fingerprint(task_info, root)
    name = _model_image_tag(task_info, root)

    def build(dockerfile: str, args: dict[str, str], parent: tuple[str, str]) -> str:
        with tempfile.TemporaryDirectory(prefix="acprof-model-build-") as directory:
            iidfile = Path(directory) / "image-id"
            command = ["docker", "build", "--platform", "linux/amd64", "--iidfile", str(iidfile),
                       "-f", str(root / "dockerfiles" / dockerfile)]
            for key, value in args.items():
                command += ["--build-arg", f"{key}={value}"]
            if dockerfile == "runtime-model.Dockerfile" and (os.environ.get("HF_TOKEN") or "").strip():
                command += ["--secret", "id=hf_token,env=HF_TOKEN"]
            command.append(str(root))
            require_image_source(*parent)
            if request_fingerprint(task_info, root) != fingerprint:
                raise RuntimeError("构建期间代码或依赖配置发生变化，尚未发布镜像标签")
            result = _run(command, check=False, capture=False)
            if result.returncode:
                raise RuntimeError(f"Docker 构建失败: {dockerfile} (exit={result.returncode})")
            require_image_source(*parent)
            if request_fingerprint(task_info, root) != fingerprint:
                raise RuntimeError("构建期间代码或依赖配置发生变化，未发布镜像标签")
            image_id = iidfile.read_text().strip()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                raise RuntimeError("构建未返回不可变 image ID")
            return image_id

    dependency = prepare_environment_image(profile.environment, root)
    runtime_id = dependency.image_id
    runtime_source = "acprof-build-source:" + runtime_id.split(":", 1)[1]
    _run(["docker", "tag", runtime_id, runtime_source])
    model_key = model_fingerprint(task_info, runtime_id, root)
    model_tag = f"acprof-weights-{profile.family}-{_sanitize_model_id(task_info.model_id)}:{model_key[:20]}"
    model_identity = inspect_identity(model_tag)
    if model_identity is None:
        candidate = build("runtime-model.Dockerfile", {
            "RUNTIME_IMAGE": runtime_source, "MODEL_ID": task_info.model_id,
            "MODEL_REVISION": task_info.model_revision, "HF_ENDPOINT": HF_MIRROR_ENDPOINT,
            "TASK_FAMILY": task_info.task_family, "RUNTIME_BACKEND": task_info.runtime_backend,
            "MODEL_ADAPTER": profile.adapter, "MODEL_DOWNLOAD_POLICY": download_policy(task_info),
            "MODEL_FILES_KEY": model_key,
        }, (runtime_source, runtime_id))
        model_identity = inspect_identity(candidate)
        if model_identity is None or (model_identity.get("labels") or {}).get(MODEL_KEY_LABEL) != model_key:
            raise RuntimeError("模型文件层指纹不匹配；拒绝发布缓存")
        _run(["docker", "tag", candidate, model_tag])
    if (model_identity.get("labels") or {}).get(MODEL_KEY_LABEL) != model_key:
        raise RuntimeError("模型文件层指纹不匹配；拒绝复用")
    model_id = model_identity["image_id"]
    model_source = "acprof-build-source:" + model_id.split(":", 1)[1]
    _run(["docker", "tag", model_id, model_source])
    final_id = build("runtime-final.Dockerfile", {
        "MODEL_IMAGE": model_source, "RUNTIME_PROFILE": profile.profile_id,
        "MODEL_ADAPTER": profile.adapter, "BUILD_FINGERPRINT": build_fingerprint(fingerprint, model_id),
        "REQUEST_FINGERPRINT": fingerprint, "PLATFORM_IMAGE_ID": dependency.platform_image_id,
        "ENVIRONMENT_IMAGE_ID": runtime_id, "MODEL_IMAGE_ID": model_id,
        "DEPENDENCY_LOCK_SHA256": dependency.manifest["dependency_lock_sha256"],
        "MODEL_SPEC_B64": encode_model_spec(task_model_spec(task_info)),
    }, (model_source, model_id))
    image = verified_image(task_info, final_id, fingerprint, root)
    if image is None:
        raise RuntimeError("构建后未找到模型镜像")
    _run(["docker", "tag", final_id, name])
    return ImageInfo(tag=final_id, name=name, runtime_environment=image.runtime_environment)

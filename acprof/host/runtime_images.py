"""按运行环境、模型 revision 和代码内容构建并核验不可变镜像。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from acprof.runtime_profiles import RuntimeProfile, select_runtime_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT_LABEL = "org.acprof.build-fingerprint"


def build_fingerprint(task_info: Any, project_dir: str | Path = PROJECT_ROOT) -> str:
    root = Path(project_dir)
    profile = select_runtime_profile(task_info)
    overrides = {
        key: os.environ.get(key, "").strip()
        for key in ("ACPROF_NLP_TORCH_INDEX_URL", "ACPROF_NLP_TORCH_SPEC")
    } if not profile.requirements_lock else {}
    digest = hashlib.sha256(json.dumps({
        "schema_version": 1, "model_id": task_info.model_id,
        "model_revision": task_info.model_revision,
        "task": task_info.pipeline_tag, "backend": task_info.runtime_backend,
        "profile": profile.to_dict(),
        "build_overrides": overrides,
    }, sort_keys=True).encode())
    paths = sorted((root / "acprof").rglob("*.py"))
    paths += sorted((root / "dockerfiles").glob("*.Dockerfile"))
    if profile.requirements_lock:
        paths.append(root / profile.requirements_lock)
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_fingerprint(profile: RuntimeProfile, project_dir: str | Path = PROJECT_ROOT) -> str:
    root = Path(project_dir)
    digest = hashlib.sha256(json.dumps(profile.to_dict(), sort_keys=True).encode())
    digest.update((root / profile.requirements_lock).read_bytes())
    digest.update((root / "dockerfiles/runtime.Dockerfile").read_bytes())
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


def verified_image(task_info: Any, name: str, fingerprint: str):
    from acprof.host.docker_runtime import ImageInfo, _run

    identity = inspect_identity(name)
    if identity is None:
        return None
    labels = identity.get("labels") or {}
    if labels.get(FINGERPRINT_LABEL) != fingerprint:
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
        "build_fingerprint": fingerprint, "profile_id": profile.profile_id,
        "adapter": profile.adapter, "model_id": task_info.model_id,
        "model_revision": task_info.model_revision,
    }
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("镜像环境、适配器或模型 revision 与本次任务不匹配")
    if manifest.get("model_snapshot_revision") != task_info.model_revision:
        raise RuntimeError("镜像内实际 snapshot revision 与本次任务不匹配")
    return ImageInfo(tag=identity["image_id"], name=name, runtime_environment=manifest)


def prepare_runtime_image(task_info: Any, project_dir: str, *, reuse_existing: bool = False):
    from acprof.host.docker_runtime import _model_image_tag, build_image

    profile = select_runtime_profile(task_info)
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
    fingerprint = build_fingerprint(task_info, project_dir)
    name = _model_image_tag(task_info, project_dir)
    print(f"[runtime] 环境: {profile.profile_id}；适配器: {profile.adapter}", flush=True)
    if reuse_existing:
        image = verified_image(task_info, name, fingerprint)
        if image is not None:
            print(f"[build] 指纹核验通过，跳过构建并复用：{name}", flush=True)
            return image
        print(f"[build] 未找到本地模型镜像 {name}；将自动构建，旧版 :latest 不用于本次采集。", flush=True)
    return build_image(task_info, project_dir)


def build_runtime_image(task_info: Any, project_dir: str):
    from acprof.host.docker_runtime import (
        _build_legacy_image, _model_image_tag, _run,
    )
    from acprof.config import HF_MIRROR_ENDPOINT
    profile = select_runtime_profile(task_info)
    task_info.runtime_profile_id, task_info.model_adapter = profile.profile_id, profile.adapter
    if not re.fullmatch(r"[0-9a-f]{40}", task_info.model_revision or ""):
        raise RuntimeError("镜像构建要求固定 model revision；请通过 prepare_image 准备镜像")
    root = Path(project_dir)
    fingerprint = build_fingerprint(task_info, root)
    name = _model_image_tag(task_info, root)

    def build(dockerfile: str, target: str, args: dict[str, str]) -> None:
        command = ["docker", "build", "-f", str(root / "dockerfiles" / dockerfile)]
        for key, value in args.items():
            command += ["--build-arg", f"{key}={value}"]
        if (os.environ.get("HF_TOKEN") or "").strip():
            command += ["--secret", "id=hf_token,env=HF_TOKEN"]
        command += ["-t", target, str(root)]
        # Build output is streamed outside all measurement windows.
        result = _run(command, check=False, capture=False)
        if result.returncode:
            raise RuntimeError(f"Docker 构建失败: {dockerfile} (exit={result.returncode})")

    if profile.requirements_lock:
        runtime_tag = f"acprof-runtime-{profile.profile_id}:{runtime_fingerprint(profile, root)[:20]}"
        if inspect_identity(runtime_tag) is None:
            build("runtime.Dockerfile", runtime_tag, {
                "PYTHON_BASE_IMAGE": profile.python_base_image,
                "REQUIREMENTS_LOCK": profile.requirements_lock,
                "TORCH_INDEX_URL": profile.torch_index_url,
            })
        runtime_id = inspect_identity(runtime_tag)["image_id"]
        runtime_source = "acprof-build-source:" + runtime_id.split(":", 1)[1]
        _run(["docker", "tag", runtime_id, runtime_source])
        model_tag = name + "-weights"
        build("runtime-model.Dockerfile", model_tag, {
            "RUNTIME_IMAGE": runtime_source, "MODEL_ID": task_info.model_id,
            "MODEL_REVISION": task_info.model_revision, "HF_ENDPOINT": HF_MIRROR_ENDPOINT,
        })
    else:
        _build_legacy_image(task_info, project_dir)
        model_tag = name
    model_id = inspect_identity(model_tag)["image_id"]
    model_source = "acprof-build-source:" + model_id.split(":", 1)[1]
    _run(["docker", "tag", model_id, model_source])
    lock_hash = hashlib.sha256((root / profile.requirements_lock).read_bytes()).hexdigest() if profile.requirements_lock else ""
    if build_fingerprint(task_info, root) != fingerprint:
        raise RuntimeError("构建期间代码或依赖配置发生变化，请重新构建以固定版本")
    build("runtime-final.Dockerfile", name, {
        "MODEL_IMAGE": model_source, "RUNTIME_PROFILE": profile.profile_id,
        "MODEL_ADAPTER": profile.adapter, "BUILD_FINGERPRINT": fingerprint,
        "DEPENDENCY_LOCK_SHA256": lock_hash,
    })
    if build_fingerprint(task_info, root) != fingerprint:
        _run(["docker", "image", "rm", name], check=False)
        raise RuntimeError("构建期间代码发生变化，已撤销本次镜像标签；请重新构建")
    image = verified_image(task_info, name, fingerprint)
    if image is None:
        raise RuntimeError("构建后未找到模型镜像")
    return image

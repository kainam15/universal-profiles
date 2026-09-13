"""按依赖内容准备平台和环境缓存；主构建与离线验证共用此入口。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

from acprof.dependency_locks import (
    content_digest, package_versions, python_lock_text, require_exact_packages,
    system_lock_identity,
)
from acprof.runtime_profiles import (
    DependencyEnvironment, PlatformSpec, environment_identity, platform_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_LABEL = "org.acprof.platform-build-fingerprint"
ENVIRONMENT_LABEL = "org.acprof.environment-build-fingerprint"
BUILD_HELPERS = ("dockerfiles/environment_tools.py", "acprof/dependency_locks.py")


@dataclass(frozen=True)
class PreparedEnvironment:
    image_id: str
    name: str
    platform_image_id: str
    manifest: dict


def recipe_identity(root: Path, recipe: str) -> dict:
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in ("dockerfiles/" + recipe, *BUILD_HELPERS)}


def platform_fingerprint(platform: PlatformSpec, project_dir=PROJECT_ROOT) -> str:
    root = Path(project_dir)
    return _platform_fingerprint(platform_identity(platform, root), root)


def _platform_fingerprint(identity: dict, root: Path) -> str:
    return content_digest({"platform": identity,
                           "recipe": recipe_identity(root, "platform.Dockerfile")})


def runtime_fingerprint(environment: DependencyEnvironment, project_dir=PROJECT_ROOT) -> str:
    """构建声明指纹；实际环境镜像的缓存键另加入固定平台 image ID。"""
    root = Path(project_dir)
    identity = environment_identity(environment, root)
    return _runtime_fingerprint(identity, _platform_fingerprint(identity["platform"], root), root)


def _runtime_fingerprint(identity: dict, platform_key: str, root: Path) -> str:
    return content_digest({"environment": identity, "platform_build": platform_key,
                           "recipe": recipe_identity(root, "runtime.Dockerfile")})


def read_image_manifest(image_id: str, path: str) -> dict:
    from acprof.host.docker_runtime import _run
    result = _run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat", image_id, path], check=False)
    if result.returncode:
        raise RuntimeError(f"镜像缺少依赖清单：{image_id}: {path}")
    try:
        manifest = json.loads(result.stdout)
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not an object")
        return manifest
    except (ValueError, TypeError) as error:
        raise RuntimeError("镜像依赖清单无效") from error


def verify_manifest(manifest: dict, expected: dict, identity: dict) -> None:
    if manifest.get("schema_version") != 1 or any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("镜像平台或依赖环境身份不匹配；拒绝复用")
    system = identity.get("system", identity.get("platform", {}).get("system"))
    if manifest.get("system_lock_sha256") != content_digest(system_lock_identity(system)):
        raise RuntimeError("镜像系统依赖 lock 不匹配")
    require_exact_packages(package_versions(identity["packages"]), manifest.get("packages", {}))
    require_exact_packages(system["packages"], manifest.get("system_packages", {}), kind="system")


def verify_labels(image: dict, expected: dict, kind: str) -> None:
    required = {"org.acprof.image-kind": kind, "org.acprof.platform": expected["platform_id"],
                PLATFORM_LABEL: expected["platform_build_fingerprint"]}
    if kind != "platform":
        required.update({"org.acprof.environment": expected["environment_id"],
                         ENVIRONMENT_LABEL: expected["environment_build_fingerprint"]})
    if any((image.get("labels") or {}).get(key) != value for key, value in required.items()):
        raise RuntimeError("依赖缓存标签内容不匹配")


def require_image_source(source: str, image_id: str) -> None:
    from acprof.host.runtime_images import inspect_identity
    actual = inspect_identity(source)
    if actual is None or actual["image_id"] != image_id:
        raise RuntimeError("构建父镜像引用发生变化，拒绝发布缓存")


def checked_image(name: str, label: str, fingerprint: str, expected: dict, identity: dict, kind: str):
    from acprof.host.runtime_images import inspect_identity
    image = inspect_identity(name)
    if image is None:
        return None
    if (image.get("labels") or {}).get(label) != fingerprint:
        raise RuntimeError(f"依赖缓存标签内容不匹配：{name}")
    verify_labels(image, expected, kind)
    manifest = read_image_manifest(image["image_id"], f"/opt/acprof/{kind}-manifest.json")
    verify_manifest(manifest, expected, identity)
    return image["image_id"], manifest


def build_dependency(root: Path, recipe: str, name: str, arguments: dict, expected: dict,
                     identity: dict, current_fingerprint, original_fingerprint: str) -> None:
    from acprof.host.docker_runtime import _run
    from acprof.host.runtime_images import inspect_identity
    with tempfile.TemporaryDirectory(prefix="acprof-dependency-build-") as directory:
        context = Path(directory)
        for relative in ("dockerfiles/" + recipe, *BUILD_HELPERS):
            (context / Path(relative).name).write_bytes((root / relative).read_bytes())
        (context / "requirements.lock").write_text(python_lock_text(identity["packages"]))
        system = identity.get("system", identity.get("platform", {}).get("system"))
        (context / "system.lock").write_text(json.dumps(system, sort_keys=True) + "\n")
        (context / "expectation.json").write_text(json.dumps(expected, sort_keys=True) + "\n")
        iidfile = context / "image-id"
        command = ["docker", "build", "--platform", "linux/amd64", "--iidfile", str(iidfile),
                   "-f", str(context / recipe)]
        for key, value in arguments.items():
            command += ["--build-arg", f"{key}={value}"]
        command.append(str(context))
        if current_fingerprint() != original_fingerprint:
            raise RuntimeError("依赖构建输入发生变化，尚未构建镜像")
        result = _run(command, check=False, capture=False)
        if result.returncode:
            raise RuntimeError(f"Docker 构建失败: {recipe} (exit={result.returncode})")
        image_id = iidfile.read_text().strip()
        image = inspect_identity(image_id)
        if image is None or image["image_id"] != image_id:
            raise RuntimeError("构建没有返回不可变 image ID")
        if current_fingerprint() != original_fingerprint:
            raise RuntimeError("构建期间依赖或配方发生变化，未发布缓存标签")
        kind = "platform" if recipe == "platform.Dockerfile" else "environment"
        verify_labels(image, expected, kind)
        verify_manifest(read_image_manifest(image_id, f"/opt/acprof/{kind}-manifest.json"), expected, identity)
        _run(["docker", "tag", image_id, name])


def prepare_environment_image(environment: DependencyEnvironment, project_dir=PROJECT_ROOT) -> PreparedEnvironment:
    from acprof.host.docker_runtime import _run
    from acprof.host.runtime_images import inspect_identity
    root = Path(project_dir)
    identity = environment_identity(environment, root)
    platform = environment.platform
    platform_data = identity["platform"]
    # 构建键与输入目录必须来自同一份声明，避免读锁和计算键之间的变更被误标为新缓存。
    platform_key = _platform_fingerprint(platform_data, root)
    platform_name = f"acprof-platform-{platform.platform_id}:{platform_key[:20]}"
    expected_platform = {"platform_id": platform.platform_id, "platform_definition_id": content_digest(platform_data),
                         "platform_build_fingerprint": platform_key, "python_version": platform.python_version,
                         "python_base_image": platform.python_base_image, "architecture": platform.architecture}
    existing = checked_image(platform_name, PLATFORM_LABEL, platform_key, expected_platform, platform_data, "platform")
    if existing is None:
        build_dependency(root, "platform.Dockerfile", platform_name,
                         {"PYTHON_BASE_IMAGE": platform.python_base_image, "PLATFORM_ID": platform.platform_id,
                          "PLATFORM_BUILD_FINGERPRINT": platform_key}, expected_platform, platform_data,
                         lambda: platform_fingerprint(platform, root), platform_key)
        existing = checked_image(platform_name, PLATFORM_LABEL, platform_key, expected_platform, platform_data, "platform")
    if existing is None:
        raise RuntimeError("平台构建后未找到缓存镜像")
    platform_image_id = existing[0]
    spec_key = _runtime_fingerprint(identity, platform_key, root)
    if runtime_fingerprint(environment, root) != spec_key:
        raise RuntimeError("依赖构建输入发生变化，尚未构建环境镜像")
    environment_key = content_digest({"definition": spec_key, "platform_image_id": platform_image_id})
    name = f"acprof-runtime-env:{environment_key[:20]}"
    expected = {**expected_platform, "platform_image_id": platform_image_id,
                "environment_id": content_digest(identity), "environment_build_fingerprint": environment_key}
    existing = checked_image(name, ENVIRONMENT_LABEL, environment_key, expected, identity, "environment")
    if existing is None:
        source = "acprof-build-source:" + platform_image_id.split(":", 1)[1]
        _run(["docker", "tag", platform_image_id, source])
        def current_environment_fingerprint():
            require_image_source(source, platform_image_id)
            return runtime_fingerprint(environment, root)
        build_dependency(root, "runtime.Dockerfile", name,
                         {"PLATFORM_IMAGE": source, "ENVIRONMENT_ID": expected["environment_id"],
                          "ENVIRONMENT_BUILD_FINGERPRINT": environment_key}, expected, identity,
                         current_environment_fingerprint, spec_key)
        existing = checked_image(name, ENVIRONMENT_LABEL, environment_key, expected, identity, "environment")
    if existing is None:
        raise RuntimeError("依赖环境构建后未找到缓存镜像")
    if runtime_fingerprint(environment, root) != spec_key:
        raise RuntimeError("依赖构建输入发生变化，拒绝使用缓存")
    if inspect_identity(name)["image_id"] != existing[0]:
        raise RuntimeError("依赖缓存标签在核验期间发生变化")
    return PreparedEnvironment(existing[0], name, platform_image_id, existing[1])

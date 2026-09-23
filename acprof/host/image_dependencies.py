"""用镜像身份匹配依赖锁，描述逻辑构建层的包增量；不扫描或运行镜像。"""

from __future__ import annotations

from dataclasses import replace

from acprof.host.image_management import ImageInventory


def dependency_stage(rows: list[dict]) -> str:
    """核对最近的文件操作，拒绝继承已知标签后又安装包的自定义镜像。

    history 只在读取时检查，返回阶段标识；不保留可能含凭据的原始命令。
    """
    commands = []
    for row in rows:
        command = row["command"].split("#(nop)")[-1].strip().removesuffix(" # buildkit")
        if command.startswith("RUN ") and "/bin/sh -c " in command:
            command = "RUN " + command.partition("/bin/sh -c ")[2]
        command = " ".join(command.split())
        if command.partition(" ")[0] not in {
            "ARG", "ENV", "LABEL", "CMD", "ENTRYPOINT", "EXPOSE", "USER", "WORKDIR",
            "STOPSIGNAL", "HEALTHCHECK", "SHELL", "VOLUME", "ONBUILD",
        }:
            commands.append(command)
        if len(commands) == 4:
            break
    stages = {
        "base": ["RUN python /build/environment_tools.py platform",
                 "COPY requirements.lock expectation.json /opt/acprof/",
                 "RUN python /build/environment_tools.py system /opt/acprof/system.lock",
                 "COPY system.lock /opt/acprof/system.lock"],
        "runtime": ["RUN python /build/environment_tools.py environment",
                    "COPY requirements.lock expectation.json /opt/acprof/"],
        "weights": ['RUN if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; python /opt/acprof/download_model.py',
                    "COPY acprof/container/download_model.py acprof/container/model_files.py /opt/acprof/"],
        "model": ["RUN python -m acprof.container.runtime_manifest", "COPY acprof/ /app/acprof/"],
    }
    return next((kind for kind, expected in stages.items() if commands[:len(expected)] == expected), "")


def describe_dependencies(inventory: ImageInventory) -> ImageInventory:
    from acprof.dependency_locks import content_digest, package_versions
    from acprof.host.dependency_images import _platform_fingerprint
    from acprof.runtime_profiles import ENVIRONMENTS, PLATFORMS, environment_identity, platform_identity

    from acprof.installation import resource_root
    root = resource_root()
    platforms, environments = {}, {}
    for spec in PLATFORMS.values():
        try:
            identity = platform_identity(spec, root)
            platforms[(spec.platform_id, _platform_fingerprint(identity, root))] = identity
        except (OSError, ValueError, KeyError):
            continue
    for spec in ENVIRONMENTS.values():
        try:
            identity = environment_identity(spec, root)
            environments[(spec.platform.platform_id, content_digest(identity))] = identity
        except (OSError, ValueError, KeyError):
            continue

    indexed = {}
    for item in sorted(inventory.images, key=lambda image: len(image.ancestor_ids)):
        item = replace(item, python_dependencies=(), system_dependencies=(), dependency_source="unknown")
        platform = platforms.get((item.platform_id, item.platform_key))
        if item.dependency_stage == item.kind and item.parent_source not in {"ambiguous", "conflict"}:
            if item.kind == "base" and platform:
                # 系统完整包表含上游镜像已有包；artifacts 才是本层实际安装的制品。
                system = tuple(sorted((f'{entry["name"]}:{entry["architecture"]}', entry["version"])
                                      for entry in platform["system"]["artifacts"]))
                item = replace(item, python_dependencies=tuple(sorted(package_versions(platform["packages"]).items())),
                               system_dependencies=system, dependency_source="platform-lock")
            elif item.kind == "runtime" and platform:
                environment = environments.get((item.platform_id, item.environment_id))
                if environment and environment["platform"] == platform:
                    inherited = package_versions(platform["packages"])
                    delta = tuple(sorted((name, version) for name, version in package_versions(environment["packages"]).items()
                                         if inherited.get(name) != version))
                    item = replace(item, python_dependencies=delta, dependency_source="environment-lock")
            elif item.kind in {"weights", "model"}:
                parent = indexed.get(item.parent_id)
                expected = "runtime" if item.kind == "weights" else "weights"
                if (parent and parent.kind == expected and parent.dependency_source != "unknown"
                        and item.environment_id and item.environment_key and item.platform_key
                        and (item.environment_id, item.environment_key, item.platform_key)
                        == (parent.environment_id, parent.environment_key, parent.platform_key)):
                    item = replace(item, dependency_source="inherited")
        indexed[item.image_id] = item
    return replace(inventory, images=tuple(indexed[item.image_id] for item in inventory.images))

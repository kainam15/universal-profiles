"""按实际 image ID 管理 Docker 镜像；所有查询由用户操作触发。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import re
import subprocess


class ImageManagementError(RuntimeError):
    """Docker 不可访问或待删除清单已经变化。"""

    def __init__(self, reason: str, detail: str = ""):
        self.reason, self.detail = reason, detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class DockerConnection:
    arguments: tuple[str, ...]
    name: str


@dataclass(frozen=True)
class ManagedImage:
    image_id: str
    tags: tuple[str, ...]
    size_bytes: int
    created: str
    kind: str
    model_id: str = ""
    containers: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()
    acprof: bool = False
    platform_id: str = ""
    environment_id: str = ""
    profiles: tuple[str, ...] = ()
    platform_key: str = ""
    environment_key: str = ""
    model_files_key: str = ""
    parent_id: str = ""
    parent_source: str = "unknown"
    ancestor_ids: tuple[str, ...] = ()
    descendant_ids: tuple[str, ...] = ()
    inherited_bytes: int | None = None
    added_bytes: int | None = None
    shared_bytes: int | None = None
    unique_bytes: int | None = None
    space_source: str = ""
    layer_sizes: tuple[int | None, ...] = ()

    @property
    def name(self) -> str:
        return next((tag for tag in self.tags if not tag.startswith("acprof-build-source:")),
                    self.tags[0] if self.tags else self.image_id[7:19])

    @property
    def model_key(self) -> str:
        return self.model_id.casefold().replace("/", "--")

    @property
    def display_name(self) -> str:
        if self.kind == "runtime" and self.environment_id:
            label = " / ".join(self.profiles) or "env-" + self.environment_id[:12]
            if self.platform_id and self.platform_id not in label:
                label += " · " + self.platform_id
            return label
        if self.model_id:
            return self.model_id
        return self.name.rsplit(":", 1)[0] if self.tags else self.name


@dataclass(frozen=True)
class ImageLayer:
    chain_id: str
    diff_id: str
    size_bytes: int | None
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class ImageInventory:
    connection: DockerConnection
    daemon_id: str
    images: tuple[ManagedImage, ...]
    layers: tuple[ImageLayer, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageRemoval:
    image_id: str
    success: bool
    detail: str


def _run(arguments: tuple[str, ...] | list[str], *, timeout: int = 30) -> str:
    # 不使用输出命令的 host._run，避免破坏 TUI；Docker CLI 沿用其凭据和连接配置。
    try:
        result = subprocess.run(
            ["docker", *arguments], capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ImageManagementError("Docker 操作超时，请刷新后检查状态") from exc
    except OSError as exc:
        raise ImageManagementError("无法执行 Docker", str(exc)) from exc
    if result.returncode:
        raise ImageManagementError("Docker 操作失败", (result.stderr or result.stdout).strip())
    return result.stdout.strip()


def _connection() -> DockerConnection:
    # DOCKER_CONTEXT 优先于 DOCKER_HOST；显式 CLI 参数固定此次列表及删除的目标。
    context = os.environ.get("DOCKER_CONTEXT", "").strip()
    host = os.environ.get("DOCKER_HOST", "").strip()
    if not context and host:
        return DockerConnection(("--host", host), host)
    context = context or _run(("context", "show"))
    if not context:
        raise ImageManagementError("无法确定 Docker 环境")
    return DockerConnection(("--context", context), context)


def _inspect(connection: DockerConnection, resource: str, references: list[str]) -> list[dict]:
    records: list[dict] = []
    for start in range(0, len(references), 100):
        try:
            batch = json.loads(_run((*connection.arguments, resource, "inspect", *references[start:start + 100])))
        except json.JSONDecodeError as exc:
            raise ImageManagementError("Docker 返回了无效的镜像信息") from exc
        if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
            raise ImageManagementError("Docker 返回了无效的镜像信息")
        records.extend(batch)
    return records


def _kind(tags: tuple[str, ...], labels: dict) -> str:
    repositories = [tag.rsplit(":", 1)[0] for tag in tags]
    explicit_kind = {"platform": "base", "environment": "runtime", "weights": "weights", "model": "model"}.get(
        labels.get("org.acprof.image-kind"))
    if explicit_kind:
        return explicit_kind
    if any(name.startswith("acprof-platform-") for name in repositories):
        return "base"
    if any(name.startswith("acprof-runtime-") for name in repositories):
        return "runtime"
    if "acprof-base" in repositories:
        return "base"
    if any(name.startswith("acprof-weights-") for name in repositories):
        return "weights"
    if any("dependency-check" in name or "reuse-base" in name or
           re.match(r"acprof-(massif|nsys|ncu)-", name) for name in repositories):
        return "debug"
    if "org.acprof.build-fingerprint" in labels:
        return "model"
    if "org.acprof.model-files-key" in labels:
        return "weights"
    if any(re.match(r"acprof-(audio|cv|nlp|diffusion|multimodal|structured|timeseries)-", name)
           for name in repositories):
        return "model"
    return "other" if tags else "untagged"


def list_images(connection: DockerConnection | None = None, *, include_space: bool = True) -> ImageInventory:
    """按 ID 去重；大小沿用 Docker 的完整 Size，不累计共享层为独占空间。"""
    connection = connection or _connection()
    daemon_id = _run((*connection.arguments, "info", "--format", "{{.ID}}"))
    if not daemon_id:
        raise ImageManagementError("无法确定 Docker 环境")
    ids = list(dict.fromkeys(_run((*connection.arguments, "image", "ls", "--all", "--quiet", "--no-trunc")).splitlines()))
    if any(not re.fullmatch(r"sha256:[0-9a-f]{64}", key) for key in ids):
        raise ImageManagementError("Docker 返回了无效的镜像信息")
    records = _inspect(connection, "image", ids)
    container_ids = _run((*connection.arguments, "container", "ls", "--all", "--quiet", "--no-trunc")).splitlines()
    containers: dict[str, list[str]] = {}
    for row in _inspect(connection, "container", container_ids):
        try:
            containers.setdefault(row["Image"], []).append(
                f'{row["Name"].lstrip("/")} ({row["State"]["Status"]})')
        except (KeyError, TypeError, AttributeError) as exc:
            raise ImageManagementError("无法核验容器引用") from exc
    images: list[ManagedImage] = []
    try:
        for row in records:
            image_id, size = row["Id"], row["Size"]
            raw_tags = row.get("RepoTags") or []
            if (image_id not in ids or isinstance(size, bool) or not isinstance(size, int) or size < 0
                    or not isinstance(raw_tags, list)
                    or any(not isinstance(tag, str) or not tag or tag.startswith("-") or
                           any(char.isspace() for char in tag) for tag in raw_tags)):
                raise ValueError("invalid image identity, tags or size")
            tags = tuple(sorted(set(raw_tags)))
            config = row.get("Config") or {}
            labels = config.get("Labels") or {}
            # 只保留 MODEL_ID，不向界面或日志传递其它环境变量及任意 label。
            model = next((value.partition("=")[2] for value in config.get("Env") or []
                          if value.startswith("MODEL_ID=")), "")
            kind = _kind(tags, labels)
            env = dict(value.split("=", 1) for value in config.get("Env") or [] if "=" in value)
            parent = (env.get("ACPROF_MODEL_IMAGE_ID", "") if kind == "model" else "") or row.get("Parent") or ""
            if not model and kind in {"model", "weights"}:
                for tag in tags:
                    match = re.match(r"acprof-(?:weights-)?(?:audio|cv|nlp|diffusion|multimodal|structured|timeseries)-(.+):[^:]+$", tag)
                    if match:
                        model = match[1]
                        break
            images.append(ManagedImage(
                image_id, tags, size, str(row.get("Created") or ""), kind, model,
                tuple(sorted(containers.get(image_id, []))), tuple(row.get("RootFS", {}).get("Layers", [])),
                any(tag.startswith("acprof-") for tag in tags) or any(key.startswith("org.acprof.") for key in labels),
                platform_id=str(labels.get("org.acprof.platform") or ""),
                environment_id=str(labels.get("org.acprof.environment") or ""),
                profiles=(str(labels["org.acprof.runtime-profile"]),) if labels.get("org.acprof.runtime-profile") else (),
                platform_key=str(labels.get("org.acprof.platform-build-fingerprint") or ""),
                environment_key=str(labels.get("org.acprof.environment-build-fingerprint") or ""),
                model_files_key=str(labels.get("org.acprof.model-files-key") or ""),
                parent_id=parent,
                parent_source="recorded" if parent else "unknown",
            ))
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ImageManagementError("Docker 返回了无效的镜像信息", str(exc)) from exc
    if set(ids) != {item.image_id for item in images}:
        raise ImageManagementError("Docker 镜像清单不完整，请刷新")
    inventory = ImageInventory(connection, daemon_id, tuple(sorted(images, key=lambda item: item.name)))
    if not include_space:
        return inventory
    # 仅手动刷新需要完整空间信息；删除前的身份核验不重复扫描 history。
    from acprof.host.image_graph import describe_inventory
    inventory = _read_space(inventory)
    return describe_inventory(inventory)


def _size_bytes(value: object) -> int | None:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(B|kB|MB|GB|TB)", str(value))
    if not match:
        return None
    return round(float(match[1]) * {"B": 1, "kB": 1000, "MB": 10**6, "GB": 10**9, "TB": 10**12}[match[2]])


def _read_space(inventory: ImageInventory) -> ImageInventory:
    usage = {}
    warnings = []
    try:
        rows = json.loads(_run((*inventory.connection.arguments, "system", "df", "-v", "--format", "{{json .Images}}")))
        if not isinstance(rows, list):
            raise ValueError("invalid disk usage")
        usage = {row["ID"]: row for row in rows}
    except (ImageManagementError, ValueError, KeyError, TypeError):
        warnings.append("Docker 空间统计不可用；共享大小由可核验的层计算。")
    images = []
    for item in inventory.images:
        sizes: tuple[int | None, ...] = (None,) * len(item.layers)
        if item.layers:
            try:
                history = _run((*inventory.connection.arguments, "image", "history", "--no-trunc",
                                "--human=false", "--format", '{"size":{{.Size}},"command":{{json .CreatedBy}}}', item.image_id))
                rows = [json.loads(value) for value in history.splitlines()]
                nonempty = []
                for row in reversed(rows):
                    size = row["size"]
                    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                        raise ValueError("invalid history size")
                    command = row["command"].split("#(nop)")[-1].strip().upper()
                    metadata = re.match(r"^(ARG|ENV|LABEL|CMD|ENTRYPOINT|EXPOSE|USER|STOPSIGNAL|HEALTHCHECK|SHELL|VOLUME|ONBUILD)\b", command)
                    if size or not metadata:
                        nonempty.append(size)
                # history 不提供 empty_layer。剔除已知元数据指令后，仅在层数/总量均吻合时
                # 映射；WORKDIR/RUN 等真实零字节层保留，模糊历史仍显示未知。
                if len(nonempty) == len(item.layers) and sum(nonempty) == item.size_bytes:
                    sizes = nonempty
            except (ImageManagementError, ValueError, KeyError, TypeError, AttributeError):
                pass
        row = usage.get(item.image_id, {})
        shared, unique = _size_bytes(row.get("SharedSize")), _size_bytes(row.get("UniqueSize"))
        images.append(replace(item, layer_sizes=tuple(sizes), shared_bytes=shared, unique_bytes=unique,
                              space_source="docker-df" if shared is not None and unique is not None else ""))
    if any(any(size is None for size in item.layer_sizes) for item in images):
        warnings.append("部分层大小无法核验，显示未知；不会把缺失值当作零。")
    return replace(inventory, images=tuple(images), warnings=tuple(warnings))


def delete_images(inventory: ImageInventory, image_ids: tuple[str, ...]) -> tuple[ImageRemoval, ...]:
    """复核整个清单后删除全部选定标签；不强制、不 prune 未选择的父镜像。"""
    previous = {item.image_id: item for item in inventory.images}
    selected = set(image_ids)
    if not selected or not selected <= previous.keys():
        raise ImageManagementError("请选择列表中的镜像")
    current = list_images(inventory.connection, include_space=False)
    if current.daemon_id != inventory.daemon_id:
        raise ImageManagementError("Docker 环境已改变，请刷新后重新选择")
    indexed = {item.image_id: item for item in current.images}
    for image_id in selected:
        item = indexed.get(image_id)
        if item is None or item.tags != previous[image_id].tags:
            raise ImageManagementError("镜像或标签已改变，请刷新后重新选择", previous[image_id].name)
        if item.containers:
            raise ImageManagementError("镜像仍被容器引用，请先单独处理容器", ", ".join(item.containers))
    outcomes = []
    # 较深的文件层先处理，使同一次选择中的最终镜像先于权重/环境镜像删除。
    for item in sorted((indexed[key] for key in selected), key=lambda item: (-len(item.layers), item.name)):
        try:
            if _run((*inventory.connection.arguments, "info", "--format", "{{.ID}}")) != inventory.daemon_id:
                raise ImageManagementError("Docker 环境已改变，请刷新后重新选择")
            # 批次执行期间也复核引用，防止另一个进程移动标签后删到其它镜像。
            references = list(item.tags or (item.image_id,))
            checked = _inspect(inventory.connection, "image", references)
            if len(checked) != len(references) or any(row.get("Id") != item.image_id for row in checked):
                raise ImageManagementError("镜像或标签已改变，请刷新后重新选择", item.name)
            detail = _run((*inventory.connection.arguments, "image", "rm", "--no-prune", *references), timeout=60)
            outcomes.append(ImageRemoval(item.image_id, True, detail))
        except ImageManagementError as exc:
            outcomes.append(ImageRemoval(item.image_id, False, str(exc)))
    return tuple(outcomes)

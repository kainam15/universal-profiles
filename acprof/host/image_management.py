"""按实际 image ID 管理 Docker 镜像；所有查询由用户操作触发。"""

from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def name(self) -> str:
        return next((tag for tag in self.tags if not tag.startswith("acprof-build-source:")),
                    self.tags[0] if self.tags else self.image_id[7:19])

    @property
    def model_key(self) -> str:
        return self.model_id.casefold().replace("/", "--")


@dataclass(frozen=True)
class ImageInventory:
    connection: DockerConnection
    daemon_id: str
    images: tuple[ManagedImage, ...]


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


def list_images(connection: DockerConnection | None = None) -> ImageInventory:
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
            ))
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ImageManagementError("Docker 返回了无效的镜像信息", str(exc)) from exc
    if set(ids) != {item.image_id for item in images}:
        raise ImageManagementError("Docker 镜像清单不完整，请刷新")
    return ImageInventory(connection, daemon_id, tuple(sorted(images, key=lambda item: item.name)))


def delete_images(inventory: ImageInventory, image_ids: tuple[str, ...]) -> tuple[ImageRemoval, ...]:
    """复核整个清单后删除全部选定标签；不强制、不 prune 未选择的父镜像。"""
    previous = {item.image_id: item for item in inventory.images}
    selected = set(image_ids)
    if not selected or not selected <= previous.keys():
        raise ImageManagementError("请选择列表中的镜像")
    current = list_images(inventory.connection)
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

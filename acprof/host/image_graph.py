"""从已读取的镜像元数据和层链解析关系；不访问 Docker 或启动容器。"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from acprof.host.image_management import ImageInventory, ImageLayer, ManagedImage
from acprof.installation import resource_root


def _environment_names() -> dict[str, tuple[str, ...]]:
    from acprof.dependency_locks import content_digest
    from acprof.runtime_profiles import ENVIRONMENTS, environment_identity

    names: dict[str, list[str]] = {}
    for name, environment in ENVIRONMENTS.items():
        try:
            identity = content_digest(environment_identity(environment, resource_root()))
        except (OSError, ValueError, KeyError):
            continue  # 历史或不完整 checkout 无法对上锁时保留环境摘要，不能猜 profile。
        names.setdefault(identity, []).append(name)
    return {identity: tuple(sorted(values)) for identity, values in names.items()}


def _prefix(parent: ManagedImage, child: ManagedImage, *, strict: bool = True) -> bool:
    return (bool(parent.layers) and parent.image_id != child.image_id
            and (len(parent.layers) < len(child.layers) if strict else len(parent.layers) <= len(child.layers))
            and child.layers[:len(parent.layers)] == parent.layers)


def _parent(item: ManagedImage, indexed: dict[str, ManagedImage]) -> tuple[str, str]:
    if item.parent_id:
        parent = indexed.get(item.parent_id)
        if parent is None:
            return item.parent_id, "missing"
        return (parent.image_id, "recorded") if _prefix(parent, item, strict=False) else ("", "conflict")
    kind, attribute = {"runtime": ("base", "platform_key"), "weights": ("runtime", "environment_key"),
                       "model": ("weights", "model_files_key")}.get(item.kind, ("", ""))
    if attribute and getattr(item, attribute):
        candidates = [parent for parent in indexed.values() if parent.kind == kind
                      and getattr(parent, attribute) == getattr(item, attribute)
                      and _prefix(parent, item, strict=False)]
        if len(candidates) == 1:
            return candidates[0].image_id, "metadata"
        if len(candidates) > 1:
            return "", "ambiguous"
    candidates = [parent for parent in indexed.values() if _prefix(parent, item)]
    if candidates:
        length = max(len(parent.layers) for parent in candidates)
        nearest = [parent for parent in candidates if len(parent.layers) == length]
        if len(nearest) == 1:
            return nearest[0].image_id, "layer-prefix"
        return "", "ambiguous"
    return "", "unknown"


def describe_inventory(inventory: ImageInventory) -> ImageInventory:
    names = _environment_names() if any(item.environment_id for item in inventory.images) else {}
    indexed = {item.image_id: item for item in inventory.images}
    parents = {key: _parent(item, indexed) for key, item in indexed.items()}
    # 不信任外部镜像的 Parent/label：拒绝环，零文件层的镜像也不会导致无限递归。
    for key in parents:
        path = {key}
        parent = parents[key][0]
        while parent in parents:
            if parent in path:
                parents[key] = ("", "conflict")
                break
            path.add(parent)
            parent = parents[parent][0]
    ancestors = {}
    for key in indexed:
        path = []
        parent = parents[key][0]
        while parent in indexed:
            path.append(parent)
            parent = parents[parent][0]
        ancestors[key] = tuple(reversed(path))

    refs: dict[str, set[str]] = {}
    sizes: dict[str, set[int]] = {}
    diffs = {}
    chains: dict[str, list[str]] = {}
    for item in inventory.images:
        chain = ""
        chains[item.image_id] = []
        for index, diff in enumerate(item.layers):
            chain = "sha256:" + sha256(f"{chain} {diff}".encode()).hexdigest() if chain else diff
            chains[item.image_id].append(chain)
            refs.setdefault(chain, set()).add(item.image_id)
            diffs[chain] = diff
            size = item.layer_sizes[index] if index < len(item.layer_sizes) else None
            if size is not None:
                sizes.setdefault(chain, set()).add(size)
    layers = tuple(ImageLayer(chain, diffs[chain], next(iter(sizes[chain])) if len(sizes.get(chain, ())) == 1 else None,
                              tuple(sorted(ids))) for chain, ids in refs.items())
    layer_index = {layer.chain_id: layer for layer in layers}
    images = []
    for item in inventory.images:
        parent_id, source = parents[item.image_id]
        parent = indexed.get(parent_id)
        inherited = parent.size_bytes if parent and parent.size_bytes <= item.size_bytes else None
        shared, unique, space_source = item.shared_bytes, item.unique_bytes, item.space_source
        own_layers = [layer_index[chain] for chain in chains[item.image_id]]
        if item.layers and all(layer.size_bytes is not None for layer in own_layers):
            shared = sum(layer.size_bytes for layer in own_layers if len(layer.image_ids) > 1)
            unique = sum(layer.size_bytes for layer in own_layers if len(layer.image_ids) == 1)
            space_source = "layers"
        elif item.layers and all(len(layer.image_ids) > 1 for layer in own_layers):
            shared, unique, space_source = item.size_bytes, 0, "layers"
        descendants = tuple(sorted((key for key in indexed if item.image_id in ancestors[key]),
                                   key=lambda key: (len(ancestors[key]), indexed[key].name)))
        images.append(replace(item, parent_id=parent_id, parent_source=source, inherited_bytes=inherited,
                              added_bytes=item.size_bytes - inherited if inherited is not None else None,
                              ancestor_ids=ancestors[item.image_id], descendant_ids=descendants,
                              profiles=names.get(item.environment_id, item.profiles) if item.kind == "runtime" else item.profiles,
                              shared_bytes=shared, unique_bytes=unique, space_source=space_source))
    return replace(inventory, images=tuple(images), layers=layers)


def reclaimable_image_bytes(inventory: ImageInventory, image_ids: tuple[str, ...]) -> int | None:
    """选中集合移除后无人引用的层总量；缓存/存储驱动未知，所以仅是镜像层估算上限。"""
    selected = set(image_ids)
    items = [item for item in inventory.images if item.image_id in selected]
    if not items or any(item.containers for item in items):
        return 0
    layers = [layer for layer in inventory.layers if set(layer.image_ids) <= selected]
    if all(item.layers for item in items) and all(layer.size_bytes is not None for layer in layers):
        return sum(layer.size_bytes for layer in layers)
    if len(items) == 1:
        return items[0].unique_bytes
    return None

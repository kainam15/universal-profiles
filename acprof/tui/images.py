"""镜像列表的筛选、文案和删除确认；不执行 Docker 命令。"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from rich.text import Text
from textual.widgets import Button, DataTable, Static, Tree

from acprof.host.image_graph import reclaimable_image_bytes
from acprof.host.image_management import ImageInventory, ImageLayer, ImageManagementError, ManagedImage
from acprof.tui.i18n import join_messages, message
from acprof.tui.views import ConfirmActionScreen


IMAGE_KINDS = {
    "base": "公共基础", "runtime": "运行依赖", "weights": "模型文件",
    "model": "推理服务", "debug": "调试镜像", "other": "其它镜像", "untagged": "无标签",
}
IMAGE_HINT = "点击刷新读取 Docker；←→ 折叠/展开，空格勾选，列表表头可排序。"


class ImageTable(DataTable):
    BINDINGS = [Binding("space", "select_cursor", "勾选镜像", show=False)]


class ImageTree(Tree[ManagedImage]):
    BINDINGS = [Binding("space", "select_cursor", show=False),
                Binding("left", "collapse_branch", show=False),
                Binding("right", "expand_branch", show=False)]

    def render_label(self, node, base_style, style):
        label = super().render_label(node, base_style, style)
        # 原生 Tree 的叶节点没有展开箭头；保留同宽空白，避免数值列左移两格。
        return label if node.allow_expand else Text("  ") + label

    def action_collapse_branch(self) -> None:
        node = self.cursor_node
        if node and node.is_expanded and node.children:
            node.collapse()
        elif node and node.parent is not self.root:
            self.move_cursor(node.parent)

    def action_expand_branch(self) -> None:
        node = self.cursor_node
        if node and node.children:
            if node.is_expanded:
                self.move_cursor(node.children[0])
            else:
                node.expand()


class ImageDeleteScreen(ConfirmActionScreen):
    """长标签清单可独立滚动，确认按钮在小终端仍保持可见。"""

    CSS = ConfirmActionScreen.CSS + """
    ImageDeleteScreen { align: center middle; }
    ImageDeleteScreen #confirm-dialog { height: 85%; max-height: 40; }
    #image-confirm-content { height: 1fr; margin-bottom: 1; }
    #image-confirm-text { height: auto; }
    """

    def compose(self) -> ComposeResult:
        tr = self.app.tr
        with Vertical(id="confirm-dialog"):
            yield Static(tr(self.dialog_title), id="confirm-title", markup=False)
            with VerticalScroll(id="image-confirm-content"):
                yield Static(tr(self.message), id="image-confirm-text", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button(tr("取消"), id="confirm-no")
                yield Button(tr(self.confirm_label), id="confirm-yes", variant="error")


def format_image_size(value: int | None) -> str:
    if value is None:
        return "?"
    for unit, scale in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6), ("kB", 1000)):
        if value >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{value} B"


def filtered_images(inventory: ImageInventory, query: str, scope: str) -> tuple[ManagedImage, ...]:
    terms = query.casefold().replace("/", "--").split()
    items = []
    for item in inventory.images:
        if scope == "acprof" and not item.acprof:
            continue
        if scope == "models" and (not item.acprof or not item.model_key):
            continue
        if scope == "runtime" and item.kind not in {"base", "runtime"}:
            continue
        if scope == "untagged" and item.tags:
            continue
        if scope in {"cpu", "cu124", "cu128"} and item.platform_id != scope:
            continue
        if scope == "base" and item.kind != "base":
            continue
        haystack = " ".join((*item.tags, item.image_id, item.model_id, item.display_name,
                             item.environment_id, item.platform_id, *item.profiles)).casefold().replace("/", "--")
        if all(term in haystack for term in terms):
            items.append(item)
    return tuple(items)


def image_detail(item: ManagedImage, inventory: ImageInventory) -> str:
    indexed = {image.image_id: image for image in inventory.images}
    def path_name(image):
        name = image.display_name
        if image.model_id:
            name = join_messages(" · ", (message(IMAGE_KINDS[image.kind]), name))
        return join_messages("", (name, " ≈" if image.parent_source == "layer-prefix" else ""))
    path = [path_name(indexed[key]) for key in item.ancestor_ids if key in indexed]
    if item.parent_source == "missing":
        path.append(item.parent_id[:19] + " (?)")
    path.append(path_name(item))
    sources = {"recorded": "构建记录", "metadata": "构建指纹与层前缀核验", "layer-prefix": "层前缀推断，未确认 FROM",
               "missing": "父镜像不在本地", "ambiguous": "存在多个候选父镜像", "conflict": "父镜像记录与层链冲突",
               "unknown": "本地父镜像未知"}
    parts = [
        message("继承路径：{0}", join_messages(" › ", path)),
        message("完整大小：{0} · 继承：{1} · 本镜像新增：{2}", format_image_size(item.size_bytes),
                format_image_size(item.inherited_bytes), format_image_size(item.added_bytes)),
        message("删除预计释放：{0}", reclaimable_text(inventory, (item.image_id,))),
    ]
    if item.inherited_bytes is not None and item.size_bytes:
        inherited_cells = min(30, round(30 * item.inherited_bytes / item.size_bytes))
        parts.append(message("空间构成：{0}  █ 继承 / ░ 新增", "█" * inherited_cells + "░" * (30 - inherited_cells)))
    parts.extend((
        message("其它镜像共享：{0} · 仅当前镜像使用：{1}", format_image_size(item.shared_bytes), format_image_size(item.unique_bytes)),
        message("父镜像依据：{0} · 后代：{1}", message(sources[item.parent_source]), len(item.descendant_ids)),
        message("空间来源：{0}", message({"layers": "层链与已核验 history 字节数", "docker-df": "Docker df 近似值",
                                         "": "未知"}[item.space_source])),
        message("? 表示未知；释放范围受构建缓存与存储驱动影响。"),
        message("镜像 ID：{0}", item.image_id),
        message("类型：{0} · 完整大小：{1}（{2} bytes）", message(IMAGE_KINDS[item.kind]),
                format_image_size(item.size_bytes), item.size_bytes),
        message("模型：{0} · 创建时间：{1}", item.model_id or "—", item.created or "—"),
        message("容器引用：{0}", ", ".join(item.containers) if item.containers else message("无")),
        message("全部标签：\n{0}", "\n".join(item.tags) if item.tags else message("无标签")),
    ))
    parts.extend(message(warning) for warning in inventory.warnings)
    return join_messages("\n", parts)


def reclaimable_text(inventory: ImageInventory, image_ids: tuple[str, ...]) -> str:
    value = reclaimable_image_bytes(inventory, image_ids)
    return message("未知") if value is None else "0 B" if value == 0 else message("0 B ～ 约 {0}", format_image_size(value))


def layer_detail(layer: ImageLayer, inventory: ImageInventory) -> str:
    indexed = {item.image_id: item for item in inventory.images}
    return join_messages("\n", (
        message("层内容 Diff ID：{0}", layer.diff_id),
        message("层链 Chain ID：{0}", layer.chain_id),
        message("层大小：{0} · 引用镜像：{1}", format_image_size(layer.size_bytes), len(layer.image_ids)),
        message("相同 Diff ID 的不同父层链分开统计；引用包含筛选外镜像，按 image ID 去重。"),
        message("使用此层的镜像：\n{0}", "\n".join(
            f"{indexed[key].display_name} · {key[7:19]}\n  {indexed[key].name}" for key in layer.image_ids)),
    ))


def render_image_tree(tree: ImageTree, inventory: ImageInventory | None, visible: tuple[ManagedImage, ...],
                      selected: set[str], current_id: str, tr, width: int) -> None:
    previous = {}
    def collect(node):
        for child in node.children:
            previous[child.data.image_id] = child.is_expanded
            collect(child)
    collect(tree.root)
    tree.clear()
    tree.show_root = False
    if inventory is None:
        return
    indexed = {item.image_id: item for item in inventory.images}
    matches = {item.image_id for item in visible}
    shown = matches | {key for item in visible for key in item.ancestor_ids}
    nodes = {}
    for item in sorted((indexed[key] for key in shown), key=lambda item: (len(item.ancestor_ids), item.display_name, item.image_id)):
        parent = nodes.get(item.parent_id, tree.root)
        marker = "✓" if item.image_id in selected else "—" if item.containers else "□"
        # 同名不同版本保留短 ID；原始 repo/tag 始终在详情和列表中可见。
        logical = f"{tr(IMAGE_KINDS[item.kind])} · {item.display_name}" if item.model_id else item.display_name
        evidence = " ≈" if item.parent_source == "layer-prefix" else " ?" if item.parent_source in {"ambiguous", "missing", "conflict"} else ""
        name = f"{marker} {logical}{evidence} · {item.image_id[7:13]}"
        name_width = max(15, width - 28 - len(item.ancestor_ids) * tree.guide_depth)
        label = Text(name, style="dim" if item.image_id not in matches else "")
        label.truncate(name_width, overflow="ellipsis", pad=True)
        label.append(f" {format_image_size(item.size_bytes):>10} {format_image_size(item.added_bytes):>10} {len(item.containers):>3}")
        nodes[item.image_id] = parent.add(label, data=item, expand=previous.get(item.image_id, True))
    for node in nodes.values():
        node.allow_expand = bool(node.children)
    target = nodes.get(current_id) or next(iter(nodes.values()), None)
    while target and target.parent is not tree.root and not target.parent.is_expanded:
        target = target.parent
    # 新节点要等 Textual 完成行布局，才有可用于 move_cursor 的行号。
    tree.call_after_refresh(tree.move_cursor, target)


def deletion_message(inventory: ImageInventory, image_ids: tuple[str, ...]) -> str:
    items = [item for item in inventory.images if item.image_id in image_ids]
    parts = [
        message("Docker 环境：{0} · 共 {1} 个镜像", inventory.connection.name, len(items)),
        message("删除全部标签后无法使用原镜像续采或补采；已保存的实验文件保留。"),
        message("删除预计释放：{0}", reclaimable_text(inventory, image_ids)),
        message("按所选集合的层引用去重估算；构建缓存仍可能保留数据。"),
    ]
    for item in items:
        parts.append(message("{0}\n镜像 ID：{1}\n完整大小：{2}",
                             "\n".join(item.tags) if item.tags else message("无标签"),
                             item.image_id, format_image_size(item.size_bytes)))
    parts.append(message("共享层和构建缓存可能继续占用空间；镜像大小不能相加为可释放空间。"))
    return join_messages("\n\n", parts)


def image_error(error: BaseException) -> str:
    if isinstance(error, ImageManagementError):
        return message("{0}：{1}", message(error.reason), error.detail) if error.detail else message(error.reason)
    return str(error)

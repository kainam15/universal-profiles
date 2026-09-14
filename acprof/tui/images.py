"""镜像列表的筛选、文案和删除确认；不执行 Docker 命令。"""

from __future__ import annotations

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import var
from textual.strip import Strip
from rich.rule import Rule
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.widgets import Button, Collapsible, DataTable, Static, Tree

from acprof.host.image_graph import reclaimable_image_bytes
from acprof.host.image_management import ImageInventory, ImageLayer, ImageManagementError, ManagedImage
from acprof.tui.i18n import join_messages, message
from acprof.tui.table import ResizableDataTable
from acprof.tui.views import COLLAPSED_SYMBOL, EXPANDED_SYMBOL, ConfirmActionScreen


IMAGE_KINDS = {
    "base": "公共基础", "runtime": "运行依赖", "weights": "模型文件",
    "model": "推理服务", "debug": "调试镜像", "other": "其它镜像", "untagged": "无标签",
}
IMAGE_HINT = "清单自动刷新；点行查看，点 □/☑ 勾选；空格切换，←→ 展开/折叠。"
IMAGE_PLATFORMS = {"cpu": "CPU", "cu124": "CUDA 12.4", "cu128": "CUDA 12.8"}
# 已知环境别名对应的版本来自依赖锁；不猜测其它名称中的数字含义。
IMAGE_RUNTIME_NAMES = {
    "moss-transformers560": "moss-transformers5.6.0",
    "multimodal-transformers4576": "multimodal-transformers4.57.6",
}


class ImageTable(ResizableDataTable):
    BINDINGS = [Binding("space", "select_cursor", "勾选镜像", show=False)]

    async def on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        if self.disabled or event.button != 1 or self._consume_resize_click(event):
            return
        # 保留原生表头、光标和滚动处理，禁止重复点当前行发出勾选事件。
        with self.prevent(DataTable.RowSelected):
            await super()._on_click(event)
        meta = event.style.meta
        if meta.get("column") == 0 and self.is_valid_row_index(meta.get("row", -1)) and not meta.get("out_of_bounds"):
            self.action_select_cursor()


class ImageTreeHeader(ResizableDataTable):
    """树表头复用相同拖动交互，并与树的内容宽度和横向滚动同步。"""

    tree: ImageTree | None = None
    minimum_name_width = 6

    def on_mount(self) -> None:
        self.tree = self.parent.query_one(ImageTree)
        self.tree.column_header = self
        self.set_columns(self.size.width)

    def minimum_column_width(self, key: str | None) -> int:
        return self.minimum_name_width if key == "name" else 1

    def set_columns(self, width: int) -> None:
        self.clear(columns=True)
        for title, key, size in (("镜像依赖", "name", max(15, width - 32)),
                                 ("完整大小", "size", 10), ("新增大小", "added", 10), ("容器", "containers", 4)):
            self.add_column(Text(self.app.tr(title)), key=key, width=size)

    @on(ResizableDataTable.ColumnResized)
    def resize_tree_columns(self, event: ResizableDataTable.ColumnResized) -> None:
        event.stop()
        if self.tree is not None:
            self.tree.set_column_widths(tuple(column.get_render_width(self) for column in self.ordered_columns))

    def watch_scroll_x(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_x(old_value, new_value)
        if self.tree is not None and self.tree.scroll_x != new_value:
            self.tree.scroll_to(x=new_value, animate=False, force=True)


class ImageTree(Tree[ManagedImage]):
    auto_expand = var(False)
    COMPONENT_CLASSES = Tree.COMPONENT_CLASSES | {"image-tree--path"}
    BINDINGS = [Binding("space", "select_cursor", show=False),
                Binding("left", "collapse_branch", show=False),
                Binding("right", "expand_branch", show=False)]
    column_header: ImageTreeHeader | None = None

    def watch_cursor_line(self, previous_line: int, line: int) -> None:
        super().watch_cursor_line(previous_line, line)
        if previous_line != line:
            # 原生 Tree 只刷新选中节点的子树；祖先连接线也需要更新。
            self.refresh()

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        scroll_x, scroll_y = self.scroll_offset
        line = y + scroll_y
        # 复用 Textual 8.2.8 的标签位置，沿用折叠、隐藏根节点和滚动的布局。
        region = self._get_label_region(line)
        if region is None or not self.show_guides:
            return strip
        spans = [(0, region.x, self.get_component_rich_style("tree--guides", partial=True))]
        node = self.cursor_node
        while node is not None and node.parent is not None:
            parent = node.parent
            if parent is self.root and not self.show_root:
                break
            if parent.line < line <= node.line:
                child_region = self._get_label_region(node.line)
                if child_region is not None:
                    x = child_region.x - self.guide_depth
                    # 经过其它分支时只亮竖线；到路径节点才延伸横线。
                    end = child_region.x - 1 if line == node.line else x + 1
                    spans.append((x, end, self.get_component_rich_style("image-tree--path", partial=True)))
                break
            node = parent
        for start, end, style in spans:
            start, end = max(0, start - scroll_x), min(strip.cell_length, end - scroll_x)
            if start < end:
                strip = Strip.join((
                    strip.crop(0, start),
                    Strip(Segment.apply_style(strip.crop(start, end), post_style=style)),
                    strip.crop(end),
                ))
        return strip

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name_labels: dict[str, Text] = {}

    def set_column_widths(self, widths: tuple[int, ...]) -> None:
        def update(node, depth):
            for child in node.children:
                item = child.data
                label = self.name_labels[item.image_id].copy()
                label.truncate(max(4, widths[0] - depth * self.guide_depth - 2), overflow="ellipsis", pad=True)
                for value, width in zip((format_image_size(item.size_bytes), format_image_size(item.added_bytes),
                                         str(len(item.containers))), widths[1:]):
                    cell = Text(value)
                    cell.truncate(width - 2, overflow="ellipsis")
                    cell.align("left", width - 2)
                    label.append(" ")
                    label.append(cell)
                    label.append(" ")
                child.set_label(label)
                update(child, depth + 1)
        update(self.root, 0)
        # TreeNode.set_label 只刷新行；重算虚拟宽度后才能滚到新增的列区域。
        self._invalidate()

    def watch_scroll_x(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_x(old_value, new_value)
        if self.column_header is not None and self.column_header.scroll_x != new_value:
            self.column_header.scroll_to(x=new_value, animate=False, force=True)

    async def on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        if self.disabled or event.button != 1:
            return
        meta = event.style.meta
        if meta.get("toggle") or meta.get("image_checkbox"):
            await super()._on_click(event)
        elif "line" in meta:
            self.cursor_line = meta["line"]

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


class ImageWorkspace(Vertical):
    """在可用空间内分配列表和详情高度，保留本次会话的手动值。"""

    _detail_height: int | None = None

    def on_resize(self) -> None:
        self.query_one(ImageDetailResizeHandle).finish_resize()
        self.call_after_refresh(self._apply_detail_height)

    def _clamp_detail_height(self, height: int) -> int:
        # 为列表保留三行、分隔条保留一行；短终端仍能独立滚动两侧。
        return max(3, min(height, self.size.height - 4))

    def set_detail_height(self, height: int | None) -> None:
        self._detail_height = None if height is None else self._clamp_detail_height(height)
        self._apply_detail_height()

    def _apply_detail_height(self) -> None:
        if not self.is_mounted or not self.size.height:
            return
        height = self._detail_height
        if height is None:
            height = 7 if self.app.has_class("short") else 9
        # 缩窗只限制显示高度，不覆盖手动值，放大后可以恢复。
        self.query_one(ImageDetailPanel).styles.height = self._clamp_detail_height(height)


class ImageDetailResizeHandle(Static, can_focus=True):
    """单行分隔条；使用屏幕坐标拖动，避免控件移动时位置跳变。"""

    BINDINGS = [
        Binding("up", "adjust(1)", show=False),
        Binding("down", "adjust(-1)", show=False),
        Binding("home", "reset_height", show=False),
        Binding("escape", "cancel_resize", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__("↕ 拖动调整详情高度", markup=False, **kwargs)
        self.tooltip = "上下拖动调整详情高度；聚焦后 ↑/↓ 微调，Home 恢复默认。本次会话保留。"
        self._drag_origin: tuple[int, int] | None = None

    def render(self) -> Rule:
        return Rule(str(self.content), style="")

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if self.disabled or event.button != 1 or event.screen_offset not in self.region:
            return
        event.prevent_default()
        event.stop()
        self.focus(scroll_visible=False)
        detail = self.parent.query_one(ImageDetailPanel)
        self._drag_origin = (event.screen_y, detail.size.height)
        self.capture_mouse()
        self.add_class("-dragging")

    def _resize_to(self, screen_y: int) -> None:
        if self._drag_origin is not None:
            origin_y, height = self._drag_origin
            self.parent.set_detail_height(height + origin_y - screen_y)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag_origin is not None:
            event.prevent_default()
            event.stop()
            self._resize_to(event.screen_y)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._drag_origin is not None and event.button == 1:
            event.prevent_default()
            event.stop()
            self._resize_to(event.screen_y)
            self.finish_resize()

    def on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()

    def finish_resize(self) -> None:
        self._drag_origin = None
        if self.app.mouse_captured is self:
            self.release_mouse()
        self.remove_class("-dragging")

    def on_mouse_release(self) -> None:
        self.finish_resize()

    def on_hide(self) -> None:
        self.finish_resize()

    def on_unmount(self) -> None:
        self.finish_resize()

    def watch_disabled(self, disabled: bool) -> None:
        super().watch_disabled(disabled)
        if disabled:
            self.finish_resize()

    def action_adjust(self, delta: int) -> None:
        if not self.disabled:
            self.finish_resize()
            self.parent.set_detail_height(self.parent.query_one(ImageDetailPanel).size.height + delta)

    def action_reset_height(self) -> None:
        if not self.disabled:
            self.finish_resize()
            self.parent.set_detail_height(None)

    def action_cancel_resize(self) -> None:
        self.finish_resize()


class ImageDetailPanel(VerticalScroll):
    """常显摘要与按需展开的详情；同一对象重绘时保留阅读状态。"""

    _detail_key: tuple[str, str] | None = None

    def compose(self) -> ComposeResult:
        yield self.app._localized_widget(Static("", id="image-detail-title", markup=False))
        yield self.app._localized_widget(Static(
            "选择一行查看镜像摘要；展开分组查看详情。", id="image-detail", markup=False,
        ))
        for group, content, title in (("dependencies", "dependency", "依赖清单"),
                                       ("metadata", "metadata", "镜像信息"),
                                       ("diagnostics", "diagnostic", "诊断信息")):
            with self.app._localized_widget(Collapsible(
                title=title, collapsed=True, id="image-" + group,
                collapsed_symbol=COLLAPSED_SYMBOL, expanded_symbol=EXPANDED_SYMBOL,
            )):
                yield self.app._localized_widget(Static("", id=f"image-{content}-detail", markup=False))

    def on_mount(self) -> None:
        self.show_empty("选择一行查看镜像摘要；展开分组查看详情。")

    def _show(self, key: tuple[str, str] | None, title: str, summary: str,
              sections: dict[str, tuple[str, str]]) -> None:
        changed = key != self._detail_key
        self._detail_key = key
        heading = self.query_one("#image-detail-title", Static)
        heading.display = bool(title)
        self.app._set_text(heading, title)
        self.app._set_text(self.query_one("#image-detail", Static), summary)
        for group in self.query(Collapsible):
            section_title, content = sections.get(group.id, ("", ""))
            group.display = bool(section_title)
            self.app._set_text(group, section_title, "title")
            self.app._set_text(group.query_one("Contents > Static", Static), content)
            if changed:
                group.collapsed = True
        if changed:
            # Collapsible 本身也会请求滚动，最后回到新对象的摘要。
            self.call_after_refresh(self.scroll_home, animate=False, immediate=True)

    def show_empty(self, text: str) -> None:
        self._show(None, "", text, {})

    def show_image(self, item: ManagedImage, inventory: ImageInventory) -> None:
        self._show(("image", item.image_id), message("镜像摘要 · {0} · {1}", message(IMAGE_KINDS[item.kind]), image_display_name(item)),
                   image_summary(item, inventory), {
                       "image-dependencies": (dependency_title(item), dependency_detail(item)),
                       "image-metadata": (message("镜像信息 · 标签 {0} · 容器 {1}", len(item.tags), len(item.containers)),
                                          image_metadata(item, inventory)),
                       "image-diagnostics": (message("诊断信息 · 有警告" if inventory.warnings else "诊断信息"),
                                             image_diagnostics(item, inventory)),
                   })

    def show_layer(self, layer: ImageLayer, inventory: ImageInventory) -> None:
        self._show(("layer", layer.chain_id), message("层摘要"),
                   message("层大小：{0} · 引用镜像：{1}", format_image_size(layer.size_bytes), len(layer.image_ids)), {
                       "image-metadata": (message("引用镜像 · {0}", len(layer.image_ids)), layer_image_detail(layer, inventory)),
                       "image-diagnostics": (message("诊断信息"), layer_diagnostics(layer)),
                   })


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


def image_display_name(item: ManagedImage, parent: ManagedImage | None = None) -> str:
    """显示平台与依赖版本；树中父节点已说明的平台无需在子节点重复。"""
    platform = IMAGE_PLATFORMS.get(item.platform_id, item.platform_id)
    if item.kind == "base" and item.platform_id in IMAGE_PLATFORMS:
        return platform
    if item.kind == "runtime" and item.environment_id:
        profiles = (profile.removesuffix("-" + item.platform_id) if item.platform_id else profile
                    for profile in item.profiles)
        label = " / ".join(IMAGE_RUNTIME_NAMES.get(profile, profile) for profile in profiles)
        label = label or "env-" + item.environment_id[:12]
        inherited = (parent is not None and parent.kind in {"base", "runtime"}
                     and parent.platform_id == item.platform_id and item.platform_id in IMAGE_PLATFORMS)
        return label + (" · " + platform if platform and not inherited else "")
    return item.display_name


def dependency_packages(item: ManagedImage) -> list[str]:
    # 先显示辨识环境最有用的包；版本一律取已匹配的锁，不从 profile 名称猜测。
    priority = ("torch", "diffusers", "transformers", "sentence-transformers", "torchvision",
                "torchaudio", "librosa", "scikit-learn", "pandas", "numpy", "triton")
    order = {name: index for index, name in enumerate(priority)}
    return [f"{name}=={version}" for name, version in sorted(
        item.python_dependencies, key=lambda pair: (order.get(pair[0], len(order)), pair[0]))]


def dependency_title(item: ManagedImage) -> str:
    if item.dependency_source == "unknown":
        return message("依赖清单 · 未知")
    if item.dependency_source == "inherited":
        return message("依赖清单 · 无新增包")
    return message("依赖清单 · Python {0} · 系统 {1}", len(item.python_dependencies), len(item.system_dependencies))


def dependency_detail(item: ManagedImage) -> str:
    if item.dependency_source == "unknown":
        return message("本层依赖：未知（镜像身份、依赖锁或构建记录无法核验）。")
    if item.dependency_source == "inherited":
        return join_messages("\n", (
            message("本层依赖：无新增包，继承父镜像。"),
            message("本层添加模型文件；包依赖由运行环境提供。" if item.kind == "weights" else
                    "本层添加推理服务代码与运行清单；包依赖由运行环境提供。"),
        ))
    packages = dependency_packages(item)
    parts = [message("平台 Python 依赖（{0}，含基础镜像已有包）：\n{1}" if item.dependency_source == "platform-lock" else
                     "本层新增 Python 依赖（{0}）：\n{1}", len(packages), "\n".join(packages) or message("无新增包"))]
    if item.system_dependencies:
        parts.append(message("本层系统安装制品（{0}）：\n{1}", len(item.system_dependencies),
                             "\n".join(f"{name}={version}" for name, version in item.system_dependencies)))
    else:
        parts.append(message("系统包继承平台，本层无新增。"))
    return join_messages("\n\n", parts)


def dependency_source_detail(item: ManagedImage) -> str:
    if item.dependency_source == "unknown":
        return message("本层依赖：未知（镜像身份、依赖锁或构建记录无法核验）。")
    return message("依赖来源：已核对的构建步骤与父镜像身份。" if item.dependency_source == "inherited" else
                   "依赖来源：与镜像身份匹配的锁文件；未执行实时包扫描。")


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
        haystack = " ".join((*item.tags, item.image_id, item.model_id, item.display_name, image_display_name(item),
                             item.environment_id, item.platform_id, *item.profiles, *dependency_packages(item),
                             *(f"{name}={version}" for name, version in item.system_dependencies))).casefold().replace("/", "--")
        if all(term in haystack for term in terms):
            items.append(item)
    return tuple(items)


def image_path(item: ManagedImage, inventory: ImageInventory) -> str:
    indexed = {image.image_id: image for image in inventory.images}
    def path_name(image):
        name = image_display_name(image, indexed.get(image.parent_id))
        if image.model_id:
            name = join_messages(" · ", (message(IMAGE_KINDS[image.kind]), name))
        return join_messages("", (name, " ≈" if image.parent_source == "layer-prefix" else ""))
    path = [path_name(indexed[key]) for key in item.ancestor_ids if key in indexed]
    if item.parent_source == "missing":
        path.append(message("父镜像不在本地"))
    path.append(path_name(item))
    return message("继承路径：{0}", join_messages(" › ", path))


def image_summary(item: ManagedImage, inventory: ImageInventory) -> str:
    return join_messages("\n", (
        message("完整大小：{0} · 继承：{1} · 本镜像新增：{2}", format_image_size(item.size_bytes),
                format_image_size(item.inherited_bytes), format_image_size(item.added_bytes)),
        message("删除预计释放：{0} · 容器引用：{1}", reclaimable_text(inventory, (item.image_id,)), len(item.containers)),
    ))


def image_metadata(item: ManagedImage, inventory: ImageInventory) -> str:
    return join_messages("\n\n", (
        image_path(item, inventory),
        message("其它镜像共享：{0} · 仅当前镜像使用：{1}", format_image_size(item.shared_bytes), format_image_size(item.unique_bytes)),
        message("模型：{0} · 创建时间：{1}", item.model_id or "—", item.created or "—"),
        message("容器引用：{0}", ", ".join(item.containers) if item.containers else message("无")),
        message("全部标签：\n{0}", "\n".join(item.tags) if item.tags else message("无标签")),
    ))


def image_diagnostics(item: ManagedImage, inventory: ImageInventory) -> str:
    sources = {"recorded": "构建记录", "metadata": "构建指纹与层前缀核验", "layer-prefix": "层前缀推断，未确认 FROM",
               "missing": "父镜像不在本地", "ambiguous": "存在多个候选父镜像", "conflict": "父镜像记录与层链冲突",
               "unknown": "本地父镜像未知"}
    identity = [message("镜像 ID：{0}", item.image_id)]
    if item.parent_id:
        identity.append(message("父镜像 ID：{0}", item.parent_id))
    evidence = (
        dependency_source_detail(item),
        message("父镜像依据：{0} · 后代：{1}", message(sources[item.parent_source]), len(item.descendant_ids)),
        message("空间来源：{0}", message({"layers": "层链与已核验 history 字节数", "docker-df": "Docker df 近似值",
                                         "": "未知"}[item.space_source])),
    )
    space = [message("完整大小：{0} bytes", item.size_bytes)]
    if item.inherited_bytes is not None and item.size_bytes:
        inherited_cells = min(30, round(30 * item.inherited_bytes / item.size_bytes))
        space.append(message("空间构成：{0}  █ 继承 / ░ 新增", "█" * inherited_cells + "░" * (30 - inherited_cells)))
    space.extend((
        message("? 表示未知；释放范围受构建缓存与存储驱动影响。"),
    ))
    parts = [join_messages("\n", identity), join_messages("\n", evidence), join_messages("\n", space)]
    parts.extend(message(warning) for warning in inventory.warnings)
    return join_messages("\n\n", parts)


def reclaimable_text(inventory: ImageInventory, image_ids: tuple[str, ...]) -> str:
    value = reclaimable_image_bytes(inventory, image_ids)
    return message("未知") if value is None else "0 B" if value == 0 else message("0 B ～ 约 {0}", format_image_size(value))


def layer_image_detail(layer: ImageLayer, inventory: ImageInventory) -> str:
    indexed = {item.image_id: item for item in inventory.images}
    return message("使用此层的镜像：\n{0}", "\n\n".join(
        f"{image_display_name(indexed[key])} · {key[7:19]}\n  {indexed[key].name}" for key in layer.image_ids))


def layer_diagnostics(layer: ImageLayer) -> str:
    return join_messages("\n", (
        message("层内容 Diff ID：{0}", layer.diff_id),
        message("层链 Chain ID：{0}", layer.chain_id),
        message("相同 Diff ID 的不同父层链分开统计；引用包含筛选外镜像，按 image ID 去重。"),
    ))


def render_image_tree(tree: ImageTree, inventory: ImageInventory | None, visible: tuple[ManagedImage, ...],
                      selected: set[str], current_id: str, tr, width: int, *, preserve_scroll: bool = False) -> None:
    offset = tree.scroll_offset
    previous = {}
    def collect(node):
        for child in node.children:
            previous[child.data.image_id] = child.is_expanded
            collect(child)
    collect(tree.root)
    tree.clear()
    tree.name_labels.clear()
    tree.show_root = False
    header = tree.screen.query_one("#image-tree-header", ImageTreeHeader)
    header.minimum_name_width = max((len(item.ancestor_ids) * tree.guide_depth + 4 for item in visible), default=6)
    header.set_columns(width)
    if inventory is None:
        return
    indexed = {item.image_id: item for item in inventory.images}
    matches = {item.image_id for item in visible}
    shown = matches | {key for item in visible for key in item.ancestor_ids}
    nodes = {}
    for item in sorted((indexed[key] for key in shown), key=lambda item: (len(item.ancestor_ids), image_display_name(item), item.image_id)):
        parent = nodes.get(item.parent_id, tree.root)
        marker = "☑" if item.image_id in selected else "—" if item.containers else "□"
        # 同名镜像仍按 image ID 分别管理；完整 ID 在详情中查看。
        logical = image_display_name(item, parent.data)
        if item.model_id:
            logical = f"{tr(IMAGE_KINDS[item.kind])} · {logical}"
        evidence = " ≈" if item.parent_source == "layer-prefix" else " ?" if item.parent_source in {"ambiguous", "missing", "conflict"} else ""
        name = f" {marker}  {logical}{evidence}"
        label = Text(name, style="dim" if item.image_id not in matches else "")
        # 复选框及左右各一格留白可点击，不覆盖箭头、名称或数值。
        if not item.containers:
            label.stylize(Style(meta={"image_checkbox": True}), 0, 3)
        tree.name_labels[item.image_id] = label
        nodes[item.image_id] = parent.add(label, data=item, expand=previous.get(item.image_id, True))
    for node in nodes.values():
        node.allow_expand = bool(node.children)
    tree.set_column_widths(tuple(column.get_render_width(header) for column in header.ordered_columns))
    target = nodes.get(current_id) or next(iter(nodes.values()), None)
    while target and target.parent is not tree.root and not target.parent.is_expanded:
        target = target.parent
    # 新节点要等 Textual 完成行布局，才有可用于 move_cursor 的行号。
    tree.call_after_refresh(tree.move_cursor, target)
    if preserve_scroll:
        tree.call_after_refresh(tree.scroll_to, x=offset.x, y=offset.y, animate=False, immediate=True, force=True)


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

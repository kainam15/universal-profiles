"""镜像列表的筛选、文案和删除确认；不执行 Docker 命令。"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Static

from acprof.host.image_management import ImageInventory, ImageManagementError, ManagedImage
from acprof.tui.i18n import join_messages, message
from acprof.tui.views import ConfirmActionScreen


IMAGE_KINDS = {
    "base": "公共基础", "runtime": "运行依赖", "weights": "模型文件",
    "model": "推理服务", "debug": "调试镜像", "other": "其它镜像", "untagged": "无标签",
}
IMAGE_HINT = "点击刷新读取 Docker；↑↓ 浏览，空格或点击勾选。"


class ImageTable(DataTable):
    BINDINGS = [Binding("space", "select_cursor", "勾选镜像", show=False)]


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


def format_image_size(value: int) -> str:
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
        haystack = " ".join((*item.tags, item.image_id, item.model_id)).casefold().replace("/", "--")
        if all(term in haystack for term in terms):
            items.append(item)
    return tuple(items)


def image_detail(item: ManagedImage) -> str:
    return join_messages("\n", (
        message("镜像 ID：{0}", item.image_id),
        message("类型：{0} · 完整大小：{1}（{2} bytes）", message(IMAGE_KINDS[item.kind]),
                format_image_size(item.size_bytes), item.size_bytes),
        message("模型：{0} · 创建时间：{1}", item.model_id or "—", item.created or "—"),
        message("容器引用：{0}", ", ".join(item.containers) if item.containers else message("无")),
        message("全部标签：\n{0}", "\n".join(item.tags) if item.tags else message("无标签")),
        message("完整大小包含共享层，不等于删除后释放的空间。"),
    ))


def deletion_message(inventory: ImageInventory, image_ids: tuple[str, ...]) -> str:
    items = [item for item in inventory.images if item.image_id in image_ids]
    parts = [
        message("Docker 环境：{0} · 共 {1} 个镜像", inventory.connection.name, len(items)),
        message("删除全部标签后无法使用原镜像续采或补采；已保存的实验文件保留。"),
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

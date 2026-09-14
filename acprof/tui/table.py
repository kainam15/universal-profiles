"""可拖动表头边界的表格；列宽仅在控件的本次会话内保留。"""

from __future__ import annotations

from collections.abc import Iterator

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.message import Message
from textual.reactive import var
from textual.strip import Strip
from textual.widgets import DataTable
from textual.widgets.data_table import CellType, ColumnKey


class ResizableDataTable(DataTable):
    """复用 Textual 的列、滚动及鼠标捕获，不重新构造行数据。"""

    resize_enabled = var(True)

    class ColumnResized(Message):
        def __init__(self, table: ResizableDataTable) -> None:
            super().__init__()
            self.data_table = table

        @property
        def control(self) -> ResizableDataTable:
            return self.data_table

    def __init__(self, *args, **kwargs) -> None:
        self._manual_widths: dict[str, int] = {}
        self._resize_column: ColumnKey | None = None
        self._resize_origin_x = 0
        self._resize_origin_width = 0
        self._suppress_resize_click = False
        super().__init__(*args, **kwargs)
        self.tooltip = "拖动表头 │ 调整列宽；本次会话保留。"

    def minimum_column_width(self, key: str | None) -> int:
        return 1

    def add_column(self, label: str | Text, *, width: int | None = None,
                   key: str | None = None, default: CellType | None = None) -> ColumnKey:
        if key is not None:
            width = self._manual_widths.get(key, width)
        if width is not None:
            width = max(self.minimum_column_width(key), width)
        return super().add_column(label, width=width, key=key, default=default)

    def clear(self, columns: bool = False) -> ResizableDataTable:
        self._finish_resize()
        return super().clear(columns=columns)

    def _header_boundaries(self) -> Iterator[tuple[int, int, ColumnKey]]:
        """返回相邻列间可见边界的 cell 坐标；固定列不随横向滚动移动。"""
        columns = self.ordered_columns
        right = self._row_label_column_width
        fixed_width = right + sum(column.get_render_width(self) for column in columns[:self.fixed_columns])
        viewport_width = self.scrollable_content_region.width
        # 末列右沿没有相邻列，不绘制手柄，也不参与拖动命中。
        for index, column in enumerate(columns[:-1]):
            left = right
            right += column.get_render_width(self)
            x = right - 1
            if index < self.fixed_columns:
                # 终端缩窄后，过宽的固定列仍须在可见右沿提供拖动入口。
                if left >= viewport_width:
                    continue
                x = min(x, viewport_width - 1)
            else:
                x -= self.scroll_offset.x
                if x < fixed_width:
                    continue
            if 0 <= x < viewport_width:
                yield x, index, column.key

    def _resize_target(self, event: events.MouseEvent) -> ColumnKey | None:
        if (self.disabled or not self.resize_enabled or not self.show_header
                or event.screen_offset not in self.scrollable_content_region):
            return None
        x, y = event.screen_offset - self.content_region.offset
        if not 0 <= y < self.header_height:
            return None
        for boundary, _, key in self._header_boundaries():
            if x == boundary:
                return key
        return None

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        if not self.show_header or y >= self.header_height:
            return strip
        parts = []
        start = 0
        for boundary, index, key in self._header_boundaries():
            parts.append(strip.crop(start, boundary))
            style = next(iter(strip.crop(boundary, boundary + 1))).style or Style()
            style += Style(meta={"row": -1, "column": index})
            if key == self._resize_column:
                style += Style(bold=True, reverse=True)
            parts.append(Strip([Segment("│", style)], 1))
            start = boundary + 1
        parts.append(strip.crop(start))
        return Strip.join(parts)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._suppress_resize_click = False
        key = self._resize_target(event)
        if event.button != 1 or key is None:
            return
        event.prevent_default()
        event.stop()
        self._resize_column = key
        self._resize_origin_x = event.screen_x
        self._resize_origin_width = self.columns[key].get_render_width(self) - 2 * self.cell_padding
        index = self.get_column_index(key)
        if index < self.fixed_columns:
            left = self._row_label_column_width + sum(
                column.get_render_width(self) for column in self.ordered_columns[:index])
            self._resize_origin_width = min(self._resize_origin_width,
                                           max(1, self.scrollable_content_region.width - left - 2 * self.cell_padding))
        self._suppress_resize_click = True
        self.capture_mouse()
        self.refresh()

    def _resize_to(self, screen_x: int) -> None:
        key = self._resize_column
        if key is None or key not in self.columns or self.disabled or not self.resize_enabled:
            self._finish_resize()
            return
        column = self.columns[key]
        width = max(self.minimum_column_width(key.value), self._resize_origin_width + screen_x - self._resize_origin_x)
        current_width = column.get_render_width(self) - 2 * self.cell_padding
        if width == current_width:
            return
        column.width = width
        column.auto_width = False
        if key.value is not None:
            self._manual_widths[key.value] = width
        # Textual 8.2.8 没有公开的 set_column_width；列宽改变后须同步缓存和滚动范围。
        # 空 new_rows 避免重新测量整张表的数据。
        self._update_count += 1
        self._clear_caches()
        self._update_dimensions(())
        self.refresh(layout=True)
        self.post_message(self.ColumnResized(self))

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._resize_column is None:
            return
        event.prevent_default()
        event.stop()
        self._resize_to(event.screen_x)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._resize_column is None or event.button != 1:
            return
        event.prevent_default()
        event.stop()
        self._resize_to(event.screen_x)
        self._finish_resize()

    def _consume_resize_click(self, event: events.Click) -> bool:
        if self._suppress_resize_click or self._resize_target(event) is not None:
            self._suppress_resize_click = False
            event.prevent_default()
            event.stop()
            return True
        return False

    def on_click(self, event: events.Click) -> None:
        self._consume_resize_click(event)

    def _finish_resize(self) -> None:
        if self._resize_column is None:
            return
        self._resize_column = None
        if self.app.mouse_captured is self:
            self.release_mouse()
        self.refresh()

    def on_mouse_release(self) -> None:
        self._finish_resize()

    def on_hide(self) -> None:
        self._finish_resize()

    def on_unmount(self) -> None:
        self._finish_resize()

    def watch_disabled(self, disabled: bool) -> None:
        super().watch_disabled(disabled)
        if disabled:
            self._finish_resize()

    def watch_resize_enabled(self, enabled: bool) -> None:
        if not enabled:
            self._finish_resize()

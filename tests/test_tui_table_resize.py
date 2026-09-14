"""通过实际表头字符、鼠标事件和滚动范围验证列宽拖动。"""

import unittest

from rich.cells import cell_len
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Static

from acprof.tui.table import ResizableDataTable


class TableApp(App):
    CSS = """
    ResizableDataTable { width: 52; height: 10; margin: 1 2; padding: 1 2; border: solid $primary; }
    """

    def __init__(self):
        super().__init__()
        self.headers = []
        self.selected = []

    def compose(self) -> ComposeResult:
        yield ResizableDataTable(id="table", fixed_columns=1, header_height=2, cursor_type="row")
        yield Static("outside", id="outside")

    def on_mount(self):
        table = self.query_one(ResizableDataTable)
        table.add_column("中文名称", width=10, key="name")
        table.add_column("Size", width=12, key="size")
        table.add_column("Description", width=30, key="description")
        for index in range(20):
            table.add_row("模型名称较长", str(index), "details " * 8, key=str(index))

    def on_data_table_header_selected(self, event):
        self.headers.append(event.column_key.value)

    def on_data_table_row_selected(self, event):
        self.selected.append(event.row_key.value)


def header_offset(table, boundary_index=0, y=0):
    text = table.render_line(y).text
    # 从渲染的分隔符定位，不使用被测控件的命中判断或元数据。
    boundaries = [cell_len(text[:index]) for index, char in enumerate(text) if char == "│"]
    inset = table.content_region.offset - table.region.offset
    return inset.x + boundaries[boundary_index], inset.y + y


async def drag(pilot, table, start, dx, dy=0, *, release_click=False):
    await pilot.mouse_down(table, offset=start)
    end = (table.region.x + start[0] + dx, table.region.y + start[1] + dy)
    await pilot.hover(offset=end)
    await pilot.mouse_up(offset=end)
    # Pilot.mouse_up 不像真实驱动那样生成释放后的 Click。
    if release_click:
        await pilot._post_mouse_events([events.Click], offset=end, button=1)
    await pilot.pause()


class ResizableTableTests(unittest.IsolatedAsyncioTestCase):
    async def test_last_header_edge_cannot_resize_and_keeps_header_clicks(self):
        for layout in ("wide", "scrolled", "single"):
            with self.subTest(layout=layout):
                app = TableApp()
                async with app.run_test(size=(80, 24)) as pilot:
                    table = app.query_one(ResizableDataTable)
                    if layout == "wide":
                        table.styles.width = 72
                    elif layout == "single":
                        table.clear(columns=True)
                        table.add_column("中文名称", width=10, key="name")
                        table.add_row("模型名称")
                    await pilot.pause()
                    if layout == "scrolled":
                        table.scroll_to(x=table.max_scroll_x, animate=False, force=True)
                        await pilot.pause()
                    columns = table.ordered_columns
                    widths = tuple(column.get_render_width(table) for column in columns)
                    edge = sum(widths) - 1 - table.scroll_offset.x
                    inset = table.content_region.offset - table.region.offset
                    start = (inset.x + edge, inset.y)
                    await pilot.mouse_down(table, offset=start)
                    self.assertIsNone(app.mouse_captured, "末列右沿不能捕获鼠标调整宽度")
                    await pilot.hover(table, offset=(start[0] - 4, start[1]))
                    await pilot.mouse_up(table, offset=(start[0] - 4, start[1]))
                    await pilot.pause()
                    self.assertEqual(tuple(column.get_render_width(table) for column in columns), widths)
                    self.assertNotEqual(table.render_line(0).crop(edge, edge + 1).text, "│")
                    if layout == "single":
                        self.assertNotIn("│", table.render_line(0).text)
                    await pilot.click(table, offset=start)
                    self.assertEqual(app.headers, [columns[-1].key.value])

    async def test_drag_updates_rendered_cells_and_scroll_extent(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            before = table.virtual_size.width
            scroll = table.max_scroll_x
            start = header_offset(table, y=1)
            await drag(pilot, table, start, 6, release_click=True)
            self.assertEqual(table.columns["name"].width, 16)
            self.assertEqual(header_offset(table)[0], start[0] + 6)
            self.assertEqual(table.virtual_size.width, before + 6)
            self.assertEqual(table.max_scroll_x, scroll + 6)
            self.assertIn("模型名称较长", table.render_line(2).text)
            self.assertFalse(app.headers)
            self.assertFalse(app.selected)
            self.assertIsNone(app.mouse_captured)

    async def test_scrolled_header_and_fixed_column_use_visible_boundaries(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            fixed = header_offset(table)
            second = header_offset(table, 1)
            table.scroll_to(x=5, y=6, animate=False, force=True)
            await pilot.pause()
            self.assertEqual(header_offset(table), fixed)
            self.assertEqual(header_offset(table, 1)[0], second[0] - 5)
            await drag(pilot, table, header_offset(table, 1), 4)
            self.assertEqual(table.columns["size"].width, 16)
            self.assertEqual(table.columns["name"].width, 10)
            self.assertEqual(table.scroll_offset.x, 5)
            self.assertEqual(table.scroll_offset.y, 6)
            await drag(pilot, table, header_offset(table), -3)
            self.assertEqual(table.columns["name"].width, 7)
            self.assertEqual(table.columns["size"].width, 16)

    async def test_drag_outside_table_clamps_width_and_releases_capture(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            start = header_offset(table)
            await pilot.mouse_down(table, offset=start)
            self.assertIs(app.mouse_captured, table)
            await pilot.hover(offset=(0, 20))
            await pilot.mouse_up(offset=(0, 20))
            self.assertEqual(table.columns["name"].width, 1)
            self.assertIsNone(app.mouse_captured)
            await pilot.hover(table, offset=start)
            self.assertEqual(table.columns["name"].width, 1)
            await drag(pilot, table, header_offset(table), 8)
            self.assertEqual(table.columns["name"].width, 9)

    async def test_body_right_button_and_plain_header_clicks_keep_normal_behavior(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            start = header_offset(table)
            await drag(pilot, table, (start[0], start[1] + 2), 4)
            self.assertEqual(table.columns["name"].width, 10)
            await pilot.mouse_down(table, offset=start, button=3)
            self.assertIsNone(app.mouse_captured)
            await pilot.hover(table, offset=(start[0] + 4, start[1]))
            self.assertEqual(table.columns["name"].width, 10)
            await pilot.click(table, offset=start)
            self.assertFalse(app.headers, "边界点击不能排序")
            await pilot.click(table, offset=(start[0] - 5, start[1]))
            self.assertEqual(app.headers, ["name"])

    async def test_interrupted_drag_cannot_leave_mouse_captured(self):
        for interruption in ("disabled", "locked", "hidden", "clear", "capture_lost"):
            with self.subTest(interruption=interruption):
                app = TableApp()
                async with app.run_test(size=(80, 24)) as pilot:
                    table = app.query_one(ResizableDataTable)
                    await pilot.pause()
                    await pilot.mouse_down(table, offset=header_offset(table))
                    self.assertIs(app.mouse_captured, table)
                    if interruption == "disabled":
                        table.disabled = True
                    elif interruption == "locked":
                        table.resize_enabled = False
                    elif interruption == "hidden":
                        table.display = False
                    elif interruption == "clear":
                        table.clear(columns=True)
                    else:
                        table.release_mouse()
                    await pilot.pause()
                    self.assertIsNone(app.mouse_captured)
                    await pilot.mouse_up(offset=(70, 20))
                    table.disabled = False
                    table.resize_enabled = True
                    table.display = True
                    if interruption == "clear":
                        table.add_column("中文名称", width=10, key="name")
                        table.add_column("Size", width=12, key="size")
                    await pilot.pause()
                    await pilot.hover(table, offset=header_offset(table))
                    self.assertEqual(table.columns["name"].width, 10)
                    await drag(pilot, table, header_offset(table), 2)
                    self.assertEqual(table.columns["name"].width, 12)

    async def test_auto_width_becomes_manual_and_survives_rebuild_by_key(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            table.clear(columns=True)
            table.add_column("中文名称", key="name")
            table.add_column("Size", width=4, key="size")
            table.add_row("模型名称较长", "1")
            await pilot.pause()
            await drag(pilot, table, header_offset(table), -4)
            self.assertEqual(table.columns["name"].width, 8)
            table.clear(columns=True)
            table.add_column("Name", key="name", width=20)
            table.add_column("Size", width=4, key="size")
            table.add_row("much longer content than the manually sized column", "1")
            await pilot.pause()
            self.assertEqual(table.columns["name"].width, 8)
            self.assertEqual(table.virtual_size.width, 16)
            self.assertEqual(len(table.columns), 2)

    async def test_oversized_fixed_column_can_be_shrunk_after_terminal_resize(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            await drag(pilot, table, header_offset(table), 26)
            self.assertEqual(table.columns["name"].width, 36)
            table.styles.width = 30
            await pilot.resize_terminal(60, 24)
            await pilot.pause()
            self.assertIn("│", table.render_line(0).text,
                          "固定列超出视口时，仍需提供可抓取的边界")
            start = header_offset(table)
            await drag(pilot, table, start, -5)
            self.assertLess(table.columns["name"].width, table.scrollable_content_region.width)
            self.assertEqual(header_offset(table)[0], start[0] - 5)
            self.assertIsNone(app.mouse_captured)


if __name__ == "__main__":
    unittest.main()

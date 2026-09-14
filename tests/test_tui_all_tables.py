"""运行矩阵、报告和镜像树也通过鼠标边界调整列宽。"""

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from textual.widgets import Collapsible, DataTable, Tree

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.reports import ReportRow, ReportView
from test_image_management import DockerFixture
from test_tui_table_resize import drag, header_offset


class AllTablesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.docker = DockerFixture()
        for context in (patch.dict(os.environ, {}, clear=True),
                        patch("acprof.host.image_management.subprocess.run", side_effect=self.docker.run),
                        patch.object(AcprofTui, "IMAGE_REFRESH_INTERVAL", 3600)):
            context.start()
            self.addCleanup(context.stop)

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "settings.json")

    async def test_matrix_header_drag_changes_width(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._activate_tab("monitor-tab")
            app.query_one("#matrix-board", Collapsible).collapsed = False
            await pilot.pause()
            table = app.query_one("#matrix-table", DataTable)
            width = table.columns["case"].get_render_width(table) - 2 * table.cell_padding
            # 首先按原生表格坐标发送事件，以便缺失功能时确实失败于列宽。
            start = (table.columns["case"].get_render_width(table) - 1, 0)
            await drag(pilot, table, start, 5)
            self.assertEqual(table.columns["case"].width, width + 5)
            app._init_matrix_for_run(RunConfig.smoke("demo/model"))
            await pilot.pause()
            self.assertEqual(table.columns["case"].width, width + 5)
            await pilot.mouse_down(table, offset=header_offset(table))
            self.assertIs(app.mouse_captured, table)
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            await pilot.pause()
            self.assertIsNone(app.mouse_captured)
            await pilot.mouse_up(table, offset=(start[0] + 8, 0))
            self.assertEqual(table.columns["case"].width, width + 5)
            app._latest_snapshot = ProgressSnapshot()
            app._set_busy(False)
            await drag(pilot, table, header_offset(table), 3)
            self.assertEqual(table.columns["case"].width, width + 8)

    async def test_report_header_drag_survives_language_switch(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._activate_tab("reports-tab")
            app._report_view = ReportView(Path("example.json"), "统计", ("指标", "均值"),
                                          (ReportRow(("Latency", "20 ms"), "来源"),), "说明")
            app._render_report_view()
            await pilot.pause()
            table = app.query_one("#report-table", DataTable)
            width = table.ordered_columns[0].get_render_width(table) - 2 * table.cell_padding
            await drag(pilot, table, (width + 1, 0), 5)
            self.assertEqual(table.ordered_columns[0].width, width + 5)
            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.pause()
            self.assertEqual(table.ordered_columns[0].width, width + 5)

    async def test_tree_header_drag_moves_data_columns_without_folding_or_selecting(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app.action_show_images()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            tree = app.query_one("#image-tree", Tree)
            before = tree.render_line(0).text.index("100 B")
            header = app.query_one("#image-tree-header", DataTable)
            self.assertIn("容器", header.render_line(0).text)
            start = header_offset(header)
            await drag(pilot, header, start, -8)
            self.assertEqual(tree.render_line(0).text.index("100 B"), before - 8)
            self.assertTrue(tree.root.children[0].is_expanded)
            self.assertFalse(app._selected_image_ids)

    async def test_tree_columns_align_and_scroll_together_after_manual_resize(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app.action_show_images()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            tree = app.query_one("#image-tree", Tree)
            header = app.query_one("#image-tree-header", DataTable)
            commands = len(self.docker.commands)
            await drag(pilot, header, header_offset(header), 12)
            width = header.columns["name"].width
            self.assertEqual(header.max_scroll_x, tree.max_scroll_x)
            tree.scroll_to(x=8, animate=False, force=True)
            await pilot.pause()
            self.assertEqual(header.scroll_offset.x, 8)
            before = header.columns["size"].width
            await drag(pilot, header, header_offset(header, 1), -3)
            self.assertEqual(header.columns["size"].width, before - 3)
            tree.scroll_to(x=tree.max_scroll_x, animate=False, force=True)
            await pilot.pause()
            for row, values in enumerate((("100 B", "?", "0"), ("400 B", "300 B", "0"),
                                           ("410 B", "10 B", "0"))):
                line = tree.render_line(row)
                for column, value in enumerate(values):
                    start = header_offset(header, column)[0] + 2
                    self.assertEqual(line.crop(start, start + cell_len(value)).text, value,
                                     "父子行的大小、未知值和容器数量应与表头统一左对齐")
            header.scroll_to(x=4, animate=False, force=True)
            await pilot.pause()
            self.assertEqual(tree.scroll_offset.x, 4)
            await pilot.click("#image-view-list")
            await pilot.click("#image-view-tree")
            self.assertEqual(header.columns["name"].width, width)
            self.assertEqual(header.columns["size"].width, before - 3)
            self.assertEqual(len(self.docker.commands), commands)

    async def test_every_table_resizes_in_both_languages_at_all_sizes(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            app = self.make_app()
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                app._init_matrix_for_run(RunConfig.smoke("demo/model"))
                app.query_one("#matrix-board", Collapsible).collapsed = False
                app._report_view = ReportView(Path("example.json"), "统计", ("指标", "均值"),
                                              (ReportRow(("Latency", "20 ms"), "来源"),), "说明")
                app._render_report_view()
                app.action_show_images()
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                for language in ("zh", "en"):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    await pilot.pause()
                    for tab, view, selector, index, delta in (
                        ("monitor-tab", "", "#matrix-table", 0, 3),
                        ("reports-tab", "", "#report-table", 0, 3),
                        ("images-tab", "tree", "#image-tree-header", 0, -3),
                        ("images-tab", "list", "#image-table", 1, -3),
                        ("images-tab", "layers", "#image-layer-table", 0, -3),
                    ):
                        with self.subTest(size=size, language=language, table=selector):
                            app._activate_tab(tab)
                            await pilot.pause()
                            await app.workers.wait_for_complete()
                            await pilot.pause()
                            commands = len(self.docker.commands)
                            if view:
                                self.assertTrue(await pilot.click("#image-view-" + view))
                            table = app.query_one(selector, DataTable)
                            column = table.ordered_columns[index]
                            width = column.get_render_width(table) - 2 * table.cell_padding
                            await drag(pilot, table, header_offset(table, index), delta)
                            self.assertEqual(table.columns[column.key].width, width + delta)
                            self.assertIn("本次会话" if language == "zh" else "this session", table.tooltip)
                            self.assertIsNone(app.mouse_captured)
                            table.scroll_to(x=table.max_scroll_x, animate=False, force=True)
                            await pilot.pause()
                            edge = sum(item.get_render_width(table) for item in table.ordered_columns) - 1
                            edge -= table.scroll_offset.x
                            self.assertLess(edge, table.scrollable_content_region.width)
                            self.assertNotEqual(table.render_line(0).crop(edge, edge + 1).text, "│",
                                                "末列右侧不能出现拖动手柄")
                            table.scroll_to(x=0, animate=False, force=True)
                            await pilot.pause()
                            self.assertEqual(len(self.docker.commands), commands)

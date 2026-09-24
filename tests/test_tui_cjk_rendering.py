"""中文浮层在最终屏幕与局部终端输出中保持完整。"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.segment import Segment
from rich.text import Text
from textual.geometry import Region, Size
from textual.strip import Strip
from textual.widgets._toast import Toast

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig
from acprof.tui.views import ConfirmActionScreen


class CjkCompositionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.app = AcprofTui(settings_path=Path(temporary.name) / "tui.json")
        self.compositor = self.app.get_default_screen()._compositor
        self.compositor.size = Size(12, 1)

    def configure_layers(self, layers):
        viewport = self.compositor.size.region
        self.compositor._visible_widgets = {
            object(): (region, viewport) for region, _ in layers
        }
        self.compositor._cuts = None
        renders = [(region, viewport, [Strip([Segment(text)])]) for region, text in layers]
        mocked = patch.object(self.compositor, "_get_renders", return_value=renders)
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_hidden_widget_edges_do_not_erase_double_width_characters(self):
        self.configure_layers([
            (Region(1, 0, 10, 1), "甲乙丙丁戊"),
            (Region(4, 0, 3, 1), "TOP"),
            (Region(0, 0, 12, 1), "." * 12),
        ])
        self.assertEqual(self.compositor.render_strips()[0].text, ".甲乙丙丁戊.")

    def test_real_occlusion_still_clips_the_underlying_wide_character(self):
        self.configure_layers([
            (Region(4, 0, 3, 1), "TOP"),
            (Region(1, 0, 10, 1), "甲乙丙丁戊"),
            (Region(0, 0, 12, 1), "." * 12),
        ])
        self.assertEqual(self.compositor.render_strips()[0].text, ".甲 TOP丁戊.")

    def test_partial_update_emits_the_whole_character_at_either_half(self):
        self.configure_layers([
            (Region(1, 0, 10, 1), "甲乙丙丁戊"),
            (Region(4, 0, 3, 1), "TOP"),
            (Region(0, 0, 12, 1), "." * 12),
        ])
        for column in (3, 4):
            with self.subTest(column=column):
                self.compositor._dirty_regions = {Region(column, 0, 1, 1)}
                update = self.compositor.render_partial_update()
                self.assertIsNotNone(update)
                self.assertIn("乙", Text.from_ansi(update.render_segments(self.app.console)).plain)
                segments = self.app.console.render(update)
                self.assertIn("乙", "".join(segment.text for segment in segments if not segment.control))
        self.assertIsNone(self.compositor.render_partial_update())

    def test_clipped_and_empty_foreground_rows_do_not_crash_or_shift_text(self):
        self.configure_layers([
            (Region(-2, 0, 10, 1), "甲乙丙丁"),
            (Region(3, 0, 3, 1), "TOP"),
            (Region(0, 0, 12, 1), "." * 12),
        ])
        self.assertEqual(self.compositor.render_strips()[0].text, "甲乙丙丁....")
        self.compositor._get_renders.return_value = [(Region(0, 0, 0, 1), Region(0, 0, 12, 1), [Strip([])])]
        self.assertEqual(self.compositor.render_strips()[0].text, "")


class TuiCjkNotificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_toast_survives_button_borders_language_changes_and_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(
                RunConfig.smoke("demo/model"), settings_path=Path(directory) / "tui.json",
            )
            with patch.object(app, "refresh_images"):
                async with app.run_test(size=(80, 24), notifications=True) as pilot:
                    await pilot.pause()
                    for language, expected in (
                        ("zh", "已恢复完整默认配置"), ("en", "Full defaults restored"),
                    ):
                        app.clear_notifications()
                        app.ui_preferences = replace(
                            app.ui_preferences, language=language, show_command_bar=False,
                        )
                        app._apply_ui_preferences()
                        await pilot.pause()
                        app.notify("已恢复完整默认配置", timeout=60)
                        await pilot.pause()
                        for size in ((80, 24), (81, 24), (120, 30), (121, 30), (150, 45), (151, 45)):
                            with self.subTest(language=language, size=size):
                                await pilot.resize_terminal(*size)
                                await pilot.pause()
                                toast = app.query_one(Toast)
                                lines = app.screen._compositor.render_strips()
                                rendered = "\n".join(lines[y].text for y in toast.region.line_range)
                                self.assertIn(expected, rendered)
                        self.assertTrue(await pilot.click(app.query_one(Toast)))
                        await pilot.pause()
                        self.assertFalse(app.query(Toast))

    async def test_titled_notifications_on_confirmation_screen_keep_text_and_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(settings_path=Path(directory) / "tui.json")
            with patch.object(app, "_launch") as launch:
                async with app.run_test(size=(81, 24), notifications=True) as pilot:
                    await pilot.pause()
                    app.clear_notifications()
                    await app.push_screen(ConfirmActionScreen("开始采集", "检查配置后再开始采集"))
                    app.notify("请等待当前任务完成", title="设置未保存", severity="error", timeout=60)
                    await pilot.pause()
                    for size in ((81, 24), (120, 30), (150, 45)):
                        with self.subTest(size=size):
                            await pilot.resize_terminal(*size)
                            await pilot.pause()
                            text = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                            self.assertIn("设置未保存", text)
                            self.assertIn("请等待当前任务完成", text)
                    await pilot.press("escape")
                    await pilot.pause()
                    self.assertNotIsInstance(app.screen, ConfirmActionScreen)
                    launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

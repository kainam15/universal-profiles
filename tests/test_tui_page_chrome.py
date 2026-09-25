"""页面滚动、缩放和交互状态不能改变操作栏位置或颜色含义。"""

import colorsys
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.themes import UI_THEMES


class TuiPageChromeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.app = AcprofTui(
            RunConfig.smoke("demo/model"),
            settings_path=Path(temporary.name) / "tui.json",
        )
        refresh = patch.object(self.app, "refresh_images")
        refresh.start()
        self.addCleanup(refresh.stop)

    def assert_reachable(self, button):
        region = button.region
        self.assertGreater(region.width, 0, button.id)
        self.assertEqual(region.height, 3, button.id)
        self.assertLessEqual(cell_len(button.label.plain), button.content_region.width, button.id)
        center_x, center_y = region.center
        for x in (region.x + 1, center_x, region.right - 2):
            self.assertIs(self.app.get_widget_at(x, center_y)[0], button, button.id)

    def assert_actions_aligned(self, pane, secondary_ids, primary):
        secondary = [pane.query_one("#" + widget_id) for widget_id in secondary_ids]
        self.assertEqual(secondary[0].region.x, pane.region.x + 2)
        for left, right in zip(secondary, secondary[1:]):
            self.assertEqual(right.region.x, left.region.right + 1, right.id)
        for action in secondary:
            self.assertEqual(action.region.y, primary.region.y, action.id)
        self.assertGreaterEqual(primary.region.x, secondary[-1].region.right + 1)

    async def test_all_pages_keep_headers_and_rightmost_actions_fixed_when_scrolling_and_resizing(self):
        app = self.app
        async with app.run_test(size=(150, 45)) as pilot:
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                for language in ("zh", "en"):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    for page, primary, secondary in (
                        ("run-tab", "start-run", ("open-run-settings", "quick-check", "probe-largest")),
                        ("monitor-tab", "stop-run", ("log-title", "copy-log", "follow-log", "expand-log", "clear-log")),
                        ("plot-tab", "plot-results", ("summarize-results",)),
                        ("reports-tab", "report-calculate", ("report-current", "report-open")),
                        ("profile-tab", "profile-run", ("profile-dry-run",)),
                        ("images-tab", "image-delete", ("image-toggle", "image-model", "image-clear")),
                        ("settings-tab", "save-ui-settings", ("restore-ui-defaults",)),
                    ):
                        with self.subTest(size=size, language=language, page=page):
                            app._activate_tab(page)
                            await pilot.pause()
                            pane = app.query_one("#" + page)
                            button = app.query_one("#" + primary, Button)
                            self.assert_reachable(button)
                            self.assertEqual(button.region.bottom, pane.region.bottom)
                            self.assertEqual(button.region.right, pane.region.right - 2)
                            self.assert_actions_aligned(pane, secondary, button)
                            header = pane.query_one(".page-header")
                            header_region, button_region = header.region, button.region
                            self.assertEqual(header_region.y, pane.region.y)
                            for scroll in pane.query(VerticalScroll):
                                if scroll.display and scroll.region.height:
                                    scroll.scroll_end(animate=False, immediate=True)
                            await pilot.pause()
                            self.assertEqual(header.region, header_region)
                            self.assertEqual(button.region, button_region)
                            self.assert_actions_aligned(pane, secondary, button)
                            for action in pane.query(".action-bar Button"):
                                if action.display:
                                    self.assert_reachable(action)
                            # Footer remains anchored when the global command box is hidden.
                            app.ui_preferences = replace(app.ui_preferences, show_command_bar=False)
                            app._apply_ui_preferences()
                            await pilot.pause()
                            self.assertEqual(button.region.bottom, pane.region.bottom)
                            self.assert_reachable(button)
                            self.assert_actions_aligned(pane, secondary, button)
                            app.ui_preferences = replace(app.ui_preferences, show_command_bar=True)
                            app._apply_ui_preferences()

    def assert_color_family(self, color, family):
        hue, saturation, _ = colorsys.rgb_to_hsv(*(channel / 255 for channel in color.rgb))
        if family == "gray":
            self.assertLess(saturation, 0.15)
        else:
            self.assertGreater(saturation, 0.2)
            low, high = {"cyan": (0.46, 0.56), "yellow": (0.10, 0.18),
                         "green": (0.35, 0.46), "red": (0.95, 1.0)}[family]
            self.assertGreaterEqual(hue, low)
            self.assertLessEqual(hue, high)

    async def test_action_semantics_survive_theme_hover_focus_and_disabled_states(self):
        app = self.app
        async with app.run_test(size=(120, 30)) as pilot:
            for theme in UI_THEMES:
                app.theme = theme
                await pilot.pause()
                for page, selector, family in (
                    ("run-tab", "#start-run", "cyan"),
                    ("profile-tab", "#profile-run", "yellow"),
                    ("images-tab", "#image-delete", "red"),
                ):
                    with self.subTest(theme=theme, selector=selector):
                        app._activate_tab(page)
                        button = app.query_one(selector, Button)
                        button.disabled = False
                        await pilot.pause()
                        original = button.styles.color
                        self.assert_color_family(original, family)
                        self.assertTrue(await pilot.hover(button))
                        await pilot.pause()
                        self.assertEqual(button.styles.color, original)
                        button.focus()
                        await pilot.pause()
                        self.assertEqual(button.styles.color, original)
                        self.assertTrue(button.styles.text_style.underline)
                        self.assertFalse(button.styles.text_style.reverse)
                        self.assertEqual(button.styles.background.a, 0)
                        button.disabled = True
                        await pilot.pause()
                        self.assert_color_family(button.styles.color, "gray")
                        self.assertFalse(button.styles.text_style.underline)
                        button.disabled = False
                        await pilot.pause()
                        self.assertEqual(button.styles.color, original)
                app._activate_tab("plot-tab")
                await pilot.pause()
                normal = app.query_one("#summarize-results", Button)
                foreground = app.current_theme.foreground
                self.assertEqual(normal.styles.color.hex, foreground.upper())
                await pilot.hover(normal)
                normal.focus()
                await pilot.pause()
                self.assertEqual(normal.styles.color.hex, foreground.upper())

    async def test_ready_success_and_failure_feedback_have_semantic_colors(self):
        app = self.app
        async with app.run_test(size=(80, 24)) as pilot:
            app._activate_tab("monitor-tab")
            for stage, family in (("服务就绪", "green"), ("已完成", "green"),
                                  ("正式测量", "cyan"), ("失败", "red")):
                with self.subTest(stage=stage):
                    app._render_snapshot(ProgressSnapshot(stage=stage))
                    await pilot.pause()
                    self.assert_color_family(app.query_one("#status-stage").styles.color, family)
            app._activate_tab("settings-tab")
            await pilot.pause()
            self.assertTrue(await pilot.click("#save-ui-settings"))
            await pilot.pause()
            status = app.query_one("#settings-status", Static)
            self.assertIn("已保存", str(status.content))
            self.assert_color_family(status.styles.color, "green")
            with patch("acprof.tui.app.save_settings", side_effect=OSError("read only")):
                self.assertTrue(await pilot.click("#save-ui-settings"))
                await pilot.pause()
            self.assertIn("保存失败", str(status.content))
            self.assert_color_family(status.styles.color, "red")

    async def test_confirmation_keeps_the_action_color_and_cancel_does_not_launch(self):
        app = self.app
        with patch.object(app, "_launch") as launch:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("f5")
                await pilot.pause()
                self.assert_color_family(app.screen.query_one("#confirm-yes").styles.color, "cyan")
                await pilot.press("escape")
                app._activate_tab("profile-tab")
                app.query_one("#result-dir", Input).value = str(app.settings_path.parent)
                await pilot.pause()
                self.assertTrue(await pilot.click("#profile-run"))
                await pilot.pause()
                self.assert_color_family(app.screen.query_one("#confirm-yes").styles.color, "yellow")
                await pilot.press("escape")
                app._process_kind = "run"
                app._set_busy(True)
                app.action_request_stop()
                await pilot.pause()
                self.assert_color_family(app.screen.query_one("#confirm-yes").styles.color, "red")
                await pilot.press("escape")
                launch.assert_not_called()
                app._process_kind = ""
                app._set_busy(False)

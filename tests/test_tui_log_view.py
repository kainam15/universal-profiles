from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.console import Console
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Tabs
from textual.widgets.text_area import Selection

from acprof.cli.tui import AcprofTui, LogPanel
from acprof.cli.tui_core import ProgressSnapshot, RunConfig
from acprof.cli.tui_log import SelectableLog
from acprof.cli.tui_scrollbar import SolidScrollBarRender


class TuiLogViewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings_path = Path(temporary.name) / "tui.json"

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)

    def assert_button_reachable(self, app, button_id):
        button = app.query_one(f"#{button_id}", Button)
        region = button.region
        self.assertGreater(region.width, 0, button_id)
        self.assertGreater(region.height, 0, button_id)
        self.assertGreaterEqual(region.x, 0, button_id)
        self.assertGreaterEqual(region.y, 0, button_id)
        self.assertLessEqual(region.right, app.size.width, button_id)
        self.assertLessEqual(region.bottom, app.size.height, button_id)
        self.assertIs(app.get_widget_at(region.x + 3, region.y)[0], button)

    async def test_matrix_stays_collapsed_during_run_and_log_has_reading_space(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            with self.subTest(size=size):
                app = self.make_app()
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    app._activate_tab("monitor-tab")
                    app._init_matrix_for_run(RunConfig(model="demo/model"))
                    app._render_snapshot(ProgressSnapshot(
                        stage="正式测量", current_case=1, total_cases=32,
                        measurement_active=True,
                    ))
                    await pilot.pause()
                    self.assertTrue(app.query_one("#matrix-board", Collapsible).collapsed)
                    log = app.query_one("#run-log", SelectableLog)
                    self.assertGreaterEqual(log.content_region.height, 5)
                    if size[1] >= 30:
                        self.assertGreaterEqual(log.content_region.height, 10)
                    self.assertLessEqual(log.region.bottom, app.query_one("#bottom-panel").region.y)
                    self.assert_button_reachable(app, "expand-log")

    async def test_expand_and_restore_keep_one_live_log_and_reachable_controls(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._activate_tab("monitor-tab")
            await pilot.pause()
            log = app.query_one("#run-log", SelectableLog)
            panel = app.query_one("#log-panel", LogPanel)
            normal_height = log.content_region.height
            log.clear().write("[build] first line")
            app.query_one("#stop-run", Button).disabled = False

            self.assertTrue(await pilot.click("#expand-log", offset=(3, 0)))
            await pilot.pause()
            self.assertIs(app.screen.maximized, panel)
            self.assertGreater(log.content_region.height, normal_height + 8)
            for button_id in ("copy-log", "follow-log", "restore-log", "stop-run"):
                self.assert_button_reachable(app, button_id)

            # Exercise the existing process-output delivery while maximized;
            # no collection command or external process may start in this test.
            with patch("acprof.tui.app.subprocess.Popen", side_effect=AssertionError("unexpected process")):
                app._process_started(12345, "test")
                app._consume_process_line("[build] second line", None, False)
                await pilot.pause()
                self.assertIs(app.query_one("#run-log", SelectableLog), log)
                self.assertIn("[build] second line", log.text)
                log.selection = Selection((0, 0), (0, len("[build] first line")))
                self.assertTrue(await pilot.click("#copy-log", offset=(3, 0)))
                await pilot.pause()
                self.assertEqual(app.clipboard, "[build] first line")
                self.assertFalse(log.following)
                self.assertTrue(await pilot.click("#follow-log", offset=(3, 0)))
                await pilot.pause()
                self.assertTrue(log.following)
                self.assertTrue(log.selection.is_empty)
                with patch.object(app, "action_request_stop") as stop:
                    self.assertTrue(await pilot.click("#stop-run", offset=(3, 0)))
                    await pilot.pause()
                    stop.assert_called_once_with()

            self.assertTrue(await pilot.click("#restore-log", offset=(3, 0)))
            await pilot.pause()
            self.assertIsNone(app.screen.maximized)
            self.assertIs(app.query_one("#run-log", SelectableLog), log)
            self.assertEqual(log.content_region.height, normal_height)
            self.assertIn("[build] second line", log.text)

    async def test_f8_and_escape_toggle_log_from_experiment_and_focused_log(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            panel = app.query_one("#log-panel", LogPanel)
            await pilot.press("f8")
            await pilot.pause()
            self.assertIs(app.screen.maximized, panel)
            self.assertIs(app.focused, app.query_one("#run-log", SelectableLog))
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsNone(app.screen.maximized)
            await pilot.press("f8")
            await pilot.pause()
            self.assertIs(app.screen.maximized, panel)
            await pilot.press("f8")
            await pilot.pause()
            self.assertIsNone(app.screen.maximized)

    async def test_resizing_rewraps_log_without_changing_copied_original_text(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("f8")
            await pilot.pause()
            log = app.query_one("#run-log", SelectableLog)
            raw = "[cmd] docker build --build-arg MODEL_ID=demo/model " + "中英文 payload " * 35
            log.clear().write(raw)
            log.selection = Selection((0, 0), log.document.end)
            await pilot.pause()
            narrow_rows = log.virtual_size.height
            self.assertGreater(narrow_rows, 3)
            await pilot.press("ctrl+c")
            self.assertEqual(app.clipboard, raw)
            await pilot.resize_terminal(150, 45)
            await pilot.pause()
            self.assertLess(log.virtual_size.height, narrow_rows)
            self.assertEqual(log.selected_text, raw)
            self.assertEqual(log.text, raw)
            self.assertTrue(await pilot.click("#copy-log", offset=(3, 0)))
            await pilot.pause()
            self.assertEqual(app.clipboard, raw)
            self.assertEqual(log.max_scroll_x, 0)

    async def test_real_scrollbars_use_solid_renderer_and_theme_backgrounds(self):
        app = self.make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._activate_tab("monitor-tab")
            await pilot.pause()
            log = app.query_one("#run-log", SelectableLog)
            log.wrap = False
            log.clear().write("\n".join(f"line {index:03d} " + "payload " * 30 for index in range(80)))
            for theme in ("acprof-dark", "acprof-light"):
                with self.subTest(theme=theme):
                    app.theme = theme
                    await pilot.pause()
                    active_tab = app.query_one(Tabs).active_tab
                    self.assertNotEqual(active_tab.styles.color, active_tab.styles.background)
                    scrollable_widgets = [widget for widget in app.query(Widget) if widget.is_scrollable]
                    self.assertTrue(scrollable_widgets)
                    for widget in scrollable_widgets:
                        for scrollbar in (widget.vertical_scrollbar, widget.horizontal_scrollbar):
                            self.assertIs(scrollbar.renderer, SolidScrollBarRender, repr(widget))
                        self.assertEqual(widget.styles.scrollbar_corner_color, widget.styles.scrollbar_background)
                        self.assertNotEqual(widget.styles.scrollbar_background.rgb, (0, 0, 0))
                    self.assertGreater(log.max_scroll_x, 0)
                    self.assertGreater(log.max_scroll_y, 0)
                    for scrollbar in (log.vertical_scrollbar, log.horizontal_scrollbar):
                        self.assertTrue(scrollbar.display)
                        console = Console(width=scrollbar.size.width, height=scrollbar.size.height)
                        segments = [segment for segment in console.render(scrollbar.render()) if segment.text != "\n"]
                        self.assertTrue(segments)
                        self.assertTrue(all(not segment.text.strip() for segment in segments))
                        self.assertTrue(all(not segment.style.reverse for segment in segments))
                        self.assertTrue(all(segment.style.bgcolor is not None for segment in segments))
                        self.assertTrue(all(segment.style.bgcolor.get_truecolor() != (0, 0, 0) for segment in segments))
                        self.assertIn("grab", [segment.style.meta.get("@mouse.down") for segment in segments])


if __name__ == "__main__":
    unittest.main()

import asyncio
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from rich.segment import Segment
from textual.app import App
from textual.driver import Driver
from textual.geometry import Offset
from textual.widgets import Button, Collapsible, Input

from acprof.tui.app import AcprofTui
from acprof.tui.views import ConfirmActionScreen
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.commands import RunConfig
from acprof.tui.input import BarCursorInput


class RecordingDriver(Driver):
    def __init__(self, app):
        super().__init__(app)
        self.output = []
        self.visibility_events = asyncio.Queue()

    def write(self, data):
        self.output.append(data)
        for control in re.findall(r"\x1b\[\?25([hl])", data):
            self.visibility_events.put_nowait((control == "h", time.monotonic()))

    def start_application_mode(self):
        self.output.append("START")

    def stop_application_mode(self):
        self.output.append("STOP")

    def disable_input(self):
        pass


class TuiInputTests(unittest.IsolatedAsyncioTestCase):
    def make_app(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return AcprofTui(
            RunConfig.smoke("demo/model"),
            settings_path=Path(temporary.name) / "tui.json",
        )

    async def test_ctrl_a_selects_all_and_typing_replaces_entire_value(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            with self.subTest(size=size):
                app = self.make_app()
                async with app.run_test(size=size) as pilot:
                    field = app.query_one("#model", Input)
                    for value, position in (
                        ("OpenMOSS-Team/MOSS-Transcribe-Diarize", 0),
                        ("模型/中文abc", 3),
                        ("模型abc/" * 24, 144),
                        ("", 0),
                    ):
                        with self.subTest(value=value):
                            field.value = value
                            field.cursor_position = position
                            await pilot.pause()
                            await pilot.press("ctrl+a")
                            self.assertEqual(field.selected_text, value)
                            self.assertEqual(field.selection, (0, len(value)))
                            self.assertEqual(field.value, value)
                            await pilot.press("X")
                            self.assertEqual(field.value, "X")

    async def test_ctrl_a_supports_delete_paste_and_home_navigation(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            field = app.query_one("#model", Input)
            for key in ("backspace", "delete"):
                with self.subTest(key=key):
                    field.value = "模型/abc"
                    await pilot.press("end", "ctrl+a", key)
                    self.assertEqual(field.value, "")

            field.value = "old/model"
            app.copy_to_clipboard("新的/model")
            await pilot.press("ctrl+a", "ctrl+a", "ctrl+v")
            self.assertEqual(field.value, "新的/model")
            await pilot.press("home")
            self.assertEqual(field.selection, (0, 0))
            await pilot.press("end")
            self.assertEqual(field.selection, (len(field.value), len(field.value)))
            await pilot.press("ctrl+shift+a")
            self.assertEqual(field.selected_text, field.value)

    async def test_caret_tracks_editing_wide_text_and_horizontal_scroll(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            with self.subTest(size=size):
                app = self.make_app()
                async with app.run_test(size=size) as pilot:
                    self.assertTrue(all(
                        isinstance(field, BarCursorInput) and not field.cursor_blink
                        for field in app.query(Input)
                    ))
                    field = app.query_one("#model", Input)
                    for value, position in (
                        ("", 0), ("abc", 0), ("abc", 1), ("abc", 3),
                        ("模型abc", 2), ("模型abc", 5), ("abcdef" * 20, 120),
                    ):
                        field.value = value
                        field.cursor_position = position
                        await pilot.pause()
                        expected = field.content_region.offset + Offset(
                            cell_len(value[:position]) - field.scroll_offset.x, 0,
                        )
                        self.assertEqual(app._input_cursor_offset(), expected)
                        self.assertTrue(field.content_region.contains(*expected))
                        self.assertEqual(field.value, value)

                    field.value = "abcd"
                    await pilot.click("#model", offset=(3, 1))
                    await pilot.press("home", "right", "X")
                    self.assertEqual(field.value, "aXbcd")
                    self.assertEqual(field.cursor_position, 2)
                    await pilot.press("shift+right", "shift+right", "Y")
                    self.assertEqual(field.value, "aXYd")
                    await pilot.resize_terminal(100, 28)
                    await pilot.pause()
                    self.assertEqual(
                        app._input_cursor_offset(),
                        field.content_region.offset + Offset(3, 0),
                    )

    async def test_cursor_does_not_overpaint_placeholder_or_selected_text(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            field = app.query_one("#model", Input)
            for value in ("", "abc", "模型abc"):
                field.value = value
                await pilot.press("ctrl+a")
                await pilot.pause()
                field._cursor_visible = True
                visible = list(Segment.simplify(field.render_line(0)))
                field._cursor_visible = False
                hidden = list(Segment.simplify(field.render_line(0)))
                self.assertEqual(visible, hidden)
                self.assertEqual(field.selected_text, value)

    async def test_cursor_hides_outside_visible_enabled_focused_input(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            field = app.query_one("#model", Input)
            await pilot.pause()
            self.assertIsNotNone(app._input_cursor_offset())
            app.query_one("#start-run", Button).focus()
            await pilot.pause()
            self.assertIsNone(app._input_cursor_offset())
            field.focus()
            await pilot.pause()
            app.app_focus = False
            self.assertIsNone(app._input_cursor_offset())
            app.app_focus = True
            app.query_one("#command-details", Collapsible).collapsed = False
            await pilot.pause()
            app.query_one("#run-form").scroll_end(animate=False, immediate=True)
            await pilot.pause()
            self.assertIsNone(app._input_cursor_offset())
            app.query_one("#run-form").scroll_home(animate=False, immediate=True)
            await pilot.pause()
            self.assertIsNotNone(app._input_cursor_offset())
            app.push_screen(ConfirmActionScreen("确认", "测试弹窗"))
            await pilot.pause()
            self.assertIsNone(app._input_cursor_offset())
            app.pop_screen()
            await pilot.pause()
            self.assertIsNotNone(app._input_cursor_offset())
            previous_offset = app._input_cursor_offset()
            field.disabled = True
            # Textual may immediately focus the next enabled input.
            self.assertIsNot(app.focused, field)
            self.assertNotEqual(app._input_cursor_offset(), previous_offset)

    async def test_terminal_output_shape_visibility_and_lifecycle(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            field = app.query_one("#model", Input)
            field.value = "abc"
            field.cursor_position = 1
            await pilot.pause()
            with patch.object(App, "get_driver_class", return_value=RecordingDriver):
                driver = app.get_driver_class()(app)
            with patch.object(app, "_driver", driver):
                app._begin_update()
                app._end_update()
                caret = app._input_cursor_offset()
                self.assertEqual(driver.output, [
                    f"\x1b[6 q\x1b[{caret.y + 1};{caret.x + 1}H\x1b[?25h",
                ])
                app._begin_update()
                app._end_update()
                self.assertEqual(driver.output[1], "\x1b[?25l")
                self.assertNotIn("\x1b[6 q", driver.output[2])
                with patch.object(app, "_input_cursor_offset", return_value=None):
                    app._begin_update()
                    app._end_update()
                self.assertEqual(driver.output[-1], "\x1b[?25l")

                driver.suspend_application_mode()
                self.assertEqual(driver.output[-2:], ["\x1b[?25l\x1b[0 q", "STOP"])
                self.assertIsNone(app._input_cursor_timer)
                output = driver.output.copy()
                app._sync_input_cursor()
                self.assertEqual(driver.output, output)
                driver.resume_application_mode()
                app._begin_update()
                app._end_update()
                self.assertIn("\x1b[6 q", driver.output[-1])
                driver.stop_application_mode()
                self.assertEqual(driver.output[-2:], ["\x1b[?25l\x1b[0 q", "STOP"])
                with patch.object(app, "_reset_input_cursor", side_effect=OSError):
                    with self.assertRaises(OSError):
                        driver.stop_application_mode()
                self.assertEqual(driver.output[-1], "STOP")

    async def test_idle_cursor_repeatedly_blinks_without_widget_redraws(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.2)
            driver = RecordingDriver(app)
            with patch.object(app, "_driver", driver):
                try:
                    app._sync_input_cursor()
                    visible, previous_time = driver.visibility_events.get_nowait()
                    self.assertTrue(visible)
                    field = app.query_one("#model", Input)
                    with (
                        patch.object(app, "_display", wraps=app._display) as display,
                        patch.object(field, "refresh", wraps=field.refresh) as refresh,
                    ):
                        for expected in (False, True, False, True):
                            visible, changed_at = await asyncio.wait_for(
                                driver.visibility_events.get(), timeout=2,
                            )
                            self.assertEqual(visible, expected)
                            self.assertGreater(changed_at - previous_time, 0.3)
                            self.assertLess(changed_at - previous_time, 1.2)
                            previous_time = changed_at
                        display.assert_not_called()
                        refresh.assert_not_called()
                finally:
                    app._reset_input_cursor()

    async def test_redraw_preserves_hidden_phase_and_edit_or_click_restarts_it(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            field = app.query_one("#model", Input)
            with patch.object(app, "_driver", RecordingDriver(app)):
                try:
                    app._sync_input_cursor()
                    timer = app._input_cursor_timer
                    app._toggle_input_cursor()
                    self.assertFalse(app._input_cursor_visible)
                    app._begin_update()
                    app._end_update()
                    self.assertFalse(app._input_cursor_visible)
                    self.assertIs(app._input_cursor_timer, timer)
                    field.value = "abc"
                    app._sync_input_cursor()
                    self.assertTrue(app._input_cursor_visible)
                    self.assertIsNot(app._input_cursor_timer, timer)
                    app._toggle_input_cursor()
                    self.assertFalse(app._input_cursor_visible)
                    await pilot.click("#model", offset=(3, 1))
                    self.assertTrue(app._input_cursor_visible)
                    app._toggle_input_cursor()
                    await pilot.press("X")
                    self.assertTrue(app._input_cursor_visible)
                    app.query_one("#quit-app", Button).focus()
                    await pilot.pause()
                    self.assertIsInstance(app.focused, Button)
                    app._sync_input_cursor()
                    self.assertFalse(app._input_cursor_visible)
                    self.assertIsNone(app._input_cursor_timer)
                finally:
                    app._reset_input_cursor()

    async def test_measurement_window_stops_cursor_timer_and_resumes_afterward(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "_driver", RecordingDriver(app)):
                try:
                    app._sync_input_cursor()
                    self.assertIsNotNone(app._input_cursor_timer)
                    app._consume_process_line(
                        "", ProgressSnapshot(measurement_active=True), False,
                    )
                    self.assertIsNone(app._input_cursor_timer)
                    self.assertTrue(app._input_cursor_visible)
                    app._toggle_input_cursor()
                    self.assertTrue(app._input_cursor_visible)
                    app._consume_process_line(
                        "", ProgressSnapshot(measurement_active=False), False,
                    )
                    self.assertIsNotNone(app._input_cursor_timer)
                    app.set_input_cursor_blink_enabled(False)
                    app._set_busy(False)
                    self.assertTrue(app._input_cursor_blink_enabled)
                finally:
                    app._reset_input_cursor()

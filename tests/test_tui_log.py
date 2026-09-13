import unittest

from textual.app import App, ComposeResult
from textual.scrollbar import ScrollTo, ScrollUp
from textual.widgets.text_area import Selection

from acprof.tui.log import SelectableLog


class LogApp(App):
    CSS = "SelectableLog { width: 1fr; height: 1fr; border: none; padding: 0; }"

    def __init__(self, **log_options):
        super().__init__()
        self.log_options = log_options

    def compose(self) -> ComposeResult:
        yield SelectableLog(id="log", **self.log_options)


class SelectableLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_mouse_drag_and_ctrl_c_copy_original_text_read_only(self):
        app = LogApp()
        async with app.run_test(size=(50, 10)) as pilot:
            log = app.query_one(SelectableLog)
            log.write("alpha beta gamma")
            await pilot.pause()
            await pilot.mouse_down(log, offset=(0, 0))
            await pilot.hover(log, offset=(10, 0))
            await pilot.mouse_up(log, offset=(10, 0))
            await pilot.pause()
            self.assertEqual(log.selected_text, "alpha beta")
            self.assertFalse(log.following)
            await pilot.press("ctrl+c")
            self.assertEqual(app.clipboard, "alpha beta")
            before = log.text
            await pilot.press("x", "backspace", "ctrl+x", "ctrl+v", "ctrl+z")
            self.assertEqual(log.text, before)

    async def test_ctrl_a_and_double_click_select_text(self):
        app = LogApp()
        async with app.run_test(size=(50, 10)) as pilot:
            log = app.query_one(SelectableLog)
            log.write("alpha beta gamma\nnext line")
            await pilot.pause()
            await pilot.click(log, offset=(7, 0), times=2)
            await pilot.pause()
            self.assertEqual(log.selected_text, "beta")
            await pilot.press("ctrl+a", "ctrl+c")
            self.assertEqual(app.clipboard, "alpha beta gamma\nnext line")

    async def test_wrapping_and_resize_preserve_original_line_and_selection(self):
        app = LogApp()
        async with app.run_test(size=(40, 10)) as pilot:
            log = app.query_one(SelectableLog)
            raw = "[cmd] docker build --build-arg MODEL_ID=模型/例子 " * 5
            log.write(raw)
            await pilot.pause()
            narrow_rows = log.virtual_size.height
            self.assertGreater(narrow_rows, 1)
            self.assertEqual(log.max_scroll_x, 0)
            log.selection = Selection((0, 0), log.document.end)
            await pilot.press("ctrl+c")
            self.assertEqual(app.clipboard, raw)
            await pilot.resize_terminal(100, 15)
            await pilot.pause()
            self.assertLess(log.virtual_size.height, narrow_rows)
            self.assertEqual(log.selected_text, raw)
            self.assertEqual(tuple(log.lines), (raw,))
            log.wrap = False
            await pilot.pause()
            self.assertGreater(log.max_scroll_x, 0)
            log.wrap = True
            await pilot.pause()
            self.assertEqual(log.max_scroll_x, 0)
            self.assertEqual(log.selected_text, raw)

    async def test_new_output_preserves_reading_view_until_explicit_follow(self):
        app = LogApp()
        async with app.run_test(size=(60, 10)) as pilot:
            log = app.query_one(SelectableLog)
            log.write("\n".join(f"line {index:02}" for index in range(50)))
            await pilot.pause()
            self.assertTrue(log.is_vertical_scroll_end)
            log.scroll_to(y=12, animate=False, immediate=True)
            self.assertFalse(log.following)
            log.selection = Selection((12, 0), (13, 7))
            selected = log.selected_text
            old_y = log.scroll_y
            log.write("new output\nmore output")
            await pilot.pause()
            self.assertEqual(log.selected_text, selected)
            self.assertEqual(log.scroll_y, old_y)
            log.follow_tail()
            await pilot.pause()
            self.assertTrue(log.following)
            self.assertTrue(log.selection.is_empty)
            self.assertTrue(log.is_vertical_scroll_end)
            log.write("latest")
            await pilot.pause()
            self.assertTrue(log.is_vertical_scroll_end)

    async def test_scroll_intent_then_immediate_output_keeps_requested_position(self):
        app = LogApp()
        async with app.run_test(size=(60, 10)) as pilot:
            log = app.query_one(SelectableLog)
            log.write("\n".join(f"line {index:02}" for index in range(80)))
            await pilot.pause()
            actions = (
                log.action_cursor_up,
                log.action_cursor_page_up,
                log.action_cursor_line_start,
                lambda: log._on_scroll_up(ScrollUp()),
                lambda: log._on_scroll_to(ScrollTo(y=20)),
            )
            for action in actions:
                log.follow_tail()
                await pilot.pause()
                action()
                expected_y = log.scroll_y
                self.assertFalse(log.following)
                self.assertLess(expected_y, log.max_scroll_y)
                # No await: output arrives before another frame or animation.
                log.write("output arriving during navigation")
                await pilot.pause()
                self.assertEqual(log.scroll_y, expected_y)
                self.assertFalse(log.following)
            log.action_cursor_down()
            log.action_cursor_page_down()
            log.action_cursor_line_end()
            self.assertTrue(log.is_vertical_scroll_end)
            self.assertFalse(log.following)
            last_y = log.scroll_y
            log.write("End must not implicitly resume following")
            await pilot.pause()
            self.assertEqual(log.scroll_y, last_y)
            self.assertFalse(log.following)
            log.follow_tail()
            log.write("explicitly resumed")
            await pilot.pause()
            self.assertTrue(log.following)
            self.assertTrue(log.is_vertical_scroll_end)

    async def test_retention_trims_immediately_without_losing_surviving_selection(self):
        app = LogApp(max_lines=20)
        async with app.run_test(size=(50, 8)) as pilot:
            log = app.query_one(SelectableLog)
            log.write("\n".join(f"line {index:02}" for index in range(20)))
            await pilot.pause()
            log.scroll_to(y=8, animate=False, immediate=True)
            log.selection = Selection((10, 0), (11, 7))
            selected = log.selected_text
            old_y = log.scroll_y
            log.write("line 20\nline 21")
            await pilot.pause()
            self.assertEqual(len(log.lines), 20)
            self.assertEqual(log.lines[0], "line 02")
            self.assertEqual(log.selected_text, selected)
            self.assertEqual(log.scroll_y, old_y - 2)
            log.max_lines = 12
            await pilot.pause()
            self.assertEqual(len(log.lines), 12)
            self.assertEqual(log.lines[0], "line 10")
            self.assertEqual(log.selected_text, selected)
            self.assertEqual(log.history.undo_stack, [])
            self.assertEqual(log.history.redo_stack, [])
            self.assertFalse(log.cursor_blink)
            self.assertFalse(log.show_cursor)

    async def test_copy_all_and_clear_keep_real_newlines(self):
        app = LogApp()
        async with app.run_test(size=(40, 10)) as pilot:
            log = app.query_one(SelectableLog)
            self.assertFalse(log.copy_all())
            log.write("first\n\n").write("second\n")
            await pilot.pause()
            self.assertTrue(log.copy_all())
            self.assertEqual(app.clipboard, "first\n\nsecond")
            log.clear()
            await pilot.pause()
            self.assertEqual(tuple(log.lines), ())
            self.assertEqual(log.text, "")
            self.assertTrue(log.following)
            self.assertFalse(log.copy_selection())
            self.assertFalse(log.copy_all())


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from textual.widgets import Checkbox, Input, Select

from acprof.tui.app import AcprofTui
from acprof.tui.views import ConfirmActionScreen
from acprof.tui.commands import RunConfig


class TuiProfileToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.result_dir = Path(temporary.name)
        self.settings_path = self.result_dir / "tui.json"

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)

    async def click_visible(self, app, pilot, selector):
        widget = app.screen.query_one(selector)
        widget.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        self.assertTrue(await pilot.click(selector, offset=(2, 1)))
        await pilot.pause()

    async def test_mouse_and_space_selection_drive_plan_and_confirmed_run(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._activate_tab("profile-tab")
            app.query_one("#result-dir", Input).value = str(self.result_dir)
            await pilot.pause()
            checkboxes = list(app.query("#profile-tools Checkbox"))
            self.assertEqual(len(checkboxes), 4)
            self.assertEqual([box.name for box in checkboxes if box.value], ["torch", "ncu"])
            for tool in ("torch", "ncu", "nsys"):
                await self.click_visible(app, pilot, f"#profile-tool-{tool}")
            app.query_one("#profile-tool-massif", Checkbox).focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()

            with patch.object(app, "_launch") as launch:
                await self.click_visible(app, pilot, "#profile-dry-run")
                launch.assert_called_once()
                planned = launch.call_args.args[0]
                self.assertEqual(planned.kind, "profile-dry-run")
                self.assertEqual(planned.command[planned.command.index("--tools") + 1], "nsys,massif")
                self.assertIn("--dry-run", planned.command)

                launch.reset_mock()
                await self.click_visible(app, pilot, "#profile-run")
                self.assertIsInstance(app.screen, ConfirmActionScreen)
                pending = app._pending_launch
                self.assertEqual(pending.command[pending.command.index("--tools") + 1], "nsys,massif")
                self.assertNotIn("--dry-run", pending.command)
                launch.assert_not_called()
                await self.click_visible(app, pilot, "#confirm-yes")
                launch.assert_called_once_with(pending)

            for tool in ("torch", "ncu"):
                await self.click_visible(app, pilot, f"#profile-tool-{tool}")
            command = app._profile_command(dry_run=True)
            self.assertEqual(command[command.index("--tools") + 1], "torch,ncu,nsys,massif")

    async def test_empty_selection_blocks_launch_but_explicit_slash_tools_still_work(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._activate_tab("profile-tab")
            app.query_one("#result-dir", Input).value = str(self.result_dir)
            await pilot.pause()
            self.assertEqual(len(app.query("#profile-tools Checkbox")), 4)
            for tool in ("torch", "ncu"):
                await self.click_visible(app, pilot, f"#profile-tool-{tool}")
            with patch.object(app, "_launch") as launch, patch.object(app, "notify") as notify:
                for button in ("profile-dry-run", "profile-run"):
                    await self.click_visible(app, pilot, f"#{button}")
                    launch.assert_not_called()
                    self.assertIsNone(app._pending_launch)
                    self.assertIn("请至少勾选一个补采工具", str(notify.call_args))
                command_input = app.query_one("#slash-command", Input)
                command_input.value = f"/profile {self.result_dir} nsys,massif"
                command_input.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                launch.assert_called_once()
                command = launch.call_args.args[0].command
                self.assertEqual(command[command.index("--tools") + 1], "nsys,massif")

    async def test_options_fit_after_resize_and_language_change_and_lock_during_run(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            self.assertEqual(len(app.query("#profile-tools Checkbox")), 4)
            for size in ((150, 45), (120, 30), (80, 24)):
                await pilot.resize_terminal(*size)
                for language in ("en", "zh"):
                    with self.subTest(size=size, language=language):
                        app.action_show_settings()
                        app.query_one("#ui-language", Select).value = language
                        await pilot.pause()
                        app._activate_tab("profile-tab")
                        await pilot.pause()
                        for tool in ("torch", "ncu", "nsys", "massif"):
                            box = app.query_one(f"#profile-tool-{tool}", Checkbox)
                            box.scroll_visible(animate=False, immediate=True)
                            await pilot.pause()
                            self.assertGreaterEqual(box.region.x, 0)
                            self.assertLessEqual(box.region.right, size[0])
                            self.assertGreaterEqual(box.region.y, 0)
                            self.assertLessEqual(box.region.bottom, app.query_one("#bottom-panel").region.y)
                            self.assertLessEqual(cell_len(box.label.plain) + 4, box.content_region.width)
                            self.assertIs(app.get_widget_at(box.region.x + 2, box.region.y + 1)[0], box)
                            previous = box.value
                            self.assertTrue(await pilot.click(box, offset=(2, 1)))
                            await pilot.pause()
                            self.assertEqual(box.value, not previous)
            app._set_busy(True)
            self.assertTrue(all(box.disabled for box in app.query("#profile-tools Checkbox")))
            app._set_busy(False)
            self.assertTrue(all(not box.disabled for box in app.query("#profile-tools Checkbox")))


if __name__ == "__main__":
    unittest.main()

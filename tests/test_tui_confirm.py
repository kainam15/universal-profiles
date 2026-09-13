from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import Button

from acprof.tui.app import AcprofTui
from acprof.tui.views import ConfirmActionScreen
from acprof.tui.commands import RunConfig


class TuiConfirmTests(unittest.IsolatedAsyncioTestCase):
    def make_app(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return AcprofTui(
            RunConfig.smoke("demo/model"),
            settings_path=Path(temporary.name) / "tui.json",
        )

    def assert_highlighted(self, button, highlighted):
        self.assertEqual(bool(button.styles.text_style.underline), highlighted)
        self.assertFalse(button.styles.text_style.reverse)
        self.assertEqual(button.styles.background.a, 0)

    async def test_mouse_highlight_only_lasts_while_hovering_confirmation(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            for theme in ("acprof-dark", "acprof-light"):
                with self.subTest(size=size, theme=theme):
                    app = self.make_app()
                    app.theme = theme
                    with patch.object(app, "_launch") as launch:
                        async with app.run_test(size=size) as pilot:
                            await pilot.pause()
                            self.assertTrue(await pilot.click("#start-run"))
                            await pilot.pause()
                            self.assertIsInstance(app.screen, ConfirmActionScreen)
                            cancel = app.screen.query_one("#confirm-no", Button)
                            confirm = app.screen.query_one("#confirm-yes", Button)
                            self.assert_highlighted(cancel, False)
                            self.assert_highlighted(confirm, False)
                            self.assertIsNone(app.screen.focused)

                            for hovered, other in ((cancel, confirm), (confirm, cancel)):
                                self.assertTrue(await pilot.hover(hovered))
                                await pilot.pause()
                                self.assert_highlighted(hovered, True)
                                self.assert_highlighted(other, False)
                                self.assertIsNone(app.screen.focused)
                                self.assertTrue(await pilot.hover("#confirm-title"))
                                await pilot.pause()
                                self.assert_highlighted(hovered, False)

                            # Resuming or resizing must not restore automatic focus.
                            app.app_focus = False
                            await pilot.pause()
                            app.app_focus = True
                            await pilot.resize_terminal(120, 30)
                            await pilot.pause()
                            self.assertIsNone(app.screen.focused)
                            self.assert_highlighted(cancel, False)
                            self.assert_highlighted(confirm, False)
                            self.assertTrue(await pilot.click(cancel))
                            await pilot.pause()
                            self.assertNotIsInstance(app.screen, ConfirmActionScreen)
                            self.assertIsNone(app._pending_launch)
                            launch.assert_not_called()

    async def test_keyboard_can_select_cancel_or_confirm_after_opening_dialog(self):
        app = self.make_app()
        with patch.object(app, "_launch") as launch:
            async with app.run_test(size=(120, 30)) as pilot:
                await pilot.press("f5")
                await pilot.pause()
                dialog = app.screen
                self.assertIsInstance(dialog, ConfirmActionScreen)
                await pilot.press("enter", "space")
                await pilot.pause()
                self.assertIs(app.screen, dialog)
                launch.assert_not_called()

                cancel = dialog.query_one("#confirm-no", Button)
                confirm = dialog.query_one("#confirm-yes", Button)
                for key, focused, other in (
                    ("tab", cancel, confirm),
                    ("tab", confirm, cancel),
                    ("shift+tab", cancel, confirm),
                ):
                    await pilot.press(key)
                    await pilot.pause()
                    self.assertIs(dialog.focused, focused)
                    self.assert_highlighted(focused, True)
                    self.assert_highlighted(other, False)
                await pilot.press("enter")
                await pilot.pause()
                self.assertNotIsInstance(app.screen, ConfirmActionScreen)
                self.assertIsNone(app._pending_launch)
                launch.assert_not_called()

                await pilot.press("f5", "escape")
                await pilot.pause()
                self.assertNotIsInstance(app.screen, ConfirmActionScreen)
                self.assertIsNone(app._pending_launch)
                launch.assert_not_called()

                await pilot.press("f5", "tab", "tab")
                await pilot.pause()
                self.assertEqual(app.screen.focused.id, "confirm-yes")
                await pilot.press("enter")
                await pilot.pause()
                self.assertNotIsInstance(app.screen, ConfirmActionScreen)
                self.assertIsNone(app._pending_launch)
                launch.assert_called_once()
                self.assertEqual(launch.call_args.args[0].kind, "run")

    async def test_mouse_can_confirm_without_prior_keyboard_selection(self):
        app = self.make_app()
        with patch.object(app, "_launch") as launch:
            async with app.run_test(size=(120, 30)) as pilot:
                await pilot.pause()
                self.assertTrue(await pilot.click("#start-run"))
                await pilot.pause()
                self.assertTrue(await pilot.click("#confirm-yes"))
                await pilot.pause()
                self.assertNotIsInstance(app.screen, ConfirmActionScreen)
                self.assertIsNone(app._pending_launch)
                launch.assert_called_once()
                self.assertEqual(launch.call_args.args[0].kind, "run")


if __name__ == "__main__":
    unittest.main()

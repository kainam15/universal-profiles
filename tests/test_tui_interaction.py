import shlex
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from textual.widgets import Button, Input, Select, Static

from acprof.cli.tui import AcprofTui
from acprof.cli.tui_core import ProgressSnapshot, RunConfig, format_command


PROJECT_DIR = Path(__file__).resolve().parents[1]


class TuiInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_rapid_clicks_are_not_discarded_by_button_active_effect(self):
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 45)) as pilot:
            app._activate_tab("settings-tab")
            await pilot.pause()
            button = app.query_one("#restore-ui-defaults", Button)
            # Deliver two Click events in the same event-loop turn. Pilot
            # deliberately pauses between events, which can hide this issue.
            with patch.object(button, "press", wraps=button.press) as press:
                await button._on_click(Mock())
                await button._on_click(Mock())
                self.assertEqual(press.call_count, 2)
            await pilot.pause()
            self.assertFalse(button.has_class("-active"))
            self.assertTrue(all(not field.cursor_blink for field in app.query(Input)))

    async def test_burst_of_form_changes_collects_once_and_keeps_latest_values(self):
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            with patch.object(app, "_collect_config", wraps=app._collect_config) as collect:
                app.query_one("#cpus", Input).value = "1,2"
                app.query_one("#cpus", Input).value = "1,3"
                app.query_one("#model", Input).value = "demo/latest"
                await pilot.pause(0.12)
                self.assertEqual(collect.call_count, 1)
            preview = str(app.query_one("#command-preview", Static).render())
            self.assertIn("--cpus 1,3", preview)
            self.assertIn("--model demo/latest", preview)
            self.assertEqual(app.query_one("#run-preset", Select).value, "custom")

    async def test_applying_preset_does_not_queue_redundant_preview_updates(self):
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            with patch.object(
                app, "_refresh_command_preview", wraps=app._refresh_command_preview
            ) as refresh:
                app._apply_config(RunConfig.main_matrix("demo/model"), preset="main")
                await pilot.pause(0.12)
                self.assertEqual(refresh.call_count, 1)
            self.assertIsNone(app._preview_timer)
            self.assertEqual(app.query_one("#run-preset", Select).value, "main")
            self.assertEqual(app.query_one("#cpus", Input).value, "1,2,4,8")

    async def test_starting_run_cancels_preview_and_locks_configuration(self):
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            with patch.object(app, "_refresh_command_preview") as refresh:
                app._configuration_changed()
                self.assertIsNotNone(app._preview_timer)
                app._set_busy(True)
                self.assertIsNone(app._preview_timer)
                await pilot.pause(0.12)
                refresh.assert_not_called()
            controls = list(app.query(
                ".config-control, #run-preset, .ui-preference, "
                "#save-run-default, #restore-ui-defaults, #save-ui-settings"
            ))
            self.assertTrue(controls)
            self.assertTrue(all(control.disabled for control in controls))
            app._set_busy(False)
            self.assertTrue(all(not control.disabled for control in controls))

    async def test_measurement_pauses_elapsed_timer_and_resumes_after_window(self):
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            timer = Mock()
            app._elapsed_timer = timer
            with patch.object(app, "_render_snapshot"):
                app._consume_process_line(
                    "", ProgressSnapshot(stage="正式测量", measurement_active=True), True
                )
                timer.pause.assert_called_once_with()
                timer.resume.assert_not_called()
                with patch.object(app.query_one("#status-elapsed", Static), "update") as update:
                    app._tick_elapsed()
                    update.assert_not_called()
                app._consume_process_line(
                    "", ProgressSnapshot(stage="清理 case", measurement_active=False), True
                )
                timer.resume.assert_called_once_with()
            app._elapsed_timer = None


class CommandPreviewTests(unittest.TestCase):
    def test_preview_only_resolves_entrypoint_paths_and_preserves_shell_arguments(self):
        command = [
            str(PROJECT_DIR / ".venv/bin/python"), "-u", str(PROJECT_DIR / "run.py"),
            "--model", "demo/model", "--output-dir", "/remote mount/results;literal",
            "--input-scales", "64,128",
        ]
        original_resolve = Path.resolve
        with patch.object(Path, "resolve", autospec=True, side_effect=original_resolve) as resolve:
            preview = format_command(command, project_dir=PROJECT_DIR)
        self.assertEqual(resolve.call_count, 2)
        expected = command.copy()
        expected[2] = "run.py"
        self.assertEqual(shlex.split(preview), expected)


if __name__ == "__main__":
    unittest.main()

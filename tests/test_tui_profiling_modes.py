import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from textual.widgets import ContentSwitcher, Select, TabbedContent

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig, build_run_command
from acprof.tui.diagnostics import quick_preflight


class TuiProfilingModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_refresh_timer_update_pauses_after_tabs_are_unmounted(self):
        with tempfile.TemporaryDirectory() as root:
            app = AcprofTui(RunConfig(model="test/model"), settings_path=Path(root) / "settings.json")
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                app._image_refresh_timer.pause()
                app._image_refresh_timer = Mock()
                try:
                    await app.query_one("#main-tabs", TabbedContent).remove()
                    app._sync_image_refresh_timer()
                finally:
                    app._form_ready = False
                app._image_refresh_timer.pause.assert_called()
                app._image_refresh_timer.reset.assert_not_called()

    async def test_mode_selector_updates_command_in_both_languages_and_sizes(self):
        for language in ("zh", "en"):
            for size in ((80, 24), (120, 30), (150, 45)):
                with self.subTest(language=language, size=size), tempfile.TemporaryDirectory() as root:
                    app = AcprofTui(RunConfig(model="test/model"), settings_path=Path(root) / "settings.json")
                    async with app.run_test(size=size) as pilot:
                        app.query_one("#ui-language", Select).value = language
                        await pilot.pause()
                        control = app.query_one("#profiling-mode", Select)
                        app.query_one("#main-tabs", TabbedContent).active = "run-tab"
                        app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
                        await pilot.pause()
                        control.scroll_visible(immediate=True, force=True)
                        control.focus()
                        await pilot.press("enter", "down", "enter")
                        await pilot.pause()
                        self.assertEqual(control.value, "basic")
                        self.assertGreater(control.region.height, 0)
                        config = app._collect_config()
                        self.assertEqual(config.profiling_mode, "basic")
                        command = build_run_command(config, project_dir=Path.cwd())
                        self.assertEqual(command[command.index("--profiling-mode") + 1], "basic")

    def test_basic_diagnostics_do_not_probe_unrequested_energy_or_perf(self):
        with tempfile.TemporaryDirectory() as root, patch("acprof.tui.diagnostics.probe_cpu_energy", side_effect=AssertionError("RAPL not requested")), patch(
            "acprof.tui.diagnostics.probe_perf_instructions", side_effect=AssertionError("perf not requested")
        ), patch("acprof.tui.diagnostics.shutil.which", return_value=None):
            checks = quick_preflight(RunConfig(model="test", profiling_mode="basic", gpus="off"), project_dir=root)
        for label in ("CPU RAPL", "perf instructions"):
            check = next(item for item in checks if item.label == label)
            self.assertEqual(check.capability_status, "not_requested")
            self.assertIn("not_requested", check.detail)

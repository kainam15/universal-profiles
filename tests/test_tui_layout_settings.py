from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import (
    Button, Checkbox, Collapsible, ContentSwitcher, Input, RichLog, Select, TabbedContent,
)

from acprof.cli.tui import AcprofTui, main
from acprof.cli.tui_core import ProgressSnapshot, RunConfig
from acprof.cli.tui_settings import (
    TuiSettings, UiPreferences, default_settings_path, load_settings, save_settings,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class TuiLayoutSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings_path = Path(self.temporary.name) / "tui.json"

    def assert_button_reachable(self, app, button_id):
        button = app.query_one(f"#{button_id}", Button)
        region = button.region
        bottom = app.query_one("#bottom-panel").region.y
        self.assertGreater(region.width, 0, button_id)
        self.assertGreater(region.height, 0, button_id)
        self.assertGreaterEqual(region.x, 0, button_id)
        self.assertGreaterEqual(region.y, 0, button_id)
        self.assertLessEqual(region.right, app.size.width, button_id)
        self.assertLessEqual(region.bottom, bottom, button_id)
        center = (region.x + region.width // 2, region.y + region.height // 2)
        widget, _ = app.get_widget_at(*center)
        self.assertIs(widget, button, f"{button_id} is obscured by {widget}")

    async def remember_experiment(self, app, pilot):
        app._activate_tab("run-tab")
        await pilot.pause()
        if app.query_one("#experiment-pages", ContentSwitcher).current != "advanced-form":
            self.assertTrue(await pilot.click("#open-run-settings", offset=(3, 1)))
            await pilot.pause()
        app.query_one("#advanced-form").scroll_end(animate=False, immediate=True)
        await pilot.pause()
        self.assert_button_reachable(app, "save-run-default")
        self.assertTrue(await pilot.click("#save-run-default", offset=(3, 1)))
        await pilot.pause()

    async def test_action_bar_stays_reachable_and_advanced_settings_click_works(self):
        for size in ((120, 30), (80, 24), (150, 45)):
            with self.subTest(size=size):
                app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    action_region = app.query_one("#run-actions").region
                    for button_id in ("open-run-settings", "quick-check", "probe-largest", "start-run"):
                        self.assert_button_reachable(app, button_id)
                    app.query_one("#command-details", Collapsible).collapsed = False
                    app.query_one("#run-form").scroll_end(animate=False, immediate=True)
                    await pilot.pause()
                    self.assertEqual(app.query_one("#run-actions").region, action_region)
                    for button_id in ("open-run-settings", "quick-check", "probe-largest", "start-run"):
                        self.assert_button_reachable(app, button_id)
                    self.assertTrue(await pilot.click("#open-run-settings", offset=(3, 1)))
                    await pilot.pause()
                    self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "run-tab")
                    self.assertEqual(app.query_one("#experiment-pages", ContentSwitcher).current, "advanced-form")
                    for button_id in ("open-run-settings", "quick-check", "probe-largest", "start-run"):
                        self.assert_button_reachable(app, button_id)
                    self.assertEqual(str(app.query_one("#open-run-settings", Button).label), "返回基本配置")
                    app.query_one("#advanced-form").scroll_end(animate=False, immediate=True)
                    await pilot.pause()
                    self.assertEqual(app.query_one("#run-actions").region, action_region)
                    self.assert_button_reachable(app, "save-run-default")
                    self.assertTrue(await pilot.click("#open-run-settings", offset=(3, 1)))
                    await pilot.pause()
                    self.assertEqual(app.query_one("#experiment-pages", ContentSwitcher).current, "run-form")
                    self.assertEqual(str(app.query_one("#open-run-settings", Button).label), "高级参数")
                    await pilot.press("f2")
                    await pilot.pause()
                    for button_id in ("restore-ui-defaults", "save-ui-settings"):
                        self.assert_button_reachable(app, button_id)

    async def test_f2_changes_page_from_focused_input(self):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            app.query_one("#model", Input).focus()
            await pilot.pause()
            self.assertIs(app.focused, app.query_one("#model", Input))
            await pilot.press("f2")
            await pilot.pause()
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "settings-tab")
            settings_tab = app.query_one("#settings-tab")
            self.assertEqual(len(settings_tab.query(".config-control")), 0)
            self.assertEqual(len(settings_tab.query("#save-run-default")), 0)
            self.assertEqual(len(settings_tab.query(".ui-preference")), 4)
            self.assertEqual(app.query_one("#experiment-pages", ContentSwitcher).current, "run-form")

    async def test_monitor_matrix_toggle_keeps_log_and_stop_button_reachable(self):
        for size in ((120, 30), (80, 24), (150, 45)):
            with self.subTest(size=size):
                config = RunConfig(model="demo/model")
                app = AcprofTui(config, settings_path=self.settings_path)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    app._activate_tab("monitor-tab")
                    app._init_matrix_for_run(config)
                    app._render_snapshot(ProgressSnapshot(
                        stage="正式测量", detail="正在执行工作负载", current_case=1,
                        total_cases=32, cpu="1", mem="2", gpu="off", measurement_active=True,
                    ))
                    app.query_one("#stop-run", Button).disabled = False
                    log = app.query_one("#run-log", RichLog)
                    log.write("[case] Running workload...")
                    for collapsed in (False, True):
                        with self.subTest(collapsed=collapsed):
                            app.query_one("#matrix-board", Collapsible).collapsed = collapsed
                            await pilot.pause()
                            self.assert_button_reachable(app, "stop-run")
                            self.assert_button_reachable(app, "clear-log")
                            self.assertGreaterEqual(log.region.height, 3)
                            self.assertLessEqual(log.region.bottom, app.query_one("#bottom-panel").region.y)
                            center = (
                                log.content_region.x + log.content_region.width // 2,
                                log.content_region.y + log.content_region.height // 2,
                            )
                            self.assertIs(app.get_widget_at(*center)[0], log)
                            with patch.object(app, "action_request_stop") as stop:
                                self.assertTrue(await pilot.click("#stop-run", offset=(3, 1)))
                                await pilot.pause()
                                stop.assert_called_once_with()

    async def test_wrapped_log_fits_narrow_terminal_without_horizontal_overflow(self):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._activate_tab("monitor-tab")
            await pilot.pause()
            log = app.query_one("#run-log", RichLog)
            log.write("[case] 正在执行确定性输入请求 processing deterministic workload " * 8)
            await pilot.pause()
            self.assertTrue(log.wrap)
            self.assertGreater(len(log.lines), 1)
            self.assertLessEqual(log.virtual_size.width, log.scrollable_content_region.width)
            self.assertEqual(log.max_scroll_x, 0)

    async def test_empty_model_can_be_remembered_without_allowing_collection(self):
        app = AcprofTui(settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            await self.remember_experiment(app, pilot)
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            self.assertEqual(warning, "")
            self.assertIsNotNone(saved.run_defaults)
            self.assertEqual(saved.run_defaults.model, "")
            self.assertEqual(saved.run_defaults, RunConfig())
            with patch.object(app, "_launch") as launch:
                app.action_request_run()
                await pilot.pause()
                launch.assert_not_called()
            self.assertIsNone(app._pending_launch)

    async def test_custom_gpu_order_survives_loading_and_form_collection(self):
        config = replace(RunConfig.smoke("demo/gpu-order"), gpus="on,off")
        save_settings(self.settings_path, TuiSettings(run_defaults=config), PROJECT_DIR)
        app = AcprofTui(settings_path=self.settings_path)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            self.assertEqual(app.query_one("#gpus", Select).value, "on,off")
            self.assertEqual(app._collect_config(), config)
            await self.remember_experiment(app, pilot)
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            self.assertEqual(warning, "")
            self.assertEqual(saved.run_defaults.gpus, "on,off")

    async def test_ui_preferences_apply_save_and_restore_after_restart(self):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.press("f2")
            app.query_one("#ui-theme", Select).value = "acprof-light"
            app.query_one("#ui-log-lines", Select).value = 1000
            app.query_one("#ui-log-wrap", Checkbox).value = False
            await pilot.pause()
            self.assertTrue(await pilot.click("#ui-command-bar", offset=(2, 1)))
            await pilot.pause()
            self.assertEqual(app.theme, "acprof-light")
            self.assertFalse(app.query_one("#run-log", RichLog).wrap)
            self.assertEqual(app.query_one("#run-log", RichLog).max_lines, 1000)
            self.assertFalse(app.query_one("#slash-command-bar").display)
            self.assertEqual(app.query_one("#bottom-panel").region.height, 1)
            self.assertFalse(self.settings_path.exists())
            self.assertTrue(await pilot.click("#save-ui-settings", offset=(3, 1)))
            await pilot.pause()
            self.assertTrue(self.settings_path.is_file())
        restarted = AcprofTui(settings_path=self.settings_path)
        async with restarted.run_test(size=(150, 45)) as pilot:
            await pilot.pause()
            self.assertEqual(restarted.theme, "acprof-light")
            self.assertFalse(restarted.query_one("#slash-command-bar").display)
            self.assertFalse(restarted.query_one("#run-log", RichLog).wrap)
            self.assertEqual(restarted.query_one("#run-log", RichLog).max_lines, 1000)
            self.assertEqual(restarted.initial_config, RunConfig())

    async def test_saving_ui_preserves_remembered_experiment_and_saving_experiment_preserves_ui(self):
        config = RunConfig.smoke("demo/remembered")
        app = AcprofTui(config, settings_path=self.settings_path)
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.press("f2")
            app.query_one("#ui-theme", Select).value = "acprof-light"
            await pilot.pause()
            self.assertTrue(await pilot.click("#save-ui-settings", offset=(3, 1)))
            await pilot.pause()
            # Remembering the experiment must preserve the saved theme,
            # even when a different theme is currently previewed in memory.
            app.query_one("#ui-theme", Select).value = "acprof-dark"
            await pilot.pause()
            await self.remember_experiment(app, pilot)
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            self.assertEqual(warning, "")
            self.assertEqual(saved.run_defaults, config)
            self.assertEqual(saved.ui.theme, "acprof-light")
            self.assertEqual(app.ui_preferences.theme, "acprof-dark")

            # Unsaved form edits must not replace the explicitly remembered run
            # when the user only chooses to save interface preferences.
            app.query_one("#model", Input).value = "demo/unsaved"
            app.query_one("#cpus", Input).value = "1,3"
            app.query_one("#ui-log-lines", Select).value = 500
            await pilot.pause(0.1)
            await pilot.press("f2")
            await pilot.pause()
            self.assertTrue(await pilot.click("#save-ui-settings", offset=(3, 1)))
            await pilot.pause()
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            self.assertEqual(warning, "")
            self.assertEqual(saved.run_defaults, config)
            self.assertEqual(saved.ui.theme, "acprof-dark")
            self.assertEqual(saved.ui.log_max_lines, 500)
        restarted = AcprofTui(settings_path=self.settings_path)
        self.assertEqual(restarted.initial_config, config)
        self.assertEqual(restarted.ui_preferences.log_max_lines, 500)
        self.assertEqual(restarted.ui_preferences.theme, "acprof-dark")


class TuiMainSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.xdg_patch = patch.dict(os.environ, {"XDG_CONFIG_HOME": self.temporary.name})
        self.xdg_patch.start()
        self.addCleanup(self.xdg_patch.stop)
        self.settings_path = default_settings_path(PROJECT_DIR)
        self.saved_config = replace(RunConfig.smoke("demo/saved"), cpus="1,3", repeat=7)
        save_settings(
            self.settings_path,
            TuiSettings(ui=UiPreferences(theme="acprof-light"), run_defaults=self.saved_config),
            PROJECT_DIR,
        )

    def test_cli_explicit_arguments_override_saved_defaults_without_rewriting_them(self):
        cases = (
            ([], self.saved_config),
            (["--model", "demo/explicit"], replace(self.saved_config, model="demo/explicit")),
            (["--preset", "smoke"], RunConfig.smoke("demo/saved")),
            (["--preset", "main"], RunConfig.main_matrix("demo/saved")),
            (["--preset", "default"], RunConfig(model="demo/saved")),
            (["--model", "demo/explicit", "--preset", "smoke"], RunConfig.smoke("demo/explicit")),
        )
        original = self.settings_path.read_bytes()
        for argv, expected in cases:
            with self.subTest(argv=argv):
                with patch.object(AcprofTui, "run", autospec=True) as run:
                    main(argv)
                run.assert_called_once()
                app = run.call_args.args[0]
                self.assertEqual(app.initial_config, expected)
                self.assertEqual(app.ui_preferences.theme, "acprof-light")
                self.assertEqual(app._initial_preset, app._infer_preset(expected))
                self.assertEqual(self.settings_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

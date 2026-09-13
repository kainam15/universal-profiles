from dataclasses import replace
from pathlib import Path
import re
from string import Formatter
import tempfile
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Input,
    Select,
    Static,
    TabbedContent,
)
from textual.widgets.text_area import Selection

from acprof.tui.app import AcprofTui
from acprof.tui.app import PROJECT_DIR
from acprof.tui.diagnostics import PreflightCheck, ResultSummary
from acprof.tui.progress import ProgressSnapshot, RunProgressTracker
from acprof.tui.commands import RunConfig, TuiConfigError, build_run_command
from acprof.tui.i18n import ENGLISH, error_message, message, translate
from acprof.tui.log import SelectableLog
from acprof.tui.settings import TuiSettings, load_settings, save_settings


class TranslationTests(unittest.TestCase):
    def test_templates_translate_nested_ui_text_and_preserve_user_values(self):
        value = message("已记住：{0} · CPU {1} · 内存 {2} GB", "等待/{模型}", "1,3", "4")
        self.assertEqual(translate(value, "en"), "Saved: 等待/{模型} · CPU 1,3 · Memory 4 GB")
        self.assertEqual(str(value), "已记住：等待/{模型} · CPU 1,3 · 内存 4 GB")
        nested = message("{0}必须是{1}", message("每窗口请求数"), message("整数"))
        self.assertEqual(translate(nested, "en"), "Requests per window must be an integer")
        self.assertEqual(translate(message("{0}", "等待"), "en"), "等待")
        self.assertEqual(translate("unknown {raw} text", "en"), "unknown {raw} text")

    def test_validation_keeps_translatable_errors_without_changing_config_contract(self):
        for config, expected in (
            (RunConfig(model=""), "Model ID must not be empty"),
            (RunConfig(model="demo/model", mems="invalid"), "Memory list must be comma-separated integers"),
            (RunConfig(model="demo/model", repeat_in_window="oops"), "Requests per window must be an integer"),
            (RunConfig(model="demo/model", workload_spec="等待/{raw}"), "Workload manifest does not exist: 等待/{raw}"),
        ):
            with self.subTest(config=config):
                with self.assertRaises(TuiConfigError) as caught:
                    config.validate(project_dir=PROJECT_DIR)
                self.assertIn(expected, translate(error_message(caught.exception), "en"))
                self.assertRegex(str(caught.exception), r"[\u4e00-\u9fff]")

    def test_probe_progress_is_language_independent_and_keeps_duration_semantics(self):
        tracker = RunProgressTracker()
        snapshot = tracker.feed("[largest-probe] MEMORY_RESULT mem=4 status=startup_oom")
        self.assertEqual(snapshot.stage, "内存可行性探测")
        self.assertEqual(translate(snapshot.detail, "en"), "4GB · Startup OOM; trying next candidate")
        snapshot = tracker.feed(
            "[largest-probe] RESULT status=ok input_scale=512 cpu=1 mem=8 gpu=off "
            "cold_start_s=nan request_s=1.25 ready_plus_request_s=3.5"
        )
        self.assertEqual(snapshot.stage, "探测完成")
        self.assertIn("Single request 1.250s · Cold start unavailable · Ready + request 3.500s", translate(snapshot.detail, "en"))
        self.assertFalse(snapshot.measurement_active)

    def test_catalog_preserves_interpolation_fields_and_numeric_formats(self):
        def fields(text):
            return sorted((name, spec, conversion or "") for _, name, spec, conversion in Formatter().parse(text) if name is not None)
        for source, translated in ENGLISH.items():
            with self.subTest(source=source):
                self.assertEqual(fields(source), fields(translated))
                self.assertTrue(translated.strip())


class TuiLanguageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = PROJECT_DIR / "internal-testing"
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="tui-language-", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.settings_path = Path(temporary.name) / "tui.json"

    def make_app(self, config=None):
        return AcprofTui(config, settings_path=self.settings_path)

    async def switch(self, app, pilot, language):
        app.action_show_settings()
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        self.assertEqual(app.ui_preferences.language, language)

    async def test_language_selector_accepts_keyboard_choices_in_both_directions(self):
        app = self.make_app(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("f2")
            await pilot.pause()
            # Select routes clicks through its child label; Pilot.click's
            # return value only matches the exact target widget.
            await pilot.click("#ui-language", offset=(4, 1))
            self.assertTrue(app.query_one("#ui-language", Select).expanded)
            await pilot.press("end", "enter")
            await pilot.pause()
            self.assertEqual(app.ui_preferences.language, "en")
            self.assertEqual(app.query_one("#ui-language SelectCurrent #label", Static).content, "English")
            await pilot.click("#ui-language", offset=(4, 1))
            self.assertTrue(app.query_one("#ui-language", Select).expanded)
            await pilot.press("home", "enter")
            await pilot.pause()
            self.assertEqual(app.ui_preferences.language, "zh")
            self.assertEqual(app.query_one("#ui-language SelectCurrent #label", Static).content, "简体中文")
            self.assertEqual(app.query_one("#run-preset", Select).value, "smoke")
            self.assertFalse(self.settings_path.exists())

    async def test_switch_keeps_drafts_presets_log_selection_matrix_and_loaded_results(self):
        config = replace(RunConfig.smoke("demo/model"), gpus="on,off")
        app = self.make_app(config)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app.open_run_settings()
            model = app.query_one("#model", Input)
            model.value = "等待/{模型}"
            model.cursor_position = 2
            await pilot.pause()
            # This test preserves an already rendered preview; debounce timing
            # is covered by the form interaction tests.
            app._cancel_preview_timer()
            app._refresh_command_preview(notify=False, sync_preset=True)
            command = build_run_command(app._collect_config(), project_dir=PROJECT_DIR)
            preview = app.query_one("#command-preview", Static).content
            preset = app.query_one("#run-preset", Select).value
            app._activate_tab("monitor-tab")
            log = app.query_one("#run-log", SelectableLog)
            log.write("\n".join(f"等待 原始日志 {index}" for index in range(100)))
            await pilot.pause()
            log.selection = Selection((30, 0), (31, 4))
            log.scroll_to(y=25, animate=False, immediate=True)
            await pilot.pause()
            old_text, old_selection, old_scroll = log.text, log.selection, log.scroll_y
            self.assertFalse(log.following)
            app._init_matrix_for_run(config)
            snapshot = ProgressSnapshot(
                stage="case 完成", current_case=1, completed_cases=1, total_cases=2,
                cpu="1", mem="4", gpu="on", detail=message("正在准备 case {0}/{1}", 1, 2),
            )
            app._latest_snapshot = snapshot
            app._render_snapshot(snapshot)
            summary = ResultSummary(3, 2, 1, 1, 2, 0.01, 0.03, 0.02)
            with patch("acprof.tui.app.summarize_result_csv", return_value=summary) as read_results:
                app._update_result_summary("read-once.csv", notify=False)
                remembered = self.settings_path.read_bytes()
                self.assertEqual(
                    load_settings(self.settings_path, PROJECT_DIR),
                    (TuiSettings(last_result_csv=str(PROJECT_DIR / "read-once.csv")), ""),
                )
                await self.switch(app, pilot, "en")
                self.assertEqual(read_results.call_count, 1)
                self.assertIn("Application latency (mean): 20.0ms", app.query_one("#result-summary", Static).content)
            self.assertEqual(model.value, "等待/{模型}")
            self.assertEqual(model.cursor_position, 2)
            self.assertEqual(app.query_one("#run-preset", Select).value, preset)
            self.assertEqual(app.query_one("#gpus", Select).value, "on,off")
            self.assertEqual(app.query_one("#ui-theme SelectCurrent #label", Static).content, "Ocean blue · Dark")
            self.assertEqual(build_run_command(app._collect_config(), project_dir=PROJECT_DIR), command)
            self.assertEqual(app.query_one("#command-preview", Static).content, preview)
            self.assertEqual(app.query_one("#experiment-pages", ContentSwitcher).current, "advanced-form")
            self.assertEqual(str(app.query_one("#open-run-settings", Button).label), "Basic settings")
            self.assertEqual(app.query_one("#status-stage", Static).content, "Case completed")
            self.assertTrue(app.query_one("#status-stage").has_class("stage-success"))
            self.assertIs(app._latest_snapshot, snapshot)
            table = app.query_one("#matrix-table", DataTable)
            self.assertEqual(table.get_row(app._matrix_rows[1]), ["1", "1", "4", "on", "✓ Done"])
            self.assertEqual(table.get_cell(app._matrix_rows[2], "status"), "⋯ Waiting")
            self.assertEqual(table.columns["status"].label.plain, "Status")
            self.assertEqual((log.text, log.selection, log.scroll_y), (old_text, old_selection, old_scroll))
            self.assertFalse(log.following)
            self.assertIn("Reading history", app.query_one("#log-hint", Static).content)
            self.assertEqual(self.settings_path.read_bytes(), remembered)
            self.assertEqual(app.query_one("#result-csv", Input).value, str(PROJECT_DIR / "read-once.csv"))
            await self.switch(app, pilot, "zh")
            self.assertEqual(str(app.query_one("#open-run-settings", Button).label), "返回基本配置")
            self.assertEqual(table.get_cell(app._matrix_rows[1], "status"), "✓ 完成")
            self.assertEqual((log.text, log.selection, log.scroll_y), (old_text, old_selection, old_scroll))
            self.assertEqual(self.settings_path.read_bytes(), remembered)

    async def test_explicit_save_restart_reset_and_model_memory_keep_their_boundaries(self):
        defaults = RunConfig.smoke("demo/saved")
        save_settings(self.settings_path, TuiSettings(run_defaults=defaults), PROJECT_DIR)
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            await self.switch(app, pilot, "en")
            app._remember_last_used(model="demo/latest")
            saved, warning = load_settings(self.settings_path, PROJECT_DIR)
            self.assertFalse(warning)
            self.assertEqual(saved.ui.language, "zh")
            self.assertIn("Model for next launch: demo/latest", app.query_one("#saved-run-summary", Static).content)
            self.assertTrue(await pilot.click("#save-ui-settings"))
            await pilot.pause()
        restarted = self.make_app()
        async with restarted.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            self.assertEqual(restarted.ui_preferences.language, "en")
            self.assertEqual(str(restarted.query_one("#start-run", Button).label), "Start run")
            self.assertEqual(restarted.query_one("#model", Input).value, "demo/latest")
            self.assertEqual(restarted._saved_settings.run_defaults, defaults)
            await pilot.press("f2")
            self.assertTrue(await pilot.click("#restore-ui-defaults"))
            await pilot.pause()
            self.assertEqual(restarted.ui_preferences.language, "zh")
            self.assertEqual(restarted.query_one("#ui-language", Select).value, "zh")
            self.assertEqual(load_settings(self.settings_path, PROJECT_DIR)[0].ui.language, "en")

    async def test_english_validation_dialog_notifications_and_raw_output(self):
        app = self.make_app(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            await self.switch(app, pilot, "en")
            app.query_one("#mems", Input).value = "oops"
            await pilot.pause()
            app._refresh_command_preview(notify=False)
            self.assertIn("Memory list must be comma-separated integers", app.query_one("#config-summary", Static).content)
            app.action_request_run()
            await pilot.pause()
            self.assertTrue(any(n.title == "Invalid configuration" and "Memory list" in n.message for n in app._notifications))
            app.query_one("#mems", Input).value = "4"
            await pilot.pause()
            app.action_request_probe()
            await pilot.pause()
            self.assertEqual(str(app.screen.query_one("#confirm-no", Button).label), "Cancel")
            self.assertEqual(str(app.screen.query_one("#confirm-yes", Button).label), "Start probe")
            self.assertIn("Memory candidates: 4GB", app.screen.query_one("#confirm-message", Static).content)
            await pilot.press("escape")
            app._show_quick_check([PreflightCheck(message("原生 Linux"), "ok", message("统一层级可用"))], "")
            log = app.query_one("#run-log", SelectableLog)
            self.assertIn("[Passed] Native Linux: Unified hierarchy available", log.text)
            app._consume_process_line("等待 原始子进程输出 {raw}", None, False)
            self.assertIn("等待 原始子进程输出 {raw}", log.text)
            app._process_started(123, "probe")
            self.assertIn("probe process started, PID=123", log.text)

    async def test_both_languages_fit_common_terminal_sizes_and_busy_tasks_lock_language(self):
        app = self.make_app(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.pause()
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                await pilot.pause()
                self.assertEqual(app.has_class("narrow"), size[0] < 110)
                self.assertEqual(app.has_class("short"), size[1] < 35)
                for language in ("en", "zh"):
                    with self.subTest(size=size, language=language):
                        await self.switch(app, pilot, language)
                        for tab, ids in (
                            ("settings-tab", ("ui-language", "restore-ui-defaults", "save-ui-settings")),
                            ("run-tab", ("open-run-settings", "quick-check", "probe-largest", "start-run")),
                            ("monitor-tab", ("copy-log", "follow-log", "expand-log", "clear-log", "stop-run")),
                            ("plot-tab", ("result-csv", "summarize-results", "plot-results")),
                            ("profile-tab", ("result-dir", "profile-dry-run", "profile-run")),
                        ):
                            tabs = app.query_one("#main-tabs", TabbedContent)
                            tab_button = tabs.get_tab(tab)
                            self.assertGreater(tab_button.region.width, 0, tab)
                            self.assertGreaterEqual(tab_button.region.x, 0, tab)
                            self.assertLessEqual(tab_button.region.right, size[0], tab)
                            self.assertLessEqual(cell_len(tab_button.label.plain), tab_button.content_region.width, tab)
                            self.assertTrue(await pilot.click(tab_button))
                            await pilot.pause()
                            self.assertEqual(tabs.active, tab)
                            for widget_id in ids:
                                widget = app.query_one(f"#{widget_id}")
                                region = widget.region
                                self.assertGreater(region.width, 0, widget_id)
                                self.assertGreaterEqual(region.x, 0, widget_id)
                                self.assertLessEqual(region.right, size[0], widget_id)
                                self.assertLessEqual(region.bottom, app.query_one("#bottom-panel").region.y, widget_id)
                                if isinstance(widget, Button):
                                    self.assertLessEqual(cell_len(widget.label.plain), widget.content_region.width, widget_id)
                                    self.assertLessEqual(region.right, widget.parent.content_region.right, widget_id)
                                    self.assertGreaterEqual(region.x, widget.parent.content_region.x, widget_id)
                            if tab == "monitor-tab" and language == "en":
                                self.assertGreaterEqual(app.query_one("#log-title").content_region.width, 3)
                        if language == "en":
                            for (widget, attribute), source in app._localized_text.items():
                                rendered = widget.content if attribute == "content" else getattr(widget, attribute)
                                self.assertIsNone(re.search(r"[\u4e00-\u9fff]", str(rendered)), (widget, attribute))
                            for widget in app._localized_selects:
                                displayed = widget.query_one("SelectCurrent #label", Static).content
                                self.assertIsNone(re.search(r"[\u4e00-\u9fff]", str(displayed)), widget)
            app._process_kind = "run"
            app._set_busy(True)
            self.assertTrue(app.query_one("#ui-language", Select).disabled)
            app.query_one("#ui-language", Select).value = "en"
            await pilot.pause()
            self.assertEqual(app.ui_preferences.language, "zh")
            app._process_kind = ""
            app._set_busy(False)
            self.assertFalse(app.query_one("#ui-language", Select).disabled)


if __name__ == "__main__":
    unittest.main()

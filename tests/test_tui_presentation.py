"""实验参数可独立编辑，空值与正在执行的任务有不同显示。"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import DataTable, Input, Label, Static

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig, TuiConfigError, build_run_command
from acprof.tui.i18n import translate
from acprof.tui.images import ImageTree, format_image_size, image_metadata
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.settings import load_settings
from acprof.host.image_management import DockerConnection, ImageInventory, ImageLayer, ManagedImage


PROJECT_DIR = Path(__file__).resolve().parents[1]


class TuiPresentationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.app = AcprofTui(RunConfig(model="demo/model"), settings_path=self.directory / "ui.json")

    async def test_independent_numbers_reach_command_and_saved_defaults(self):
        app = self.app
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.click("#open-run-settings")
            await pilot.pause()
            self.assertEqual(len(app.query("#warmup")), 1, "Warmup 应有独立的输入框")
            warmup, repeat = app.query_one("#warmup", Input), app.query_one("#repeat", Input)
            self.assertEqual((warmup.value, repeat.value), ("2", "5"))
            self.assertEqual(app.query_one("#sample-hz", Input).value, "20")
            self.assertEqual(app.query_one("#request-timeout-seconds", Input).value, "300")
            self.assertTrue(await pilot.click(warmup))
            await pilot.press("ctrl+a", "3", "tab", "ctrl+a", "7")
            self.assertIs(app.focused, repeat)
            sample = app.query_one("#sample-hz", Input)
            self.assertTrue(await pilot.click(sample))
            await pilot.press("ctrl+a", *"20.1256789")
            config = app._collect_config()
            self.assertEqual((config.warmup, config.repeat, config.sample_hz), (3, 7, 20.1256789))
            command = build_run_command(config, project_dir=PROJECT_DIR)
            for flag, value in (("--warmup", "3"), ("--repeat", "7"), ("--sample-hz", "20.1256789")):
                self.assertEqual(command[command.index(flag) + 1], value)
            warmup.value = "-1"
            with self.assertRaisesRegex(TuiConfigError, "Warmup"):
                app._collect_config()
            warmup.value = "3"
            repeat.value = "0"
            with self.assertRaisesRegex(TuiConfigError, "Repeat"):
                app._collect_config()
            repeat.value = "7"
            app.query_one("#advanced-form").scroll_end(animate=False, immediate=True)
            await pilot.pause()
            self.assertTrue(await pilot.click("#save-run-default"))
            await pilot.pause()
            saved, warning = load_settings(app.settings_path, PROJECT_DIR)
            self.assertEqual(warning, "")
            self.assertEqual(saved.run_defaults, config)
            assert saved.run_defaults is not None
            app._apply_config(RunConfig.smoke("demo/model"), preset="smoke")
            self.assertEqual((warmup.value, repeat.value), ("0", "1"))
            app._apply_config(saved.run_defaults)
            self.assertEqual(sample.value, "20.1256789", "显示不能丢失已保存的小数精度")

    async def test_parameter_units_remain_outside_editable_inputs_across_resize_and_languages(self):
        app = self.app
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.click("#open-run-settings")
            await pilot.pause()
            self.assertEqual(len(app.query("#repeat")), 1, "Repeat 应有独立的输入框")
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                for language in ("zh", "en"):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    for name, unit in (("warmup", "次" if language == "zh" else "runs"),
                                       ("repeat", "次" if language == "zh" else "runs"),
                                       ("sample-hz", "Hz"), ("idle-seconds", "s"),
                                       ("idle-cooldown-seconds", "s"), ("request-timeout-seconds", "s")):
                        with self.subTest(size=size, language=language, field=name):
                            field = app.query_one("#" + name, Input)
                            field.scroll_visible(animate=False, immediate=True)
                            await pilot.pause()
                            assert field.parent is not None
                            label = field.parent.query_one(".field-unit", Label)
                            self.assertEqual(str(label.content), unit)
                            self.assertGreaterEqual(label.region.x, field.region.right)
                            self.assertLessEqual(label.region.right, app.size.width - 2)
                            self.assertTrue(await pilot.click(field))
                            self.assertIs(app.focused, field)
                            self.assertNotIn(unit, field.value)
            app._set_busy(True)
            self.assertTrue(app.query_one("#warmup", Input).disabled)
            self.assertTrue(app.query_one("#repeat", Input).disabled)
            app._set_busy(False)
            self.assertFalse(app.query_one("#repeat", Input).disabled)

    async def test_monitor_distinguishes_no_case_from_missing_case_resources(self):
        app = self.app
        async with app.run_test() as pilot:
            app._render_snapshot(ProgressSnapshot())
            self.assertEqual(str(app.query_one("#status-elapsed", Static).content), "—")
            self.assertEqual(str(app.query_one("#status-resource", Static).content), "CPU=—  MEM=—  GPU=—")
            app._latest_snapshot = ProgressSnapshot(current_case=1, total_cases=2, cpu="2")
            app._render_snapshot(app._latest_snapshot)
            self.assertEqual(str(app.query_one("#status-resource", Static).content), "CPU=2  MEM=未知  GPU=未知")
            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.pause()
            self.assertIn("MEM=Unknown", str(app.query_one("#status-resource", Static).content))

    async def test_loading_report_indicator_is_cleared_on_failure(self):
        app = self.app
        async with app.run_test() as pilot:
            with patch.object(app, "_execute_report_read"):
                app._open_report(str(self.directory / "report.json"))
                self.assertTrue(app._is_busy())
                self.assertTrue(str(app.query_one("#report-status", Static).content).startswith("…"))
                app._show_report(None, "broken JSON")
                await pilot.pause()
                self.assertFalse(app._is_busy())
                text = str(app.query_one("#report-status", Static).content)
                self.assertIn("报告读取失败", text)
                self.assertNotIn("…", text)

    async def test_unknown_values_translate_in_image_tree_list_and_layers(self):
        app = self.app
        item = ManagedImage("sha256:base", ("acprof-platform-cpu:base",), 0, "", "base", acprof=True)
        layer = ImageLayer("chain", "diff", None, (item.image_id,))
        inventory = ImageInventory(DockerConnection((), "local"), "daemon", (item,), (layer,))
        with patch.object(app, "refresh_images"):
            async with app.run_test(size=(120, 30)) as pilot:
                app._activate_tab("images-tab")
                app._image_inventory = inventory
                app._render_images()
                await pilot.pause()
                for language, unknown in (("zh", "未知"), ("en", "Unknown")):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    await pilot.pause()
                    cells = [str(cell) for cell in app.query_one("#image-table", DataTable).get_row_at(0)]
                    self.assertEqual(cells[2:6], ["0 B", unknown, "0", unknown])
                    self.assertEqual(app.query_one("#image-layer-table", DataTable).get_row_at(0)[1], unknown)
                    tree = app.query_one("#image-tree", ImageTree)
                    label = tree.root.children[0].label.plain
                    self.assertIn("0 B", label)
                    self.assertIn(unknown, label)
                    self.assertNotIn("?", label)


class DisplayValueTests(unittest.TestCase):
    def test_unknown_image_size_keeps_real_zero_and_translates(self):
        self.assertEqual(format_image_size(None), "未知")
        self.assertEqual(translate(format_image_size(None), "en"), "Unknown")
        self.assertEqual(format_image_size(0), "0 B")

    def test_absent_model_and_unknown_created_time_are_distinct(self):
        item = ManagedImage("sha256:base", (), 0, "", "base")
        # 路径与共享空间不需要 Docker；空清单保留镜像本身的信息。
        inventory = ImageInventory(DockerConnection((), "local"), "daemon", (item,))
        self.assertIn("模型：— · 创建时间：未知", image_metadata(item, inventory))

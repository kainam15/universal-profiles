from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import Button, Checkbox, Input, Static, TabbedContent, TabPane

from acprof.cli.tui import AcprofTui
from acprof.cli.tui_core import RunConfig


class TuiResultToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.csv_path = self.directory / "绘图结果.csv"
        self.csv_path.write_text("status,warmup,latency_app_s\nok,0,0.02\n", encoding="utf-8")
        self.settings_path = self.directory / "tui.json"

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)

    async def test_separate_tabs_keep_drafts_and_plot_uses_its_csv(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            self.assertEqual(
                [tabs.get_tab(pane).label.plain for pane in tabs.query(TabPane)],
                ["实验配置", "运行监控", "绘图工具", "统计报告", "补采工具", "设置"],
            )

            self.assertTrue(await pilot.click(tabs.get_tab("profile-tab")))
            await pilot.pause()
            self.assertEqual(tabs.active, "profile-tab")
            self.assertFalse(app.query("#profile-tab #result-csv, #profile-tab #plot-results"))
            directory_draft = str(self.directory / "未提交的补采目录")
            app.query_one("#result-dir", Input).value = directory_draft
            self.assertTrue(await pilot.click("#profile-tool-nsys", offset=(2, 1)))
            await pilot.pause()

            self.assertTrue(await pilot.click(tabs.get_tab("plot-tab")))
            await pilot.pause()
            self.assertEqual(tabs.active, "plot-tab")
            self.assertFalse(app.query("#plot-tab #result-dir, #plot-tab .profile-tool, #plot-tab #profile-run"))
            app.query_one("#result-csv", Input).value = str(self.csv_path)
            self.assertTrue(await pilot.click("#summarize-results", offset=(2, 1)))
            await pilot.pause()
            self.assertIn("成功 1", app.query_one("#result-summary", Static).content)
            self.assertEqual(app.query_one("#result-dir", Input).value, directory_draft)

            self.assertTrue(await pilot.click(tabs.get_tab("profile-tab")))
            await pilot.pause()
            self.assertTrue(app.query_one("#profile-tool-nsys", Checkbox).value)
            self.assertEqual(app.query_one("#result-dir", Input).value, directory_draft)
            self.assertTrue(await pilot.click(tabs.get_tab("plot-tab")))
            await pilot.pause()
            self.assertEqual(app.query_one("#result-csv", Input).value, str(self.csv_path))

            with patch.object(app, "_execute_command") as execute:
                self.assertTrue(await pilot.click("#plot-results", offset=(2, 1)))
                await pilot.pause()
                execute.assert_called_once()
                command, kind = execute.call_args.args
                self.assertEqual(kind, "plot")
                self.assertEqual(Path(command[2]).name, "plot.py")
                self.assertEqual(command[3], str(self.csv_path))
                self.assertEqual(tabs.active, "monitor-tab")
                controls = list(app.query(
                    "#summarize-results, #plot-results, #profile-dry-run, #profile-run, .profile-tool"
                ))
                self.assertTrue(all(widget.disabled for widget in controls))
                app._process_finished("plot", 0, None, "")
                await pilot.pause()
                self.assertTrue(all(not widget.disabled for widget in controls))

    async def test_summary_commands_show_the_plotting_page(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            for command in ("results", "summary"):
                with self.subTest(command=command):
                    field = app.query_one("#slash-command", Input)
                    field.value = f'/{command} "{self.csv_path}"'
                    field.focus()
                    await pilot.pause()
                    await pilot.press("enter")
                    await pilot.pause()
                    self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "plot-tab")
                    summary = app.query_one("#result-summary", Static)
                    self.assertIn("成功 1", summary.content)
                    self.assertGreater(summary.region.height, 0)
                    self.assertFalse(app.query_one("#plot-results", Button).disabled)


if __name__ == "__main__":
    unittest.main()

"""验证用户从 TUI 计算和查看报告，保持原始结果与测量隔离。"""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import Button, DataTable, Input, Static, TabbedContent, TabPane

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig
from acprof.tui.progress import ProgressSnapshot


class TuiReportsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.csv_path = self.directory / "模型结果.csv"
        fields = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx",
                  "status", "repeat_in_window", "latency_app_s", "latency_s",
                  "container_attributed_energy_eff_j"]
        with self.csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, latency in enumerate((0.01, 0.02, 0.03, 99, 999)):
                writer.writerow(dict(cpu_cores=2, mem_cap_gb=8, gpu_mode="off", input_scale=64,
                                     warmup=int(index == 3), repeat_idx=index,
                                     status="error" if index == 4 else "ok",
                                     repeat_in_window=1000 if index == 0 else 1,
                                     latency_app_s=latency, latency_s=latency,
                                     container_attributed_energy_eff_j=0.1))

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json")

    async def open_tab(self, app, pilot):
        await pilot.pause()
        tabs = app.query_one("#main-tabs", TabbedContent)
        self.assertIn("reports-tab", [pane.id for pane in tabs.query(TabPane)],
                      "TUI 应提供统计报告页，让用户查看已有报告和计算窗口统计")
        app._activate_tab("reports-tab")
        await pilot.pause()

    async def finish_workers(self, app, pilot):
        # stats 子进程完成后会调度读取报告的 worker。
        for _ in range(2):
            await app.workers.wait_for_complete()
            await pilot.pause()

    def overhead_report(self, name="开销.json", **changes):
        data = dict(schema_version=1, kind="monitor_overhead_diagnostic", successful=True,
                    gpu_mode="off", source_run_id="original-run", image_id="sha256:example",
                    comparisons=[dict(scenario="monitors-20", paired_rounds=5,
                                      paired_mean_change_pct=2.0, ci_low_pct=-1.0, ci_high_pct=3.0)])
        data.update(changes)
        path = self.directory / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    async def test_calculate_uses_formal_windows_and_opens_saved_report_without_changing_csv(self):
        original = self.csv_path.read_bytes()
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#report-source", Input).value = str(self.csv_path)
            self.assertTrue(await pilot.click("#report-calculate"))
            await self.finish_workers(app, pilot)
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "reports-tab")
            output = Path(app.query_one("#report-source", Input).value)
            self.assertEqual(output.suffix, ".json")
            data = json.loads(output.read_text())
            latency = next(row for row in data["groups"] if row["metric"] == "latency_app_s")
            self.assertEqual(latency["n_windows"], 3)
            self.assertAlmostEqual(latency["mean"], 0.02)
            table = app.query_one("#report-table", DataTable)
            self.assertEqual(table.row_count, 3)
            cells = [str(cell) for cell in table.get_row_at(0)]
            self.assertIn("20 ms", cells)
            self.assertFalse(app._is_busy())
            self.assertFalse(app.query_one("#start-run", Button).disabled)
        self.assertEqual(self.csv_path.read_bytes(), original)

    async def test_open_overhead_at_all_sizes_and_language_switch_preserves_data_and_draft(self):
        report = self.overhead_report()
        original = report.read_bytes()
        for size in ((80, 24), (120, 30), (150, 45)):
            with self.subTest(size=size):
                app = self.make_app()
                async with app.run_test(size=size) as pilot:
                    await self.open_tab(app, pilot)
                    source = app.query_one("#report-source", Input)
                    source.value = str(report)
                    self.assertTrue(await pilot.click("#report-open"))
                    await self.finish_workers(app, pilot)
                    table = app.query_one("#report-table", DataTable)
                    self.assertEqual(table.row_count, 1)
                    self.assertIn("+2%", [str(cell) for cell in table.get_row_at(0)])
                    self.assertIn("方向不确定", [str(cell) for cell in table.get_row_at(0)])
                    for selector in ("#report-open", "#report-calculate", "#report-current", "#report-table"):
                        region = app.query_one(selector).region
                        self.assertGreater(region.height, 0)
                        self.assertLessEqual(region.bottom, size[1] - 3)
                    table.focus()
                    await pilot.pause()
                    await pilot.press("right", "left", "down")
                    app.ui_preferences = replace(app.ui_preferences, language="en")
                    app._apply_ui_preferences()
                    await pilot.pause()
                    self.assertEqual(source.value, str(report))
                    self.assertIn("Uncertain direction", [str(cell) for cell in table.get_row_at(0)])
                    self.assertIn("Report", app.query_one("#main-tabs", TabbedContent).get_tab("reports-tab").label.plain)
        self.assertEqual(report.read_bytes(), original)

    async def test_invalid_or_failed_reports_clear_old_table_and_show_error_on_report_page(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            source = app.query_one("#report-source", Input)
            source.value = str(self.overhead_report())
            await pilot.click("#report-open")
            await self.finish_workers(app, pilot)
            self.assertEqual(app.query_one("#report-table", DataTable).row_count, 1)
            invalid = self.directory / "损坏.json"
            for content in ("{broken", '{"schema_version":99}',
                            '{"schema_version":1,"kind":"ui_overhead","successful":false,"error":"interrupted"}'):
                invalid.write_text(content)
                source.value = str(invalid)
                await pilot.click("#report-open")
                await self.finish_workers(app, pilot)
                self.assertEqual(app.query_one("#report-table", DataTable).row_count, 0)
                self.assertIn("报告读取失败", str(app.query_one("#report-status", Static).content))
                self.assertFalse(app._is_busy())
                self.assertFalse(app.query_one("#report-open", Button).disabled)

    async def test_measurement_blocks_report_io_and_statistics_including_slash_commands(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#report-source", Input).value = str(self.csv_path)
            app._process_kind = "run"
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            with patch.object(app, "_execute_command") as execute:
                for selector in ("#report-open", "#report-calculate", "#report-current", "#report-source"):
                    self.assertTrue(app.query_one(selector).disabled)
                for command in ("/stats", "/report"):
                    field = app.query_one("#slash-command", Input)
                    field.value = command
                    field.focus()
                    await pilot.pause()
                    await pilot.press("enter")
                    await pilot.pause()
                execute.assert_not_called()
                self.assertFalse(list(app.workers))
            app._process_kind = ""
            app._latest_snapshot = ProgressSnapshot()
            app._set_busy(False)
            self.assertFalse(app.query_one("#report-open", Button).disabled)

    async def test_current_result_follows_completed_run_but_preserves_an_explicit_report_path(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app._process_finished("run", 0, ProgressSnapshot(final_csv=str(self.csv_path)), "")
            await pilot.pause()
            source = app.query_one("#report-source", Input)
            self.assertEqual(source.value, str(self.csv_path))
            self.assertFalse(list(app.workers))
            source.value = str(self.overhead_report())
            app._remember_last_used(result_csv=str(self.directory / "another.csv"))
            self.assertTrue(source.value.endswith("开销.json"))
            await pilot.click("#report-current")
            await pilot.pause()
            self.assertEqual(source.value, str(self.directory / "another.csv"))


if __name__ == "__main__":
    unittest.main()

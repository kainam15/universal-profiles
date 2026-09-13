"""报告展示保留单位、真实零和统计边界；损坏报告不能伪装成有效结果。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from acprof.tui.i18n import translate
from acprof.tui.reports import read_report


class ReportViewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "统计.json"
        self.group = dict(cpu_cores=2.0, mem_cap_gb=8.0, gpu_mode="off", input_scale=64.0,
                          metric="latency_app_s", unit="s", n_windows=3, missing_windows=0,
                          mean=0.02, std=0.01, ci_low=0.01, ci_high=0.03, reason="")
        self.window = dict(schema_version=1, resampling_unit="csv_request_window", confidence=0.95,
                           filter="status=ok and warmup=0", groups=[self.group], result_csv="原始/结果.csv")

    def read(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.path.read_bytes()
        result = read_report(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        return result

    def test_window_units_confidence_level_zero_and_missing_values_are_explicit(self):
        data = deepcopy(self.window)
        data["confidence"] = 0.9
        data["groups"] += [dict(self.group, metric="cpu_energy_total_j", unit="J/request",
                                mean=0, std=0, ci_low=0, ci_high=0),
                           dict(self.group, n_windows=1, missing_windows=2, std=None,
                                ci_low=None, ci_high=None, reason="insufficient_windows"),
                           dict(self.group, n_windows=0, missing_windows=3, mean=None, std=None,
                                ci_low=None, ci_high=None, reason="insufficient_windows")]
        view = self.read(data)
        self.assertIn("90%", str(view.title))
        self.assertEqual(view.rows[0].cells[2:4], ("20 ms", "[10, 30] ms"))
        self.assertEqual(view.rows[1].cells[2:4], ("0 J/request", "[0, 0] J/request"))
        self.assertEqual(view.rows[2].cells[2:], ("20 ms", "—", "1/2", "窗口不足"))
        self.assertEqual(view.rows[3].cells[2:], ("—", "—", "0/3", "无有效窗口"))
        self.assertIn("原始/结果.csv", str(view.note))

    def test_empty_formal_results_are_viewable_without_inventing_samples(self):
        view = self.read(dict(self.window, groups=[]))
        self.assertEqual(view.rows, ())
        self.assertIn("0 项", str(view.title))

    def test_ui_comparison_keeps_headless_and_terminal_scopes_distinct(self):
        data = dict(schema_version=1, kind="ui_overhead", successful=True, ui="headless",
                    pairs=[{"round": 0}, {"round": 1}, {"round": 2}], paired_mean_change_pct=-0.16,
                    ci_low_pct=-0.57, ci_high_pct=0.35)
        view = self.read(data)
        self.assertEqual(view.rows[0].cells[1:4], ("3", "-0.16%", "[-0.57%, 0.35%]"))
        self.assertEqual(translate(view.rows[0].cells[-1], "en"), "Uncertain direction")
        self.assertIn("excludes terminal rendering", translate(view.rows[0].detail, "en"))
        terminal = self.read(dict(data, ui="terminal"))
        self.assertIn("Includes terminal rendering", translate(terminal.rows[0].detail, "en"))

    def test_invalid_numbers_intervals_and_counts_are_rejected(self):
        for changes in ({"mean": float("nan")}, {"ci_low": float("inf")}, {"ci_low": 1.0},
                        {"ci_low": None}, {"n_windows": True}, {"n_windows": -1},
                        {"n_windows": 0}, {"mean": None}, {"n_windows": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.read(dict(self.window, groups=[dict(self.group, **changes)]))

    def test_failed_unknown_and_truncated_reports_are_not_successful_views(self):
        for data in (dict(schema_version=1, kind="ui_overhead", successful=False, error="interrupted"),
                     dict(self.window, schema_version=2), dict(self.window, schema_version=True),
                     dict(self.window, filter="warmup=1"), {"schema_version": 1, "kind": "audit"}, []):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.read(data)
        self.path.write_text('{"schema_version":1,')
        with self.assertRaisesRegex(ValueError, "JSON"):
            read_report(self.path)


if __name__ == "__main__":
    unittest.main()

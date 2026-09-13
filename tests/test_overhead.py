"""开销对照必须按同一轮配对，不能混用不完整实验。"""
import unittest
from unittest.mock import Mock, patch


class OverheadSummaryTests(unittest.TestCase):
    def test_known_paired_increase_and_zero_change(self):
        from scripts.measure_overhead import summarize_overhead
        rows = []
        for index, latency in enumerate((1, 2, 4)):
            rows.extend(({"round": index, "scenario": "none", "latency_app_s": latency},
                         {"round": index, "scenario": "monitors-20", "latency_app_s": latency * 1.25}))
        result = summarize_overhead(rows, seed=7)
        self.assertEqual(result[0]["paired_mean_change_pct"], 25)
        self.assertEqual((result[0]["ci_low_pct"], result[0]["ci_high_pct"]), (25, 25))

    def test_missing_baseline_duplicate_and_bad_values_are_rejected(self):
        from scripts.measure_overhead import summarize_overhead
        samples = [
            [{"round": 0, "scenario": "monitors-20", "latency_app_s": 2}],
            [{"round": 0, "scenario": "none", "latency_app_s": 1}] * 2,
            [{"round": 0, "scenario": "none", "latency_app_s": 0}],
        ]
        for rows in samples:
            with self.assertRaises(ValueError):
                summarize_overhead(rows)

    def test_request_failure_stops_and_closes_every_started_monitor(self):
        from scripts.measure_overhead import measure_window
        monitors = [Mock(), Mock()]
        for monitor in monitors:
            monitor.stop.return_value = (None, "", [1, 2])
        with patch("requests.post", side_effect=RuntimeError("request failed")):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                measure_window("http://example.invalid", {}, count=1, monitors=monitors, token="test")
        for monitor in monitors:
            monitor.stop.assert_called_once()
            monitor.close.assert_called_once()

    def test_monitor_stop_failure_does_not_leave_other_threads_running(self):
        from scripts.measure_overhead import measure_window
        monitors = [Mock(), Mock()]
        monitors[0].stop.return_value = (None, "", [1, 2])
        monitors[1].stop.side_effect = RuntimeError("stop failed")
        with patch("requests.post", return_value=Mock()):
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                measure_window("http://example.invalid", {}, count=1, monitors=monitors, token="test")
        for monitor in monitors:
            monitor.stop.assert_called_once()
            monitor.close.assert_called_once()

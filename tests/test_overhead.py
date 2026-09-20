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
        self.assertAlmostEqual(result[0]["paired_mean_change_s"], 7 / 12)
        self.assertAlmostEqual(result[0]["latency_mean_s"], 35 / 12)
        self.assertGreater(result[0]["latency_stdev_s"], 0)

    def test_failed_or_unfinished_response_is_not_a_latency_sample(self):
        from scripts.measure_overhead import measure_window
        for status, body in ((202, {"status": "pending"}), (200, {"error": "background failure"})):
            response = Mock(status_code=status)
            response.json.return_value = body
            with self.subTest(status=status), patch("requests.post", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "completed"):
                    measure_window("http://example.invalid", {}, count=1, monitors=[], token="test")

    def test_source_thread_settings_are_restored_without_inheriting_unrecorded_overrides(self):
        import os
        import scripts.measure_overhead as overhead
        with patch.dict(os.environ, {"ACPROF_RUNTIME_THREADS": "8", "ACPROF_ONNX_INTRA_OP_THREADS": "6"}):
            with overhead.source_runtime_environment({"ACPROF_RUNTIME_THREADS": "1"}):
                self.assertEqual(os.environ["ACPROF_RUNTIME_THREADS"], "1")
                self.assertNotIn("ACPROF_ONNX_INTRA_OP_THREADS", os.environ)
            self.assertEqual(os.environ["ACPROF_RUNTIME_THREADS"], "8")
            self.assertEqual(os.environ["ACPROF_ONNX_INTRA_OP_THREADS"], "6")

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

    def test_perf_failure_still_stops_sampling_threads(self):
        from scripts.measure_overhead import measure_window
        monitor, perf = Mock(), Mock()
        monitor.stop.return_value = (None, "", [1, 2])
        perf.stop.side_effect = RuntimeError("instructions unavailable")
        response = Mock(status_code=200)
        response.json.return_value = {"workload_contract": {"input": {"actual_scale": 5}}}
        with patch("requests.post", return_value=response) as post:
            with self.assertRaisesRegex(RuntimeError, "instructions unavailable"):
                measure_window("http://example.invalid", {}, count=1, monitors=[monitor],
                               token="test", perf_monitor=perf, timeout=17)
        self.assertEqual(post.call_args.kwargs["timeout"], 17)
        monitor.stop.assert_called_once()
        monitor.close.assert_called_once()
        perf.close.assert_called_once()

    def test_idle_failure_closes_collectors_before_any_request(self):
        from scripts.measure_overhead import measure_window
        monitors = [Mock(), Mock()]
        perf = Mock()
        with patch("requests.post") as post:
            with self.assertRaisesRegex(RuntimeError, "idle failure"):
                measure_window("http://example.invalid", {}, count=1, monitors=monitors,
                               token="test", perf_monitor=perf,
                               control_window=Mock(side_effect=RuntimeError("idle failure")))
        post.assert_not_called()
        for monitor in [*monitors, perf]:
            monitor.close.assert_called_once()

    def test_truncated_capture_cannot_pass_full_comparison(self):
        import scripts.measure_overhead as overhead
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.pcap'
            result = SimpleNamespace(returncode=0, stdout=json.dumps({
                'requests': {'round-0-full:4': {'latency_s': 0.1}}}), stderr='')
            with patch('subprocess.run', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'coverage'):
                    overhead.validate_capture(['parser'], path, token='round-0-full', count=2)
            self.assertFalse(path.with_suffix('.packets.json').exists())

    def test_monitor_stop_failure_does_not_leave_other_threads_running(self):
        from scripts.measure_overhead import measure_window
        monitors = [Mock(), Mock()]
        monitors[0].stop.return_value = (None, "", [1, 2])
        monitors[1].stop.side_effect = RuntimeError("stop failed")
        response = Mock(status_code=200)
        response.json.return_value = {}
        with patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                measure_window("http://example.invalid", {}, count=1, monitors=monitors, token="test")
        for monitor in monitors:
            monitor.stop.assert_called_once()
            monitor.close.assert_called_once()

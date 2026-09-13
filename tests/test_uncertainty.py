"""从独立手算结果验证重采样单位、过滤和可复现性。"""
import unittest

from acprof.analysis.uncertainty import summarize_windows


class WindowUncertaintyTests(unittest.TestCase):
    def rows(self, values):
        return [{"cpu_cores": "1", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "64",
                 "warmup": "0", "repeat_idx": str(index), "status": "ok", "latency_app_s": value,
                 "repeat_in_window": 1000 if index == 0 else 1}
                for index, value in enumerate(values)]

    def test_windows_get_equal_weight_despite_different_request_counts(self):
        report = summarize_windows(self.rows([1, 4, 7]), ["latency_app_s"])
        group = report["groups"][0]
        self.assertEqual(group["mean"], 4)
        self.assertEqual(group["n_windows"], 3)
        self.assertGreaterEqual(group["ci_low"], 1)
        self.assertLessEqual(group["ci_high"], 7)

    def test_filtering_and_one_window_do_not_manufacture_confidence(self):
        rows = self.rows([2, 100, 1000])
        rows[1]["warmup"] = "1"
        rows[2]["status"] = "warn"
        report = summarize_windows(rows, ["latency_app_s"])
        group = report["groups"][0]
        self.assertEqual(group["n_windows"], 1)
        self.assertEqual(group["mean"], 2)
        self.assertIsNone(group["ci_low"])
        self.assertEqual(group["reason"], "insufficient_windows")

    def test_constant_windows_have_exact_interval_and_seed_repeats(self):
        first = summarize_windows(self.rows([3, 3, 3, 3, 3]), ["latency_app_s"], seed=42)
        self.assertEqual(first, summarize_windows(self.rows([3] * 5), ["latency_app_s"], seed=42))
        self.assertEqual((first["groups"][0]["ci_low"], first["groups"][0]["ci_high"]), (3, 3))

    def test_reused_profiler_values_and_duplicate_windows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "独立|窗口"):
            summarize_windows(self.rows([1, 2, 3]), ["cpu_heap_peak_bytes_massif"])
        rows = self.rows([1, 2, 3])
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_windows(rows, ["latency_app_s"])

    def test_invalid_settings_and_blocks_with_insufficient_units(self):
        for settings in ({"confidence": 1}, {"resamples": 0}, {"block_size": 0}):
            with self.assertRaises(ValueError):
                summarize_windows(self.rows([1, 2, 3]), ["latency_app_s"], **settings)
        result = summarize_windows(self.rows([1, 2, 3]), ["latency_app_s"], block_size=2)
        self.assertEqual(result["groups"][0]["reason"], "insufficient_windows")


if __name__ == "__main__":
    unittest.main()

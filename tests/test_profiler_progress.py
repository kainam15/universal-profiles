import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock

from acprof.host.profiler_progress import report_profiler_completion


class ProfilerProgressTests(unittest.TestCase):
    def test_counts_source_samples_once_regardless_of_repeat(self) -> None:
        callback = Mock()
        report_profiler_completion(
            callback,
            profiler="Nsys",
            profiles=[
                {"repeat": 10, "entries": [{"error": ""}, {"error": ""}]},
                {"repeat": 10, "entries": [{"error": "timeout"}, {"error": ""}]},
            ],
            elapsed_seconds=65.0,
        )
        callback.assert_called_once()
        event = callback.call_args.args[0]
        self.assertEqual(event.profiler, "Nsys")
        self.assertEqual(event.status, "partial")
        self.assertEqual(event.total_samples, 4)
        self.assertEqual(event.error_samples, 1)
        self.assertEqual(event.elapsed_seconds, 65.0)
        self.assertEqual(event.detail, "timeout")

    def test_reports_empty_and_failed_results_without_claiming_success(self) -> None:
        for profiles, status, total, failed in (
            ([{"entries": [{"error": ""}]}], "success", 1, 0),
            ([{"entries": []}], "no_results", 0, 0),
            ([{"error": "tool_missing", "entries": []}], "failed", 0, 0),
            ([{"error": "tool_failed", "entries": [{}]}], "failed", 1, 1),
            ([{"entries": [{"error": "OOM"}]}], "failed", 1, 1),
            ([{"error": "OOM", "entries": [{"error": "OOM"}, {}]}], "partial", 2, 1),
        ):
            with self.subTest(status=status, profiles=profiles):
                callback = Mock()
                report_profiler_completion(
                    callback, profiler="CPU Torch", profiles=profiles, elapsed_seconds=0.5
                )
                event = callback.call_args.args[0]
                self.assertEqual(event.status, status)
                self.assertEqual(event.total_samples, total)
                self.assertEqual(event.error_samples, failed)

    def test_disabled_callback_does_not_inspect_results(self) -> None:
        profiles = Mock()
        report_profiler_completion(
            None, profiler="NCU", profiles=profiles, elapsed_seconds=1.0
        )
        self.assertEqual(profiles.mock_calls, [])

    def test_callback_error_does_not_escape_or_log_its_contents(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            report_profiler_completion(
                Mock(side_effect=RuntimeError("private request details")),
                profiler="Massif",
                profiles=[{"entries": [{"error": ""}]}],
                elapsed_seconds=1.0,
            )
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertNotIn("private request details", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

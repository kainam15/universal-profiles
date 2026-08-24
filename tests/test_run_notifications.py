import csv
import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from acprof.cli import run
from acprof.host import orchestrator
from acprof.notifications import NotificationConfigError, NotificationEvent


class RunNotificationLifecycleTests(unittest.TestCase):
    def tearDown(self) -> None:
        run._ACTIVE_RUN_NOTIFICATION = None
        run._ACTIVE_TMUX_TERMINAL_LOG = None

    @staticmethod
    def _context(notifier, *, total_cases=2):
        return run._RunNotificationContext(
            notifier=notifier,
            model_id="org/model",
            output_dir="/tmp/results/org--model",
            started_at=time.perf_counter() - 1.0,
            run_command="python run.py --model org/model",
            total_cases=total_cases,
        )

    @staticmethod
    def _success_event() -> NotificationEvent:
        return NotificationEvent(
            status="success",
            model_id="org/model",
            output_dir="/tmp/results/org--model",
            elapsed_seconds=1.0,
            total_cases=2,
            completed_cases=2,
            result_rows=4,
            error_rows=0,
            final_csv="/tmp/results/org--model/result_all.csv",
            host="test-host",
        )

    def test_default_mode_auto_enables_configured_wecom(self) -> None:
        notifier = Mock()
        self.assertEqual(run.DEFAULT_NOTIFY_PROVIDER, "auto")

        with patch(
            "acprof.cli.run.WeComWebhookNotifier.from_env",
            return_value=notifier,
        ) as from_env:
            run._activate_run_notification(
                provider=run.DEFAULT_NOTIFY_PROVIDER,
                model_id="org/model",
                output_dir="/tmp/results/org--model",
                started_at=time.perf_counter(),
                run_command="python run.py --model org/model",
            )

        from_env.assert_called_once_with()
        self.assertIsNotNone(run._ACTIVE_RUN_NOTIFICATION)
        self.assertIs(run._ACTIVE_RUN_NOTIFICATION.notifier, notifier)

    def test_default_mode_stays_silent_without_wecom_config(self) -> None:
        stderr = io.StringIO()
        with patch(
            "acprof.cli.run.WeComWebhookNotifier.from_env",
            side_effect=NotificationConfigError("missing"),
        ), redirect_stderr(stderr):
            run._activate_run_notification(
                provider=run.DEFAULT_NOTIFY_PROVIDER,
                model_id="org/model",
                output_dir="/tmp/results/org--model",
                started_at=time.perf_counter(),
                run_command="python run.py --model org/model",
            )

        self.assertIsNone(run._ACTIVE_RUN_NOTIFICATION)
        self.assertEqual(stderr.getvalue(), "")

    def test_start_notification_contains_current_command(self) -> None:
        captured = []

        class CapturingNotifier:
            def send(self, event):
                captured.append(event)

        run._ACTIVE_RUN_NOTIFICATION = self._context(CapturingNotifier())
        run._notify_run_started()

        self.assertEqual(len(captured), 1)
        event = captured[0]
        self.assertEqual(event.status, "started")
        self.assertEqual(event.run_command, "python run.py --model org/model")
        self.assertIn("环境预检", event.detail)
        self.assertGreaterEqual(event.elapsed_seconds, 0.0)
        self.assertIsNone(run._ACTIVE_RUN_NOTIFICATION.event)

    def test_run_main_sends_start_before_native_preflight(self) -> None:
        argv = [
            "run.py",
            "--model",
            "org/model with space",
            "--cpus",
            "1",
        ]
        expected_command = run._format_run_command(argv)
        order = []

        def activate(**kwargs):
            order.append(("activate", kwargs["run_command"]))

        def preflight():
            order.append(("preflight", None))
            raise RuntimeError("stop after ordering check")

        with patch.object(sys, "argv", argv), patch(
            "acprof.cli.run.bootstrap_project_env"
        ), patch(
            "acprof.cli.run._activate_run_notification",
            side_effect=activate,
        ), patch(
            "acprof.cli.run._start_tmux_terminal_log",
            side_effect=lambda *_args: order.append(("tmux", None)),
        ), patch(
            "acprof.cli.run._notify_run_started",
            side_effect=lambda: order.append(("started", None)),
        ), patch(
            "acprof.cli.run.require_native_linux_host",
            side_effect=preflight,
        ):
            with self.assertRaisesRegex(RuntimeError, "ordering check"):
                run._run_main()

        self.assertEqual(
            order,
            [
                ("activate", expected_command),
                ("tmux", None),
                ("started", None),
                ("preflight", None),
            ],
        )

    def test_completion_marks_error_rows_as_partial(self) -> None:
        notifier = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            result_csv = os.path.join(tmp, "result_all.csv")
            with open(result_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("status", "error"))
                writer.writeheader()
                writer.writerow({"status": "ok", "error": ""})
                writer.writerow({"status": "error", "error": "timeout"})

            run._ACTIVE_RUN_NOTIFICATION = self._context(notifier)
            run._record_run_completion(
                final_csv=result_csv,
                completed_cases=2,
            )

        event = run._ACTIVE_RUN_NOTIFICATION.event
        self.assertIsNotNone(event)
        self.assertEqual(event.status, "partial")
        self.assertEqual(event.result_rows, 2)
        self.assertEqual(event.error_rows, 1)
        notifier.send.assert_not_called()

    def test_case_progress_reports_current_case_after_csv_is_complete(self) -> None:
        captured = []

        class CapturingNotifier:
            def send(self, event):
                captured.append(event)

        with tempfile.TemporaryDirectory() as tmp:
            result_csv = os.path.join(tmp, "result_case.csv")
            with open(result_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("status", "error"))
                writer.writeheader()
                writer.writerow({"status": "ok", "error": ""})

            run._ACTIVE_RUN_NOTIFICATION = self._context(CapturingNotifier())
            run._notify_case_progress(
                orchestrator.MatrixProgress(
                    completed_cases=1,
                    total_cases=2,
                    cpu=2,
                    mem=4,
                    gpu="off",
                    result_csv=result_csv,
                )
            )

        self.assertEqual(len(captured), 1)
        event = captured[0]
        self.assertEqual(event.status, "progress")
        self.assertEqual(event.completed_cases, 1)
        self.assertEqual(event.total_cases, 2)
        self.assertEqual(event.result_rows, 1)
        self.assertEqual(event.error_rows, 0)
        self.assertIn("CPU=2, MEM=4GB, GPU=off", event.detail)

    def test_matrix_progress_callback_runs_after_each_case(self) -> None:
        order = []
        progress_events = []

        def fake_case(**kwargs):
            order.append(("case", kwargs["cpu"], kwargs["gpu"]))
            return f"/tmp/{kwargs['cpu']}-{kwargs['gpu']}.csv"

        def progress(event):
            order.append(("progress", event.cpu, event.gpu))
            progress_events.append(event)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.orchestrator.run_single_case",
            side_effect=fake_case,
        ):
            result_csvs = orchestrator.run_matrix(
                task_info=Mock(),
                image_info=Mock(),
                cpu_list=[1, 2],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                project_dir=tmp,
                progress_callback=progress,
            )

        self.assertEqual(len(result_csvs), 4)
        self.assertEqual(len(progress_events), 4)
        self.assertEqual(
            order,
            [
                ("case", 1, "off"),
                ("progress", 1, "off"),
                ("case", 1, "on"),
                ("progress", 1, "on"),
                ("case", 2, "off"),
                ("progress", 2, "off"),
                ("case", 2, "on"),
                ("progress", 2, "on"),
            ],
        )
        self.assertEqual(progress_events[-1].completed_cases, 4)
        self.assertEqual(progress_events[-1].total_cases, 4)

    def test_matrix_ignores_progress_callback_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.orchestrator.run_single_case",
            return_value="/tmp/result.csv",
        ), redirect_stderr(io.StringIO()):
            result_csvs = orchestrator.run_matrix(
                task_info=Mock(),
                image_info=Mock(),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                output_dir=tmp,
                project_dir=tmp,
                progress_callback=Mock(side_effect=RuntimeError("notify failed")),
            )

        self.assertEqual(result_csvs, ["/tmp/result.csv"])

    def test_main_finalizes_tmux_before_sending_success(self) -> None:
        order = []
        captured = []

        class CapturingNotifier:
            def send(self, event):
                order.append("notify")
                captured.append(event)

        terminal_log = ("%3", "/tmp/tmux_all.log.part", "/tmp/tmux_all.log")

        def finish_run():
            context = self._context(CapturingNotifier())
            context.event = self._success_event()
            run._ACTIVE_RUN_NOTIFICATION = context
            run._ACTIVE_TMUX_TERMINAL_LOG = terminal_log

        def stop_log(value):
            self.assertEqual(value, terminal_log)
            order.append("tmux")
            return True

        with patch("acprof.cli.run._run_main", side_effect=finish_run), patch(
            "acprof.cli.run._stop_tmux_terminal_log",
            side_effect=stop_log,
        ):
            run.main()

        self.assertEqual(order, ["tmux", "notify"])
        self.assertEqual(captured[0].terminal_log, terminal_log[2])

    def test_main_preserves_runtime_error_and_sends_failure(self) -> None:
        captured = []

        class CapturingNotifier:
            def send(self, event):
                captured.append(event)

        def fail_run():
            run._ACTIVE_RUN_NOTIFICATION = self._context(CapturingNotifier())
            raise RuntimeError("profiling failed")

        with patch("acprof.cli.run._run_main", side_effect=fail_run):
            with self.assertRaisesRegex(RuntimeError, "profiling failed"):
                run.main()

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].status, "failed")
        self.assertIn("RuntimeError: profiling failed", captured[0].detail)

    def test_main_preserves_system_exit_code_and_sends_failure(self) -> None:
        captured = []

        class CapturingNotifier:
            def send(self, event):
                captured.append(event)

        def fail_run():
            run._ACTIVE_RUN_NOTIFICATION = self._context(CapturingNotifier())
            raise SystemExit(7)

        with patch("acprof.cli.run._run_main", side_effect=fail_run):
            with self.assertRaises(SystemExit) as raised:
                run.main()

        self.assertEqual(raised.exception.code, 7)
        self.assertEqual(captured[0].status, "failed")
        self.assertIn("退出码 7", captured[0].detail)

    def test_unknown_delivery_error_does_not_change_success_result(self) -> None:
        unsafe_key = "secret-webhook-key"

        class FailingNotifier:
            def send(self, _event):
                raise RuntimeError(
                    "failed at https://qyapi.weixin.qq.com/cgi-bin/webhook/"
                    f"send?key={unsafe_key}"
                )

        def finish_run():
            context = self._context(FailingNotifier())
            context.event = self._success_event()
            run._ACTIVE_RUN_NOTIFICATION = context
            return 23

        stderr = io.StringIO()
        with patch("acprof.cli.run._run_main", side_effect=finish_run), redirect_stderr(
            stderr
        ):
            result = run.main()

        self.assertEqual(result, 23)
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertNotIn(unsafe_key, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

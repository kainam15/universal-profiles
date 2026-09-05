import csv
import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from acprof.cli import run
from acprof.host import orchestrator
from acprof.host.detect import TaskInfo
from acprof.host.profiler_progress import ProfilerProgress
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

    def test_profiler_completion_preserves_status_counts_and_final_event(self) -> None:
        cases = (
            ("success", 3, 0, ""),
            ("partial", 3, 1, "one scale failed"),
            ("failed", 3, 3, "probe failed"),
            ("no_results", 0, 0, ""),
        )
        for status, total_samples, error_samples, detail in cases:
            with self.subTest(status=status):
                notifier = Mock()
                context = self._context(notifier)
                final_event = self._success_event()
                context.event = final_event
                run._ACTIVE_RUN_NOTIFICATION = context

                run._notify_profiler_completion(
                    ProfilerProgress(
                        profiler="CPU Torch",
                        status=status,
                        elapsed_seconds=0.25,
                        total_samples=total_samples,
                        error_samples=error_samples,
                        detail=detail,
                    )
                )

                notifier.send.assert_called_once()
                event = notifier.send.call_args.args[0]
                self.assertEqual(event.status, f"profiler_{status}")
                self.assertEqual(event.profiler, "CPU Torch")
                self.assertEqual(event.profile_elapsed_seconds, 0.25)
                self.assertEqual(event.profile_samples, total_samples)
                self.assertEqual(event.profile_error_samples, error_samples)
                self.assertEqual(event.detail, detail or None)
                self.assertEqual(event.model_id, context.model_id)
                self.assertEqual(event.output_dir, context.output_dir)
                self.assertGreaterEqual(event.elapsed_seconds, 1.0)
                self.assertIs(context.event, final_event)

    def test_profiler_completion_is_silent_without_active_notifier(self) -> None:
        run._ACTIVE_RUN_NOTIFICATION = None
        with patch("acprof.cli.run._send_notification_event") as send:
            run._notify_profiler_completion(
                ProfilerProgress("NCU", "success", 2.0, 1, 0)
            )

        send.assert_not_called()

    def test_profiler_notification_failure_preserves_final_event_and_hides_secret(self) -> None:
        unsafe_key = "synthetic-profiler-notification-secret"
        notifier = Mock()
        notifier.send.side_effect = RuntimeError(
            "https://qyapi.weixin.qq.com/cgi-bin/webhook/"
            f"send?key={unsafe_key}"
        )
        context = self._context(notifier)
        final_event = self._success_event()
        context.event = final_event
        run._ACTIVE_RUN_NOTIFICATION = context
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            run._notify_profiler_completion(
                ProfilerProgress("Nsys", "success", 2.0, 1, 0)
            )

        notifier.send.assert_called_once()
        self.assertIs(context.event, final_event)
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertNotIn(unsafe_key, stderr.getvalue())

    def test_main_wires_profiler_notifications_before_matrix_with_resolved_model(self) -> None:
        self._assert_main_profiler_notifications(provider="auto")

    def test_main_disables_both_profiler_callbacks_when_notify_is_none(self) -> None:
        self._assert_main_profiler_notifications(provider="none")

    def _assert_main_profiler_notifications(self, *, provider: str) -> None:
        task_info = TaskInfo(
            model_id="org/resolved-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        order = []
        events = []
        notifier = Mock()

        def send(event):
            order.append(event.profiler or event.status)
            events.append(event)

        notifier.send.side_effect = send

        def collect(profilers, **kwargs):
            callback = kwargs.get("progress_callback")
            if callback is not None:
                for profiler in profilers:
                    callback(ProfilerProgress(profiler, "success", 2.0, 1, 0))
            return os.path.join(kwargs["output_dir"], "mock_profile_plan.json")

        def run_matrix(**_kwargs):
            order.append("matrix")
            return []

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            sys,
            "argv",
            [
                "run.py", "--model", "org/requested-model", "--skip-build",
                "--compute-profile-tool", "both", "--execution-profile-tool", "both",
                "--cpus", "1", "--mems", "4", "--gpus", "off,on",
                "--notify", provider, "--output-dir", tmp_dir,
            ],
        ), patch.multiple(
            run,
            bootstrap_project_env=Mock(),
            require_native_linux_host=Mock(),
            require_native_docker=Mock(),
            require_cgroup_prerequisites=Mock(return_value="v2"),
            require_packet_latency_prerequisites=Mock(),
            require_cpu_energy_prerequisites=Mock(),
            require_mips_prerequisites=Mock(),
            _start_tmux_terminal_log=Mock(return_value=None),
        ), patch(
            "acprof.cli.run.WeComWebhookNotifier.from_env", return_value=notifier,
        ) as from_env, patch(
            "acprof.host.detect.detect_task", return_value=task_info,
        ), patch(
            "acprof.host.orchestrator.collect_static_meta", return_value=SimpleNamespace(),
        ), patch(
            "acprof.host.orchestrator.write_static_meta_json",
        ), patch(
            "acprof.host.orchestrator.plan_input_scales",
            return_value=orchestrator.PlannedInputScales(
                scales=[1.0], source="unit", plan_file=None,
            ),
        ), patch(
            "acprof.host.compute_profile.collect_compute_profile_plan",
            side_effect=lambda **kwargs: collect(("CPU Torch", "GPU Torch", "NCU"), **kwargs),
        ) as compute, patch(
            "acprof.host.execution_profile.collect_execution_profile_plan",
            side_effect=lambda **kwargs: collect(("Massif", "Nsys"), **kwargs),
        ) as execution, patch(
            "acprof.host.orchestrator.run_matrix", side_effect=run_matrix,
        ), redirect_stdout(io.StringIO()):
            run.main()

        compute.assert_called_once()
        execution.assert_called_once()
        expected_callback = (
            None if provider == "none" else run._notify_profiler_completion
        )
        for collector in (compute, execution):
            self.assertIs(
                collector.call_args.kwargs["progress_callback"], expected_callback,
            )
        if provider == "none":
            from_env.assert_not_called()
            notifier.send.assert_not_called()
            self.assertEqual(order, ["matrix"])
        else:
            from_env.assert_called_once_with()
            self.assertEqual(
                order,
                ["started", "CPU Torch", "GPU Torch", "NCU", "Massif", "Nsys", "matrix", "no_results"],
            )
            for event in events[1:]:
                self.assertEqual(event.model_id, task_info.model_id)
                self.assertEqual(
                    event.output_dir, os.path.join(tmp_dir, "org--resolved-model"),
                )
            self.assertTrue(all(event.status == "profiler_success" for event in events[1:-1]))
            self.assertEqual(events[-1].total_cases, 2)

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

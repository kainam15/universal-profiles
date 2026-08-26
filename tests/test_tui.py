import asyncio
import csv
from pathlib import Path
import sys
import tempfile
import unittest

from textual.widgets import DataTable, Input, Select, Static
from textual.css.query import NoMatches

from acprof.cli.tui import (
    AcprofTui,
    ConfirmActionScreen,
    PendingLaunch,
    StatusCheckbox,
)
from acprof.cli.tui_core import (
    _readable_rapl_paths,
    RunConfig,
    RunProgressTracker,
    TuiConfigError,
    build_probe_command,
    build_profile_command,
    build_run_command,
    format_command,
    parse_slash_command,
    summarize_result_csv,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class TuiCoreTests(unittest.TestCase):
    def test_default_config_disables_compute_profiler(self):
        config = RunConfig(model="demo/model")
        command = build_run_command(
            config,
            project_dir=PROJECT_DIR,
            python_executable="python",
        )

        self.assertEqual(config.compute_profile_tool, "none")
        self.assertEqual(
            command[command.index("--compute-profile-tool") + 1],
            "none",
        )

    def test_smoke_command_delegates_to_existing_run_entrypoint(self):
        config = RunConfig.smoke("google-bert/bert-base-uncased")
        command = build_run_command(
            config,
            project_dir=PROJECT_DIR,
            python_executable=PROJECT_DIR / ".venv/bin/python",
        )

        self.assertEqual(command[0], str(PROJECT_DIR / ".venv/bin/python"))
        self.assertEqual(command[1:3], ["-u", str(PROJECT_DIR / "run.py")])
        self.assertEqual(command[command.index("--cpus") + 1], "1")
        self.assertEqual(command[command.index("--mems") + 1], "4")
        self.assertEqual(command[command.index("--input-scales") + 1], "64")
        self.assertEqual(
            command[command.index("--compute-profile-tool") + 1],
            "none",
        )
        self.assertNotIn("--allow-cgroup-v1", command)
        self.assertIn("run.py", format_command(command, project_dir=PROJECT_DIR))

    def test_probe_command_uses_matrix_bounds_without_collection_options(self):
        config = RunConfig(
            model="demo/model",
            cpus="1,4",
            mems="2,8",
            gpus="off,on",
            input_scales="64,512",
            warmup=9,
            repeat=11,
            idle_seconds=99,
            skip_build=True,
        )

        command = build_probe_command(
            config,
            project_dir=PROJECT_DIR,
            python_executable="python",
        )

        self.assertEqual(command[1:3], ["-u", str(PROJECT_DIR / "probe.py")])
        self.assertEqual(command[command.index("--cpus") + 1], "1,4")
        self.assertEqual(command[command.index("--mems") + 1], "2,8")
        self.assertEqual(command[command.index("--input-scales") + 1], "64,512")
        self.assertIn("--skip-build", command)
        self.assertNotIn("--warmup", command)
        self.assertNotIn("--repeat", command)
        self.assertNotIn("--idle-seconds", command)
        self.assertNotIn("--compute-profile-tool", command)
        self.assertIn("probe.py", format_command(command, project_dir=PROJECT_DIR))

    def test_invalid_matrix_is_rejected_before_launch(self):
        with self.assertRaises(TuiConfigError) as context:
            RunConfig(
                model="demo/model",
                cpus="1,1",
                mems="4,8",
                gpus="off",
            ).validate(project_dir=PROJECT_DIR)
        self.assertIn("CPU 列表不能重复", str(context.exception))

    def test_optional_overrides_and_flags_are_preserved(self):
        config = RunConfig(
            model="demo/model",
            task="text-generation",
            task_family="nlp",
            backend="transformers_pipeline",
            cpus="2",
            mems="8",
            gpus="off",
            input_scales="64,128",
            skip_build=True,
            prune_startup_oom=False,
            idle_debug=True,
        )
        command = build_run_command(
            config,
            project_dir=PROJECT_DIR,
            python_executable="python",
        )
        self.assertIn("--task", command)
        self.assertIn("--task-family", command)
        self.assertIn("--backend", command)
        self.assertIn("--skip-build", command)
        self.assertIn("--no-prune-startup-oom", command)
        self.assertIn("--idle-debug", command)

    def test_progress_tracker_marks_measurement_and_completion(self):
        tracker = RunProgressTracker()
        tracker.feed("Resource matrix: 1 CPUs x 1 MEMs x 1 GPUs = 1 cases")
        tracker.feed("# Case 1/1: CPU=2, MEM=8GB, GPU=off")
        measuring = tracker.feed("[case] Running workload...")
        self.assertTrue(measuring.measurement_active)
        self.assertEqual(measuring.stage, "正式测量")

        tracker.feed("[case][WARN] recoverable warning")
        tracker.feed("[case] Stopping container...")
        tracker.feed("[case] Done. Output: result.csv")
        tracker.feed("[merge] Final CSV: /tmp/result_all.csv (3 rows)")
        complete = tracker.feed("Profiling complete!")

        self.assertFalse(complete.measurement_active)
        self.assertEqual(complete.completed_cases, 1)
        self.assertEqual(complete.warnings, 1)
        self.assertEqual(complete.final_csv, "/tmp/result_all.csv")
        self.assertEqual(complete.stage, "已完成")

    def test_progress_tracker_reports_largest_scale_probe_timings(self):
        tracker = RunProgressTracker()
        starting = tracker.feed(
            "[largest-probe] Starting minimum configuration: "
            "CPU=1, MEM=2GB, GPU=off, input_scale=512"
        )
        self.assertEqual(starting.stage, "启动探测容器")
        self.assertEqual((starting.cpu, starting.mem, starting.gpu), ("1", "2", "off"))
        measuring = tracker.feed(
            "[largest-probe] Running one largest-scale request..."
        )
        self.assertTrue(measuring.measurement_active)

        complete = tracker.feed(
            "[largest-probe] RESULT status=ok input_scale=512 cpu=1 mem=2 "
            "gpu=off cold_start_s=12.500000 request_s=4.321000 "
            "ready_plus_request_s=16.821000"
        )
        complete = tracker.feed(
            "[largest-probe] Summary JSON: /tmp/probes/largest_scale_probe.json"
        )

        self.assertFalse(complete.measurement_active)
        self.assertEqual(complete.stage, "探测完成")
        self.assertEqual(complete.completed_cases, 1)
        self.assertIn("单次请求 4.321s", complete.detail)
        self.assertEqual(
            complete.probe_summary,
            "/tmp/probes/largest_scale_probe.json",
        )

    def test_progress_tracker_shows_memory_scan_before_timing(self):
        tracker = RunProgressTracker()
        scan = tracker.feed(
            "[largest-probe] MEMORY_SCAN cpu=1 gpu=off "
            "candidates=2,4,8 input_scale=512"
        )
        self.assertEqual(scan.stage, "准备内存探测")
        self.assertEqual(scan.total_cases, 3)

        tracker.feed(
            "[largest-probe] MEMORY_TRY current=1 total=3 "
            "cpu=1 mem=2 gpu=off input_scale=512"
        )
        first = tracker.feed(
            "[largest-probe] MEMORY_RESULT mem=2 status=startup_oom"
        )
        self.assertFalse(first.measurement_active)
        self.assertIn("启动 OOM", first.detail)
        self.assertEqual(first.completed_cases, 1)

        tracker.feed(
            "[largest-probe] MEMORY_TRY current=2 total=3 "
            "cpu=1 mem=4 gpu=off input_scale=512"
        )
        measuring = tracker.feed(
            "[largest-probe] Running one largest-scale request..."
        )
        self.assertTrue(measuring.measurement_active)
        found = tracker.feed(
            "[largest-probe] MEMORY_RESULT mem=4 status=ok"
        )
        self.assertEqual(found.stage, "找到最低可用内存")
        self.assertIn("4GB", found.detail)

        complete = tracker.feed(
            "[largest-probe] RESULT status=ok input_scale=512 cpu=1 mem=4 "
            "gpu=off cold_start_s=5 request_s=2 "
            "ready_plus_request_s=7"
        )
        self.assertEqual(complete.stage, "探测完成")
        self.assertEqual(complete.total_cases, 2)
        self.assertEqual(complete.completed_cases, 2)
        self.assertIn("最低可用内存 4GB", complete.detail)

    def test_result_summary_and_profile_dry_run(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_dir = Path(temporary_dir)
            result_csv = result_dir / "result_all.csv"
            with result_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "status",
                        "warmup",
                        "cpu_cores",
                        "mem_cap_gb",
                        "gpu_mode",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "status": "ok",
                            "warmup": "1",
                            "cpu_cores": "1",
                            "mem_cap_gb": "4",
                            "gpu_mode": "off",
                        },
                        {
                            "status": "error",
                            "warmup": "0",
                            "cpu_cores": "1",
                            "mem_cap_gb": "4",
                            "gpu_mode": "off",
                        },
                    )
                )

            summary = summarize_result_csv(result_csv)
            self.assertEqual(summary.rows, 2)
            self.assertEqual(summary.ok_rows, 1)
            self.assertEqual(summary.error_rows, 1)
            self.assertEqual(summary.warmup_rows, 1)
            self.assertEqual(summary.cases, 1)

            profile = build_profile_command(
                result_dir,
                tools="torch,ncu",
                dry_run=True,
                project_dir=PROJECT_DIR,
                python_executable="python",
            )
            self.assertEqual(profile[-1], "--dry-run")

    def test_summarize_result_csv_with_latencies(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_csv = Path(temporary_dir) / "result_all.csv"
            with result_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "status",
                        "warmup",
                        "cpu_cores",
                        "mem_cap_gb",
                        "gpu_mode",
                        "latency_app_s",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "status": "ok",
                            "warmup": "1",
                            "cpu_cores": "1",
                            "mem_cap_gb": "4",
                            "gpu_mode": "off",
                            "latency_app_s": "0.100",
                        },
                        {
                            "status": "ok",
                            "warmup": "0",
                            "cpu_cores": "1",
                            "mem_cap_gb": "4",
                            "gpu_mode": "off",
                            "latency_app_s": "0.050",
                        },
                        {
                            "status": "ok",
                            "warmup": "0",
                            "cpu_cores": "2",
                            "mem_cap_gb": "4",
                            "gpu_mode": "off",
                            "latency_app_s": "0.030",
                        },
                    )
                )
            summary = summarize_result_csv(result_csv)
            self.assertEqual(summary.rows, 3)
            self.assertEqual(summary.ok_rows, 3)
            self.assertEqual(summary.warmup_rows, 1)
            self.assertEqual(summary.cases, 2)
            self.assertAlmostEqual(summary.min_latency_s, 0.030)
            self.assertAlmostEqual(summary.max_latency_s, 0.050)
            self.assertAlmostEqual(summary.avg_latency_s, 0.040)

    def test_rapl_check_does_not_follow_cyclic_sysfs_links(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            package = root / "intel-rapl:0"
            package.mkdir()
            (package / "energy_uj").write_text("123\n", encoding="utf-8")
            (package / "device").symlink_to(package, target_is_directory=True)
            subdomain = root / "intel-rapl:0:0"
            subdomain.mkdir()
            (subdomain / "energy_uj").write_text("456\n", encoding="utf-8")

            paths = _readable_rapl_paths(root)

            self.assertEqual(paths, [str(package / "energy_uj")])

    def test_slash_command_parser_does_not_execute_shell_text(self):
        command, args = parse_slash_command('/plot "results/a b/result_all.csv"')
        self.assertEqual(command, "plot")
        self.assertEqual(args, ["results/a b/result_all.csv"])
        with self.assertRaises(TuiConfigError):
            parse_slash_command("plot result.csv")


class TuiAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_mounts_and_requires_confirmation_before_run(self):
        app = AcprofTui(RunConfig.smoke("google-bert/bert-base-uncased"))
        async with app.run_test(size=(150, 52)) as pilot:
            await pilot.pause()
            self.assertFalse(app.ENABLE_COMMAND_PALETTE)
            self.assertFalse(app.use_command_palette)
            self.assertFalse(app.query_one("HeaderIcon").display)
            self.assertFalse(
                app.query_one("#allow-cgroup-v1", StatusCheckbox).display
            )
            command_bar = app.query_one("#slash-command-bar")
            self.assertEqual(command_bar.styles.padding.top, 1)
            self.assertEqual(command_bar.styles.padding.right, 2)
            self.assertEqual(command_bar.styles.padding.bottom, 1)
            self.assertEqual(command_bar.styles.padding.left, 2)
            slash_command = app.query_one("#slash-command", Input)
            self.assertEqual(slash_command.parent.id, "slash-command-bar")
            self.assertTrue(slash_command.placeholder.startswith("快捷命令："))
            self.assertIn("/help", slash_command.placeholder)
            footer = app.query_one("Footer")
            self.assertEqual(command_bar.region.bottom, footer.region.y)
            self.assertEqual(slash_command.region.y - command_bar.region.y, 1)
            self.assertEqual(footer.region.y - slash_command.region.bottom, 1)
            preview = str(app.query_one("#command-preview", Static).render())
            self.assertIn("/.venv/bin/python", preview)
            self.assertIn("--warmup 0", preview)
            self.assertIn("--idle-seconds 20", preview)
            self.assertIn("--idle-cooldown-seconds 5", preview)
            with self.assertRaises(NoMatches):
                app.query_one("#preview-command")
            with self.assertRaises(NoMatches):
                app.query_one("#back-config")
            self.assertTrue(app.query_one("#advanced-settings").collapsed)
            self.assertTrue(app.query_one("#command-details").collapsed)
            self.assertEqual(
                str(app.query_one("#probe-largest").label),
                "探测最大输入",
            )

            prune = app.query_one("#prune-startup-oom", StatusCheckbox)
            reuse = app.query_one("#skip-build", StatusCheckbox)
            self.assertEqual(StatusCheckbox.BUTTON_INNER, "✓")
            self.assertTrue(prune.value)
            self.assertTrue(prune.has_class("-on"))
            self.assertFalse(reuse.value)
            self.assertFalse(reuse.has_class("-on"))

            app.query_one("#cpus", Input).value = "1,3"
            await pilot.pause(0.1)
            auto_preview = str(app.query_one("#command-preview", Static).render())
            self.assertIn("--cpus 1,3", auto_preview)
            self.assertEqual(app.query_one("#run-preset").value, "custom")

            app.query_one("#run-preset", Select).value = "main"
            await pilot.pause(0.1)
            preset_preview = str(app.query_one("#command-preview", Static).render())
            self.assertEqual(app.query_one("#cpus", Input).value, "1,2,4,8")
            self.assertIn("--cpus 1,2,4,8", preset_preview)
            self.assertIn("--compute-profile-tool none", preset_preview)

            app.action_request_probe()
            await pilot.pause()
            self.assertIsInstance(app.screen, ConfirmActionScreen)
            self.assertIsNotNone(app._pending_launch)
            self.assertEqual(app._pending_launch.kind, "probe")
            await pilot.press("escape")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, ConfirmActionScreen)

            app.action_request_run()
            await pilot.pause()
            self.assertIsInstance(app.screen, ConfirmActionScreen)
            self.assertFalse(app._is_busy())
            await pilot.press("escape")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, ConfirmActionScreen)

    async def test_subprocess_progress_is_event_driven(self):
        script = "\n".join(
            (
                "print('Resource matrix: x = 1 cases', flush=True)",
                "print('# Case 1/1: CPU=1, MEM=4GB, GPU=off', flush=True)",
                "print('[case] Running workload...', flush=True)",
                "print('routine measurement detail', flush=True)",
                "print('[case] Stopping container...', flush=True)",
                "print('[case] Done. Output: result.csv', flush=True)",
                "print('Profiling complete!', flush=True)",
            )
        )
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            app._launch(
                PendingLaunch(
                    (sys.executable, "-u", "-c", script),
                    "run",
                    RunConfig.smoke("demo/model"),
                )
            )
            for _ in range(40):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break
            self.assertFalse(app._is_busy())
            self.assertEqual(app._latest_snapshot.stage, "已完成")
            self.assertEqual(app._latest_snapshot.completed_cases, 1)

    async def test_probe_subprocess_keeps_timing_result_in_monitor(self):
        script = "\n".join(
            (
                "print('[largest-probe] Starting minimum configuration: "
                "CPU=1, MEM=2GB, GPU=off, input_scale=512', flush=True)",
                "print('[largest-probe] Running one largest-scale request...', flush=True)",
                "print('[largest-probe] RESULT status=ok input_scale=512 "
                "cpu=1 mem=2 gpu=off cold_start_s=12.5 request_s=4.321 "
                "ready_plus_request_s=16.821', flush=True)",
                "print('[largest-probe] Summary JSON: /tmp/probe.json', flush=True)",
            )
        )
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            app._launch(
                PendingLaunch(
                    (sys.executable, "-u", "-c", script),
                    "probe",
                    RunConfig.smoke("demo/model"),
                )
            )
            for _ in range(40):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break

            self.assertFalse(app._is_busy())
            self.assertEqual(app._latest_snapshot.stage, "探测完成")
            self.assertIn("单次请求 4.321s", app._latest_snapshot.detail)
            self.assertEqual(app._latest_snapshot.probe_summary, "/tmp/probe.json")

    async def test_elapsed_timer_skips_during_measurement(self):
        """Elapsed ticker must not cause widget updates during measurement."""
        script = "\n".join(
            (
                "import time",
                "print('Resource matrix: x = 1 cases', flush=True)",
                "print('# Case 1/1: CPU=1, MEM=4GB, GPU=off', flush=True)",
                "print('[case] Running workload...', flush=True)",
                "time.sleep(0.2)",
                "print('[case] Stopping container...', flush=True)",
                "print('[case] Done. Output: result.csv', flush=True)",
                "print('Profiling complete!', flush=True)",
            )
        )
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            app._launch(
                PendingLaunch(
                    (sys.executable, "-u", "-c", script),
                    "run",
                    RunConfig.smoke("demo/model"),
                )
            )
            # The elapsed timer should be active during the run.
            self.assertIsNotNone(app._elapsed_timer)
            for _ in range(40):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break
            # After completion the timer is stopped.
            self.assertIsNone(app._elapsed_timer)
            self.assertFalse(app._is_busy())

    async def test_matrix_board_populates_for_run(self):
        """The resource matrix DataTable should populate on run launch."""
        config = RunConfig.smoke("demo/model")
        app = AcprofTui(config)
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            table = app.query_one("#matrix-table", DataTable)
            self.assertEqual(table.row_count, 0)
            script = "\n".join(
                (
                    "print('Resource matrix: x = 1 cases', flush=True)",
                    "print('# Case 1/1: CPU=1, MEM=4GB, GPU=off', flush=True)",
                    "print('[case] Running workload...', flush=True)",
                    "print('[case] Stopping container...', flush=True)",
                    "print('[case] Done. Output: result.csv', flush=True)",
                    "print('Profiling complete!', flush=True)",
                )
            )
            app._launch(
                PendingLaunch(
                    (sys.executable, "-u", "-c", script),
                    "run",
                    config,
                )
            )
            await pilot.pause(0.1)
            self.assertEqual(table.row_count, 1)
            for _ in range(40):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break
            self.assertFalse(app._is_busy())

    async def test_stage_gets_color_class(self):
        """The status-stage widget should receive CSS classes for visual state."""
        from acprof.cli.tui_core import ProgressSnapshot
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            stage_widget = app.query_one("#status-stage", Static)
            app._render_snapshot(ProgressSnapshot(stage="已完成"))
            await pilot.pause()
            self.assertTrue(stage_widget.has_class("stage-success"))
            app._render_snapshot(ProgressSnapshot(stage="失败"))
            await pilot.pause()
            self.assertTrue(stage_widget.has_class("stage-error"))
            self.assertFalse(stage_widget.has_class("stage-success"))
            app._render_snapshot(ProgressSnapshot(stage="正式测量"))
            await pilot.pause()
            self.assertTrue(stage_widget.has_class("stage-measuring"))

    async def test_matrix_slash_command_toggles_board(self):
        """Slash command /matrix toggles the matrix board collapsible."""
        from textual.widgets import Collapsible
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            board = app.query_one("#matrix-board", Collapsible)
            self.assertTrue(board.collapsed)
            input_widget = app.query_one("#slash-command", Input)
            input_widget.focus()
            input_widget.value = "/matrix"
            await pilot.press("enter")
            await pilot.pause()
            self.assertFalse(board.collapsed)


if __name__ == "__main__":
    unittest.main()

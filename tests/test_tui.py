import asyncio
import csv
from pathlib import Path
import sys
import tempfile
import unittest

from textual.widgets import Input, Select, Static
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
    build_profile_command,
    build_run_command,
    format_command,
    parse_slash_command,
    summarize_result_csv,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class TuiCoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

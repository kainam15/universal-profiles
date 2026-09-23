"""工具入口必须可以脱离源码工作目录运行。"""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shlex
from unittest.mock import patch

from acprof.tui.commands import RunConfig, build_run_command


ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def invoke(self, arguments, cwd):
        return subprocess.run(arguments, cwd=cwd, text=True, capture_output=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=30)

    def test_module_help_from_an_empty_workspace(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = self.invoke([sys.executable, "-m", "acprof", "--help"], workspace)
            self.assertEqual(result.returncode, 0, result.stderr)
            for command in ("run", "tui", "probe", "plot", "doctor"):
                self.assertIn(command, result.stdout)
            self.assertEqual(list(Path(workspace).iterdir()), [])

    def test_unknown_command_is_an_argument_error(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = self.invoke([sys.executable, "-m", "acprof", "unknown"], workspace)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_doctor_json_is_machine_readable_even_when_prerequisites_are_missing(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = self.invoke([sys.executable, "-m", "acprof", "doctor", "--json",
                                  "--profiling-mode", "basic"], workspace)
            self.assertIn(result.returncode, (0, 1), result.stderr)
            self.assertTrue(result.stdout.startswith("{"), result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["profiling_mode"], "basic")
            self.assertEqual(report["ready"], result.returncode == 0)
            self.assertTrue(report["checks"])

    def test_tui_run_command_works_without_root_scripts(self):
        with tempfile.TemporaryDirectory() as workspace:
            command = build_run_command(RunConfig.smoke("example/model"),
                                        project_dir=Path(workspace))
            result = self.invoke([*command, "--help"], workspace)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--profiling-mode", result.stdout)

    def test_recorded_frozen_command_is_executable(self):
        from acprof.cli.run import _format_run_command
        with patch("sys.frozen", True, create=True), patch("sys.executable", "/opt/AC Prof/acprof"):
            command = _format_run_command(["acprof run", "--model", "example/model"])
        self.assertEqual(shlex.split(command), ["/opt/AC Prof/acprof", "run", "--model", "example/model"])

    def test_posthoc_recognizes_installed_runs_in_their_own_workspace(self):
        from acprof.host.posthoc.storage import find_active_processes
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = root / "proc" / "999999999"
            process.mkdir(parents=True)
            (process / "cwd").symlink_to(root, target_is_directory=True)
            for prefix in (["/opt/python", "/opt/bin/acprof", "run"],
                           ["/opt/acprof-linux-x86_64", "run"],
                           ["/opt/python", "-m", "acprof", "run"]):
                with self.subTest(prefix=prefix):
                    (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in
                        [*prefix, "--model", "org/model", "--output-dir", "results"]))
                    with patch("acprof.host.posthoc.storage.Path", side_effect=lambda value:
                               root / "proc" if value == "/proc" else Path(value)):
                        matches = find_active_processes(root / "results" / "org--model", model_id="org/model")
                    self.assertEqual([pid for pid, _ in matches], [999999999])


if __name__ == "__main__":
    unittest.main()

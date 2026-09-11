"""Quick checks must exercise the same perf access paths as formal collection."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from acprof.cli.tui_core import RunConfig, quick_preflight
from acprof.monitors.perf_mips import PERF_PROBE_TIMEOUT_S


class TuiPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project_dir = Path(temporary.name)
        environment = patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        which = patch(
            "acprof.tui.diagnostics.shutil.which",
            side_effect=lambda command, **kwargs: f"/usr/bin/{command}",
        )
        self.which = which.start()
        self.addCleanup(which.stop)
        host_platform = patch("acprof.tui.diagnostics.platform.platform", return_value="test Linux")
        host_platform.start()
        self.addCleanup(host_platform.stop)
        rapl = patch("acprof.tui.diagnostics._readable_rapl_paths", return_value=["/fake/energy_uj"])
        rapl.start()
        self.addCleanup(rapl.stop)

    @staticmethod
    def result(returncode=0, stderr="1000,,instructions,100.00,,\n"):
        return subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)

    @staticmethod
    def host_command(command, **kwargs):
        tool = Path(command[0]).name
        if tool == "docker":
            if tuple(command[1:3]) == ("context", "show"):
                stdout = "default"
            elif tuple(command[1:3]) == ("context", "inspect"):
                stdout = "unix:///var/run/docker.sock"
            else:
                stdout = "test-host|Linux"
        elif tool == "ip":
            stdout = "docker0"
        elif tool == "nvidia-smi":
            stdout = "test GPU"
        elif tool == "perf":
            raise AssertionError("quick checks must use the shared perf resolver")
        else:
            raise AssertionError(f"unexpected host command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    def run_check(self, responses):
        with patch("acprof.monitors.perf_mips.subprocess.run", side_effect=responses) as run:
            checks = quick_preflight(
                RunConfig(model="", gpus="on"), project_dir=self.project_dir,
                command_runner=self.host_command,
            )
        # A perf failure must leave the remaining independent checks usable.
        self.assertEqual(next(check for check in checks if check.label == "NVIDIA GPU").status, "ok")
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["timeout"], PERF_PROBE_TIMEOUT_S)
        return next(check for check in checks if check.label == "perf instructions"), run

    def test_direct_perf_success_does_not_try_sudo(self):
        check, run = self.run_check([self.result()])
        self.assertEqual(check.status, "ok")
        self.assertIn("普通用户 perf 可用", check.detail)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "perf")
        self.assertEqual(run.call_args.kwargs["input"], "")

    def test_noninteractive_sudo_is_reported_as_available(self):
        check, run = self.run_check([
            self.result(1, "Access to performance monitoring is limited."), self.result(),
        ])
        self.assertEqual(check.status, "ok")
        self.assertIn("sudo perf 可用（无需交互输入）", check.detail)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0][:3], ["sudo", "-n", "perf"])

    def test_local_password_sudo_is_used_and_reloaded_without_retaining_credentials(self):
        for password in ("test-first-password", "test-updated-password"):
            with self.subTest(password_source="updated local file"):
                (self.project_dir / ".env.local").write_text(
                    f"ACPROF_SUDO_PASSWORD={password}\n", encoding="utf-8",
                )
                check, run = self.run_check([
                    self.result(1, "Permission denied"),
                    self.result(1, "sudo: a password is required"),
                    self.result(),
                ])
                self.assertEqual(check.status, "ok")
                self.assertIn("sudo perf 可用（已配置凭据）", check.detail)
                self.assertEqual(run.call_count, 3)
                self.assertEqual(run.call_args.args[0][:5], ["sudo", "-S", "-p", "", "perf"])
                self.assertEqual(run.call_args.kwargs["input"], password + "\n")
                self.assertEqual(run.call_args.kwargs["env"]["ACPROF_SUDO_PASSWORD"], password)
                self.assertNotIn(password, str(run.call_args.args[0]))
                self.assertNotIn(password, check.detail)
                self.assertNotIn("ACPROF_SUDO_PASSWORD", os.environ)

    def test_explicit_process_credential_takes_precedence_over_local_file(self):
        (self.project_dir / ".env.local").write_text(
            "ACPROF_SUDO_PASSWORD=test-file-password\n", encoding="utf-8",
        )
        with patch.dict(os.environ, {"ACPROF_SUDO_PASSWORD": "test-process-password"}):
            check, run = self.run_check([
                self.result(1, "Permission denied"),
                self.result(1, "sudo: a password is required"),
                self.result(),
            ])
            self.assertEqual(check.status, "ok")
            self.assertEqual(run.call_args.kwargs["input"], "test-process-password\n")
            self.assertEqual(os.environ["ACPROF_SUDO_PASSWORD"], "test-process-password")

    def test_failures_keep_all_attempt_details_and_redact_password(self):
        password = "test-diagnostic-password"
        (self.project_dir / ".env.local").write_text(
            f"ACPROF_SUDO_PASSWORD={password}\n", encoding="utf-8",
        )
        check, run = self.run_check([
            self.result(1, "Permission denied\nperf_event_paranoid setting is 4"),
            self.result(1, "sudo: a password is required"),
            self.result(1, f"rejected {password}\ninstructions event not supported"),
        ])
        self.assertEqual(check.status, "fail")
        self.assertEqual(run.call_count, 3)
        self.assertIn("Permission denied\nperf_event_paranoid setting is 4", check.detail)
        self.assertIn("sudo -n perf: sudo: a password is required", check.detail)
        self.assertIn("sudo -S perf:", check.detail)
        self.assertIn("instructions event not supported", check.detail)
        self.assertNotIn(password, check.detail)

    def test_zero_exit_without_an_instruction_count_is_failure(self):
        check, _ = self.run_check([
            self.result(0, "<not supported>,,instructions,0.00,,\n"),
            self.result(0, "<not counted>,,instructions,0.00,,\n"),
        ])
        self.assertEqual(check.status, "fail")
        self.assertIn("<not supported>", check.detail)
        self.assertIn("<not counted>", check.detail)

    def test_timeout_and_missing_sudo_are_visible_without_stopping_gpu_check(self):
        check, _ = self.run_check([
            subprocess.TimeoutExpired(["perf", "stat"], PERF_PROBE_TIMEOUT_S),
            FileNotFoundError("sudo not installed"),
        ])
        self.assertEqual(check.status, "fail")
        self.assertIn("TimeoutExpired", check.detail)
        self.assertIn("sudo not installed", check.detail)

    def test_missing_perf_is_reported_without_running_a_probe(self):
        self.which.side_effect = lambda command, **kwargs: None if command == "perf" else f"/usr/bin/{command}"
        check, run = self.run_check([])
        self.assertEqual(check.status, "fail")
        self.assertIn("perf command was not found", check.detail)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from acprof.monitors import perf_mips


class PerfMIPSTests(unittest.TestCase):
    def test_parses_perf_stat_csv_output(self) -> None:
        parsed = perf_mips.parse_perf_stat_output(
            """
123456789,,instructions,100.00,,
1.250000000 seconds time elapsed
"""
        )

        self.assertEqual(parsed.instructions_total, 123_456_789)
        self.assertAlmostEqual(parsed.perf_elapsed_s, 1.25)

    def test_preflight_accepts_modern_perf_csv_without_elapsed_line(self) -> None:
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["perf", "stat"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="1000,,instructions,633068,100.00,,\n",
                )
            raise AssertionError(f"unexpected command: {cmd}")

        with patch("acprof.monitors.perf_mips.shutil.which", return_value="/usr/bin/perf"), patch(
            "acprof.monitors.perf_mips.subprocess.run",
            side_effect=fake_run,
        ):
            prefix = perf_mips.resolve_perf_command_prefix()

        self.assertEqual(prefix, ["perf"])

    def test_monitor_uses_direct_perf_when_preflight_allows_it(self) -> None:
        popen_cmds = []

        class FakeProcess:
            returncode = 0

            def __init__(self, cmd, **kwargs):
                popen_cmds.append(cmd)
                self.stderr = io.StringIO(
                    "500000,,instructions,100.00,,\n"
                    "0.250000000 seconds time elapsed\n"
                )

            def send_signal(self, sig):
                self.signal = sig

            def poll(self):
                return None

            def communicate(self, timeout=None):
                self.returncode = 0
                return "", self.stderr.getvalue()

            def kill(self):
                self.returncode = -9

        fake_pid = SimpleNamespace(returncode=0, stdout="1234\n", stderr="")
        with patch("acprof.monitors.perf_mips.subprocess.run", return_value=fake_pid), patch(
            "acprof.monitors.perf_mips.subprocess.Popen",
            side_effect=lambda cmd, **kwargs: FakeProcess(cmd, **kwargs),
        ):
            monitor = perf_mips.PerfMIPSMonitor(
                container_name="case_container",
                command_prefix=["perf"],
            )
            monitor.start()
            result = monitor.stop(repeat_in_window=2, latency_app_s=0.125)

        self.assertEqual(popen_cmds[0][:5], ["perf", "stat", "--no-big-num", "-x", ","])
        self.assertIn("-p", popen_cmds[0])
        self.assertIn("1234", popen_cmds[0])
        self.assertEqual(result.instructions_total, 500_000)
        self.assertEqual(result.instructions_per_request, 250_000.0)
        self.assertAlmostEqual(result.cpu_mips_app, 2.0)

    def test_monitor_uses_password_sudo_when_pid_attach_needs_privilege(self) -> None:
        popen_cmds = []
        run_calls = []

        class FakeStdin:
            def write(self, text):
                self.text = text

            def flush(self):
                return None

            def close(self):
                return None

        class FakeProcess:
            returncode = 0

            def __init__(self, cmd, **kwargs):
                popen_cmds.append(cmd)
                self.stdin = FakeStdin()

            def send_signal(self, sig):
                self.signal = sig

            def poll(self):
                return None

            def communicate(self, timeout=None):
                self.returncode = 0
                return "", (
                    "500000,,instructions,100.00,,\n"
                    "0.250000000 seconds time elapsed\n"
                )

            def kill(self):
                self.returncode = -9

        def fake_run(cmd, **kwargs):
            run_calls.append((cmd, kwargs))
            if cmd[:2] == ["docker", "inspect"]:
                return SimpleNamespace(returncode=0, stdout="1234\n", stderr="")
            if cmd[:2] == ["perf", "stat"] and "-p" not in cmd:
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="1000,,instructions,633068,100.00,,\n",
                )
            if cmd[:2] == ["perf", "stat"] and "-p" in cmd:
                return SimpleNamespace(
                    returncode=255,
                    stdout="",
                    stderr="Access to performance monitoring and observability operations is limited.",
                )
            if cmd[:3] == ["sudo", "-n", "perf"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="sudo password required")
            if cmd[:4] == ["sudo", "-S", "-p", ""]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="<not counted>,,instructions,0,100.00,,\n",
                )
            raise AssertionError(f"unexpected command: {cmd}")

        with patch.dict("acprof.monitors.perf_mips.os.environ", {"ACPROF_SUDO_PASSWORD": "secret"}), patch(
            "acprof.monitors.perf_mips.subprocess.run",
            side_effect=fake_run,
        ), patch(
            "acprof.monitors.perf_mips.subprocess.Popen",
            side_effect=lambda cmd, **kwargs: FakeProcess(cmd, **kwargs),
        ):
            monitor = perf_mips.PerfMIPSMonitor(container_name="case_container")
            monitor.start()
            result = monitor.stop(repeat_in_window=2, latency_app_s=0.125)

        self.assertEqual(popen_cmds[0][:5], ["sudo", "-S", "-p", "", "perf"])
        self.assertEqual(result.instructions_per_request, 250_000.0)
        sudo_attach_probe = [
            kwargs.get("input")
            for cmd, kwargs in run_calls
            if cmd[:4] == ["sudo", "-S", "-p", ""]
        ]
        self.assertEqual(sudo_attach_probe, ["secret\n"])

    def test_monitor_uses_wall_elapsed_when_perf_omits_elapsed_line(self) -> None:
        class FakeProcess:
            returncode = 0

            def poll(self):
                return None

            def send_signal(self, sig):
                self.signal = sig

            def communicate(self, timeout=None):
                self.returncode = 0
                return "", "500000,,instructions,633068,100.00,,\n"

            def kill(self):
                self.returncode = -9

        fake_pid = SimpleNamespace(returncode=0, stdout="1234\n", stderr="")
        with patch("acprof.monitors.perf_mips.subprocess.run", return_value=fake_pid), patch(
            "acprof.monitors.perf_mips.subprocess.Popen",
            side_effect=lambda cmd, **kwargs: FakeProcess(),
        ), patch("acprof.monitors.perf_mips.time.perf_counter", side_effect=[10.0, 10.25]):
            monitor = perf_mips.PerfMIPSMonitor(
                container_name="case_container",
                command_prefix=["perf"],
            )
            monitor.start()
            result = monitor.stop(repeat_in_window=2, latency_app_s=0.125)

        self.assertEqual(result.instructions_total, 500_000)
        self.assertAlmostEqual(result.perf_elapsed_s, 0.25)

    def test_monitor_writes_sudo_password_without_breaking_stop(self) -> None:
        popen_stdin_closed = []

        class FakeStdin:
            def write(self, text):
                self.text = text

            def flush(self):
                return None

            def close(self):
                popen_stdin_closed.append(True)

        class FakeProcess:
            returncode = 0

            def __init__(self, cmd, **kwargs):
                self.stdin = FakeStdin()

            def poll(self):
                return None

            def send_signal(self, sig):
                self.signal = sig

            def communicate(self, timeout=None):
                self.returncode = 0
                return "", (
                    "1000000,,instructions,100.00,,\n"
                    "0.500000000 seconds time elapsed\n"
                )

            def kill(self):
                self.returncode = -9

        fake_pid = SimpleNamespace(returncode=0, stdout="1234\n", stderr="")
        with patch.dict("acprof.monitors.perf_mips.os.environ", {"ACPROF_SUDO_PASSWORD": "secret"}), patch(
            "acprof.monitors.perf_mips.subprocess.run",
            return_value=fake_pid,
        ), patch(
            "acprof.monitors.perf_mips.subprocess.Popen",
            side_effect=lambda cmd, **kwargs: FakeProcess(cmd, **kwargs),
        ):
            monitor = perf_mips.PerfMIPSMonitor(
                container_name="case_container",
                command_prefix=["sudo", "-S", "-p", "", "perf"],
            )
            monitor.start()
            result = monitor.stop(repeat_in_window=2, latency_app_s=0.25)

        self.assertEqual(popen_stdin_closed, [True])
        self.assertEqual(result.instructions_per_request, 500_000.0)

    def test_preflight_uses_password_sudo_when_noninteractive_sudo_fails(self) -> None:
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[:2] == ["perf", "stat"]:
                return SimpleNamespace(returncode=255, stdout="", stderr="no permission")
            if cmd[:3] == ["sudo", "-n", "perf"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="sudo password required")
            if cmd[:4] == ["sudo", "-S", "-p", ""]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="1000,,instructions,100.00,,\n0.010000000 seconds time elapsed\n",
                )
            raise AssertionError(f"unexpected command: {cmd}")

        with patch.dict("acprof.monitors.perf_mips.os.environ", {"ACPROF_SUDO_PASSWORD": "secret"}), patch(
            "acprof.monitors.perf_mips.shutil.which",
            return_value="/usr/bin/perf",
        ), patch("acprof.monitors.perf_mips.subprocess.run", side_effect=fake_run):
            prefix = perf_mips.resolve_perf_command_prefix()

        self.assertEqual(prefix, ["sudo", "-S", "-p", "", "perf"])
        self.assertEqual(calls[-1][1]["input"], "secret\n")

    def test_preflight_failure_prints_friendly_remediation(self) -> None:
        stderr = io.StringIO()

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=255, stdout="", stderr="perf_event_paranoid setting is 4")

        with patch("acprof.monitors.perf_mips.shutil.which", return_value="/usr/bin/perf"), patch(
            "acprof.monitors.perf_mips.subprocess.run",
            side_effect=fake_run,
        ), patch("acprof.monitors.perf_mips.read_perf_event_paranoid", return_value="4"), self.assertRaises(
            SystemExit
        ) as raised, redirect_stderr(stderr):
            perf_mips.require_mips_prerequisites()

        self.assertEqual(raised.exception.code, 1)
        message = stderr.getvalue()
        self.assertIn("[mips][ERROR]", message)
        self.assertIn("MIPS profiling requires Linux perf access", message)
        self.assertIn("perf_event_paranoid=4", message)
        self.assertIn("echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid", message)
        self.assertIn("ACPROF_SUDO_PASSWORD", message)
        self.assertIn("Avoid `sudo python run.py ...`", message)


if __name__ == "__main__":
    unittest.main()

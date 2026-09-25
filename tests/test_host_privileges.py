"""Runtime checks must not elevate privileges or install capabilities."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.host import env_utils, packet_capture
from acprof.monitors import perf_mips


class HostPrivilegeTests(unittest.TestCase):
    def test_perf_permission_failure_never_retries_with_sudo(self):
        failed = subprocess.CompletedProcess([], 1, "", "Permission denied")
        with patch.object(perf_mips.shutil, "which", return_value="/usr/bin/perf"), patch.object(
            perf_mips.subprocess, "run", return_value=failed,
        ) as run, self.assertRaises(perf_mips.MIPSProfilingError):
            perf_mips.resolve_perf_command_prefix(env={"PATH": "/usr/bin"})
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("sudo", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["input"], "")

    def test_pid_attach_failure_never_retries_with_sudo(self):
        failed = subprocess.CompletedProcess([], 1, "", "Permission denied")
        with patch.object(perf_mips.shutil, "which", return_value="/usr/bin/perf"), patch.object(
            perf_mips.subprocess, "run", return_value=failed,
        ) as run, self.assertRaises(perf_mips.MIPSProfilingError):
            perf_mips.resolve_perf_command_prefix_for_pid(1234)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("sudo", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["input"], "")

    def test_packet_permission_failure_does_not_grant_capability(self):
        failed = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(packet_capture.shutil, "which", side_effect=lambda name: "/usr/bin/" + name), patch.object(
            packet_capture.os, "geteuid", return_value=1000,
        ), patch.object(packet_capture, "_run", return_value=failed) as run:
            with self.assertRaises(packet_capture.PacketLatencyError):
                packet_capture._resolve_packet_latency_runtime(".", "/tmp/test.pcap", "docker0")
        self.assertTrue(all(call.args[0][0] == "getcap" for call in run.call_args_list))

    def test_tcpdump_needs_only_effective_net_raw_and_disables_promiscuous_mode(self):
        with patch.object(packet_capture.shutil, "which", side_effect=lambda name: "/usr/bin/" + name), patch.object(
            packet_capture.os, "geteuid", return_value=1000,
        ), patch.object(packet_capture, "_run", return_value=subprocess.CompletedProcess(
            [], 0, "/usr/bin/tcpdump cap_net_raw=ep\n", "",
        )):
            runtime = packet_capture._resolve_packet_latency_runtime(".", "/tmp/test.pcap", "docker0")
        self.assertEqual(runtime.tcpdump_cmd[0], "/usr/bin/tcpdump")
        self.assertIn("-p", runtime.tcpdump_cmd)

    def test_local_password_setting_is_rejected_without_exposing_value(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            Path(directory, ".env.local").write_text("ACPROF_SUDO_PASSWORD=test-retired-secret\n")
            with self.assertRaisesRegex(ValueError, "ACPROF_SUDO_PASSWORD") as raised:
                env_utils.load_project_env(directory)
            self.assertNotIn("test-retired-secret", str(raised.exception))
            self.assertNotIn("ACPROF_SUDO_PASSWORD", os.environ)

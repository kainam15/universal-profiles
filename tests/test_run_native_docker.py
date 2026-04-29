import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run


class NativeDockerGuardTests(unittest.TestCase):
    def test_docker_desktop_context_exits_before_docker_info(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="desktop-linux\n", stderr="")

        with patch("run.subprocess.run", return_value=context) as mock_run, patch(
            "builtins.print"
        ) as mock_print:
            with self.assertRaises(SystemExit) as raised:
                run.require_native_docker()

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(mock_run.call_count, 1)
        message = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("Docker Desktop", message)
        self.assertIn("native Docker", message)

    def test_docker_desktop_info_exits_with_native_docker_hint(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="default\n", stderr="")
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                "Name=docker-desktop\n"
                "OperatingSystem=Docker Desktop\n"
                "DockerRootDir=/var/lib/docker\n"
            ),
            stderr="",
        )

        with patch("run.subprocess.run", side_effect=[context, completed]), patch(
            "builtins.print"
        ) as mock_print:
            with self.assertRaises(SystemExit) as raised:
                run.require_native_docker()

        self.assertEqual(raised.exception.code, 1)
        message = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("Docker Desktop", message)
        self.assertIn("native Docker", message)
        self.assertIn("DOCKER_HOST=unix:///var/run/docker.sock", message)

    def test_native_linux_docker_info_is_allowed(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="default\n", stderr="")
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                "Name=kainam-Jiaolong16S-Series-GM6HG0X\n"
                "OperatingSystem=Ubuntu 24.04.4 LTS\n"
                "DockerRootDir=/var/lib/docker\n"
            ),
            stderr="",
        )

        with patch("run.subprocess.run", side_effect=[context, completed]):
            run.require_native_docker()

    def test_main_invokes_native_docker_guard_after_parsing_args(self) -> None:
        with patch.object(sys, "argv", ["run.py", "--model", "dummy-model"]), patch(
            "run.require_native_docker",
            side_effect=RuntimeError("guard called"),
        ):
            with self.assertRaisesRegex(RuntimeError, "guard called"):
                run.main()

    def test_main_runs_packet_latency_preflight_before_task_detection(self) -> None:
        with patch.object(sys, "argv", ["run.py", "--model", "dummy-model"]), patch(
            "run.bootstrap_project_env",
            return_value=None,
        ), patch("run.require_native_docker"), patch(
            "run.require_packet_latency_prerequisites",
            side_effect=SystemExit(2),
        ) as preflight, patch(
            "detect.detect_task",
            side_effect=AssertionError("detect_task should not run before preflight"),
        ):
            with self.assertRaises(SystemExit) as raised:
                run.main()

        self.assertEqual(raised.exception.code, 2)
        preflight.assert_called_once_with(
            project_dir=run.PROJECT_DIR,
            sniff_iface="docker0",
        )


if __name__ == "__main__":
    unittest.main()

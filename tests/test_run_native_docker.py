import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import orchestrator
from acprof.cli import run
from acprof.host.detect import TaskInfo


class NativeDockerGuardTests(unittest.TestCase):
    def test_native_linux_host_allows_ubuntu(self) -> None:
        with patch("acprof.cli.run.platform.system", return_value="Linux"), patch(
            "acprof.cli.run.platform.release",
            return_value="7.0.0-28-generic",
        ), patch.dict("acprof.cli.run.os.environ", {}, clear=True):
            run.require_native_linux_host()

    def test_native_linux_host_rejects_wsl(self) -> None:
        stderr = io.StringIO()

        with patch("acprof.cli.run.platform.system", return_value="Linux"), patch(
            "acprof.cli.run.platform.release",
            return_value="6.6.87.2-microsoft-standard-WSL2",
        ), patch.dict(
            "acprof.cli.run.os.environ",
            {"WSL_DISTRO_NAME": "Ubuntu"},
            clear=True,
        ), self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_native_linux_host()

        self.assertEqual(raised.exception.code, 1)
        message = stderr.getvalue()
        self.assertIn("native Linux host", message)
        self.assertIn("WSL was detected", message)
        self.assertIn("source .venv/bin/activate", message)

    def test_native_linux_host_rejects_windows(self) -> None:
        stderr = io.StringIO()

        with patch("acprof.cli.run.platform.system", return_value="Windows"), patch.dict(
            "acprof.cli.run.os.environ",
            {},
            clear=True,
        ), self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_native_linux_host()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("detected host OS Windows", stderr.getvalue())

    def test_cpu_energy_preflight_exits_with_remediation_when_unavailable(self) -> None:
        stderr = io.StringIO()

        with patch("acprof.monitors.energy_cpu.detect_cpu_power_source", return_value="unavailable"), patch(
            "acprof.monitors.energy_cpu.detect_vcpu_power_method",
            return_value="unavailable",
        ), self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_cpu_energy_prerequisites()

        self.assertEqual(raised.exception.code, 1)
        message = stderr.getvalue()
        self.assertIn("[cpu-energy][ERROR]", message)
        self.assertIn("CPU/vCPU energy profiling is required", message)
        self.assertIn("cpu_power_source=unavailable", message)
        self.assertIn("sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj", message)
        self.assertIn("/etc/tmpfiles.d/acprof-rapl.conf", message)

    def test_cpu_energy_preflight_allows_rapl_cgroup_share(self) -> None:
        with patch("acprof.monitors.energy_cpu.detect_cpu_power_source", return_value="rapl"), patch(
            "acprof.monitors.energy_cpu.detect_vcpu_power_method",
            return_value="rapl_cgroup_cpu_share",
        ):
            run.require_cpu_energy_prerequisites()

    def test_main_runs_cpu_energy_preflight_before_task_detection(self) -> None:
        with patch.object(sys, "argv", ["acprof.cli.run.py", "--model", "dummy-model"]), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch("acprof.cli.run.require_native_linux_host"), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites",
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites",
            side_effect=SystemExit(3),
        ) as preflight, patch(
            "acprof.cli.run.require_mips_prerequisites",
        ), patch(
            "acprof.host.detect.detect_task",
            side_effect=AssertionError("detect_task should not run before CPU energy preflight"),
        ):
            with self.assertRaises(SystemExit) as raised:
                run.main()

        self.assertEqual(raised.exception.code, 3)
        preflight.assert_called_once_with()

    def test_main_runs_mips_preflight_before_task_detection(self) -> None:
        with patch.object(sys, "argv", ["acprof.cli.run.py", "--model", "dummy-model"]), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch("acprof.cli.run.require_native_linux_host"), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites",
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites",
        ), patch(
            "acprof.cli.run.require_mips_prerequisites",
            side_effect=SystemExit(4),
        ) as preflight, patch(
            "acprof.host.detect.detect_task",
            side_effect=AssertionError("detect_task should not run before MIPS preflight"),
        ):
            with self.assertRaises(SystemExit) as raised:
                run.main()

        self.assertEqual(raised.exception.code, 4)
        preflight.assert_called_once_with()

    def test_docker_desktop_context_exits_before_docker_info(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="desktop-linux\n", stderr="")

        with patch.dict("acprof.cli.run.os.environ", {}, clear=True), patch(
            "acprof.cli.run.subprocess.run",
            return_value=context,
        ) as mock_run, patch(
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
        endpoint = SimpleNamespace(
            returncode=0,
            stdout="unix:///var/run/docker.sock\n",
            stderr="",
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                "Name=docker-desktop\n"
                "OperatingSystem=Docker Desktop\n"
                "DockerRootDir=/var/lib/docker\n"
            ),
            stderr="",
        )

        with patch.dict("acprof.cli.run.os.environ", {}, clear=True), patch(
            "acprof.cli.run.subprocess.run",
            side_effect=[context, endpoint, completed],
        ), patch(
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
        endpoint = SimpleNamespace(
            returncode=0,
            stdout="unix:///var/run/docker.sock\n",
            stderr="",
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                "Name=kainam-Jiaolong16S-Series-GM6HG0X\n"
                "OperatingSystem=Ubuntu 24.04.4 LTS\n"
                "DockerRootDir=/var/lib/docker\n"
            ),
            stderr="",
        )

        with patch.dict("acprof.cli.run.os.environ", {}, clear=True), patch(
            "acprof.cli.run.subprocess.run",
            side_effect=[context, endpoint, completed],
        ):
            run.require_native_docker()

    def test_remote_docker_context_exits_before_docker_info(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="remote-lab\n", stderr="")
        endpoint = SimpleNamespace(
            returncode=0,
            stdout="tcp://192.0.2.10:2376\n",
            stderr="",
        )
        stderr = io.StringIO()

        with patch.dict("acprof.cli.run.os.environ", {}, clear=True), patch(
            "acprof.cli.run.subprocess.run",
            side_effect=[context, endpoint],
        ) as mock_run, self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_native_docker()

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(mock_run.call_count, 2)
        self.assertIn("tcp://192.0.2.10:2376", stderr.getvalue())
        self.assertIn("/var/run/docker.sock", stderr.getvalue())

    def test_wsl_native_socket_override_is_rejected(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="default\n", stderr="")
        stderr = io.StringIO()

        with patch.dict(
            "acprof.cli.run.os.environ",
            {"DOCKER_HOST": "unix:///var/run/docker-native.sock"},
            clear=True,
        ), patch(
            "acprof.cli.run.subprocess.run",
            return_value=context,
        ) as mock_run, self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_native_docker()

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(mock_run.call_count, 1)
        self.assertIn("docker-native.sock", stderr.getvalue())

    def test_main_invokes_native_linux_guard_after_parsing_args(self) -> None:
        with patch.object(sys, "argv", ["acprof.cli.run.py", "--model", "dummy-model"]), patch(
            "acprof.cli.run.require_native_linux_host",
            side_effect=RuntimeError("host guard called"),
        ), patch(
            "acprof.cli.run.require_native_docker",
            side_effect=AssertionError("Docker guard must run after host guard"),
        ):
            with self.assertRaisesRegex(RuntimeError, "host guard called"):
                run.main()

    def test_main_invokes_native_docker_guard_after_host_guard(self) -> None:
        with patch.object(sys, "argv", ["acprof.cli.run.py", "--model", "dummy-model"]), patch(
            "acprof.cli.run.require_native_linux_host",
        ), patch(
            "acprof.cli.run.require_native_docker",
            side_effect=RuntimeError("guard called"),
        ):
            with self.assertRaisesRegex(RuntimeError, "guard called"):
                run.main()

    def test_main_runs_packet_latency_preflight_before_task_detection(self) -> None:
        with patch.object(sys, "argv", ["acprof.cli.run.py", "--model", "dummy-model"]), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch("acprof.cli.run.require_native_linux_host"), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites",
            side_effect=SystemExit(2),
        ) as preflight, patch(
            "acprof.cli.run.require_cpu_energy_prerequisites",
        ), patch(
            "acprof.host.detect.detect_task",
            side_effect=AssertionError("detect_task should not run before preflight"),
        ):
            with self.assertRaises(SystemExit) as raised:
                run.main()

        self.assertEqual(raised.exception.code, 2)
        preflight.assert_called_once_with(
            project_dir=run.PROJECT_DIR,
            sniff_iface="docker0",
        )

    def test_main_defaults_to_auto_repeat_window_and_reports_energy_abort(self) -> None:
        task_info = TaskInfo(
            model_id="dummy-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            sys,
            "argv",
            [
                "acprof.cli.run.py",
                "--model",
                "dummy-model",
                "--skip-build",
                "--no-compute-profile",
                "--cpus",
                "1",
                "--mems",
                "2",
                "--gpus",
                "on",
                "--output-dir",
                tmp_dir,
            ],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch(
            "acprof.cli.run.require_native_linux_host"
        ), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites"
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites"
        ), patch(
            "acprof.cli.run.require_mips_prerequisites"
        ), patch(
            "acprof.host.detect.detect_task",
            return_value=task_info,
        ), patch(
            "acprof.host.orchestrator.collect_static_meta",
            return_value=SimpleNamespace(),
        ), patch(
            "acprof.host.orchestrator.write_static_meta_csv"
        ), patch(
            "acprof.host.orchestrator.plan_input_scales",
            return_value=orchestrator.PlannedInputScales(
                scales=[1.0],
                source="unit",
                plan_file=None,
            ),
        ), patch(
            "acprof.host.orchestrator.run_matrix",
            side_effect=orchestrator.EnergyProfilingError("gpu_idle_power_w unstable"),
        ) as run_matrix, self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            run.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("[energy][ERROR] gpu_idle_power_w unstable", stderr.getvalue())
        _, kwargs = run_matrix.call_args
        self.assertEqual(kwargs["repeat_in_window"], 0)
        self.assertEqual(kwargs["repeat_window_seconds"], 10.0)

    def test_main_defaults_to_dual_compute_profiles_and_keeps_artifacts(self) -> None:
        task_info = TaskInfo(
            model_id="dummy-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            sys,
            "argv",
            [
                "acprof.cli.run.py",
                "--model",
                "dummy-model",
                "--skip-build",
                "--cpus",
                "1",
                "--mems",
                "2",
                "--gpus",
                "off,on",
                "--output-dir",
                tmp_dir,
            ],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch(
            "acprof.cli.run.require_native_linux_host"
        ), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites"
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites"
        ), patch(
            "acprof.cli.run.require_mips_prerequisites"
        ), patch(
            "acprof.host.detect.detect_task",
            return_value=task_info,
        ), patch(
            "acprof.host.orchestrator.collect_static_meta",
            return_value=SimpleNamespace(),
        ), patch(
            "acprof.host.orchestrator.write_static_meta_csv"
        ), patch(
            "acprof.host.orchestrator.plan_input_scales",
            return_value=orchestrator.PlannedInputScales(
                scales=[1.0],
                source="unit",
                plan_file=None,
            ),
        ), patch(
            "acprof.host.compute_profile.collect_compute_profile_plan",
            return_value=f"{tmp_dir}/compute_profile_plan.json",
        ) as collect_compute_profile_plan, patch(
            "acprof.host.orchestrator.run_matrix",
            return_value=[],
        ):
            run.main()

        _, kwargs = collect_compute_profile_plan.call_args
        self.assertEqual(kwargs["compute_profile_tool"], "both")
        self.assertEqual(kwargs["torch_profiler_repeat"], 1)
        self.assertEqual(kwargs["ncu_repeat"], 1)
        self.assertTrue(kwargs["keep_profiles"])

    def test_main_records_invocation_command_in_static_meta(self) -> None:
        task_info = TaskInfo(
            model_id="dummy-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--model",
                "dummy-model",
                "--skip-build",
                "--no-compute-profile",
                "--cpus",
                "1",
                "--mems",
                "2",
                "--gpus",
                "off",
                "--output-dir",
                tmp_dir,
            ],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch(
            "acprof.cli.run.require_native_linux_host"
        ), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites"
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites"
        ), patch(
            "acprof.cli.run.require_mips_prerequisites"
        ), patch(
            "acprof.host.detect.detect_task",
            return_value=task_info,
        ), patch(
            "acprof.host.orchestrator.collect_static_meta",
            return_value=SimpleNamespace(),
        ) as collect_static_meta, patch(
            "acprof.host.orchestrator.write_static_meta_csv"
        ), patch(
            "acprof.host.orchestrator.plan_input_scales",
            return_value=orchestrator.PlannedInputScales(
                scales=[1.0],
                source="unit",
                plan_file=None,
            ),
        ), patch(
            "acprof.host.orchestrator.run_matrix",
            return_value=[],
        ):
            run.main()

        _, kwargs = collect_static_meta.call_args
        self.assertEqual(
            kwargs["run_command"],
            "python run.py --model dummy-model --skip-build --no-compute-profile "
            "--cpus 1 --mems 2 --gpus off --output-dir "
            + tmp_dir,
        )


if __name__ == "__main__":
    unittest.main()

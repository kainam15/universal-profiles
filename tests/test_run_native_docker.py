import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import orchestrator
from acprof.cli import run
from acprof.host.detect import TaskInfo


class TmuxTerminalLogTests(unittest.TestCase):
    def test_terminal_log_is_not_started_outside_tmux(self) -> None:
        with patch.dict(
            "acprof.cli.run.os.environ",
            {},
            clear=True,
        ), patch("acprof.cli.run.subprocess.run") as mock_run:
            terminal_log = run._start_tmux_terminal_log(
                "/tmp/acprof-results",
                ["run.py", "--model", "dummy-model"],
            )

        self.assertIsNone(terminal_log)
        mock_run.assert_not_called()

    def test_terminal_log_records_and_atomically_finalizes_tmux_output(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        pipe_status = SimpleNamespace(returncode=0, stdout="0\n", stderr="")
        commands = []

        def fake_run(command, **_kwargs):
            commands.append(command)
            if command[1] == "display-message":
                return pipe_status
            return completed

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "acprof.cli.run.os.environ",
            {
                "TMUX": "/tmp/tmux-1000/default,123,0",
                "TMUX_PANE": "%7",
            },
            clear=True,
        ), patch(
            "acprof.cli.run.subprocess.run",
            side_effect=fake_run,
        ):
            output_dir = os.path.join(tmp, "results", "org--model")
            terminal_log = run._start_tmux_terminal_log(
                output_dir,
                ["run.py", "--model", "org/model"],
            )
            self.assertIsNotNone(terminal_log)
            pane_id, partial_path, log_path = terminal_log

            with open(partial_path, "a", encoding="utf-8") as f:
                f.write("experiment output\n")

            finalized = run._stop_tmux_terminal_log(terminal_log)

            self.assertTrue(finalized)
            self.assertEqual(pane_id, "%7")
            self.assertFalse(os.path.exists(partial_path))
            with open(log_path, "r", encoding="utf-8") as f:
                terminal_text = f.read()

        self.assertIn("$ python run.py --model org/model", terminal_text)
        self.assertIn("experiment output", terminal_text)
        self.assertEqual(commands[0][1], "display-message")
        self.assertEqual(commands[1][1], "pipe-pane")
        self.assertIn("-O", commands[1])
        self.assertEqual(commands[2], ["tmux", "pipe-pane", "-t", "%7"])

    def test_main_finalizes_tmux_log_when_profiling_raises(self) -> None:
        terminal_log = ("%3", "/tmp/tmux_all.log.part", "/tmp/tmux_all.log")

        def fail_after_starting_log():
            run._ACTIVE_TMUX_TERMINAL_LOG = terminal_log
            raise RuntimeError("profiling failed")

        with patch(
            "acprof.cli.run._run_main",
            side_effect=fail_after_starting_log,
        ), patch(
            "acprof.cli.run._stop_tmux_terminal_log",
            return_value=True,
        ) as stop_log:
            with self.assertRaisesRegex(RuntimeError, "profiling failed"):
                run.main()

        stop_log.assert_called_once_with(terminal_log)
        self.assertIsNone(run._ACTIVE_TMUX_TERMINAL_LOG)


class NativeDockerGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        # Local developer credentials must never make CLI tests send messages.
        notification_env = patch.dict(
            "acprof.cli.run.os.environ",
            {"ACPROF_WECOM_WEBHOOK_URL": ""},
        )
        notification_env.start()
        self.addCleanup(notification_env.stop)

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

    def test_detect_cgroup_version_distinguishes_v2_and_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cgroup_root = os.path.join(tmp, "cgroup")
            proc_self_cgroup = os.path.join(tmp, "self.cgroup")
            os.makedirs(cgroup_root)
            with open(
                os.path.join(cgroup_root, "cgroup.controllers"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("cpu io memory\n")
            with open(proc_self_cgroup, "w", encoding="utf-8") as f:
                f.write("0::/user.slice/test.scope\n")

            self.assertEqual(
                run.detect_cgroup_version(
                    cgroup_root=cgroup_root,
                    proc_self_cgroup_path=proc_self_cgroup,
                ),
                "v2",
            )

            os.remove(os.path.join(cgroup_root, "cgroup.controllers"))
            with open(proc_self_cgroup, "w", encoding="utf-8") as f:
                f.write("2:cpu,cpuacct:/docker/test\n")
                f.write("3:memory:/docker/test\n")

            self.assertEqual(
                run.detect_cgroup_version(
                    cgroup_root=cgroup_root,
                    proc_self_cgroup_path=proc_self_cgroup,
                ),
                "v1",
            )

    def test_cgroup_preflight_requires_v2_by_default(self) -> None:
        stderr = io.StringIO()
        with patch(
            "acprof.cli.run.detect_cgroup_version",
            return_value="v1",
        ), self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_cgroup_prerequisites()

        self.assertEqual(raised.exception.code, 1)
        message = stderr.getvalue()
        self.assertIn("requires the unified cgroup v2", message)
        self.assertIn("cgroup_version=v1", message)
        self.assertIn("--allow-cgroup-v1", message)

    def test_cgroup_preflight_allows_explicit_v1_compatibility(self) -> None:
        stderr = io.StringIO()
        with patch(
            "acprof.cli.run.detect_cgroup_version",
            return_value="v1",
        ), redirect_stderr(stderr):
            version = run.require_cgroup_prerequisites(allow_cgroup_v1=True)

        self.assertEqual(version, "v1")
        self.assertIn("legacy compatibility", stderr.getvalue())
        self.assertIn("do not mix", stderr.getvalue())

    def test_partial_results_require_matching_cgroup_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "result_case_model_1c_2g_off.csv")
            static_meta_path = os.path.join(tmp, "static_meta.json")
            with open(partial_path, "w", encoding="utf-8") as f:
                f.write("status\nok\n")
            with open(static_meta_path, "w", encoding="utf-8") as f:
                json.dump({"cgroup_version": "v1"}, f)

            stderr = io.StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
                run.require_result_cgroup_compatibility(
                    tmp,
                    cgroup_version="v2",
                )

            self.assertEqual(raised.exception.code, 1)
            message = stderr.getvalue()
            self.assertIn("Current cgroup_version:  v2", message)
            self.assertIn("Existing cgroup_version: v1", message)
            self.assertIn("will not mix", message)

            run.require_result_cgroup_compatibility(
                tmp,
                cgroup_version="v1",
            )

    def test_partial_results_without_cgroup_provenance_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                os.path.join(tmp, "result_case_model_1c_2g_off.csv"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("status\nok\n")

            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                run.require_result_cgroup_compatibility(
                    tmp,
                    cgroup_version="v2",
                )

    def test_main_passes_legacy_cgroup_flag_to_preflight(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["acprof.cli.run.py", "--model", "dummy-model", "--allow-cgroup-v1"],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch(
            "acprof.cli.run.require_native_linux_host",
        ), patch(
            "acprof.cli.run.require_native_docker",
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            side_effect=SystemExit(7),
        ) as preflight, patch(
            "acprof.host.detect.detect_task",
            side_effect=AssertionError("task detection must follow cgroup preflight"),
        ):
            with self.assertRaises(SystemExit) as raised:
                run.main()

        self.assertEqual(raised.exception.code, 7)
        preflight.assert_called_once_with(allow_cgroup_v1=True)

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
        with patch.object(
            sys,
            "argv",
            ["acprof.cli.run.py", "--model", "dummy-model", "--notify", "none"],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch("acprof.cli.run.require_native_linux_host"), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
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
        with patch.object(
            sys,
            "argv",
            ["acprof.cli.run.py", "--model", "dummy-model", "--notify", "none"],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch("acprof.cli.run.require_native_linux_host"), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
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
        with patch.object(
            sys,
            "argv",
            ["acprof.cli.run.py", "--model", "dummy-model", "--notify", "none"],
        ), patch(
            "acprof.cli.run.require_native_linux_host",
            side_effect=RuntimeError("host guard called"),
        ), patch(
            "acprof.cli.run.require_native_docker",
            side_effect=AssertionError("Docker guard must run after host guard"),
        ):
            with self.assertRaisesRegex(RuntimeError, "host guard called"):
                run.main()

    def test_main_invokes_native_docker_guard_after_host_guard(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["acprof.cli.run.py", "--model", "dummy-model", "--notify", "none"],
        ), patch(
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
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
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
                "--compute-profile-tool",
                "none",
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
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
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
            "acprof.host.orchestrator.write_static_meta_json"
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
        self.assertEqual(kwargs["idle_seconds"], 20.0)
        self.assertEqual(kwargs["idle_cooldown_seconds"], 5.0)
        self.assertEqual(kwargs["compute_profile_plan_file"], "")
        self.assertTrue(kwargs["prune_startup_oom"])
        self.assertFalse(
            collect_static_meta.call_args.kwargs["compute_profile_enabled"]
        )

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
                "--no-prune-startup-oom",
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
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
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
            "acprof.host.orchestrator.write_static_meta_json"
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
        ) as run_matrix:
            run.main()

        _, kwargs = collect_compute_profile_plan.call_args
        self.assertEqual(kwargs["compute_profile_tool"], "both")
        self.assertEqual(kwargs["torch_profiler_repeat"], 1)
        self.assertEqual(kwargs["ncu_repeat"], 1)
        self.assertTrue(kwargs["keep_profiles"])
        self.assertFalse(run_matrix.call_args.kwargs["prune_startup_oom"])

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
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
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
            "acprof.host.orchestrator.write_static_meta_json"
        ) as write_static_meta_json, patch(
            "acprof.cli.run.write_collection_history_json"
        ) as write_collection_history_json, patch(
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
        self.assertEqual(kwargs["cgroup_version"], "v2")
        self.assertEqual(kwargs["cgroup_collection_mode"], "strict_v2")
        self.assertTrue(write_static_meta_json.called)
        self.assertEqual(
            write_static_meta_json.call_args_list[0].args[1],
            os.path.join(tmp_dir, "dummy-model", "static_meta.json"),
        )
        self.assertEqual(
            write_collection_history_json.call_args.args[1],
            os.path.join(tmp_dir, "dummy-model", "collection_history.json"),
        )
        self.assertEqual(
            write_collection_history_json.call_args.args[0]["schema_version"],
            1,
        )


if __name__ == "__main__":
    unittest.main()

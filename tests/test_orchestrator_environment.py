import unittest
from types import SimpleNamespace
from unittest.mock import patch

import orchestrator
from config import STATIC_META_FIELDS
from detect import TaskInfo


class DetectEnvironmentTests(unittest.TestCase):
    def test_detect_environment_windows_11_with_wsl_kernel(self) -> None:
        with patch("orchestrator.platform.system", return_value="Windows"), patch(
            "orchestrator.platform.release", return_value="11"
        ), patch.dict("orchestrator.os.environ", {}, clear=True), patch(
            "orchestrator._run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="6.6.87.2-microsoft-standard-WSL2\n",
                stderr="",
            ),
        ):
            self.assertEqual(orchestrator._detect_environment(), "windows11+wsl")

    def test_detect_environment_linux_ubuntu_without_wsl(self) -> None:
        with patch("orchestrator.platform.system", return_value="Linux"), patch(
            "orchestrator.platform.freedesktop_os_release",
            return_value={"ID": "ubuntu", "VERSION_ID": "24.04"},
        ), patch.dict("orchestrator.os.environ", {}, clear=True), patch(
            "orchestrator._run",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="docker unavailable"),
        ):
            self.assertEqual(orchestrator._detect_environment(), "ubuntu24.04")

    def test_collect_static_meta_includes_environment(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        with patch("orchestrator._detect_environment", return_value="windows11+wsl"), patch(
            "orchestrator._get_gpu_name", return_value="Test GPU"
        ), patch(
            "orchestrator._docker_model_weight_bytes", return_value=123
        ), patch("orchestrator._docker_image_size_bytes", return_value=456), patch(
            "orchestrator._cpu_power_metadata",
            return_value=("rapl", "rapl_cgroup_cpu_share"),
        ):
            meta = orchestrator.collect_static_meta(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                batch_size=1,
                input_scale_type="seq_length",
            )

        self.assertEqual(meta.environment, "windows11+wsl")
        self.assertEqual(meta.cpu_power_source, "rapl")
        self.assertEqual(meta.vcpu_power_method, "rapl_cgroup_cpu_share")

    def test_static_meta_cpu_power_fields_are_trailing(self) -> None:
        self.assertEqual(
            STATIC_META_FIELDS[-3:],
            ["environment", "cpu_power_source", "vcpu_power_method"],
        )

    def test_run_single_case_passes_container_name_to_client(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        captured_env = {}

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and str(cmd[-1]).endswith("client.py"):
                captured_env.update(kwargs.get("env", {}))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "orchestrator._stop_container_session"
        ), patch("orchestrator._run", side_effect=fake_run):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir="results/test-unit",
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1,
                input_scales="64",
            )

        self.assertEqual(
            captured_env["CONTAINER_NAME"],
            "case_google-bert--bert-base-uncased_1c_4g_off",
        )

    def test_resolve_packet_latency_runtime_uses_wsl_tools_on_windows_wsl(self) -> None:
        project_dir = r"D:\DOR\universal-profiles"
        pcap_file = r"D:\DOR\universal-profiles\results\test\sniff_case.pcap"

        def fake_which(name: str) -> str | None:
            if name.lower() in {"wsl", "wsl.exe"}:
                return r"C:\Windows\System32\wsl.exe"
            return None

        with patch("orchestrator._detect_environment", return_value="windows11+wsl"), patch(
            "orchestrator.shutil.which",
            side_effect=fake_which,
        ), patch(
            "orchestrator._wsl_has_command",
            return_value=True,
        ):
            runtime = orchestrator._resolve_packet_latency_runtime(
                project_dir=project_dir,
                pcap_file=pcap_file,
                sniff_iface="docker0",
            )

        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(runtime.mode, "wsl")
        self.assertEqual(runtime.tcpdump_cmd[:3], ["wsl.exe", "-e", "sudo"])
        self.assertIn("tcpdump", runtime.tcpdump_cmd)
        self.assertIn("/mnt/d/DOR/universal-profiles/results/test/sniff_case.pcap", runtime.tcpdump_cmd)
        self.assertEqual(runtime.parse_cmd[:3], ["wsl.exe", "-e", "python3"])
        self.assertIn("/mnt/d/DOR/universal-profiles/sniff_parse_pcap.py", runtime.parse_cmd)


if __name__ == "__main__":
    unittest.main()

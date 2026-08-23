import csv
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import orchestrator
from acprof.config import (
    CSV_FIELDS,
    STATIC_META_FIELDS,
    STATIC_META_SCHEMA_VERSION,
)
from acprof.host.detect import TaskInfo


def _write_gpu_case_csv(
    path: str,
    idle_power_values: list[float],
    cpu_idle_power_values: list[float] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu_values = cpu_idle_power_values or [5.0 for _ in idle_power_values]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for idx, idle_power_w in enumerate(idle_power_values):
            row = {field: "nan" for field in CSV_FIELDS}
            row.update({
                "gpu_mode": "on",
                "gpu_idle_power_w": str(idle_power_w),
                "cpu_idle_power_w": str(cpu_values[idx]),
                "repeat_idx": str(idx),
                "warmup": "0",
                "status": "ok",
                "error": "",
            })
            writer.writerow(row)


def _write_cpu_case_csv(path: str, idle_power_values: list[float], gpu_mode: str = "off") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for idx, idle_power_w in enumerate(idle_power_values):
            row = {field: "nan" for field in CSV_FIELDS}
            row.update({
                "gpu_mode": gpu_mode,
                "cpu_idle_power_w": str(idle_power_w),
                "repeat_idx": str(idx),
                "warmup": "0",
                "status": "ok",
                "error": "",
            })
            writer.writerow(row)


class DetectEnvironmentTests(unittest.TestCase):
    def test_host_mem_total_bytes_uses_physical_page_count(self) -> None:
        values = {
            "SC_PAGE_SIZE": 4096,
            "SC_PHYS_PAGES": 8_000_000,
        }

        with patch(
            "acprof.host.orchestrator.os.sysconf",
            side_effect=lambda name: values[name],
        ):
            total = orchestrator._host_mem_total_bytes()

        self.assertEqual(total, 32_768_000_000)

    def test_host_swap_metadata_reads_capacity_usage_type_and_swappiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meminfo_path = os.path.join(tmp, "meminfo")
            swaps_path = os.path.join(tmp, "swaps")
            swappiness_path = os.path.join(tmp, "swappiness")
            with open(meminfo_path, "w", encoding="utf-8") as f:
                f.write("SwapTotal:       2097148 kB\n")
                f.write("SwapFree:        2096124 kB\n")
            with open(swaps_path, "w", encoding="utf-8") as f:
                f.write("Filename Type Size Used Priority\n")
                f.write("/swapfile file 2097148 1024 -2\n")
            with open(swappiness_path, "w", encoding="utf-8") as f:
                f.write("60\n")

            metadata = orchestrator._host_swap_metadata(
                proc_meminfo_path=meminfo_path,
                proc_swaps_path=swaps_path,
                swappiness_path=swappiness_path,
            )

        self.assertEqual(metadata["host_swap_total_bytes"], 2_147_479_552)
        self.assertEqual(metadata["host_swap_used_bytes_at_start"], 1_048_576)
        self.assertEqual(metadata["host_swap_type"], "file")
        self.assertEqual(metadata["host_vm_swappiness"], 60)

    def test_host_swap_metadata_reports_none_when_swap_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meminfo_path = os.path.join(tmp, "meminfo")
            swaps_path = os.path.join(tmp, "swaps")
            swappiness_path = os.path.join(tmp, "swappiness")
            with open(meminfo_path, "w", encoding="utf-8") as f:
                f.write("SwapTotal:             0 kB\n")
                f.write("SwapFree:              0 kB\n")
            with open(swaps_path, "w", encoding="utf-8") as f:
                f.write("Filename Type Size Used Priority\n")
            with open(swappiness_path, "w", encoding="utf-8") as f:
                f.write("0\n")

            metadata = orchestrator._host_swap_metadata(
                proc_meminfo_path=meminfo_path,
                proc_swaps_path=swaps_path,
                swappiness_path=swappiness_path,
            )

        self.assertEqual(metadata["host_swap_total_bytes"], 0)
        self.assertEqual(metadata["host_swap_used_bytes_at_start"], 0)
        self.assertEqual(metadata["host_swap_type"], "none")
        self.assertEqual(metadata["host_vm_swappiness"], 0)

    def test_docker_storage_metadata_uses_daemon_root_backing_filesystem(self) -> None:
        with patch(
            "acprof.host.orchestrator._docker_root_dir",
            return_value="/var/lib/docker",
        ), patch(
            "acprof.host.orchestrator.shutil.disk_usage",
            return_value=SimpleNamespace(total=1_000, used=400, free=600),
        ) as disk_usage, patch(
            "acprof.host.orchestrator._docker_mount_metadata",
            return_value=("/dev/nvme0n1p2", "ext4"),
        ), patch(
            "acprof.host.orchestrator._block_device_storage_type",
            return_value="nvme_ssd",
        ):
            metadata = orchestrator._docker_storage_metadata()

        disk_usage.assert_called_once_with("/var/lib/docker")
        self.assertEqual(
            metadata,
            {
                "docker_storage_total_bytes": 1_000,
                "docker_storage_available_bytes_at_start": 600,
                "docker_storage_filesystem": "ext4",
                "docker_storage_device": "/dev/nvme0n1p2",
                "docker_storage_type": "nvme_ssd",
            },
        )

    def test_block_device_type_uses_transport_and_rotational_flag(self) -> None:
        cases = (
            ({"tran": "nvme", "rota": False}, "nvme_ssd"),
            ({"tran": "sata", "rota": False}, "ssd"),
            ({"tran": "sata", "rota": True}, "hdd"),
            ({"tran": None, "rota": None}, "unknown"),
        )
        with patch(
            "acprof.host.orchestrator.shutil.which",
            return_value="/usr/bin/lsblk",
        ):
            for device_metadata, expected in cases:
                with self.subTest(expected=expected), patch(
                    "acprof.host.orchestrator._run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps({"blockdevices": [device_metadata]}),
                        stderr="",
                    ),
                ):
                    self.assertEqual(
                        orchestrator._block_device_storage_type("/dev/test"),
                        expected,
                    )

    def test_docker_storage_metadata_is_unknown_when_daemon_root_is_unavailable(self) -> None:
        with patch(
            "acprof.host.orchestrator._docker_root_dir",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator.shutil.disk_usage",
        ) as disk_usage:
            metadata = orchestrator._docker_storage_metadata()

        disk_usage.assert_not_called()
        self.assertIsNone(metadata["docker_storage_total_bytes"])
        self.assertIsNone(metadata["docker_storage_available_bytes_at_start"])
        self.assertEqual(metadata["docker_storage_type"], "unknown")

    def test_select_nlp_torch_index_url_uses_cu124_for_cuda_12_4_driver(self) -> None:
        with patch("acprof.host.orchestrator.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "acprof.host.orchestrator._run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="Driver Version: 550.78    CUDA Version: 12.4\n",
                stderr="",
            ),
        ):
            self.assertEqual(
                orchestrator._select_nlp_torch_index_url(),
                orchestrator.CUDA124_NLP_TORCH_INDEX_URL,
            )

    def test_select_nlp_torch_index_url_respects_explicit_override(self) -> None:
        with patch.dict(
            "acprof.host.orchestrator.os.environ",
            {"ACPROF_NLP_TORCH_INDEX_URL": "https://example.invalid/torch"},
            clear=True,
        ):
            self.assertEqual(
                orchestrator._select_nlp_torch_index_url(),
                "https://example.invalid/torch",
            )

    def test_select_nlp_torch_spec_uses_compatible_range_for_cu124(self) -> None:
        self.assertEqual(
            orchestrator._select_nlp_torch_spec(orchestrator.CUDA124_NLP_TORCH_INDEX_URL),
            orchestrator.CUDA124_NLP_TORCH_SPEC,
        )

    def test_select_nlp_torch_spec_accepts_cu124_index_with_trailing_slash(self) -> None:
        self.assertEqual(
            orchestrator._select_nlp_torch_spec(
                orchestrator.CUDA124_NLP_TORCH_INDEX_URL + "/"
            ),
            orchestrator.CUDA124_NLP_TORCH_SPEC,
        )

    def test_select_nlp_torch_spec_respects_explicit_override(self) -> None:
        with patch.dict(
            "acprof.host.orchestrator.os.environ",
            {"ACPROF_NLP_TORCH_SPEC": "torch==9.9.9"},
            clear=True,
        ):
            self.assertEqual(
                orchestrator._select_nlp_torch_spec(),
                "torch==9.9.9",
            )

    def test_build_image_passes_nlp_torch_index_build_arg(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
            parameter_count=110_106_428,
            precision_dtype="FP32",
            parameter_dtype_counts={"FP32": 110_106_428},
            quantized=False,
            model_license="apache-2.0",
            model_metadata_source="huggingface_hub",
        )
        commands = []

        def fake_run(cmd, check=True, capture=True, **kwargs):
            commands.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator._select_nlp_torch_index_url",
            return_value=orchestrator.CUDA124_NLP_TORCH_INDEX_URL,
        ), patch("acprof.host.orchestrator.os.path.exists", return_value=True), patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ):
            orchestrator.build_image(task_info, ".")

        self.assertIn(
            f"TORCH_INDEX_URL={orchestrator.CUDA124_NLP_TORCH_INDEX_URL}",
            commands[1],
        )
        self.assertIn(
            f"TORCH_PACKAGE_SPEC={orchestrator.CUDA124_NLP_TORCH_SPEC}",
            commands[1],
        )

    def test_build_image_passes_hf_token_as_buildkit_secret(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        commands = []

        def fake_run(cmd, check=True, capture=True, **kwargs):
            del check, capture, kwargs
            commands.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.dict(
            "acprof.host.orchestrator.os.environ",
            {"HF_TOKEN": "test-secret-value"},
            clear=True,
        ), patch(
            "acprof.host.orchestrator.os.path.exists",
            return_value=True,
        ), patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ), patch(
            "acprof.host.orchestrator._select_nlp_torch_index_url",
            return_value=orchestrator.CUDA124_NLP_TORCH_INDEX_URL,
        ):
            orchestrator.build_image(task_info, ".")

        family_build = commands[1]
        self.assertIn("--secret", family_build)
        self.assertIn("id=hf_token,env=HF_TOKEN", family_build)
        self.assertNotIn("test-secret-value", family_build)
        self.assertFalse(
            any(
                arg == "HF_TOKEN" or arg.startswith("HF_TOKEN=")
                for arg in family_build
            )
        )

    def test_runtime_container_is_offline_and_does_not_receive_hf_token(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="0123456789abcdef",
            detection_method="hub_api",
        )
        commands = []
        ready_response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "status": "ok",
                "model_id": task_info.model_id,
                "device": "cpu",
                "load_time_s": 1.0,
            },
        )

        with patch(
            "acprof.host.orchestrator._run",
            side_effect=lambda cmd, **_kwargs: (
                commands.append(cmd)
                or SimpleNamespace(returncode=0, stdout="", stderr="")
            ),
        ), patch("requests.get", return_value=ready_response):
            orchestrator._start_container_session(
                task_info=task_info,
                cpu=1,
                mem=2,
                gpu="off",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                container_name="offline-test",
                log_prefix="[test]",
            )

        docker_run = next(cmd for cmd in commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("HF_HUB_OFFLINE=1", docker_run)
        self.assertIn("TRANSFORMERS_OFFLINE=1", docker_run)
        self.assertIn("MODEL_LOCAL_PATH=/models/model-snapshot", docker_run)
        self.assertNotIn("HF_TOKEN", docker_run)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", docker_run)

    def test_start_container_session_reports_oom_before_ready_timeout(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        commands = []

        def fake_run(cmd, **_kwargs):
            commands.append(cmd)
            stdout = "[server] Loading model\n" if cmd[:2] == ["docker", "logs"] else ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        container_state = {
            "Status": "exited",
            "Running": False,
            "Restarting": False,
            "OOMKilled": True,
            "ExitCode": 137,
            "Error": "",
        }
        with patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ), patch(
            "acprof.host.orchestrator._inspect_container_state",
            return_value=container_state,
        ), patch(
            "requests.get",
            side_effect=ConnectionError("connection refused"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                orchestrator._start_container_session(
                    task_info=task_info,
                    cpu=1,
                    mem=2,
                    gpu="off",
                    image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                    container_name="oom-test",
                    log_prefix="[test]",
                )

        message = str(raised.exception)
        self.assertIn("container_oom_killed", message)
        self.assertIn("memory_limit=2g", message)
        self.assertIn("exit_code=137", message)
        self.assertIn(
            ["docker", "rm", "-f", "oom-test"],
            commands,
        )

    def test_detect_environment_windows_11_with_wsl_kernel(self) -> None:
        with patch("acprof.host.orchestrator.platform.system", return_value="Windows"), patch(
            "acprof.host.orchestrator.platform.release", return_value="11"
        ), patch.dict("acprof.host.orchestrator.os.environ", {}, clear=True), patch(
            "acprof.host.orchestrator._run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="6.6.87.2-microsoft-standard-WSL2\n",
                stderr="",
            ),
        ):
            self.assertEqual(orchestrator._detect_environment(), "windows11+wsl")

    def test_detect_environment_linux_ubuntu_without_wsl(self) -> None:
        with patch("acprof.host.orchestrator.platform.system", return_value="Linux"), patch(
            "acprof.host.orchestrator.platform.freedesktop_os_release",
            return_value={"ID": "ubuntu", "VERSION_ID": "24.04"},
        ), patch.dict("acprof.host.orchestrator.os.environ", {}, clear=True), patch(
            "acprof.host.orchestrator._run",
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
            parameter_count=110_106_428,
            parameter_bytes=440_425_712,
            precision_dtype="FP32",
            parameter_dtype_counts={"FP32": 110_106_428},
            quantized=False,
            model_license="apache-2.0",
            model_metadata_source="huggingface_hub",
        )

        with patch("acprof.host.orchestrator._detect_environment", return_value="windows11+wsl"), patch(
            "acprof.host.orchestrator._get_gpu_name", return_value="Test GPU"
        ), patch(
            "acprof.host.orchestrator._get_gpu_mem_total_bytes", return_value=987654321
        ), patch(
            "acprof.host.orchestrator._host_mem_total_bytes", return_value=64_000_000_000
        ), patch(
            "acprof.host.orchestrator._host_swap_metadata",
            return_value={
                "host_swap_total_bytes": 2_000_000_000,
                "host_swap_used_bytes_at_start": 100_000_000,
                "host_swap_type": "file",
                "host_vm_swappiness": 60,
            },
        ), patch(
            "acprof.host.orchestrator._docker_model_cache_bytes", return_value=123
        ), patch("acprof.host.orchestrator._docker_image_size_bytes", return_value=456), patch(
            "acprof.host.orchestrator._docker_storage_metadata",
            return_value={
                "docker_storage_total_bytes": 1_000_000,
                "docker_storage_available_bytes_at_start": 600_000,
                "docker_storage_filesystem": "ext4",
                "docker_storage_device": "/dev/nvme0n1p2",
                "docker_storage_type": "nvme_ssd",
            },
        ), patch(
            "acprof.host.orchestrator._cpu_power_metadata",
            return_value=("rapl", "rapl_cgroup_cpu_share"),
        ), patch(
            "acprof.host.orchestrator._cpu_frequency_policy_metadata",
            return_value=("performance", "on"),
        ):
            meta = orchestrator.collect_static_meta(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                batch_size=1,
                input_scale_type="seq_length",
                run_command="python run.py --model google-bert/bert-base-uncased",
                cgroup_version="v2",
                cgroup_collection_mode="strict_v2",
            )
            disabled_meta = orchestrator.collect_static_meta(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                batch_size=1,
                input_scale_type="seq_length",
                run_command=(
                    "python run.py --model google-bert/bert-base-uncased "
                    "--no-compute-profile"
                ),
                compute_profile_enabled=False,
            )

        self.assertEqual(meta.environment, "windows11+wsl")
        self.assertEqual(
            meta.run_command,
            "python run.py --model google-bert/bert-base-uncased",
        )
        self.assertEqual(meta.gpu_mem_total_bytes, 987654321)
        self.assertEqual(meta.host_mem_total_bytes, 64_000_000_000)
        self.assertEqual(meta.host_swap_total_bytes, 2_000_000_000)
        self.assertEqual(meta.host_swap_used_bytes_at_start, 100_000_000)
        self.assertEqual(meta.host_swap_type, "file")
        self.assertEqual(meta.host_vm_swappiness, 60)
        self.assertEqual(meta.docker_storage_total_bytes, 1_000_000)
        self.assertEqual(meta.docker_storage_available_bytes_at_start, 600_000)
        self.assertEqual(meta.docker_storage_filesystem, "ext4")
        self.assertEqual(meta.docker_storage_device, "/dev/nvme0n1p2")
        self.assertEqual(meta.docker_storage_type, "nvme_ssd")
        self.assertEqual(meta.cpu_power_source, "rapl")
        self.assertEqual(meta.vcpu_power_method, "rapl_cgroup_cpu_share")
        self.assertEqual(meta.cpu_governor, "performance")
        self.assertEqual(meta.cpu_boost, "on")
        self.assertEqual(meta.cgroup_version, "v2")
        self.assertEqual(meta.cgroup_collection_mode, "strict_v2")
        self.assertEqual(meta.parameter_count, 110_106_428)
        self.assertEqual(meta.parameter_bytes, 440_425_712)
        self.assertEqual(meta.model_cache_bytes, 123)
        self.assertEqual(meta.precision_dtype, "FP32")
        self.assertEqual(
            meta.inference_precision_by_device,
            {"cpu": "FP32", "gpu": "FP16"},
        )
        self.assertFalse(meta.quantized)
        self.assertEqual(meta.model_license, "apache-2.0")
        self.assertEqual(
            meta.input_format["json_schema"]["required"],
            ["text"],
        )
        self.assertIn(
            "n_results",
            meta.output_format["json_schema"]["properties"],
        )
        self.assertEqual(disabled_meta.compute_profile_tools, [])
        self.assertFalse(disabled_meta.compute_profiles_retained)
        self.assertEqual(disabled_meta.compute_profile_provenance, "disabled")
        self.assertEqual(meta.execution_profile_schema_version, 1)
        self.assertEqual(meta.execution_profile_tools, [])
        self.assertFalse(meta.execution_profiles_retained)
        self.assertEqual(meta.execution_profile_provenance, "disabled")

    def test_static_meta_compute_profile_fields_follow_host_metadata(self) -> None:
        self.assertEqual(STATIC_META_SCHEMA_VERSION, 6)
        self.assertIn("parameter_bytes", STATIC_META_FIELDS)
        self.assertIn("model_cache_bytes", STATIC_META_FIELDS)
        self.assertNotIn("model_weight_bytes", STATIC_META_FIELDS)
        self.assertIn("gpu_mem_total_bytes", STATIC_META_FIELDS)
        self.assertIn("host_mem_total_bytes", STATIC_META_FIELDS)
        self.assertIn("host_swap_total_bytes", STATIC_META_FIELDS)
        self.assertIn("host_swap_used_bytes_at_start", STATIC_META_FIELDS)
        self.assertIn("host_swap_type", STATIC_META_FIELDS)
        self.assertIn("host_vm_swappiness", STATIC_META_FIELDS)
        self.assertIn("docker_storage_total_bytes", STATIC_META_FIELDS)
        self.assertIn("cgroup_version", STATIC_META_FIELDS)
        self.assertIn("cgroup_collection_mode", STATIC_META_FIELDS)
        self.assertLess(
            STATIC_META_FIELDS.index("gpu_mem_total_bytes"),
            STATIC_META_FIELDS.index("environment"),
        )
        self.assertLess(
            STATIC_META_FIELDS.index("docker_storage_type"),
            STATIC_META_FIELDS.index("environment"),
        )
        self.assertLess(
            STATIC_META_FIELDS.index("cgroup_collection_mode"),
            STATIC_META_FIELDS.index("cpu_power_source"),
        )
        self.assertLess(
            STATIC_META_FIELDS.index("cpu_boost"),
            STATIC_META_FIELDS.index("compute_profile_tools"),
        )
        self.assertNotIn("compute_profile_schema_version", STATIC_META_FIELDS)
        self.assertLess(
            STATIC_META_FIELDS.index("compute_profile_provenance"),
            STATIC_META_FIELDS.index("execution_profile_schema_version"),
        )
        self.assertIn("massif_sampling_strategy", STATIC_META_FIELDS)
        self.assertIn("nsys_sampling_strategy", STATIC_META_FIELDS)
        self.assertEqual(STATIC_META_FIELDS[-1], "execution_profile_provenance")

    def test_enrich_static_meta_preserves_native_json_types(self) -> None:
        base = orchestrator.StaticMeta(
            model_name="model",
            model_revision="main",
            task_family="nlp",
            pipeline_tag="fill-mask",
            runtime_backend="transformers_pipeline",
            image_tag="image",
            batch_size=1,
            input_scale_type="seq_length",
            run_command="python run.py",
            model_download_url="https://example.invalid/model",
            gpu="GPU",
            gpu_mem_total_bytes=123,
            host_mem_total_bytes=1_024,
            host_swap_total_bytes=512,
            host_swap_used_bytes_at_start=64,
            host_swap_type="file",
            host_vm_swappiness=60,
            model_cache_bytes=456,
            docker_image_bytes=789,
            docker_storage_total_bytes=10_000,
            docker_storage_available_bytes_at_start=4_000,
            docker_storage_filesystem="ext4",
            docker_storage_device="/dev/nvme0n1p2",
            docker_storage_type="nvme_ssd",
            environment="ubuntu24.04",
            cgroup_version="v2",
            cgroup_collection_mode="strict_v2",
            cpu_power_source="rapl",
            vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance",
            cpu_boost="off",
        )

        enriched = orchestrator.enrich_static_meta(
            base,
            {
                "compute_profile_tools": ["torch_profiler_eager", "ncu"],
                "ncu_metrics": ["gpu__time_duration.sum", "metric.sum"],
                "compute_profiles_retained": True,
            },
        )

        self.assertEqual(
            enriched.compute_profile_tools,
            ["torch_profiler_eager", "ncu"],
        )
        self.assertEqual(
            enriched.ncu_metrics,
            ["gpu__time_duration.sum", "metric.sum"],
        )
        self.assertTrue(enriched.compute_profiles_retained)
        self.assertEqual(enriched.run_command, base.run_command)

        execution_enriched = orchestrator.enrich_static_meta(
            enriched,
            {
                "execution_profile_schema_version": 1,
                "execution_profile_tools": ["massif", "nsys"],
                "execution_profiles_retained": True,
                "execution_profile_provenance": "collected",
            },
        )
        self.assertEqual(execution_enriched.execution_profile_schema_version, 1)
        self.assertEqual(
            execution_enriched.execution_profile_tools,
            ["massif", "nsys"],
        )
        self.assertTrue(execution_enriched.execution_profiles_retained)
        self.assertEqual(
            execution_enriched.execution_profile_provenance,
            "collected",
        )

    def test_write_static_meta_json_includes_enriched_fields_atomically(self) -> None:
        meta = orchestrator.StaticMeta(
            model_name="model",
            model_revision="main",
            task_family="nlp",
            pipeline_tag="fill-mask",
            runtime_backend="transformers_pipeline",
            image_tag="image",
            batch_size=1,
            input_scale_type="seq_length",
            run_command="python run.py --model model",
            model_download_url="https://example.invalid/model",
            gpu="GPU",
            gpu_mem_total_bytes=123,
            host_mem_total_bytes=1_024,
            host_swap_total_bytes=512,
            host_swap_used_bytes_at_start=64,
            host_swap_type="file",
            host_vm_swappiness=60,
            model_cache_bytes=456,
            docker_image_bytes=789,
            docker_storage_total_bytes=10_000,
            docker_storage_available_bytes_at_start=4_000,
            docker_storage_filesystem="ext4",
            docker_storage_device="/dev/nvme0n1p2",
            docker_storage_type="nvme_ssd",
            environment="ubuntu24.04",
            cgroup_version="v2",
            cgroup_collection_mode="strict_v2",
            cpu_power_source="rapl",
            vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance",
            cpu_boost="off",
            parameter_count=42,
            parameter_bytes=168,
            precision_dtype="FP32",
            parameter_dtype_counts={"FP32": 42},
            input_format={"media_type": "application/json"},
            output_format={"media_type": "application/json"},
            quantized=False,
            model_license="mit",
            compute_profile_tools=["torch_profiler_eager", "ncu"],
            compute_profile_provenance="direct",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "static_meta.json")
            orchestrator.write_static_meta_json(meta, path)
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            leftovers = [
                name
                for name in os.listdir(tmp)
                if name.startswith(".static_meta.json.")
            ]

        self.assertEqual(list(payload), STATIC_META_FIELDS)
        self.assertNotIn("compute_profile_schema_version", payload)
        self.assertEqual(payload["parameter_count"], 42)
        self.assertEqual(payload["parameter_bytes"], 168)
        self.assertEqual(payload["model_cache_bytes"], 456)
        self.assertNotIn("model_weight_bytes", payload)
        self.assertEqual(payload["host_mem_total_bytes"], 1_024)
        self.assertEqual(payload["host_swap_total_bytes"], 512)
        self.assertEqual(payload["host_swap_used_bytes_at_start"], 64)
        self.assertEqual(payload["host_swap_type"], "file")
        self.assertEqual(payload["host_vm_swappiness"], 60)
        self.assertEqual(payload["docker_storage_total_bytes"], 10_000)
        self.assertEqual(payload["cgroup_version"], "v2")
        self.assertEqual(payload["cgroup_collection_mode"], "strict_v2")
        self.assertEqual(
            payload["docker_storage_available_bytes_at_start"],
            4_000,
        )
        self.assertEqual(payload["docker_storage_type"], "nvme_ssd")
        self.assertEqual(payload["parameter_dtype_counts"], {"FP32": 42})
        self.assertFalse(payload["quantized"])
        self.assertEqual(
            payload["compute_profile_tools"],
            ["torch_profiler_eager", "ncu"],
        )
        self.assertEqual(payload["compute_profile_provenance"], "direct")
        self.assertEqual(leftovers, [])

    def test_compute_plan_adds_static_flops_by_input_scale(self) -> None:
        meta = orchestrator.StaticMeta(
            model_name="model",
            model_revision="main",
            task_family="nlp",
            pipeline_tag="fill-mask",
            runtime_backend="transformers_pipeline",
            image_tag="image",
            batch_size=1,
            input_scale_type="seq_length",
            run_command="python run.py",
            model_download_url="https://example.invalid/model",
            gpu="GPU",
            gpu_mem_total_bytes=123,
            model_cache_bytes=456,
            docker_image_bytes=789,
            environment="ubuntu24.04",
            cpu_power_source="rapl",
            vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance",
            cpu_boost="off",
        )
        plan = {
            "static_metadata": {
                "compute_profile_tools": ["torch_profiler_eager"],
                "torch_profiler_eager_flop_semantics": (
                    "logical_operator_shape_flops"
                ),
            },
            "profiles": {
                "gpu": {
                    "torch_profiler_eager": {
                        "flop_semantics": "logical_operator_shape_flops",
                        "entries": [
                            {
                                "input_scale": 64,
                                "model_logical_mflop_per_request_torch_profiler_eager": 12.5,
                                "error": "",
                            },
                            {
                                "input_scale": 128,
                                "model_logical_mflop_per_request_torch_profiler_eager": 25.25,
                                "error": "",
                            },
                        ],
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "compute_profile_plan.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(plan, f)
            enriched = orchestrator.enrich_static_meta_from_compute_plan(
                meta,
                path,
            )

        self.assertEqual(
            enriched.static_flops,
            {
                "source": "torch_profiler_eager",
                "profile": "gpu",
                "semantics": "logical_operator_shape_flops",
                "unit": "FLOP/request",
                "input_scale_type": "seq_length",
                "batch_size": 1,
                "values": [
                    {"input_scale": 64, "flops_per_request": 12_500_000},
                    {"input_scale": 128, "flops_per_request": 25_250_000},
                ],
            },
        )
        self.assertIsNone(enriched.static_macs)

    def test_cpu_frequency_policy_metadata_reads_governor_and_boost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpu0_cpufreq = os.path.join(tmp, "cpu0", "cpufreq")
            cpu1_cpufreq = os.path.join(tmp, "cpu1", "cpufreq")
            cpufreq = os.path.join(tmp, "cpufreq")
            os.makedirs(cpu0_cpufreq)
            os.makedirs(cpu1_cpufreq)
            os.makedirs(cpufreq)

            for path in (
                os.path.join(cpu0_cpufreq, "scaling_governor"),
                os.path.join(cpu1_cpufreq, "scaling_governor"),
            ):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("performance\n")
            with open(os.path.join(cpufreq, "boost"), "w", encoding="utf-8") as f:
                f.write("1\n")

            with patch("acprof.host.orchestrator.CPU_SYSFS_ROOT", tmp):
                self.assertEqual(
                    orchestrator._cpu_frequency_policy_metadata(),
                    ("performance", "on"),
                )

    def test_manual_nlp_scales_write_effective_scale_plan(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        session = orchestrator.RunningContainer(
            name="probe_google-bert--bert-base-uncased_1c_4g_off",
            base_url="http://127.0.0.1:8106",
            host_port=8106,
            cold_start_s=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.orchestrator._start_probe_session",
            return_value=session,
        ), patch(
            "acprof.host.orchestrator._stop_container_session",
        ), patch(
            "acprof.host.orchestrator._post_probe_payload",
            return_value={
                "effective_input_scale": 254.0,
                "truncated_by_limit": False,
                "reason": "within_model_limit",
                "payload": {"text": "hello [MASK]", "params": {}},
            },
        ):
            planned = orchestrator.plan_input_scales(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
                input_scales="64",
            )

            self.assertEqual(planned.scales, [254.0])
            self.assertEqual(planned.source, "manual")
            self.assertIsNotNone(planned.plan_file)
            assert planned.plan_file is not None
            self.assertTrue(os.path.exists(planned.plan_file))
            with open(planned.plan_file, "r", encoding="utf-8") as f:
                plan = json.load(f)

        self.assertEqual(plan["entries"][0]["input_scale"], 254.0)
        self.assertEqual(plan["entries"][0]["payload"], {"text": "hello [MASK]", "params": {}})

    def test_manual_non_nlp_scales_write_reusable_payload_plan(self) -> None:
        class FakeWorkloadGenerator:
            def generate(self, scale: float) -> dict:
                return {"value": float(scale)}

            def effective_input_scale(
                self,
                scale: float,
                payload: dict | None = None,
            ) -> float:
                return float(scale)

            def scale_label(self, scale: float) -> str:
                return f"scale{scale:g}"

        for task_family in ("cv", "timeseries"):
            with self.subTest(task_family=task_family), tempfile.TemporaryDirectory() as tmp, patch(
                "acprof.workloads.get_generator",
                return_value=FakeWorkloadGenerator(),
            ):
                task_info = TaskInfo(
                    model_id=f"test/{task_family}",
                    pipeline_tag="test-task",
                    task_family=task_family,
                    runtime_backend="transformers_pipeline",
                    library_name="transformers",
                    model_revision="main",
                    detection_method="unit",
                )

                planned = orchestrator.plan_input_scales(
                    task_info=task_info,
                    image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                    cpu_list=[1],
                    mem_list=[4],
                    gpu_list=["off"],
                    batch_size=1,
                    output_dir=tmp,
                    input_scales="1,2",
                )

                self.assertEqual(planned.scales, [1.0, 2.0])
                self.assertEqual(planned.source, "manual")
                self.assertEqual(
                    planned.plan_file,
                    os.path.join(tmp, "input_scale_plan.json"),
                )
                assert planned.plan_file is not None
                with open(planned.plan_file, "r", encoding="utf-8") as f:
                    plan = json.load(f)

                self.assertEqual(
                    plan["entries"],
                    [
                        {
                            "input_scale": 1.0,
                            "scale_label": "scale1",
                            "input_metadata": {},
                            "payload": {"value": 1.0},
                        },
                        {
                            "input_scale": 2.0,
                            "scale_label": "scale2",
                            "input_metadata": {},
                            "payload": {"value": 2.0},
                        },
                    ],
                )
                self.assertEqual(plan["schema_version"], 2)
                self.assertRegex(planned.plan_sha256, r"^[0-9a-f]{64}$")

    def test_auto_audio_scales_use_workload_manifest_defaults(self) -> None:
        class FakeWorkloadGenerator:
            def default_input_scales(self) -> list[float]:
                return [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]

        task_info = TaskInfo(
            model_id="test/audio",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        expected = orchestrator.PlannedInputScales(
            scales=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            source="workload_spec",
            plan_file="/tmp/input_scale_plan.json",
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ), patch.object(
            orchestrator,
            "_plan_audio_scales",
            return_value=expected,
        ) as plan_audio:
            planned = orchestrator.plan_input_scales(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
            )

        self.assertIs(planned, expected)
        self.assertEqual(
            plan_audio.call_args.kwargs["scales"],
            [1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
        )

    def test_audio_scale_meta_records_fixed_frontend_and_decoder_limit(self) -> None:
        response = {
            "input_scale_type": "duration_s",
            "required_sampling_rate": 16000,
            "max_short_form_duration_s": 30,
            "model_input_num_samples": 480000,
            "model_input_frames": 3000,
            "short_form_fixed_padding": True,
            "fixed_frontend_num_samples": 480000,
            "fixed_frontend_num_frames": 3000,
            "frontend_feature_bins": 128,
            "encoder_positions": 1500,
            "decoder_output_token_limit": 448,
            "model_type": "whisper",
            "reason": "fixed frontend; 448 is an output limit",
        }
        session = orchestrator.RunningContainer(
            name="probe",
            base_url="http://127.0.0.1:1",
            host_port=1,
            cold_start_s=0.0,
        )
        with patch.object(
            orchestrator,
            "_request_scale_meta",
            return_value=response,
        ):
            metadata = orchestrator._request_audio_scale_meta(session, {})

        self.assertTrue(metadata["short_form_fixed_padding"])
        self.assertEqual(metadata["fixed_frontend_num_samples"], 480000)
        self.assertEqual(metadata["fixed_frontend_num_frames"], 3000)
        self.assertEqual(metadata["frontend_feature_bins"], 128)
        self.assertEqual(metadata["decoder_output_token_limit"], 448)

    def test_audio_output_schema_describes_effective_scale_and_nullable_tokens(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        _, output_format = orchestrator._model_io_formats(task_info)
        properties = output_format["json_schema"]["properties"]

        self.assertEqual(properties["effective_input_scale"], {"type": "number"})
        self.assertEqual(
            properties["output_token_count"],
            {"type": ["integer", "null"]},
        )

    def test_workload_spec_is_rejected_for_unimplemented_families(self) -> None:
        task_info = TaskInfo(
            model_id="test/nlp",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError,
            "implemented only.*audio",
        ):
            orchestrator.plan_input_scales(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
                workload_spec_path="/tmp/not-used.json",
            )

    def test_audio_manifest_limit_is_checked_before_starting_probe(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator,
            "_start_probe_session",
        ) as start_probe, self.assertRaisesRegex(
            ValueError,
            "long-form workload",
        ):
            orchestrator._plan_audio_scales(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                scales=[31.0],
                batch_size=1,
                output_dir=tmp,
                source="manual",
                workload_spec_path=None,
            )
        start_probe.assert_not_called()

    def test_materialized_plan_v2_records_workload_and_input_metadata(self) -> None:
        class FakeWorkloadGenerator:
            def generate(self, scale: float) -> dict:
                return {"value": float(scale), "params": {"mode": "fixed"}}

            def effective_input_scale(self, scale: float, payload=None) -> float:
                return float(scale)

            def scale_label(self, scale: float) -> str:
                return f"scale{scale:g}"

            def input_metadata(self, scale: float, payload=None) -> dict:
                return {"input_num_samples": int(scale * 100)}

            def plan_metadata(self) -> dict:
                return {"workload_id": "fixture-v1"}

        task_info = TaskInfo(
            model_id="test/audio",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        constraints = {"max_short_form_duration_s": 30}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ):
            planned = orchestrator._materialize_scale_plan(
                task_info=task_info,
                scales=[1.0, 2.0],
                batch_size=1,
                output_dir=tmp,
                source="unit",
                model_constraints=constraints,
            )
            assert planned.plan_file is not None
            with open(planned.plan_file, "r", encoding="utf-8") as plan_file:
                plan = json.load(plan_file)

        self.assertEqual(plan["schema_version"], 2)
        self.assertEqual(plan["workload"], {"workload_id": "fixture-v1"})
        self.assertEqual(plan["model_constraints"], constraints)
        self.assertEqual(
            plan["entries"][0]["input_metadata"]["input_num_samples"],
            100,
        )
        self.assertEqual(
            planned.workload["model_constraints"],
            constraints,
        )
        self.assertRegex(planned.plan_sha256, r"^[0-9a-f]{64}$")

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
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run):
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
                require_packet_latency=False,
            )

        self.assertEqual(
            captured_env["CONTAINER_NAME"],
            "case_google-bert--bert-base-uncased_1c_4g_off",
        )
        self.assertEqual(captured_env["USE_MIPS"], "1")

    def test_run_single_case_passes_compute_profile_plan_to_client(self) -> None:
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
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run):
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
                compute_profile_plan_file="results/test-unit/compute_profile_plan.json",
                require_packet_latency=False,
            )

        self.assertEqual(
            captured_env["COMPUTE_PROFILE_PLAN_FILE"],
            "results/test-unit/compute_profile_plan.json",
        )

    def test_run_single_case_passes_execution_profile_plan_to_client(self) -> None:
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
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch(
            "acprof.host.orchestrator._resolve_packet_latency_runtime",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ):
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
                execution_profile_plan_file=(
                    "results/test-unit/execution_profile_plan.json"
                ),
                require_packet_latency=False,
            )

        self.assertEqual(
            captured_env["EXECUTION_PROFILE_PLAN_FILE"],
            "results/test-unit/execution_profile_plan.json",
        )

    def test_run_single_case_passes_idle_debug_settings_to_client(self) -> None:
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
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run):
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
                idle_cooldown_seconds=4.5,
                idle_debug=True,
                require_packet_latency=False,
            )

        self.assertEqual(captured_env["IDLE_DEBUG"], "1")
        self.assertEqual(captured_env["IDLE_COOLDOWN_SECONDS"], "4.5")
        self.assertEqual(
            captured_env["IDLE_DIAG_PATH"],
            os.path.join(
                os.path.dirname(captured_env["OUT_CSV"]),
                "debug_idle_diag",
                os.path.basename(captured_env["OUT_CSV"]) + ".idle_diag.jsonl",
            ),
        )

    def test_run_single_case_passes_auto_repeat_window_settings_to_client(self) -> None:
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
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_gpu_case_csv(captured_env["OUT_CSV"], [10.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="on",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir="results/test-unit",
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=0,
                repeat_window_seconds=10.0,
                input_scales="64",
                require_packet_latency=False,
            )

        self.assertEqual(captured_env["REPEAT_IN_WINDOW"], "0")
        self.assertEqual(captured_env["REPEAT_WINDOW_SECONDS"], "10.0")

    def test_run_single_case_preserves_manual_repeat_window_to_client(self) -> None:
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
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                captured_env.update(kwargs.get("env", {}))
                _write_gpu_case_csv(captured_env["OUT_CSV"], [10.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run):
            orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="on",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir="results/test-unit",
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1000,
                repeat_window_seconds=10.0,
                input_scales="64",
                require_packet_latency=False,
            )

        self.assertEqual(captured_env["REPEAT_IN_WINDOW"], "1000")
        self.assertEqual(captured_env["REPEAT_WINDOW_SECONDS"], "10.0")

    def test_run_single_case_aborts_when_client_exits_nonzero(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                return SimpleNamespace(returncode=7, stdout="", stderr="idle unstable")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run):
            with self.assertRaises(orchestrator.EnergyProfilingError) as raised:
                orchestrator.run_single_case(
                    task_info=task_info,
                    cpu=1,
                    mem=4,
                    gpu="on",
                    image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                    output_dir="results/test-unit",
                    project_dir=".",
                    warmup=0,
                    repeat=1,
                    repeat_in_window=0,
                    input_scales="64",
                    require_packet_latency=False,
                )

        self.assertIn("client.py exited with code 7", str(raised.exception))

    def test_run_single_case_records_request_timeout_and_returns(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                with open(
                    kwargs["env"]["CLIENT_ERROR_PATH"],
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "error_type": "client_request_timeout",
                            "input_scale": 64.0,
                            "request_timeout_s": 300.0,
                            "request_phase": "auto_repeat_window_warmup",
                            "request_id": "case_scale64_auto_warmup0",
                        },
                        f,
                    )
                return SimpleNamespace(
                    returncode=orchestrator.CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
                    stdout="",
                    stderr="slow inference",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch(
            "acprof.host.orchestrator._resolve_packet_latency_runtime",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator._stop_container_session"
        ) as stop_container, patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=1,
                repeat=2,
                repeat_in_window=0,
                input_scales="64,128",
                require_packet_latency=False,
            )

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["status"] for row in rows}, {"error"})
        self.assertTrue(
            all("client_request_timeout" in row["error"] for row in rows)
        )
        trigger_rows = [row for row in rows if float(row["input_scale"]) == 64.0]
        skipped_rows = [row for row in rows if float(row["input_scale"]) == 128.0]
        self.assertTrue(
            all("reason=triggering_scale_probe_timed_out" in row["error"] for row in trigger_rows)
        )
        self.assertTrue(
            all("triggering_request_latency_s>300" in row["error"] for row in trigger_rows)
        )
        self.assertTrue(
            all("reason=skipped_after_prior_scale_timeout" in row["error"] for row in skipped_rows)
        )
        self.assertTrue(
            all("planned_request_attempted=false" in row["error"] for row in skipped_rows)
        )
        self.assertEqual(
            sorted({float(row["input_scale"]) for row in rows}),
            [64.0, 128.0],
        )
        stop_container.assert_called_once()

    def test_run_single_case_preserves_completed_rows_before_request_timeout(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                out_csv = kwargs["env"]["OUT_CSV"]
                row = {field: "nan" for field in CSV_FIELDS}
                row.update({
                    "cpu_cores": "1",
                    "mem_cap_gb": "4",
                    "gpu_mode": "off",
                    "input_scale": "64",
                    "repeat_idx": "0",
                    "warmup": "0",
                    "repeat_in_window": "1",
                    "latency_app_s": "12.5",
                    "cpu_idle_power_w": "5.0",
                    "status": "ok",
                    "error": "",
                })
                with open(out_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerow(row)
                with open(
                    kwargs["env"]["CLIENT_ERROR_PATH"],
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "error_type": "client_request_timeout",
                            "input_scale": 128.0,
                            "request_timeout_s": 300.0,
                            "request_phase": "measurement_repeat",
                            "measurement_repeat_idx": 0,
                            "request_id": "case_scale128_r0:0",
                        },
                        f,
                    )
                return SimpleNamespace(
                    returncode=orchestrator.CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
                    stdout="",
                    stderr="slow inference",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch(
            "acprof.host.orchestrator._resolve_packet_latency_runtime",
            return_value=None,
        ), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=1,
                repeat_in_window=1,
                input_scales="64,128",
                require_packet_latency=False,
            )

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input_scale"], "64")
        self.assertEqual(rows[0]["latency_app_s"], "12.5")
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["error"], "")
        self.assertEqual(rows[1]["input_scale"], "128")
        self.assertEqual(rows[1]["status"], "error")
        self.assertIn("client_request_timeout", rows[1]["error"])
        self.assertIn("planned_request_attempted=true", rows[1]["error"])
        self.assertIn("measurement_row_completed=false", rows[1]["error"])
        self.assertIn("triggering_request_latency_s>300", rows[1]["error"])

    def test_run_single_case_writes_error_rows_when_container_start_fails(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator._start_container_session",
            side_effect=RuntimeError(
                "container_oom_killed during startup "
                "(container=unit-test, memory_limit=2g, status=exited, exit_code=137)"
            ),
        ):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=2,
                gpu="on",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=1,
                repeat=2,
                input_scales="85,170",
            )

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["status"] for row in rows}, {"error"})
        self.assertTrue(all(row["cpu_cores"] == "1" for row in rows))
        self.assertTrue(all(row["mem_cap_gb"] == "2" for row in rows))
        self.assertTrue(all(row["gpu_mode"] == "on" for row in rows))
        self.assertEqual(
            sorted({float(row["input_scale"]) for row in rows}),
            [85.0, 170.0],
        )
        self.assertEqual(
            sorted((row["warmup"], row["repeat_idx"]) for row in rows),
            [
                ("0", "0"),
                ("0", "0"),
                ("0", "1"),
                ("0", "1"),
                ("1", "0"),
                ("1", "0"),
            ],
        )
        self.assertTrue(
            all(
                "container_start_failed: container_oom_killed during startup"
                in row["error"]
                for row in rows
            )
        )

    def test_run_single_case_accepts_stable_idle_power_case_csv(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                _write_gpu_case_csv(kwargs["env"]["OUT_CSV"], [10.0, 10.2, 10.1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="on",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=3,
                repeat_in_window=1,
                input_scales="64",
                require_packet_latency=False,
            )

        self.assertTrue(csv_path.endswith(".csv"))

    def test_run_single_case_warns_when_gpu_idle_power_csv_is_unstable(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                _write_gpu_case_csv(kwargs["env"]["OUT_CSV"], [10.0, 10.7, 10.2])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        stdout = StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run), redirect_stdout(stdout):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="on",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=3,
                repeat_in_window=1,
                input_scales="64",
                require_packet_latency=False,
            )

        message = stdout.getvalue()
        self.assertTrue(csv_path.endswith(".csv"))
        self.assertIn("[energy][WARN]", message)
        self.assertIn("gpu_idle_power_w", message)
        self.assertIn("6.8%", message)
        self.assertIn("5.0%", message)
        self.assertIn("--idle-seconds", message)
        self.assertIn("GPU processes", message)

    def test_run_single_case_warns_when_cpu_idle_power_csv_is_unstable(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd and cmd[-2:] == ["-m", "acprof.host.client"]:
                _write_cpu_case_csv(kwargs["env"]["OUT_CSV"], [5.0, 5.4, 5.1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        stdout = StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ), patch("acprof.host.orchestrator._run", side_effect=fake_run), redirect_stdout(stdout):
            csv_path = orchestrator.run_single_case(
                task_info=task_info,
                cpu=1,
                mem=4,
                gpu="off",
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                output_dir=tmp_dir,
                project_dir=".",
                warmup=0,
                repeat=3,
                repeat_in_window=1,
                input_scales="64",
                require_packet_latency=False,
            )

        message = stdout.getvalue()
        self.assertTrue(csv_path.endswith(".csv"))
        self.assertIn("[energy][WARN]", message)
        self.assertIn("cpu_idle_power_w", message)
        self.assertIn("7.7%", message)
        self.assertIn("5.0%", message)
        self.assertIn("--idle-seconds", message)
        self.assertIn("host background processes", message)

    def test_resolve_packet_latency_runtime_requires_local_linux_tools(self) -> None:
        with patch("acprof.host.orchestrator.shutil.which", return_value=None):
            runtime = orchestrator._resolve_packet_latency_runtime(
                project_dir="/repo",
                pcap_file="/repo/results/sniff_case.pcap",
                sniff_iface="docker0",
            )

        self.assertIsNone(runtime)

    def test_resolve_packet_latency_runtime_uses_tcpdump_without_sudo_when_capable(self) -> None:
        def fake_which(name: str) -> str | None:
            return {
                "tcpdump": "/usr/bin/tcpdump",
                "tshark": "/usr/bin/tshark",
            }.get(name)

        def fake_run(cmd, check=True, capture=True, **kwargs):
            if cmd[:2] == ["getcap", "/usr/bin/tcpdump"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="/usr/bin/tcpdump cap_net_admin,cap_net_raw=eip\n",
                    stderr="",
                )
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        with patch("acprof.host.orchestrator.shutil.which", side_effect=fake_which), patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ):
            runtime = orchestrator._resolve_packet_latency_runtime(
                project_dir="/repo",
                pcap_file="/repo/results/sniff_case.pcap",
                sniff_iface="docker0",
            )

        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(runtime.mode, "local")
        self.assertEqual(runtime.tcpdump_cmd[0], "/usr/bin/tcpdump")
        self.assertNotIn("sudo", runtime.tcpdump_cmd)

    def test_resolve_packet_latency_runtime_bootstraps_tcpdump_capability(self) -> None:
        def fake_which(name: str) -> str | None:
            return {
                "tcpdump": "/usr/bin/tcpdump",
                "tshark": "/usr/bin/tshark",
            }.get(name)

        calls = []
        has_capability = False

        def fake_run(cmd, check=True, capture=True, **kwargs):
            nonlocal has_capability
            calls.append((cmd, kwargs))
            if cmd[:2] == ["getcap", "/usr/bin/tcpdump"]:
                stdout = (
                    "/usr/bin/tcpdump cap_net_admin,cap_net_raw=eip\n"
                    if has_capability
                    else ""
                )
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
            if cmd[:5] == ["sudo", "-S", "-p", "", "setcap"]:
                self.assertEqual(kwargs.get("input"), "secret\n")
                has_capability = True
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        with patch("acprof.host.orchestrator.shutil.which", side_effect=fake_which), patch(
            "acprof.host.orchestrator.os.geteuid",
            return_value=1000,
        ), patch.dict(
            "acprof.host.orchestrator.os.environ",
            {"ACPROF_SUDO_PASSWORD": "secret"},
            clear=True,
        ), patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ):
            runtime = orchestrator._resolve_packet_latency_runtime(
                project_dir="/repo",
                pcap_file="/repo/results/sniff_case.pcap",
                sniff_iface="docker0",
            )

        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(runtime.tcpdump_cmd[0], "/usr/bin/tcpdump")
        self.assertNotIn("sudo", runtime.tcpdump_cmd)
        self.assertTrue(
            any(call[0][:5] == ["sudo", "-S", "-p", "", "setcap"] for call in calls)
        )

    def test_run_single_case_fails_when_packet_latency_runtime_unavailable(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        with patch(
            "acprof.host.orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("acprof.host.orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "acprof.host.orchestrator._stop_container_session"
        ):
            with self.assertRaises(orchestrator.PacketLatencyError) as raised:
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

        message = str(raised.exception)
        self.assertIn("packet latency is required", message)
        self.assertIn("tcpdump", message)
        self.assertIn("tshark", message)
        self.assertIn("sudo setcap", message)

    def test_assert_packet_latency_csv_complete_rejects_nan_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_case.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("latency_s,status\nnan,ok\n")

            with self.assertRaises(orchestrator.PacketLatencyError) as raised:
                orchestrator._assert_packet_latency_csv_complete(csv_path)

        self.assertIn("latency_s is missing", str(raised.exception))

    def test_assert_packet_latency_csv_complete_ignores_timeout_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_case.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("latency_s,status\n0.25,ok\nnan,error\n")

            orchestrator._assert_packet_latency_csv_complete(
                csv_path,
                ignore_error_rows=True,
            )


if __name__ == "__main__":
    unittest.main()

import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import orchestrator
from config import CSV_FIELDS, STATIC_META_FIELDS
from detect import TaskInfo


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
    def test_select_nlp_torch_index_url_uses_cu124_for_cuda_12_4_driver(self) -> None:
        with patch("orchestrator.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "orchestrator._run",
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
            "orchestrator.os.environ",
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
            "orchestrator.os.environ",
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
        )
        commands = []

        def fake_run(cmd, check=True, capture=True, **kwargs):
            commands.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "orchestrator._select_nlp_torch_index_url",
            return_value=orchestrator.CUDA124_NLP_TORCH_INDEX_URL,
        ), patch("orchestrator.os.path.exists", return_value=True), patch(
            "orchestrator._run",
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
            "orchestrator._get_gpu_mem_total_bytes", return_value=987654321
        ), patch(
            "orchestrator._docker_model_weight_bytes", return_value=123
        ), patch("orchestrator._docker_image_size_bytes", return_value=456), patch(
            "orchestrator._cpu_power_metadata",
            return_value=("rapl", "rapl_cgroup_cpu_share"),
        ), patch(
            "orchestrator._cpu_frequency_policy_metadata",
            return_value=("performance", "on"),
        ):
            meta = orchestrator.collect_static_meta(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                batch_size=1,
                input_scale_type="seq_length",
            )

        self.assertEqual(meta.environment, "windows11+wsl")
        self.assertEqual(meta.gpu_mem_total_bytes, 987654321)
        self.assertEqual(meta.cpu_power_source, "rapl")
        self.assertEqual(meta.vcpu_power_method, "rapl_cgroup_cpu_share")
        self.assertEqual(meta.cpu_governor, "performance")
        self.assertEqual(meta.cpu_boost, "on")

    def test_static_meta_cpu_power_fields_are_trailing(self) -> None:
        self.assertIn("gpu_mem_total_bytes", STATIC_META_FIELDS)
        self.assertLess(
            STATIC_META_FIELDS.index("gpu_mem_total_bytes"),
            STATIC_META_FIELDS.index("environment"),
        )
        self.assertEqual(
            STATIC_META_FIELDS[-5:],
            [
                "environment",
                "cpu_power_source",
                "vcpu_power_method",
                "cpu_governor",
                "cpu_boost",
            ],
        )

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

            with patch("orchestrator.CPU_SYSFS_ROOT", tmp):
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
            "orchestrator._start_probe_session",
            return_value=session,
        ), patch(
            "orchestrator._stop_container_session",
        ), patch(
            "orchestrator._post_probe_payload",
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
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
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
            if cmd and str(cmd[-1]).endswith("client.py"):
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
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
                compute_profile_plan_file="results/test-unit/compute_profile_plan.json",
                require_packet_latency=False,
            )

        self.assertEqual(
            captured_env["COMPUTE_PROFILE_PLAN_FILE"],
            "results/test-unit/compute_profile_plan.json",
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
            if cmd and str(cmd[-1]).endswith("client.py"):
                captured_env.update(kwargs.get("env", {}))
                _write_cpu_case_csv(captured_env["OUT_CSV"], [5.0])
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
                idle_debug=True,
                require_packet_latency=False,
            )

        self.assertEqual(captured_env["IDLE_DEBUG"], "1")
        self.assertEqual(captured_env["IDLE_DIAG_PATH"], captured_env["OUT_CSV"] + ".idle_diag.jsonl")

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
            if cmd and str(cmd[-1]).endswith("client.py"):
                captured_env.update(kwargs.get("env", {}))
                _write_gpu_case_csv(captured_env["OUT_CSV"], [10.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
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
            if cmd and str(cmd[-1]).endswith("client.py"):
                captured_env.update(kwargs.get("env", {}))
                _write_gpu_case_csv(captured_env["OUT_CSV"], [10.0])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
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
            if cmd and str(cmd[-1]).endswith("client.py"):
                return SimpleNamespace(returncode=7, stdout="", stderr="idle unstable")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "orchestrator._stop_container_session"
        ), patch("orchestrator._run", side_effect=fake_run):
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
            if cmd and str(cmd[-1]).endswith("client.py"):
                _write_gpu_case_csv(kwargs["env"]["OUT_CSV"], [10.0, 10.2, 10.1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "orchestrator._stop_container_session"
        ), patch("orchestrator._run", side_effect=fake_run):
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

    def test_run_single_case_aborts_when_case_idle_power_csv_is_unstable(self) -> None:
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
            if cmd and str(cmd[-1]).endswith("client.py"):
                _write_gpu_case_csv(kwargs["env"]["OUT_CSV"], [10.0, 10.7, 10.2])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_on",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "orchestrator._stop_container_session"
        ), patch("orchestrator._run", side_effect=fake_run):
            with self.assertRaises(orchestrator.EnergyProfilingError) as raised:
                orchestrator.run_single_case(
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

        message = str(raised.exception)
        self.assertIn("gpu_idle_power_w", message)
        self.assertIn("6.8%", message)
        self.assertIn("5.0%", message)
        self.assertIn("--idle-seconds", message)
        self.assertIn("GPU processes", message)

    def test_run_single_case_aborts_when_cpu_idle_power_csv_is_unstable(self) -> None:
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
            if cmd and str(cmd[-1]).endswith("client.py"):
                _write_cpu_case_csv(kwargs["env"]["OUT_CSV"], [5.0, 5.4, 5.1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
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
            with self.assertRaises(orchestrator.EnergyProfilingError) as raised:
                orchestrator.run_single_case(
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

        message = str(raised.exception)
        self.assertIn("cpu_idle_power_w", message)
        self.assertIn("7.7%", message)
        self.assertIn("5.0%", message)
        self.assertIn("--idle-seconds", message)
        self.assertIn("host background processes", message)

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

        with patch("orchestrator.shutil.which", side_effect=fake_which), patch(
            "orchestrator._run",
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

        with patch("orchestrator.shutil.which", side_effect=fake_which), patch(
            "orchestrator.os.geteuid",
            return_value=1000,
        ), patch.dict(
            "orchestrator.os.environ",
            {"ACPROF_SUDO_PASSWORD": "secret"},
            clear=True,
        ), patch(
            "orchestrator._run",
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
            "orchestrator._start_container_session",
            return_value=orchestrator.RunningContainer(
                name="case_google-bert--bert-base-uncased_1c_4g_off",
                base_url="http://127.0.0.1:8106",
                host_port=8106,
                cold_start_s=1.0,
            ),
        ), patch("orchestrator._resolve_packet_latency_runtime", return_value=None), patch(
            "orchestrator._stop_container_session"
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


if __name__ == "__main__":
    unittest.main()

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
from acprof.config import CSV_FIELDS, STATIC_META_FIELDS
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
            "acprof.host.orchestrator._docker_model_weight_bytes", return_value=123
        ), patch("acprof.host.orchestrator._docker_image_size_bytes", return_value=456), patch(
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
        self.assertEqual(meta.cpu_power_source, "rapl")
        self.assertEqual(meta.vcpu_power_method, "rapl_cgroup_cpu_share")
        self.assertEqual(meta.cpu_governor, "performance")
        self.assertEqual(meta.cpu_boost, "on")
        self.assertEqual(meta.parameter_count, 110_106_428)
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
        self.assertIn("gpu_mem_total_bytes", STATIC_META_FIELDS)
        self.assertLess(
            STATIC_META_FIELDS.index("gpu_mem_total_bytes"),
            STATIC_META_FIELDS.index("environment"),
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
            model_weight_bytes=456,
            docker_image_bytes=789,
            environment="ubuntu24.04",
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
            model_weight_bytes=456,
            docker_image_bytes=789,
            environment="ubuntu24.04",
            cpu_power_source="rapl",
            vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance",
            cpu_boost="off",
            parameter_count=42,
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
            model_weight_bytes=456,
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

        for task_family in ("cv", "audio", "timeseries"):
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
                            "payload": {"value": 1.0},
                        },
                        {
                            "input_scale": 2.0,
                            "scale_label": "scale2",
                            "payload": {"value": 2.0},
                        },
                    ],
                )

    def test_auto_non_nlp_scales_write_reusable_payload_plan(self) -> None:
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
                return f"duration{scale:g}"

        task_info = TaskInfo(
            model_id="test/audio",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ), patch(
            "acprof.host.orchestrator._default_family_max_scale",
            return_value=6.0,
        ):
            planned = orchestrator.plan_input_scales(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
            )

            self.assertEqual(planned.scales, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            assert planned.plan_file is not None
            with open(planned.plan_file, "r", encoding="utf-8") as f:
                plan = json.load(f)

        self.assertEqual(
            [entry["input_scale"] for entry in plan["entries"]],
            planned.scales,
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
            side_effect=RuntimeError("server not ready after 180s"),
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
                "container_start_failed: server not ready after 180s" in row["error"]
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


if __name__ == "__main__":
    unittest.main()

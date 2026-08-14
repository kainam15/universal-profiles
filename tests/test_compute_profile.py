import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acprof.container.handlers import transformers_pipeline_load_kwargs
from acprof.host.detect import TaskInfo

from acprof.host import compute_profile


def _write_input_scale_plan(directory: str, input_scale: float = 8.0) -> str:
    path = os.path.join(directory, "input_scale_plan.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": "google-bert/bert-base-uncased",
                "task_family": "nlp",
                "pipeline_tag": "fill-mask",
                "entries": [
                    {
                        "input_scale": input_scale,
                        "scale_label": f"seq{input_scale:g}",
                        "payload": {
                            "text": "hello [MASK]",
                            "params": {},
                        },
                    }
                ],
            },
            f,
        )
    return path


def _ncu_resume_csv_text() -> str:
    return "\n".join([
        (
            '"ID","Kernel Name","smsp__sass_thread_inst_executed_op_'
            'ffma_pred_on.sum","sm__ops_path_tensor_src_fp16_dst_fp32.sum",'
            '"gpu__time_duration.sum"'
        ),
        '"","","inst","FLOP","usec"',
        '"0","kernel_a","10","100","1000"',
        '"1","kernel_b","20","300","2000"',
    ])


class ComputeProfileTests(unittest.TestCase):
    def test_transformers_handler_forces_eager_only_when_requested(self) -> None:
        self.assertEqual(transformers_pipeline_load_kwargs(None), {})
        self.assertEqual(
            transformers_pipeline_load_kwargs({
                "attention_implementation": "eager",
            }),
            {"model_kwargs": {"attn_implementation": "eager"}},
        )
        with self.assertRaisesRegex(ValueError, "must be 'eager'"):
            transformers_pipeline_load_kwargs({
                "attention_implementation": "sdpa",
            })

    def test_input_scale_plan_is_required(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input_scale_plan_file is required",
        ):
            compute_profile._load_input_scale_plan_entries("")

        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "input_scale_plan.json")
            with self.assertRaisesRegex(
                FileNotFoundError,
                "input scale plan not found",
            ):
                compute_profile._load_input_scale_plan_entries(missing)

    def test_v1_and_v2_input_plans_reuse_the_exact_payload(self) -> None:
        payload = {
            "audio_base64": "UklGRg==",
            "audio_format": "wav",
            "sample_rate": 16000,
            "params": {"asr_task": "transcribe"},
        }
        for schema_version in (1, 2):
            with self.subTest(schema_version=schema_version), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "input_scale_plan.json")
                plan = {
                    "entries": [
                        {
                            "input_scale": 1.0,
                            "scale_label": "dur1s",
                            "payload": payload,
                        }
                    ]
                }
                if schema_version == 2:
                    plan.update({
                        "schema_version": 2,
                        "workload": {"workload_id": "fixture"},
                        "model_constraints": {"max_short_form_duration_s": 30},
                    })
                    plan["entries"][0]["input_metadata"] = {
                        "input_num_samples": 16000
                    }
                with open(path, "w", encoding="utf-8") as plan_file:
                    json.dump(plan, plan_file)

                entries = compute_profile._load_input_scale_plan_entries(path)

                self.assertEqual(entries[0]["payload"], payload)

    def test_compute_container_is_offline_and_does_not_receive_hf_token(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="0123456789abcdef",
            detection_method="hub_api",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload_file = os.path.join(tmp_dir, "payloads.json")
            profile_root = os.path.join(tmp_dir, "profiles")
            os.makedirs(profile_root)
            with open(payload_file, "w", encoding="utf-8") as f:
                f.write("{}")

            cmd = compute_profile._base_docker_cmd(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=1,
                mem=2,
                use_gpu=False,
                payload_file=payload_file,
                profile_root=profile_root,
                tool_mount_roots=(),
            )

        self.assertIn(
            f"{os.path.abspath(payload_file)}:/payloads/input_scale_plan.json:ro",
            cmd,
        )
        self.assertIn("HF_HUB_OFFLINE=1", cmd)
        self.assertIn("TRANSFORMERS_OFFLINE=1", cmd)
        self.assertIn("MODEL_LOCAL_PATH=/models/model-snapshot", cmd)
        self.assertNotIn("HF_TOKEN", cmd)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", cmd)

    def test_parse_advisor_report_sums_self_gflop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "advisor.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Function", "Self GFLOP"])
                writer.writeheader()
                writer.writerow({"Function": "a", "Self GFLOP": "1.5"})
                writer.writerow({"Function": "b", "Self GFLOP": "2.25"})

            self.assertAlmostEqual(
                compute_profile.parse_advisor_self_gflop_csv(report_path),
                3.75,
            )

    def test_parse_advisor_report_skips_native_csv_preamble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "advisor.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                f.write('sep=,\n\n')
                f.write('"Intel(R) Advisor Command Line Tool\\n')
                f.write('Copyright (C) 2009-2025 Intel Corporation. All rights reserved."\n')
                f.write('"Survey Data version=1.1.0","delimiter=,"\n\n')
                writer = csv.DictWriter(f, fieldnames=["ID", "Self GFLOP", "Module"])
                writer.writeheader()
                writer.writerow({"ID": "1", "Self GFLOP": "1.5", "Module": "libtorch_cpu.so"})
                writer.writerow({"ID": "2", "Self GFLOP": "< 0.001", "Module": "libtorch_cpu.so"})
                writer.writerow({"ID": "3", "Self GFLOP": "2.25", "Module": "libtorch_cpu.so"})

            self.assertAlmostEqual(
                compute_profile.parse_advisor_self_gflop_csv(report_path),
                3.75,
            )

    def test_parse_ncu_raw_csv_sums_flop_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"],
                )
                writer.writeheader()
                writer.writerow({
                    "Kernel Name": "kernel_a",
                    "Metric Name": "flop_count_sp",
                    "Metric Unit": "FLOP",
                    "Metric Value": "1,000",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
                    "Metric Unit": "FLOP",
                    "Metric Value": "2.5K",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": (
                        "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_on.sum"
                    ),
                    "Metric Unit": "FLOP",
                    "Metric Value": "1.5K",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": (
                        "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum"
                    ),
                    "Metric Unit": "FLOP",
                    "Metric Value": "1K",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": "sm__ops_path_tensor_src_int8_dst_int32.sum",
                    "Metric Unit": "OP",
                    "Metric Value": "9K",
                })
                writer.writerow({
                    "Kernel Name": "kernel_c",
                    "Metric Name": "gpu__time_duration.sum",
                    "Metric Unit": "nsecond",
                    "Metric Value": "10",
                })

            self.assertAlmostEqual(
                compute_profile.parse_ncu_flop_csv(report_path),
                3500.0,
            )

    def test_parse_ncu_raw_csv_weights_sass_fma_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"],
                )
                writer.writeheader()
                writer.writerow({
                    "Kernel Name": "kernel_a",
                    "Metric Name": "smsp__sass_thread_inst_executed_op_fadd_pred_on",
                    "Metric Unit": "inst",
                    "Metric Value": "10",
                })
                writer.writerow({
                    "Kernel Name": "kernel_b",
                    "Metric Name": "smsp__sass_thread_inst_executed_op_ffma_pred_on",
                    "Metric Unit": "inst",
                    "Metric Value": "10",
                })

            self.assertAlmostEqual(
                compute_profile.parse_ncu_flop_csv(report_path),
                30.0,
            )

    def test_parse_ncu_wide_csv_sums_metric_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu_wide.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID",
                    "Kernel Name",
                    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                    "gpu__time_duration.sum",
                ])
                writer.writerow(["", "", "inst", "inst", "nsecond"])
                writer.writerow(["0", "kernel_a", "10", "20", "100"])
                writer.writerow(["1", "kernel_b", "1K", "2K", "200"])

            self.assertAlmostEqual(
                compute_profile.parse_ncu_flop_csv(report_path),
                10 + 20 * 2 + 1000 + 2000 * 2,
            )

    def test_parse_ncu_long_csv_returns_normalized_structured_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu_long.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "ID",
                        "Kernel Name",
                        "Metric Name",
                        "Metric Unit",
                        "Metric Value",
                    ],
                )
                writer.writeheader()
                metrics = [
                    ("0", "kernel_a", "smsp__sass_thread_inst_executed_op_fadd_pred_on", "inst", "10"),
                    ("0", "kernel_a", "smsp__sass_thread_inst_executed_op_ffma_pred_on", "inst", "20"),
                    ("0", "kernel_a", "sm__ops_path_tensor_src_fp16_dst_fp32.sum", "FLOP", "100"),
                    ("0", "kernel_a", "gpu__time_duration.sum", "usec", "1000"),
                    ("1", "kernel_b", "smsp__sass_thread_inst_executed_op_fadd_pred_on", "inst", "30"),
                    ("1", "kernel_b", "smsp__sass_thread_inst_executed_op_ffma_pred_on", "inst", "40"),
                    ("1", "kernel_b", "sm__ops_path_tensor_src_fp16_dst_fp32.sum", "FLOP", "300"),
                    ("1", "kernel_b", "gpu__time_duration.sum", "msecond", "2"),
                ]
                for launch_id, kernel, metric, unit, value in metrics:
                    writer.writerow({
                        "ID": launch_id,
                        "Kernel Name": kernel,
                        "Metric Name": metric,
                        "Metric Unit": unit,
                        "Metric Value": value,
                    })

            parsed = compute_profile.parse_ncu_profile_csv(
                report_path,
                repeat=2,
            )

        self.assertAlmostEqual(parsed["scalar_flops_per_request"], 80.0)
        self.assertAlmostEqual(parsed["tensor_flops_per_request"], 200.0)
        self.assertAlmostEqual(parsed["total_flops_per_request"], 280.0)
        self.assertAlmostEqual(
            parsed["tensor_share_pct"],
            (400.0 / 560.0) * 100.0,
        )
        self.assertAlmostEqual(parsed["kernel_launch_count_per_request"], 1.0)
        self.assertAlmostEqual(parsed["kernel_time_sum_ms_per_request"], 1.5)

    def test_parse_ncu_wide_csv_normalizes_duration_and_launches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "ncu_wide.csv")
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID",
                    "Kernel Name",
                    "CC",
                    "launch__sm_count",
                    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                    "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
                    "gpu__time_duration.sum",
                ])
                writer.writerow(["", "", "", "SM", "inst", "FLOP", "nsecond"])
                writer.writerow(["0", "kernel_a", "8.9", "24", "10", "100", "1000"])
                writer.writerow(["1", "kernel_b", "8.9", "24", "20", "200", "3000"])

            parsed = compute_profile.parse_ncu_profile_csv(
                report_path,
                repeat=2,
            )

        self.assertAlmostEqual(parsed["scalar_flops_per_request"], 30.0)
        self.assertAlmostEqual(parsed["tensor_flops_per_request"], 150.0)
        self.assertAlmostEqual(parsed["total_flops_per_request"], 180.0)
        self.assertAlmostEqual(parsed["kernel_launch_count_per_request"], 1.0)
        self.assertAlmostEqual(parsed["kernel_time_sum_ms_per_request"], 0.002)
        self.assertEqual(parsed["gpu_compute_capability"], "8.9")
        self.assertEqual(parsed["gpu_sm_count"], 24.0)

    def test_torch_profile_rejects_unverified_eager_result(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        runner_result = {
            "model_logical_mflop_per_request_torch_profiler_eager": 123.0,
            "attention_implementation": "sdpa",
            "attention_implementation_verified": False,
        }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._base_docker_cmd",
            return_value=["docker"],
        ), patch(
            "acprof.host.compute_profile._run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout=json.dumps(runner_result),
                stderr="",
            ),
        ):
            entry = compute_profile._run_torch_profiler_for_entry(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=1,
                mem=4,
                use_gpu=True,
                payload_file=os.path.join(tmp, "payloads.json"),
                profile_root=tmp,
                entry={"input_scale": 8.0},
                repeat=1,
            )

        self.assertIsNone(
            entry["model_logical_mflop_per_request_torch_profiler_eager"]
        )
        self.assertIn("attention_implementation_not_verified", entry["error"])

    def test_ncu_entry_exposes_exact_csv_metric_keys_and_relative_report(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        csv_text = "\n".join([
            (
                '"ID","Kernel Name","smsp__sass_thread_inst_executed_op_'
                'ffma_pred_on.sum","sm__ops_path_tensor_src_fp16_dst_fp32.sum",'
                '"gpu__time_duration.sum"'
            ),
            '"","","inst","FLOP","usec"',
            '"0","kernel_a","10","100","1000"',
            '"1","kernel_b","20","300","2000"',
        ])
        collect_result = SimpleNamespace(
            returncode=0,
            stdout=(
                '{"gpu_compute_capability":"8.9","gpu_sm_count":24}\n'
            ),
            stderr="",
        )
        import_result = SimpleNamespace(
            returncode=0,
            stdout=csv_text,
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "compute_profiles")
            os.makedirs(profile_root)
            with patch(
                "acprof.host.compute_profile._base_docker_cmd",
                return_value=["docker"],
            ), patch(
                "acprof.host.compute_profile._ncu_collect_filter_args",
                return_value=[],
            ), patch(
                "acprof.host.compute_profile._ncu_section_args",
                return_value=[],
            ), patch(
                "acprof.host.compute_profile._run",
                side_effect=[collect_result, import_result],
            ):
                result = compute_profile._run_ncu_for_entry(
                    ncu_bin="/usr/bin/ncu",
                    ncu_metrics=[
                        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                        "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
                        "gpu__time_duration.sum",
                    ],
                    task_info=task_info,
                    image_tag="acprof-test:latest",
                    cpu=1,
                    mem=4,
                    payload_file=os.path.join(tmp, "payloads.json"),
                    profile_root=profile_root,
                    tool_mount_roots=(),
                    entry={"input_scale": 8.0},
                    repeat=2,
                )

            self.assertTrue(
                os.path.isfile(os.path.join(profile_root, "ncu_scale_8.csv"))
            )

        self.assertAlmostEqual(
            result["gpu_executed_mflop_per_request_ncu"],
            0.00023,
        )
        self.assertAlmostEqual(
            result["gpu_executed_tensor_mflop_per_request_ncu"],
            0.0002,
        )
        self.assertAlmostEqual(
            result["gpu_executed_scalar_mflop_per_request_ncu"],
            0.00003,
        )
        self.assertAlmostEqual(
            result["gpu_executed_tensor_share_pct_ncu"],
            (400.0 / 460.0) * 100.0,
        )
        self.assertEqual(result["gpu_kernel_launch_count_per_request_ncu"], 1.0)
        self.assertEqual(result["gpu_kernel_time_sum_ms_per_request_ncu"], 1.5)
        self.assertNotIn("gpu_profile_report_ncu", result)
        self.assertEqual(
            result["report"],
            os.path.join("compute_profiles", "ncu_scale_8.csv"),
        )

    def test_ncu_parse_failure_preserves_report_until_artifacts_are_discarded(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        command_result = SimpleNamespace(
            returncode=0,
            stdout="not a parseable NCU CSV",
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "compute_profiles")
            os.makedirs(profile_root)
            with patch(
                "acprof.host.compute_profile._base_docker_cmd",
                return_value=["docker"],
            ), patch(
                "acprof.host.compute_profile._ncu_collect_filter_args",
                return_value=[],
            ), patch(
                "acprof.host.compute_profile._ncu_section_args",
                return_value=[],
            ), patch(
                "acprof.host.compute_profile._run",
                side_effect=[command_result, command_result],
            ):
                result = compute_profile._run_ncu_for_entry(
                    ncu_bin="/usr/bin/ncu",
                    ncu_metrics=["flop_count_sp"],
                    task_info=task_info,
                    image_tag="acprof-test:latest",
                    cpu=1,
                    mem=4,
                    payload_file=os.path.join(tmp, "payloads.json"),
                    profile_root=profile_root,
                    tool_mount_roots=(),
                    entry={"input_scale": 8.0},
                    repeat=1,
                )

            report = os.path.join("compute_profiles", "ncu_scale_8.csv")
            self.assertNotIn("gpu_profile_report_ncu", result)
            self.assertEqual(result["report"], report)
            self.assertTrue(os.path.isfile(os.path.join(tmp, report)))

            profiles = {
                "gpu": {
                    "ncu": {
                        "tool": "ncu",
                        "entries": [result],
                    },
                },
            }
            compute_profile._strip_discarded_profile_paths(profiles)

        self.assertNotIn("gpu_profile_report_ncu", result)
        self.assertIsNone(result["report"])

    def test_select_ncu_metrics_combines_sass_and_float_tensor_aggregates(self) -> None:
        fadd = "smsp__sass_thread_inst_executed_op_fadd_pred_on"
        ffma = "smsp__sass_thread_inst_executed_op_ffma_pred_on"
        tensor = "sm__ops_path_tensor_src_fp16_dst_fp32.sum"

        metrics = compute_profile._select_ncu_flop_metrics([
            fadd,
            f"{ffma}.sum",
            "flop_count_sp",
            tensor,
            "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_on.sum",
            "sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum",
            "sm__ops_path_tensor_src_int8_dst_int32.sum",
        ])

        self.assertEqual(metrics, [fadd, f"{ffma}.sum", tensor])

    def test_resolve_ncu_metrics_falls_back_when_query_requires_privileges(self) -> None:
        calls = []

        def fake_run(cmd, check=False):
            calls.append(cmd)
            if "--query-metrics-mode" in cmd:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "==ERROR== Invalid option --query-metrics-mode suffix. "
                        "Please specify along with --metrics."
                    ),
                )
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Device NVIDIA GeForce RTX 4060 Laptop GPU\n"
                    "==ERROR== ERR_NVGPUCTRPERM - The user does not have "
                    "permission to access NVIDIA GPU Performance Counters\n"
                ),
                stderr="",
            )

        with patch("acprof.host.compute_profile._run", side_effect=fake_run):
            metrics, error = compute_profile._resolve_ncu_metrics("/opt/ncu")

        self.assertEqual(metrics, list(compute_profile.NCU_SASS_FLOP_WEIGHTS))
        self.assertEqual(error, "")
        self.assertEqual(len(calls), 2)

    def test_resolve_ncu_metrics_retries_query_in_gpu_container(self) -> None:
        calls = []
        container_base_cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--cap-add=SYS_ADMIN",
            "--cap-add=SYS_PTRACE",
            "acprof-test:latest",
        ]
        fadd = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
        tensor = "sm__ops_path_tensor_src_fp16_dst_fp32.sum"

        def fake_run(cmd, check=False):
            calls.append(cmd)
            if cmd[:len(container_base_cmd)] == container_base_cmd:
                return SimpleNamespace(
                    returncode=0,
                    stdout="\n".join([
                        fadd,
                        tensor,
                        (
                            "sm__ops_path_tensor_src_fp16_dst_fp32_"
                            "sparsity_on.sum"
                        ),
                        "sm__ops_path_tensor_src_int8_dst_int32.sum",
                    ]),
                    stderr="",
                )
            if "--query-metrics-mode" in cmd:
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="unsupported query mode",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="",
                stderr=(
                    "==ERROR== ERR_NVGPUCTRPERM - The user does not have "
                    "permission to access NVIDIA GPU Performance Counters"
                ),
            )

        with patch("acprof.host.compute_profile._run", side_effect=fake_run):
            metrics, error = compute_profile._resolve_ncu_metrics(
                "/opt/ncu",
                container_base_cmd=container_base_cmd,
            )

        self.assertEqual(metrics, [fadd, tensor])
        self.assertEqual(error, "")
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            calls[-1],
            [
                *container_base_cmd,
                "/opt/ncu",
                "--query-metrics",
                "--query-metrics-mode",
                "all",
            ],
        )

    def test_gpu_profile_builds_privileged_container_for_metric_query(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        query = {}

        def fake_resolve(ncu_bin, *, container_base_cmd=None):
            query["ncu_bin"] = ncu_bin
            query["container_base_cmd"] = container_base_cmd
            return list(compute_profile.NCU_SASS_FLOP_WEIGHTS), ""

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._resolve_ncu_metrics",
            side_effect=fake_resolve,
        ), patch(
            "acprof.host.compute_profile._run_ncu_for_entry",
            return_value={
                "input_scale": 8.0,
                "tool": "ncu",
                "gpu_executed_mflop_per_request_ncu": 1.0,
                "error": "",
            },
        ):
            compute_profile._profile_gpu_entries(
                entries=[{"input_scale": 8.0}],
                ncu_bin="/opt/nvidia/nsight-compute/2025.1.0/ncu",
                ncu_root=None,
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu=2,
                mem=4,
                payload_file=os.path.join(tmp, "payloads.json"),
                profile_root=tmp,
                repeat=1,
            )

        self.assertEqual(
            query["ncu_bin"],
            "/opt/nvidia/nsight-compute/2025.1.0/ncu",
        )
        container_base_cmd = query["container_base_cmd"]
        self.assertIn("--gpus", container_base_cmd)
        self.assertIn("--cap-add=SYS_ADMIN", container_base_cmd)
        self.assertIn("--cap-add=SYS_PTRACE", container_base_cmd)
        self.assertIn("--security-opt=seccomp=unconfined", container_base_cmd)
        self.assertEqual(container_base_cmd[-1], "acprof-test:latest")

    def test_ncu_resume_reuses_csv_recovers_report_and_collects_only_missing(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="revision-1",
            detection_method="hub_api",
        )
        metrics = [
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
        ]
        collected = []

        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "compute_profiles")
            os.makedirs(profile_root)
            with open(
                os.path.join(profile_root, "ncu_scale_1.csv"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(_ncu_resume_csv_text())
            with open(
                os.path.join(profile_root, "ncu_scale_20.ncu-rep"),
                "wb",
            ) as f:
                f.write(b"existing report")

            def fake_export(**kwargs):
                self.assertTrue(kwargs["report_base"].endswith("ncu_scale_20"))
                compute_profile._write_text_atomic(
                    kwargs["host_csv"],
                    _ncu_resume_csv_text(),
                )
                return True, ""

            def fake_collect(**kwargs):
                scale = float(kwargs["entry"]["input_scale"])
                collected.append(scale)
                self.assertEqual(scale, 30.0)
                _report_base, host_csv, _host_report, _checkpoint = (
                    compute_profile._ncu_artifact_paths(profile_root, scale)
                )
                compute_profile._write_text_atomic(
                    host_csv,
                    _ncu_resume_csv_text(),
                )
                return compute_profile._ncu_entry_from_csv(
                    entry=kwargs["entry"],
                    host_csv=host_csv,
                    profile_root=profile_root,
                    repeat=kwargs["repeat"],
                    runner_payload={
                        "gpu_compute_capability": "8.9",
                        "gpu_sm_count": 24,
                    },
                )

            with patch(
                "acprof.host.compute_profile._resolve_ncu_metrics",
                return_value=(metrics, ""),
            ), patch(
                "acprof.host.compute_profile._export_ncu_report",
                side_effect=fake_export,
            ) as export_report, patch(
                "acprof.host.compute_profile._run_ncu_for_entry",
                side_effect=fake_collect,
            ):
                result = compute_profile._profile_gpu_entries(
                    entries=[
                        {"input_scale": 1.0},
                        {"input_scale": 20.0},
                        {"input_scale": 30.0},
                    ],
                    ncu_bin="/opt/nvidia/nsight-compute/ncu",
                    ncu_root=None,
                    task_info=task_info,
                    image_tag="acprof-test:latest",
                    cpu=2,
                    mem=4,
                    payload_file=os.path.join(tmp, "payloads.json"),
                    profile_root=profile_root,
                    repeat=1,
                    resume_existing=True,
                )

            self.assertEqual(collected, [30.0])
            self.assertEqual(export_report.call_count, 1)
            self.assertEqual(
                [entry["input_scale"] for entry in result["entries"]],
                [1.0, 20.0, 30.0],
            )
            self.assertTrue(all(
                compute_profile._ncu_entry_complete(entry)
                for entry in result["entries"]
            ))
            for scale in (1, 20, 30):
                self.assertTrue(os.path.isfile(os.path.join(
                    profile_root,
                    f"ncu_scale_{scale}.checkpoint.json",
                )))

    def test_ncu_resume_rejects_checkpoint_from_different_repeat(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="revision-1",
            detection_method="hub_api",
        )
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "compute_profiles")
            os.makedirs(profile_root)
            host_csv = os.path.join(profile_root, "ncu_scale_1.csv")
            compute_profile._write_text_atomic(host_csv, _ncu_resume_csv_text())
            entry = compute_profile._ncu_entry_from_csv(
                entry={"input_scale": 1.0},
                host_csv=host_csv,
                profile_root=profile_root,
                repeat=1,
            )
            checkpoint_path = os.path.join(
                profile_root,
                "ncu_scale_1.checkpoint.json",
            )
            compute_profile._write_ncu_checkpoint(
                checkpoint_path=checkpoint_path,
                task_info=task_info,
                image_tag="acprof-test:latest",
                input_scale=1.0,
                repeat=1,
                metrics=["metric", compute_profile.NCU_DURATION_METRIC],
                host_csv=host_csv,
                entry=entry,
            )

            resumed = compute_profile._resume_ncu_for_entry(
                ncu_bin="/opt/ncu",
                ncu_metrics=["metric", compute_profile.NCU_DURATION_METRIC],
                task_info=task_info,
                image_tag="acprof-test:latest",
                base_cmd=["docker"],
                profile_root=profile_root,
                entry={"input_scale": 1.0},
                repeat=2,
            )

        self.assertIsNone(resumed)

    def test_vendor_mode_missing_tools_write_nan_profiles_with_errors(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._find_executable",
            return_value=None,
        ), patch(
            "acprof.host.compute_profile._run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_tool="vendor",
            )

            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            self.assertFalse(os.path.exists(os.path.join(tmp, "compute_profiles")))
            self.assertFalse(
                os.path.exists(
                    os.path.join(tmp, "compute_profile_payloads.json")
                )
            )

        self.assertIn(
            "advisor_not_found",
            plan["profiles"]["cpu"]["intel_advisor"]["error"],
        )
        self.assertIn("ncu_not_found", plan["profiles"]["gpu"]["ncu"]["error"])
        self.assertEqual(
            plan["profiles"]["cpu"]["intel_advisor"]["entries"][0][
                "model_mflop_per_request"
            ],
            None,
        )
        self.assertEqual(
            plan["profiles"]["gpu"]["ncu"]["entries"][0][
                "gpu_executed_mflop_per_request_ncu"
            ],
            None,
        )

    def test_none_compute_profile_mode_writes_disabled_plan_without_probes(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._find_executable",
            side_effect=AssertionError("none mode must not discover tools"),
        ), patch(
            "acprof.host.compute_profile._profile_torch_entries",
            side_effect=AssertionError("none mode must not run Torch"),
        ), patch(
            "acprof.host.compute_profile._profile_gpu_entries",
            side_effect=AssertionError("none mode must not run NCU"),
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=1,
                ncu_repeat=1,
                keep_profiles=True,
                compute_profile_tool="none",
            )
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            self.assertFalse(os.path.exists(os.path.join(tmp, "compute_profiles")))

        self.assertEqual(plan["compute_profile_tool_mode"], "none")
        self.assertEqual(plan["profiles"], {})
        self.assertEqual(plan["static_metadata"]["compute_profile_tools"], [])
        self.assertFalse(plan["static_metadata"]["compute_profiles_retained"])
        self.assertEqual(
            plan["static_metadata"]["compute_profile_provenance"], "disabled"
        )

    def test_default_compute_profile_uses_torch_on_each_device_and_gpu_ncu(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_find_executable(root, names):
            calls.append(("find", names))
            if "ncu" in names:
                return "/opt/nvidia/nsight-compute/2024.1.1/ncu"
            raise AssertionError("advisor should not be resolved by default")

        def fake_torch_profile(**kwargs):
            calls.append(("torch", kwargs["profile_key"], kwargs["use_gpu"]))
            return {
                "tool": "torch_profiler_eager",
                "repeat": kwargs["repeat"],
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        "tool": "torch_profiler_eager",
                        "model_logical_mflop_per_request_torch_profiler_eager": 123.0,
                        "error": "",
                    }
                ],
            }

        def fake_gpu_profile(**kwargs):
            calls.append(("gpu", kwargs["ncu_bin"]))
            return {
                "tool": "ncu",
                "repeat": kwargs["repeat"],
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        "tool": "ncu",
                        "gpu_executed_mflop_per_request_ncu": 300.0,
                        "error": "",
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._find_executable",
            side_effect=fake_find_executable,
        ), patch(
            "acprof.host.compute_profile._profile_torch_entries",
            side_effect=fake_torch_profile,
        ), patch(
            "acprof.host.compute_profile._profile_cpu_entries",
            side_effect=AssertionError("vendor CPU profiler should not run by default"),
        ), patch(
            "acprof.host.compute_profile._profile_gpu_entries",
            side_effect=fake_gpu_profile,
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
            )

            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)

        self.assertEqual(
            calls,
            [
                ("find", ("ncu", "nv-nsight-cu-cli")),
                ("torch", "cpu", False),
                ("torch", "gpu", True),
                ("gpu", "/opt/nvidia/nsight-compute/2024.1.1/ncu"),
            ],
        )
        self.assertEqual(plan["compute_profile_tool_mode"], "both")
        self.assertEqual(
            plan["profiles"]["cpu"]["torch_profiler_eager"]["tool"],
            "torch_profiler_eager",
        )
        self.assertEqual(plan["profiles"]["gpu"]["ncu"]["tool"], "ncu")
        self.assertNotIn("tool", plan["profiles"]["cpu"])
        self.assertNotIn("tool", plan["profiles"]["gpu"])

    def test_auto_compute_profile_is_alias_for_dual_collection(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_find_executable(root, names):
            calls.append(("find", names))
            if "ncu" in names:
                return "/usr/bin/ncu"
            raise AssertionError("advisor should not be resolved in auto mode")

        def fake_torch_profile(**kwargs):
            calls.append(("torch", kwargs["profile_key"], kwargs["use_gpu"]))
            return {
                "tool": "torch_profiler_eager",
                "repeat": kwargs["repeat"],
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        "tool": "torch_profiler_eager",
                        "model_logical_mflop_per_request_torch_profiler_eager": 123.0,
                        "error": "",
                    }
                ],
            }

        def fake_gpu_profile(**kwargs):
            calls.append(("gpu", kwargs["ncu_bin"]))
            return {
                "tool": "ncu",
                "repeat": kwargs["repeat"],
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        "tool": "ncu",
                        "gpu_executed_mflop_per_request_ncu": 456.0,
                        "error": "",
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._find_executable",
            side_effect=fake_find_executable,
        ), patch(
            "acprof.host.compute_profile._profile_torch_entries",
            side_effect=fake_torch_profile,
        ), patch(
            "acprof.host.compute_profile._profile_cpu_entries",
            side_effect=AssertionError("vendor CPU profiler should not run in auto mode"),
        ), patch(
            "acprof.host.compute_profile._profile_gpu_entries",
            side_effect=fake_gpu_profile,
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_tool="auto",
            )

            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)

        self.assertEqual(
            calls,
            [
                ("find", ("ncu", "nv-nsight-cu-cli")),
                ("torch", "cpu", False),
                ("torch", "gpu", True),
                ("gpu", "/usr/bin/ncu"),
            ],
        )
        self.assertEqual(
            plan["profiles"]["cpu"]["torch_profiler_eager"]["tool"],
            "torch_profiler_eager",
        )
        self.assertEqual(plan["profiles"]["gpu"]["ncu"]["tool"], "ncu")
        self.assertEqual(
            plan["profiles"]["cpu"]["torch_profiler_eager"]["entries"][0][
                "model_logical_mflop_per_request_torch_profiler_eager"
            ],
            123.0,
        )
        self.assertEqual(
            plan["profiles"]["gpu"]["ncu"]["entries"][0][
                "gpu_executed_mflop_per_request_ncu"
            ],
            456.0,
        )

    def test_both_mode_keeps_torch_and_ncu_failures_independent(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_torch_profile(**kwargs):
            calls.append(("torch", kwargs["profile_key"]))
            raise RuntimeError("torch probe failed")

        def fake_gpu_profile(**kwargs):
            calls.append(("ncu", kwargs["repeat"]))
            return {
                "tool": "ncu",
                "repeat": kwargs["repeat"],
                "metrics": ["gpu__time_duration.sum"],
                "error": "",
                "entries": [{
                    "input_scale": 8.0,
                    "tool": "ncu",
                    "gpu_executed_mflop_per_request_ncu": 42.0,
                    "error": "",
                }],
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._find_executable",
            return_value="/usr/bin/ncu",
        ), patch(
            "acprof.host.compute_profile._profile_torch_entries",
            side_effect=fake_torch_profile,
        ), patch(
            "acprof.host.compute_profile._profile_gpu_entries",
            side_effect=fake_gpu_profile,
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=3,
                torch_profiler_repeat=2,
                keep_profiles=False,
                compute_profile_tool="both",
            )
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)

        self.assertEqual(calls, [("torch", "cpu"), ("torch", "gpu"), ("ncu", 3)])
        self.assertNotIn("schema_version", plan)
        self.assertIn(
            "torch_profiler_eager_failed",
            plan["profiles"]["gpu"]["torch_profiler_eager"]["error"],
        )
        self.assertEqual(
            plan["profiles"]["gpu"]["ncu"]["entries"][0][
                "gpu_executed_mflop_per_request_ncu"
            ],
            42.0,
        )
        metadata = plan["static_metadata"]
        self.assertNotIn("compute_profile_schema_version", metadata)
        self.assertEqual(
            metadata["compute_profile_tools"],
            ["torch_profiler_eager", "ncu"],
        )
        self.assertEqual(metadata["torch_profiler_eager_repeat_cpu"], 2)
        self.assertEqual(metadata["torch_profiler_eager_repeat_gpu"], 2)
        self.assertEqual(metadata["ncu_repeat"], 3)
        self.assertFalse(metadata["compute_profiles_retained"])

    def test_vendor_compute_profile_mode_keeps_missing_tool_errors(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._find_executable",
            return_value=None,
        ), patch(
            "acprof.host.compute_profile._run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            plan_path = compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_tool="vendor",
            )

            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)

        self.assertIn(
            "advisor_not_found",
            plan["profiles"]["cpu"]["intel_advisor"]["error"],
        )
        self.assertIn("ncu_not_found", plan["profiles"]["gpu"]["ncu"]["error"])

    def test_compute_profile_resource_overrides_are_used(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_cpu_profile(**kwargs):
            calls.append(("cpu", kwargs["cpu"], kwargs["mem"]))
            return {"tool": "intel_advisor", "repeat": 1, "error": "", "entries": []}

        def fake_gpu_profile(**kwargs):
            calls.append(("gpu", kwargs["cpu"], kwargs["mem"]))
            return {"tool": "ncu", "repeat": 1, "error": "", "entries": []}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._find_executable",
            return_value="/usr/bin/tool",
        ), patch(
            "acprof.host.compute_profile._profile_cpu_entries",
            side_effect=fake_cpu_profile,
        ), patch(
            "acprof.host.compute_profile._profile_gpu_entries",
            side_effect=fake_gpu_profile,
        ):
            compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_cpus=8,
                compute_profile_mem=16,
                compute_profile_tool="vendor",
            )

        self.assertEqual(calls, [("cpu", 8, 16), ("gpu", 8, 16)])

    def test_compute_profile_default_resources_use_host_capacity(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        calls = []

        def fake_cpu_profile(**kwargs):
            calls.append(("cpu", kwargs["cpu"], kwargs["mem"]))
            return {"tool": "intel_advisor", "repeat": 1, "error": "", "entries": []}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.compute_profile._find_executable",
            return_value="/usr/bin/tool",
        ), patch(
            "acprof.host.compute_profile._host_logical_cpus",
            return_value=12,
        ), patch(
            "acprof.host.compute_profile._host_memory_gb_fraction",
            return_value=48,
        ), patch(
            "acprof.host.compute_profile._profile_cpu_entries",
            side_effect=fake_cpu_profile,
        ):
            compute_profile.collect_compute_profile_plan(
                task_info=task_info,
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                advisor_root=None,
                ncu_root=None,
                advisor_repeat=20,
                ncu_repeat=1,
                keep_profiles=False,
                compute_profile_tool="vendor",
            )

        self.assertEqual(calls, [("cpu", 12, 48)])

    def test_find_executable_searches_default_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            compute_profile,
            "DEFAULT_TOOL_SEARCH_ROOTS",
            (tmp,),
        ):
            bin_dir = os.path.join(tmp, "latest", "bin64")
            os.makedirs(bin_dir)
            advisor_path = os.path.join(bin_dir, "advisor")
            with open(advisor_path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\n")

            self.assertEqual(
                compute_profile._find_executable(None, ("advisor", "advixe-cl")),
                advisor_path,
            )

    def test_tool_mount_root_uses_profiler_install_root(self) -> None:
        advisor_bin = "/opt/intel/oneapi/advisor/2025.5/bin64/advisor"
        ncu_bin = "/opt/nvidia/nsight-compute/2025.1.0/ncu"

        self.assertEqual(
            compute_profile._tool_mount_root(advisor_bin, None),
            "/opt/intel/oneapi/advisor/2025.5",
        )
        self.assertEqual(
            compute_profile._tool_mount_root(ncu_bin, None),
            "/opt/nvidia/nsight-compute/2025.1.0",
        )

    def test_debian_ncu_mount_roots_include_target_symlink_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib_root = os.path.join(tmp, "usr", "lib", "nsight-compute")
            arch_root = os.path.join(tmp, "usr", "lib", "x86_64-linux-gnu", "nsight-compute")
            target_root = os.path.join(lib_root, "target")
            real_target = os.path.join(arch_root, "target", "linux-desktop-glibc_2_11_3-x64")
            os.makedirs(target_root)
            os.makedirs(real_target)
            os.symlink(real_target, os.path.join(target_root, "linux-desktop-glibc_2_11_3-x64"))
            ncu_path = os.path.join(lib_root, "ncu")
            with open(ncu_path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\n")

            self.assertEqual(
                compute_profile._tool_mount_roots(ncu_path, None),
                [lib_root, arch_root],
            )


if __name__ == "__main__":
    unittest.main()

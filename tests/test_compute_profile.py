import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


class ComputeProfileTests(unittest.TestCase):
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
                "model_mflop_per_request": 1.0,
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
            self.assertFalse(
                os.path.exists(
                    os.path.join(tmp, "compute_profile_payloads.json")
                )
            )

        self.assertIn("advisor_not_found", plan["profiles"]["cpu"]["error"])
        self.assertIn("ncu_not_found", plan["profiles"]["gpu"]["error"])
        self.assertEqual(
            plan["profiles"]["cpu"]["entries"][0]["model_mflop_per_request"],
            None,
        )
        self.assertEqual(
            plan["profiles"]["gpu"]["entries"][0]["model_mflop_per_request"],
            None,
        )

    def test_default_compute_profile_uses_cpu_torch_and_gpu_ncu(self) -> None:
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
                "tool": "torch_profiler",
                "repeat": kwargs["repeat"],
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        "tool": "torch_profiler",
                        "model_mflop_per_request": 123.0,
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
                        "model_mflop_per_request": 300.0,
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
                ("gpu", "/opt/nvidia/nsight-compute/2024.1.1/ncu"),
            ],
        )
        self.assertEqual(plan["compute_profile_tool_mode"], "auto")
        self.assertEqual(plan["profiles"]["cpu"]["tool"], "torch_profiler")
        self.assertEqual(plan["profiles"]["gpu"]["tool"], "ncu")

    def test_auto_compute_profile_uses_cpu_torch_and_gpu_ncu(self) -> None:
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
                "tool": "torch_profiler",
                "repeat": kwargs["repeat"],
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        "tool": "torch_profiler",
                        "model_mflop_per_request": 123.0,
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
                        "model_mflop_per_request": 456.0,
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
                ("gpu", "/usr/bin/ncu"),
            ],
        )
        self.assertEqual(plan["profiles"]["cpu"]["tool"], "torch_profiler")
        self.assertEqual(plan["profiles"]["gpu"]["tool"], "ncu")
        self.assertEqual(
            plan["profiles"]["cpu"]["entries"][0]["model_mflop_per_request"],
            123.0,
        )
        self.assertEqual(
            plan["profiles"]["gpu"]["entries"][0]["model_mflop_per_request"],
            456.0,
        )

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

        self.assertIn("advisor_not_found", plan["profiles"]["cpu"]["error"])
        self.assertIn("ncu_not_found", plan["profiles"]["gpu"]["error"])

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

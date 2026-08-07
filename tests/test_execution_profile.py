import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host.detect import TaskInfo
from acprof.host import execution_profile


def _task_info() -> TaskInfo:
    return TaskInfo(
        model_id="google-bert/bert-base-uncased",
        pipeline_tag="fill-mask",
        task_family="nlp",
        runtime_backend="transformers_pipeline",
        library_name="transformers",
        model_revision="main",
        detection_method="test",
    )


def _write_input_scale_plan(directory: str) -> str:
    path = os.path.join(directory, "input_scale_plan.json")
    with open(path, "w", encoding="utf-8") as plan_file:
        json.dump(
            {
                "entries": [
                    {
                        "input_scale": 8.0,
                        "scale_label": "seq8",
                        "payload": {"text": "hello [MASK]", "params": {}},
                    },
                    {
                        "input_scale": 16.0,
                        "scale_label": "seq16",
                        "payload": {"text": "hello [MASK]", "params": {}},
                    },
                ]
            },
            plan_file,
        )
    return path


class ExecutionProfileTests(unittest.TestCase):
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

                entries = execution_profile._load_input_scale_plan_entries(path)

                self.assertEqual(entries[0]["payload"], payload)

    def test_parse_massif_native_snapshots_uses_independent_and_total_peaks(
        self,
    ) -> None:
        report = """\
desc: --time-unit=ms --stacks=yes
cmd: python -m acprof.container.compute_profile_runner
time_unit: ms
#-----------
snapshot=0
#-----------
time=1
mem_heap_B=100
mem_heap_extra_B=90
mem_stacks_B=10
heap_tree=empty
#-----------
snapshot=1
#-----------
time=2
mem_heap_B=150
mem_heap_extra_B=5
mem_stacks_B=5
heap_tree=empty
#-----------
snapshot=2
#-----------
time=3
mem_heap_B=110
mem_heap_extra_B=40
mem_stacks_B=100
heap_tree=peak
"""
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "massif.out")
            with open(report_path, "w", encoding="utf-8") as report_file:
                report_file.write(report)
            parsed = execution_profile.parse_massif_output(report_path)

        self.assertEqual(parsed["cpu_heap_peak_bytes_massif"], 150)
        self.assertEqual(parsed["cpu_heap_extra_peak_bytes_massif"], 90)
        self.assertEqual(parsed["cpu_stack_peak_bytes_massif"], 100)
        self.assertEqual(parsed["cpu_heap_peak_total_bytes_massif"], 250)
        self.assertEqual(parsed["cpu_heap_peak_at_ms_massif"], 3)

    def test_parse_nsys_reports_normalizes_units_repeat_and_memcpy_only(
        self,
    ) -> None:
        reports = {
            "cuda_api_sum": "\n".join(
                [
                    '"Time (%)","Total Time (us)","Num Calls","Name"',
                    '"66.7","1000","3","cudaLaunchKernel"',
                    '"33.3","500","1","cudaMemcpyAsync"',
                ]
            ),
            "cuda_gpu_kern_sum": "\n".join(
                [
                    '"Time (%)","Total Time (ns)","Instances","Name"',
                    '"100","4000000","4","kernel"',
                ]
            ),
            "cuda_gpu_mem_time_sum": "\n".join(
                [
                    '"Time (%)","Total Time (ms)","Operations","Operation"',
                    '"20","6","4","[CUDA memcpy Host-to-Device]"',
                    '"80","20","10","[CUDA memset]"',
                ]
            ),
            "cuda_gpu_mem_size_sum": "\n".join(
                [
                    '"Total (KiB)","Operations","Operation"',
                    '"2","4","[CUDA memcpy Host-to-Device]"',
                    '"100","10","[CUDA memset]"',
                ]
            ),
        }

        parsed = execution_profile.parse_nsys_stats_reports(
            reports,
            repeat=2,
        )

        self.assertAlmostEqual(
            parsed["cuda_api_time_sum_ms_per_request_nsys"],
            0.75,
        )
        self.assertAlmostEqual(
            parsed["cuda_api_call_count_per_request_nsys"],
            2.0,
        )
        self.assertAlmostEqual(
            parsed["gpu_kernel_time_sum_ms_per_request_nsys"],
            2.0,
        )
        self.assertAlmostEqual(
            parsed["gpu_kernel_launch_count_per_request_nsys"],
            2.0,
        )
        self.assertAlmostEqual(
            parsed["gpu_memcpy_time_sum_ms_per_request_nsys"],
            3.0,
        )
        self.assertAlmostEqual(
            parsed["gpu_memcpy_count_per_request_nsys"],
            2.0,
        )
        self.assertAlmostEqual(
            parsed["gpu_memcpy_bytes_per_request_nsys"],
            1024.0,
        )

    def test_run_nsys_stats_requests_csv_and_discards_sqlite_cache(
        self,
    ) -> None:
        commands = []

        def fake_run(command, check=False):
            commands.append(command)
            with open(sqlite_path, "wb") as sqlite_file:
                sqlite_file.write(b"cache")
            return SimpleNamespace(returncode=0, stdout="report", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "report.nsys-rep")
            sqlite_path = os.path.join(tmp, "report.sqlite")
            with patch(
                "acprof.host.execution_profile._run",
                side_effect=fake_run,
            ):
                outputs = execution_profile._run_nsys_stats(
                    "/opt/nsight/bin/nsys",
                    report_path,
                )
            self.assertFalse(os.path.exists(sqlite_path))

        self.assertEqual(set(outputs), set(execution_profile.NSYS_REPORTS))
        self.assertEqual(len(commands), len(execution_profile.NSYS_REPORTS))
        for command in commands:
            self.assertEqual(command[command.index("--format") + 1], "csv")
            self.assertEqual(command[command.index("--timeunit") + 1], "nsec")
            self.assertEqual(command[command.index("--output") + 1], "-")
        self.assertIn("--force-export=true", commands[0])
        self.assertEqual(commands[0][-1], report_path)
        for command in commands[1:]:
            self.assertNotIn("--force-export=true", command)
            self.assertEqual(command[-1], sqlite_path)

    def test_run_nsys_stats_discards_sqlite_cache_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "report.nsys-rep")
            sqlite_path = os.path.join(tmp, "report.sqlite")

            def fake_run(command, check=False):
                with open(sqlite_path, "wb") as sqlite_file:
                    sqlite_file.write(b"cache")
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="stats failed",
                )

            with patch(
                "acprof.host.execution_profile._run",
                side_effect=fake_run,
            ), self.assertRaisesRegex(RuntimeError, "nsys_stats_failed"):
                execution_profile._run_nsys_stats(
                    "/opt/nsight/bin/nsys",
                    report_path,
                )

            self.assertFalse(os.path.exists(sqlite_path))

    def test_nsys_no_memops_keeps_api_and_kernel_metrics(self) -> None:
        reports = {
            "cuda_api_sum": (
                '"Total Time (ns)","Num Calls","Name"\n'
                '"1000000","2","cudaLaunchKernel"\n'
            ),
            "cuda_gpu_kern_sum": (
                '"Total Time (ns)","Instances","Name"\n'
                '"500000","2","kernel"\n'
            ),
            "cuda_gpu_mem_time_sum": (
                "SKIPPED: report.sqlite does not contain GPU memory data."
            ),
            "cuda_gpu_mem_size_sum": (
                "SKIPPED: report.sqlite does not contain GPU memory data."
            ),
        }

        parsed = execution_profile.parse_nsys_stats_reports(
            reports,
            repeat=2,
        )

        self.assertEqual(
            parsed["cuda_api_time_sum_ms_per_request_nsys"],
            0.5,
        )
        self.assertEqual(
            parsed["gpu_kernel_time_sum_ms_per_request_nsys"],
            0.25,
        )
        self.assertEqual(
            parsed["gpu_memcpy_time_sum_ms_per_request_nsys"],
            0.0,
        )
        self.assertEqual(
            parsed["gpu_memcpy_count_per_request_nsys"],
            0.0,
        )
        self.assertEqual(
            parsed["gpu_memcpy_bytes_per_request_nsys"],
            0.0,
        )

    def test_nsys_discovery_recurses_and_mounts_whole_version_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version_root = os.path.join(
                tmp,
                "nsight-systems",
                "2026.1.1",
            )
            target_dir = os.path.join(version_root, "target-linux-x64")
            os.makedirs(target_dir)
            nsys_bin = os.path.join(target_dir, "nsys")
            with open(nsys_bin, "w", encoding="utf-8") as executable:
                executable.write("#!/bin/sh\n")
            os.chmod(nsys_bin, 0o755)

            discovered = execution_profile._find_nsys_executable(
                os.path.join(tmp, "nsight-systems")
            )

        self.assertEqual(discovered, os.path.realpath(nsys_bin))
        self.assertEqual(
            execution_profile._nsys_mount_root(discovered),
            os.path.realpath(version_root),
        )

    def test_nsys_container_runtime_preflight_runs_importer(self) -> None:
        commands = []
        with tempfile.TemporaryDirectory() as tmp:
            importer_dir = os.path.join(tmp, "host-linux-x64")
            os.makedirs(importer_dir)
            importer = os.path.join(importer_dir, "QdstrmImporter")
            with open(importer, "w", encoding="utf-8") as executable:
                executable.write("#!/bin/sh\n")
            os.chmod(importer, 0o755)

            def fake_run(command, check=False):
                commands.append(command)
                return SimpleNamespace(
                    returncode=0,
                    stdout="NVIDIA Nsight Systems test importer\n",
                    stderr="",
                )

            with patch(
                "acprof.host.execution_profile._run",
                side_effect=fake_run,
            ):
                version = (
                    execution_profile._validate_nsys_container_runtime(
                        "acprof-nsys-test:latest",
                        tmp,
                    )
                )

        self.assertEqual(version, "NVIDIA Nsight Systems test importer")
        self.assertEqual(commands[0][0:3], ["docker", "run", "--rm"])
        self.assertIn(f"{tmp}:{tmp}:ro", commands[0])
        self.assertEqual(commands[0][-2:], [importer, "--version"])

    def test_execution_probe_does_not_force_compute_profiler_threads(
        self,
    ) -> None:
        command = execution_profile._without_compute_thread_env([
            "docker",
            "run",
            "-e",
            "MODEL_ID=test",
            "-e",
            "OMP_NUM_THREADS=2",
            "-e",
            "MKL_NUM_THREADS=2",
            "-e",
            "OPENBLAS_NUM_THREADS=2",
            "-e",
            "NUMEXPR_NUM_THREADS=2",
            "-e",
            "TORCH_NUM_THREADS=2",
            "acprof-test:latest",
        ])

        self.assertIn("MODEL_ID=test", command)
        for name in execution_profile.COMPUTE_THREAD_ENV_NAMES:
            self.assertFalse(
                any(value.startswith(f"{name}=") for value in command)
            )
        self.assertEqual(command[-1], "acprof-test:latest")

    def test_missing_tools_fill_every_resource_and_scale_without_aborting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.execution_profile._build_massif_image",
            side_effect=RuntimeError("massif_image_build_failed:no_apt"),
        ), patch(
            "acprof.host.execution_profile._find_nsys_executable",
            return_value=None,
        ), patch(
            "acprof.host.execution_profile._profile_massif_tool",
            wraps=execution_profile._profile_massif_tool,
        ) as profile_massif, patch(
            "acprof.host.execution_profile._profile_nsys_tool",
            wraps=execution_profile._profile_nsys_tool,
        ) as profile_nsys:
            plan_path = execution_profile.collect_execution_profile_plan(
                task_info=_task_info(),
                image_tag="acprof-test:latest",
                cpu_list=[1, 2],
                mem_list=[4, 8],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                project_dir=os.path.dirname(os.path.dirname(__file__)),
                tool_mode="both",
                keep_profiles=False,
            )
            with open(plan_path, "r", encoding="utf-8") as plan_file:
                plan = json.load(plan_file)

        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(len(plan["profiles"]), 8)
        self.assertEqual(
            plan["static_metadata"]["execution_profile_tools"],
            ["massif", "nsys"],
        )
        self.assertEqual(profile_massif.call_count, 1)
        self.assertEqual(profile_nsys.call_count, 2)
        self.assertEqual(
            plan["static_metadata"]["massif_sampling_strategy"],
            "representative_per_scale",
        )
        self.assertEqual(
            plan["static_metadata"]["nsys_sampling_strategy"],
            "representative_per_cpu_scale",
        )
        for profile in plan["profiles"]:
            self.assertEqual(len(profile["tools"]), 1)
            tool = "massif" if profile["gpu_mode"] == "off" else "nsys"
            tool_profile = profile["tools"][tool]
            self.assertEqual(len(tool_profile["entries"]), 2)
            for entry in tool_profile["entries"]:
                expected_source_cpu = (
                    2 if tool == "massif" else profile["cpu_cores"]
                )
                self.assertEqual(
                    entry["profile_source_cpu_cores"],
                    expected_source_cpu,
                )
                self.assertEqual(entry["profile_source_mem_cap_gb"], 8)
                self.assertTrue(entry["error"])
                if tool == "massif":
                    self.assertIsNone(
                        entry["cpu_heap_peak_total_bytes_massif"]
                    )
                    self.assertTrue(entry["compute_profile_error_massif"])
                else:
                    self.assertIsNone(
                        entry[
                            "host_inference_wall_time_ms_per_request_nsys"
                        ]
                    )
                    self.assertTrue(entry["compute_profile_error_nsys"])

    def test_full_sampling_profiles_every_resource_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.execution_profile._build_massif_image",
            side_effect=RuntimeError("massif_image_build_failed:no_apt"),
        ), patch(
            "acprof.host.execution_profile._find_nsys_executable",
            return_value=None,
        ), patch(
            "acprof.host.execution_profile._profile_massif_tool",
            wraps=execution_profile._profile_massif_tool,
        ) as profile_massif, patch(
            "acprof.host.execution_profile._profile_nsys_tool",
            wraps=execution_profile._profile_nsys_tool,
        ) as profile_nsys:
            plan_path = execution_profile.collect_execution_profile_plan(
                task_info=_task_info(),
                image_tag="acprof-test:latest",
                cpu_list=[1, 2],
                mem_list=[4, 8],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                project_dir=os.path.dirname(os.path.dirname(__file__)),
                tool_mode="both",
                massif_sampling="full",
                nsys_sampling="full",
                keep_profiles=False,
            )
            with open(plan_path, "r", encoding="utf-8") as plan_file:
                plan = json.load(plan_file)

        self.assertEqual(profile_massif.call_count, 4)
        self.assertEqual(profile_nsys.call_count, 4)
        self.assertEqual(
            plan["static_metadata"]["massif_sampling_strategy"],
            "full_resource_matrix",
        )
        self.assertEqual(
            plan["static_metadata"]["nsys_sampling_strategy"],
            "full_resource_matrix",
        )
        for profile in plan["profiles"]:
            tool = "massif" if profile["gpu_mode"] == "off" else "nsys"
            entry = profile["tools"][tool]["entries"][0]
            self.assertEqual(
                entry["profile_source_cpu_cores"], profile["cpu_cores"]
            )
            self.assertEqual(
                entry["profile_source_mem_cap_gb"], profile["mem_cap_gb"]
            )

    def test_explicit_sampling_references_select_requested_resources(self) -> None:
        massif = execution_profile._sampled_resource_cases(
            tool="massif",
            cpus=[1, 2, 8],
            memories=[2, 4, 16],
            sampling="per-scale",
            reference_cpu=2,
            reference_mem=4,
        )
        nsys_per_cpu = execution_profile._sampled_resource_cases(
            tool="nsys",
            cpus=[1, 2, 8],
            memories=[2, 4, 16],
            sampling="per-cpu-scale",
            reference_cpu=None,
            reference_mem=4,
        )
        nsys_one = execution_profile._sampled_resource_cases(
            tool="nsys",
            cpus=[1, 2, 8],
            memories=[2, 4, 16],
            sampling="per-scale",
            reference_cpu=2,
            reference_mem=4,
        )

        self.assertEqual(massif, ([(2, 4)], 2, 4))
        self.assertEqual(
            nsys_per_cpu,
            ([(1, 4), (2, 4), (8, 4)], None, 4),
        )
        self.assertEqual(nsys_one, ([(2, 4)], 2, 4))
        with self.assertRaisesRegex(ValueError, "not present"):
            execution_profile._sampled_resource_cases(
                tool="nsys",
                cpus=[1, 2, 8],
                memories=[2, 4, 16],
                sampling="per-scale",
                reference_cpu=3,
                reference_mem=4,
            )

    def test_none_mode_writes_disabled_plan_without_profiler_side_effects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.execution_profile._build_massif_image",
        ) as build_massif, patch(
            "acprof.host.execution_profile._build_nsys_image",
        ) as build_nsys, patch(
            "acprof.host.execution_profile._find_nsys_executable",
        ) as find_nsys:
            plan_path = execution_profile.collect_execution_profile_plan(
                task_info=_task_info(),
                image_tag="acprof-test:latest",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                input_scale_plan_file=_write_input_scale_plan(tmp),
                project_dir=os.path.dirname(os.path.dirname(__file__)),
                tool_mode="none",
                keep_profiles=True,
            )
            with open(plan_path, "r", encoding="utf-8") as plan_file:
                plan = json.load(plan_file)

            self.assertFalse(
                os.path.exists(
                    os.path.join(
                        tmp,
                        execution_profile.EXECUTION_PROFILE_DIRNAME,
                    )
                )
            )

        build_massif.assert_not_called()
        build_nsys.assert_not_called()
        find_nsys.assert_not_called()
        self.assertEqual(plan["profiles"], [])
        self.assertEqual(
            plan["static_metadata"]["execution_profile_tools"],
            [],
        )
        self.assertFalse(
            plan["static_metadata"]["execution_profiles_retained"]
        )
        self.assertEqual(
            plan["static_metadata"]["execution_profile_provenance"],
            "disabled",
        )

    def test_nsys_collection_enables_unregistered_nvtx_capture(self) -> None:
        reports = {
            "cuda_api_sum": (
                '"Total Time (ns)","Num Calls","Name"\n'
                '"1000000","1","cudaLaunchKernel"\n'
            ),
            "cuda_gpu_kern_sum": (
                '"Total Time (ns)","Instances","Name"\n'
                '"500000","1","kernel"\n'
            ),
            "cuda_gpu_mem_time_sum": (
                '"Total Time (ns)","Count","Operation"\n'
                '"250000","1","[CUDA memcpy Host-to-Device]"\n'
            ),
            "cuda_gpu_mem_size_sum": (
                '"Total (B)","Count","Operation"\n'
                '"4096","1","[CUDA memcpy Host-to-Device]"\n'
            ),
        }
        commands = []

        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "execution_profiles")
            os.makedirs(profile_root)

            def fake_run(command, check=False):
                commands.append(command)
                report_path = os.path.join(
                    profile_root,
                    "nsys_cpu_2_mem_4_scale_8.nsys-rep",
                )
                with open(report_path, "wb") as report:
                    report.write(b"nsys")
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "profile_window_wall_time_ms": 20.0,
                            "profile_window_wall_time_ms_per_request": 10.0,
                        }
                    ),
                    stderr="",
                )

            with patch(
                "acprof.host.execution_profile._base_docker_cmd",
                return_value=["docker", "run", "acprof-test:latest"],
            ), patch(
                "acprof.host.execution_profile._run",
                side_effect=fake_run,
            ), patch(
                "acprof.host.execution_profile._run_nsys_stats",
                return_value=reports,
            ):
                result = execution_profile._collect_nsys_entry(
                    task_info=_task_info(),
                    image_tag="acprof-test:latest",
                    nsys_bin="/opt/nsight/2026/target-linux-x64/nsys",
                    nsys_mount_root="/opt/nsight/2026",
                    cpu=2,
                    mem=4,
                    payload_file=os.path.join(tmp, "input_scale_plan.json"),
                    profile_root=profile_root,
                    output_dir=tmp,
                    entry={"input_scale": 8.0},
                    repeat=2,
                )

        command = commands[0]
        self.assertIn("NSYS_NVTX_PROFILER_REGISTER_ONLY=0", command)
        self.assertIn("--trace=cuda,nvtx", command)
        self.assertNotIn("--trace=cuda,nvtx,osrt", command)
        self.assertIn("--capture-range=nvtx", command)
        self.assertIn("--nvtx-capture=acprof_compute", command)
        self.assertIn("--capture-range-end=stop", command)
        self.assertIn("--sample=none", command)
        self.assertIn("--cpuctxsw=none", command)
        self.assertEqual(
            result["host_inference_wall_time_ms_per_request_nsys"],
            10.0,
        )
        self.assertEqual(
            result["report"],
            os.path.join(
                "execution_profiles",
                "nsys_cpu_2_mem_4_scale_8.nsys-rep",
            ),
        )

    def test_nsys_missing_report_discards_large_raw_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = os.path.join(tmp, "execution_profiles")
            os.makedirs(profile_root)
            raw_stream = os.path.join(
                profile_root,
                "nsys_cpu_2_mem_4_scale_8.qdstrm",
            )

            def fake_run(command, check=False):
                with open(raw_stream, "wb") as stream:
                    stream.write(b"x" * 4096)
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "profile_window_wall_time_ms": 20.0,
                            "profile_window_wall_time_ms_per_request": 20.0,
                        }
                    ),
                    stderr="",
                )

            with patch(
                "acprof.host.execution_profile._base_docker_cmd",
                return_value=["docker", "run", "acprof-test:latest"],
            ), patch(
                "acprof.host.execution_profile._run",
                side_effect=fake_run,
            ):
                result = execution_profile._collect_nsys_entry(
                    task_info=_task_info(),
                    image_tag="acprof-test:latest",
                    nsys_bin="/opt/nsight/2026/target-linux-x64/nsys",
                    nsys_mount_root="/opt/nsight/2026",
                    cpu=2,
                    mem=4,
                    payload_file=os.path.join(tmp, "input_scale_plan.json"),
                    profile_root=profile_root,
                    output_dir=tmp,
                    entry={"input_scale": 8.0},
                    repeat=1,
                )
            raw_stream_exists_after = os.path.exists(raw_stream)

        self.assertFalse(raw_stream_exists_after)
        self.assertEqual(
            result["compute_profile_error_nsys"],
            "nsys_import_failed:report_not_found:"
            "discarded_qdstrm_bytes=4096",
        )


if __name__ == "__main__":
    unittest.main()

import acprof.host.collection_history as host_collection_history
import acprof.host.compute_profile_plan as host_compute_profile_plan
import acprof.host.execution_profile_plan as host_execution_profile_plan
import acprof.host.posthoc.backfill as host_posthoc_backfill
import acprof.host.posthoc.context as host_posthoc_context
import acprof.host.posthoc.plans as host_posthoc_plans
import acprof.host.posthoc.storage as host_posthoc_storage
import csv
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.cli import posthoc
from acprof.host.compute_profile_plan import TORCH_LOGICAL_MFLOP_FIELD


class PosthocProfileTests(unittest.TestCase):
    def test_load_context_uses_image_id_and_preserves_runtime_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            path = root / host_posthoc_context.STATIC_META_NAME
            metadata = json.loads(path.read_text())
            metadata["image_id"] = "sha256:" + "a" * 64
            metadata["runtime_environment"] = {"profile_id": "legacy-nlp", "adapter": "family-default"}
            path.write_text(json.dumps(metadata))
            context = host_posthoc_context.load_result_context(root)
            self.assertEqual(context.image_tag, metadata["image_id"])
            self.assertEqual(context.task_info.runtime_profile_id, "legacy-nlp")

    def test_parser_defaults_to_reduced_execution_sampling(self):
        defaults = posthoc._build_parser().parse_args(["results/example"])
        explicit = posthoc._build_parser().parse_args(
            [
                "results/example",
                "--massif-sampling",
                "full",
                "--nsys-sampling",
                "per-scale",
                "--nsys-reference-cpu",
                "4",
                "--nsys-reference-mem",
                "8",
            ]
        )

        self.assertEqual(defaults.massif_sampling, "per-scale")
        self.assertEqual(defaults.nsys_sampling, "per-cpu-scale")
        self.assertEqual(explicit.massif_sampling, "full")
        self.assertEqual(explicit.nsys_sampling, "per-scale")
        self.assertEqual(explicit.nsys_reference_cpu, 4)
        self.assertEqual(explicit.nsys_reference_mem, 8)

    def _write_fixture(
        self,
        root: Path,
        *,
        include_cpu=True,
        include_gpu=True,
        torch_complete=True,
    ):
        root.mkdir(parents=True, exist_ok=True)
        input_plan = {
            "schema_version": 2,
            "model_id": "example/model",
            "task_family": "nlp",
            "pipeline_tag": "fill-mask",
            "entries": [
                {
                    "input_scale": 8.0,
                    "scale_label": "8",
                    "payload": {"text": "hello"},
                }
            ],
        }
        plan_path = root / host_posthoc_context.INPUT_SCALE_PLAN_NAME
        plan_path.write_text(json.dumps(input_plan), encoding="utf-8")
        plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        static_meta = {
            "schema_version": 7,
            "model_name": "example/model",
            "model_revision": "revision-1",
            "task_family": "nlp",
            "pipeline_tag": "fill-mask",
            "runtime_backend": "transformers_pipeline",
            "image_tag": "acprof-nlp-example--model:latest",
            "image_id": "sha256:" + "b" * 64,
            "batch_size": 1,
            "input_scale_type": "seq_length",
            "input_scale_plan_sha256": plan_hash,
            "run_command": "python run.py --model example/model",
            "compute_profile_tools": (
                ["torch_profiler_eager"] if torch_complete else []
            ),
            "execution_profile_tools": [],
        }
        (root / host_posthoc_context.STATIC_META_NAME).write_text(
            json.dumps(static_meta), encoding="utf-8"
        )

        fieldnames = [
            "marker",
            "cpu_cores",
            "mem_cap_gb",
            "gpu_mode",
            "input_scale",
            "latency_s",
            "latency_app_s",
            *host_posthoc_context.TORCH_FIELDS,
            *host_posthoc_context.NCU_FIELDS,
            *host_posthoc_context.MASSIF_FIELDS,
            *host_posthoc_context.NSYS_FIELDS,
            "status",
        ]
        # De-duplicate fields shared through tuple expansion.
        fieldnames = list(dict.fromkeys(fieldnames))
        rows = []
        if include_cpu:
            rows.append(
                {
                    "marker": "cpu-original",
                    "cpu_cores": "2",
                    "mem_cap_gb": "4",
                    "gpu_mode": "off",
                    "input_scale": "8",
                    "latency_s": "2.0",
                    "latency_app_s": "2.5",
                    TORCH_LOGICAL_MFLOP_FIELD: (
                        "123.000000" if torch_complete else "nan"
                    ),
                    host_compute_profile_plan.TORCH_ERROR_FIELD: (
                        "" if torch_complete else "compute_profile_disabled"
                    ),
                    host_execution_profile_plan.MASSIF_HEAP_PEAK_TOTAL_FIELD: "nan",
                    "status": "ok",
                }
            )
        if include_gpu:
            rows.append(
                {
                    "marker": "gpu-original",
                    "cpu_cores": "2",
                    "mem_cap_gb": "4",
                    "gpu_mode": "on",
                    "input_scale": "8",
                    "latency_s": "0.5",
                    "latency_app_s": "0.4",
                    TORCH_LOGICAL_MFLOP_FIELD: (
                        "456.000000" if torch_complete else "nan"
                    ),
                    host_compute_profile_plan.TORCH_ERROR_FIELD: (
                        "" if torch_complete else "compute_profile_disabled"
                    ),
                    host_compute_profile_plan.NCU_TOTAL_MFLOP_FIELD: "nan",
                    host_execution_profile_plan.NSYS_HOST_WALL_TIME_FIELD: "nan",
                    "status": "ok",
                }
            )
        csv_path = root / host_posthoc_context.RESULT_CSV_NAME
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def _compute_plan(self):
        return {
            "model_id": "example/model",
            "task_family": "nlp",
            "pipeline_tag": "fill-mask",
            "runtime_backend": "transformers_pipeline",
            "static_metadata": {
                "compute_profile_tools": ["ncu"],
                "ncu_repeat": 1,
                "ncu_version": "test-ncu",
            },
            "profiles": {
                "gpu": {
                    "ncu": {
                        "tool": "ncu",
                        "error": "",
                        "entries": [
                            {
                                "input_scale": 8.0,
                                host_compute_profile_plan.NCU_TOTAL_MFLOP_FIELD: 100.0,
                                host_compute_profile_plan.NCU_TENSOR_MFLOP_FIELD: 80.0,
                                host_compute_profile_plan.NCU_SCALAR_MFLOP_FIELD: 20.0,
                                host_compute_profile_plan.NCU_TENSOR_SHARE_FIELD: 80.0,
                                host_compute_profile_plan.NCU_KERNEL_COUNT_FIELD: 5.0,
                                host_compute_profile_plan.NCU_KERNEL_TIME_FIELD: 3.0,
                                "error": "",
                            }
                        ],
                    }
                }
            },
        }

    def _torch_plan(self):
        def profile(value):
            return {
                "tool": "torch_profiler_eager",
                "flop_semantics": "logical_operator_shape_flops",
                "error": "",
                "entries": [
                    {
                        "input_scale": 8.0,
                        host_compute_profile_plan.TORCH_LOGICAL_MFLOP_FIELD: value,
                        "error": "",
                    }
                ],
            }

        return {
            "model_id": "example/model",
            "task_family": "nlp",
            "pipeline_tag": "fill-mask",
            "runtime_backend": "transformers_pipeline",
            "static_metadata": {
                "compute_profile_tools": ["torch_profiler_eager"],
                "torch_profiler_eager_flop_semantics": (
                    "logical_operator_shape_flops"
                ),
                "torch_profiler_eager_repeat_cpu": 1,
                "torch_profiler_eager_repeat_gpu": 1,
                "torch_version": "test-torch",
                "transformers_version": "test-transformers",
            },
            "profiles": {
                "cpu": {"torch_profiler_eager": profile(111.0)},
                "gpu": {"torch_profiler_eager": profile(222.0)},
            },
        }

    def _execution_plan(self):
        return {
            "schema_version": 1,
            "model_id": "example/model",
            "model_revision": "revision-1",
            "task_family": "nlp",
            "pipeline_tag": "fill-mask",
            "runtime_backend": "transformers_pipeline",
            "static_metadata": {
                "execution_profile_schema_version": 1,
                "execution_profile_tools": ["massif", "nsys"],
                "massif_repeat": 1,
                "massif_version": "test-massif",
                "nsys_repeat": 1,
                "nsys_version": "test-nsys",
            },
            "profiles": [
                {
                    "cpu_cores": 2,
                    "mem_cap_gb": 4,
                    "gpu_mode": "off",
                    "tools": {
                        "massif": {
                            "tool": "massif",
                            "error": "",
                            "entries": [
                                {
                                    "input_scale": 8.0,
                                    host_execution_profile_plan.MASSIF_HEAP_PEAK_FIELD: 1000,
                                    host_execution_profile_plan.MASSIF_HEAP_EXTRA_PEAK_FIELD: 200,
                                    host_execution_profile_plan.MASSIF_STACK_PEAK_FIELD: 50,
                                    host_execution_profile_plan.MASSIF_HEAP_PEAK_TOTAL_FIELD: 1250,
                                    host_execution_profile_plan.MASSIF_PEAK_AT_MS_FIELD: 12.5,
                                    "error": "",
                                }
                            ],
                        }
                    },
                },
                {
                    "cpu_cores": 2,
                    "mem_cap_gb": 4,
                    "gpu_mode": "on",
                    "tools": {
                        "nsys": {
                            "tool": "nsys",
                            "error": "",
                            "entries": [
                                {
                                    "input_scale": 8.0,
                                    host_execution_profile_plan.NSYS_HOST_WALL_TIME_FIELD: 20.0,
                                    host_execution_profile_plan.NSYS_CUDA_API_TIME_FIELD: 3.0,
                                    host_execution_profile_plan.NSYS_CUDA_API_CALL_COUNT_FIELD: 4,
                                    host_execution_profile_plan.NSYS_GPU_KERNEL_TIME_FIELD: 10.0,
                                    host_execution_profile_plan.NSYS_GPU_KERNEL_LAUNCH_COUNT_FIELD: 5,
                                    host_execution_profile_plan.NSYS_GPU_MEMCPY_TIME_FIELD: 1.0,
                                    host_execution_profile_plan.NSYS_GPU_MEMCPY_COUNT_FIELD: 2,
                                    host_execution_profile_plan.NSYS_GPU_MEMCPY_BYTES_FIELD: 4096,
                                    "error": "",
                                }
                            ],
                        }
                    },
                },
            ],
        }

    def _read_rows(self, path: Path):
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_load_context_and_applicable_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root)
            context = host_posthoc_context.load_result_context(root)

        self.assertEqual(context.task_info.model_id, "example/model")
        self.assertEqual(context.resource_cases, [(2, 4, "off"), (2, 4, "on")])
        applicable, skipped = host_posthoc_plans.applicable_tools(
            context, ("torch", "ncu", "nsys", "massif")
        )
        self.assertEqual(applicable, ("torch", "ncu", "nsys", "massif"))
        self.assertEqual(skipped, ())

    def test_load_context_rejects_embedded_histories_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root)
            static_path = root / host_posthoc_context.STATIC_META_NAME
            static_meta = json.loads(static_path.read_text())
            static_meta["timeout_retry_history"] = [{"completed_at": "2026-08-21T22:29:33+08:00"}]
            static_path.write_text(json.dumps(static_meta))
            before = static_path.read_bytes()
            with self.assertRaisesRegex(host_posthoc_context.PosthocError, "collection_history"):
                host_posthoc_context.load_result_context(root)
            self.assertEqual(static_path.read_bytes(), before)

    def test_backfill_updates_only_profiler_fields_for_applicable_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root)
            context = host_posthoc_context.load_result_context(root)
            _fields, rows, updated = host_posthoc_backfill.backfill_rows(
                context,
                tools=("ncu", "nsys", "massif"),
                compute_plan=self._compute_plan(),
                execution_plan=self._execution_plan(),
            )

        cpu = next(row for row in rows if row["gpu_mode"] == "off")
        gpu = next(row for row in rows if row["gpu_mode"] == "on")
        self.assertEqual(cpu["marker"], "cpu-original")
        self.assertEqual(cpu[TORCH_LOGICAL_MFLOP_FIELD], "123.000000")
        self.assertEqual(cpu[host_execution_profile_plan.MASSIF_HEAP_PEAK_TOTAL_FIELD], "1250.000000")
        self.assertEqual(gpu["marker"], "gpu-original")
        self.assertEqual(gpu[TORCH_LOGICAL_MFLOP_FIELD], "456.000000")
        self.assertEqual(gpu[host_compute_profile_plan.NCU_TOTAL_MFLOP_FIELD], "100.000000")
        self.assertEqual(gpu[host_posthoc_context.NCU_DERIVED_APP_FIELD], "250.000000")
        self.assertEqual(gpu[host_posthoc_context.NCU_DERIVED_PACKET_FIELD], "200.000000")
        self.assertEqual(gpu[host_execution_profile_plan.NSYS_HOST_WALL_TIME_FIELD], "20.000000")
        self.assertEqual(updated, {"ncu": 1, "nsys": 1, "massif": 1})

    def test_successful_existing_profile_is_preserved_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root, include_cpu=False)
            context = host_posthoc_context.load_result_context(root)
            for field in host_posthoc_context.TOOL_METRIC_FIELDS["ncu"]:
                context.rows[0][field] = "777.000000"
            context.rows[0][host_compute_profile_plan.NCU_ERROR_FIELD] = ""
            _fields, rows, updated = host_posthoc_backfill.backfill_rows(
                context,
                tools=("ncu",),
                compute_plan=self._compute_plan(),
            )

        self.assertEqual(rows[0][host_compute_profile_plan.NCU_TOTAL_MFLOP_FIELD], "777.000000")
        self.assertEqual(updated, {"ncu": 0})

    def test_torch_plan_backfills_cpu_gpu_rows_and_static_flops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root, torch_complete=False)
            (root / "compute_profile_plan.json").write_text(
                json.dumps(self._torch_plan()), encoding="utf-8"
            )
            with patch(
                "acprof.host.posthoc.service.find_active_processes", return_value=[]
            ), patch(
                "acprof.host.posthoc.service._validate_profiler_runtime",
                side_effect=AssertionError("complete Torch plan should be reused"),
            ):
                summary = posthoc.run_posthoc(root, tools="torch")

            rows = self._read_rows(root / "result_all.csv")
            cpu = next(row for row in rows if row["gpu_mode"] == "off")
            gpu = next(row for row in rows if row["gpu_mode"] == "on")
            self.assertEqual(cpu[host_compute_profile_plan.TORCH_LOGICAL_MFLOP_FIELD], "111.000000")
            self.assertEqual(gpu[host_compute_profile_plan.TORCH_LOGICAL_MFLOP_FIELD], "222.000000")
            self.assertEqual(summary.reused_tools, ("torch",))
            self.assertEqual(summary.updated_rows_by_tool, {"torch": 2})

            metadata = json.loads((root / "static_meta.json").read_text())
            self.assertIn("torch_profiler_eager", metadata["compute_profile_tools"])
            self.assertEqual(metadata["torch_version"], "test-torch")
            self.assertEqual(metadata["static_flops"]["profile"], "gpu")
            self.assertEqual(
                metadata["static_flops"]["values"],
                [{"input_scale": 8, "flops_per_request": 222_000_000}],
            )

    def test_torch_collector_uses_available_cpu_and_gpu_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root, torch_complete=False)
            context = host_posthoc_context.load_result_context(root)

            def fake_collect(**kwargs):
                plan_path = Path(kwargs["output_dir"]) / "compute_profile_plan.json"
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                plan_path.write_text(json.dumps(self._torch_plan()), encoding="utf-8")
                return str(plan_path)

            with patch(
                "acprof.host.compute_profile.collect_compute_profile_plan",
                side_effect=fake_collect,
            ) as collect:
                plan = host_posthoc_plans._collect_compute_plan(
                    context,
                    root / "posthoc_profiles" / "torch",
                    tool="torch",
                    ncu_root=None,
                    torch_repeat=3,
                    ncu_repeat=1,
                    compute_profile_cpus=8,
                    compute_profile_mem=16,
                )

        kwargs = collect.call_args.kwargs
        self.assertEqual(kwargs["compute_profile_tool"], "torch")
        self.assertEqual(kwargs["gpu_list"], ["off", "on"])
        self.assertEqual(kwargs["torch_profiler_repeat"], 3)
        self.assertEqual(kwargs["compute_profile_cpus"], 8)
        self.assertEqual(kwargs["compute_profile_mem"], 16)
        self.assertTrue(kwargs["resume_existing_ncu_profiles"])
        self.assertTrue(host_posthoc_plans.compute_plan_covers_tool(plan, context, "torch"))

    def test_torch_and_ncu_plans_merge_without_overwriting_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root, torch_complete=False)
            context = host_posthoc_context.load_result_context(root)
            merged = host_posthoc_plans.merge_compute_plans(
                context,
                {"torch": self._torch_plan(), "ncu": self._compute_plan()},
            )

        self.assertEqual(
            merged["static_metadata"]["compute_profile_tools"],
            ["torch_profiler_eager", "ncu"],
        )
        self.assertIn("torch_profiler_eager", merged["profiles"]["cpu"])
        self.assertIn("torch_profiler_eager", merged["profiles"]["gpu"])
        self.assertIn("ncu", merged["profiles"]["gpu"])
        self.assertTrue(host_posthoc_plans.compute_plan_covers_tool(merged, context, "torch"))
        self.assertTrue(host_posthoc_plans.compute_plan_covers_tool(merged, context, "ncu"))

    def test_representative_massif_plan_expands_to_all_cpu_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root, include_gpu=False)
            context = host_posthoc_context.load_result_context(root)
            context.resource_cases.append((8, 16, "off"))
            plan = self._execution_plan()
            expanded = host_posthoc_plans.expand_representative_massif_plan(
                plan,
                context,
                reference_cpu=2,
                reference_mem=4,
            )

        resources = {
            (profile["cpu_cores"], profile["mem_cap_gb"])
            for profile in expanded["profiles"]
        }
        self.assertEqual(resources, {(2, 4), (8, 16)})
        self.assertEqual(
            expanded["static_metadata"]["massif_sampling_strategy"],
            "representative_per_scale",
        )
        entry = expanded["profiles"][1]["tools"]["massif"]["entries"][0]
        self.assertEqual(entry["profile_source_cpu_cores"], 2)

    def test_one_command_reuses_plans_updates_original_names_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            csv_path = self._write_fixture(root)
            initial_history = {
                "schema_version": 1,
                "posthoc_profile_history": [],
                "timeout_retry_history": [{"completed_at": "2026-08-01T00:00:00Z"}],
                "quality_retry_history": [],
            }
            history_path = root / host_collection_history.COLLECTION_HISTORY_NAME
            history_path.write_text(json.dumps(initial_history), encoding="utf-8")
            original_csv = csv_path.read_bytes()
            original_meta = (root / host_posthoc_context.STATIC_META_NAME).read_bytes()
            original_history = history_path.read_bytes()
            (root / "compute_profile_plan.json").write_text(
                json.dumps(self._compute_plan()), encoding="utf-8"
            )
            (root / "execution_profile_plan.json").write_text(
                json.dumps(self._execution_plan()), encoding="utf-8"
            )

            with patch(
                "acprof.host.posthoc.service.find_active_processes", return_value=[]
            ), patch(
                "acprof.host.posthoc.service._validate_profiler_runtime",
                side_effect=AssertionError("runtime should not be used"),
            ):
                summary = posthoc.run_posthoc(root)

            self.assertEqual(Path(summary.result_csv), root / "result_all.csv")
            self.assertEqual(Path(summary.static_meta), root / "static_meta.json")
            self.assertEqual(set(summary.reused_tools), {"ncu", "nsys", "massif"})
            backup = Path(summary.backup_dir)
            self.assertEqual((backup / "result_all.csv").read_bytes(), original_csv)
            self.assertEqual((backup / "static_meta.json").read_bytes(), original_meta)
            self.assertEqual(
                (backup / host_collection_history.COLLECTION_HISTORY_NAME).read_bytes(),
                original_history,
            )

            rows = self._read_rows(root / "result_all.csv")
            cpu = next(row for row in rows if row["gpu_mode"] == "off")
            gpu = next(row for row in rows if row["gpu_mode"] == "on")
            self.assertEqual(cpu["marker"], "cpu-original")
            self.assertEqual(gpu["marker"], "gpu-original")
            self.assertEqual(gpu[host_compute_profile_plan.NCU_TOTAL_MFLOP_FIELD], "100.000000")
            self.assertEqual(cpu[host_execution_profile_plan.MASSIF_HEAP_PEAK_TOTAL_FIELD], "1250.000000")

            metadata = json.loads((root / "static_meta.json").read_text())
            self.assertIn("ncu", metadata["compute_profile_tools"])
            self.assertEqual(
                set(metadata["execution_profile_tools"]), {"massif", "nsys"}
            )
            self.assertEqual(metadata["nsys_version"], "test-nsys")
            self.assertEqual(metadata["massif_version"], "test-massif")
            self.assertNotIn("posthoc_profile_history", metadata)
            self.assertNotIn("posthoc_profile_last_run", metadata)
            collection_history = json.loads(history_path.read_text())
            self.assertEqual(len(collection_history["posthoc_profile_history"]), 1)
            self.assertEqual(len(collection_history["timeout_retry_history"]), 1)
            self.assertFalse((root / ".posthoc.lock").exists())

    def test_three_file_commit_restores_csv_and_meta_if_history_publish_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root)
            history_path = root / host_collection_history.COLLECTION_HISTORY_NAME
            history_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "posthoc_profile_history": [],
                        "timeout_retry_history": [],
                        "quality_retry_history": [],
                    }
                ),
                encoding="utf-8",
            )
            context = host_posthoc_context.load_result_context(root)
            backup = host_posthoc_storage.create_backup(context)
            original_csv = context.result_csv.read_bytes()
            original_meta = context.static_meta_path.read_bytes()
            original_history = history_path.read_bytes()
            rows = [dict(row) for row in context.rows]
            rows[0]["marker"] = "changed"
            static_meta = {**context.static_meta, "run_command": "changed"}
            collection_history = host_collection_history.append_collection_record(
                context.collection_history,
                "posthoc_profile_history",
                {"completed_at": "2026-08-23T00:00:00Z"},
            )
            real_replace = os.replace

            def fail_history_publish(source, destination):
                if (
                    Path(destination) == history_path
                    and ".restore." not in Path(source).name
                ):
                    raise OSError("simulated history publish failure")
                return real_replace(source, destination)

            with patch(
                "acprof.host.posthoc.storage.os.replace",
                side_effect=fail_history_publish,
            ), self.assertRaisesRegex(OSError, "simulated history publish failure"):
                host_posthoc_storage.commit_result_files(
                    context,
                    fieldnames=context.fieldnames,
                    rows=rows,
                    static_meta=static_meta,
                    collection_history=collection_history,
                    backup_dir=backup,
                )

            self.assertEqual(context.result_csv.read_bytes(), original_csv)
            self.assertEqual(context.static_meta_path.read_bytes(), original_meta)
            self.assertEqual(history_path.read_bytes(), original_history)
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_gpu_only_collection_runs_ncu_and_nsys_but_skips_massif(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root, include_cpu=False)
            with patch(
                "acprof.host.posthoc.service.find_active_processes", return_value=[]
            ), patch(
                "acprof.host.posthoc.service._validate_profiler_runtime"
            ) as validate_runtime, patch(
                "acprof.host.posthoc.service._collect_compute_plan",
                return_value=self._compute_plan(),
            ) as collect_compute, patch(
                "acprof.host.posthoc.service._collect_execution_plan",
                return_value=self._execution_plan(),
            ) as collect_execution:
                summary = posthoc.run_posthoc(root)

            self.assertEqual(set(summary.collected_tools), {"ncu", "nsys"})
            self.assertIn("massif", summary.skipped_tools)
            validate_runtime.assert_called_once()
            collect_compute.assert_called_once()
            self.assertEqual(collect_compute.call_args.kwargs["tool"], "ncu")
            collect_execution.assert_called_once()
            self.assertEqual(collect_execution.call_args.kwargs["tool"], "nsys")

    def test_dry_run_does_not_create_backup_or_change_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            csv_path = self._write_fixture(root)
            original = csv_path.read_bytes()
            with patch(
                "acprof.host.posthoc.service.find_active_processes", return_value=[]
            ):
                summary = posthoc.run_posthoc(root, dry_run=True)

            self.assertIsNone(summary.backup_dir)
            self.assertEqual(csv_path.read_bytes(), original)
            self.assertFalse((root / host_posthoc_context.BACKUP_DIRNAME).exists())
            self.assertFalse((root / host_posthoc_context.POSTHOC_DIRNAME).exists())

    def test_active_run_is_rejected_before_files_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "example--model"
            self._write_fixture(root)
            with patch(
                "acprof.host.posthoc.service.find_active_processes",
                return_value=[(123, "python run.py --model example/model")],
            ):
                with self.assertRaisesRegex(posthoc.PosthocError, "still using"):
                    posthoc.run_posthoc(root)


if __name__ == "__main__":
    unittest.main()

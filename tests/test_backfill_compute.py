import csv
import json
import os
import tempfile
import unittest

from acprof.cli.backfill_compute import (
    COMPUTE_PROFILE_FIELDS,
    backfill_compute_profile_csv,
)
from acprof.host.compute_profile_plan import find_compute_profile_entry


REMOVED_LEGACY_COMPUTE_FIELDS = (
    "compute_profile_tool",
    "model_mflop_per_request",
    "compute_mflops_app",
    "compute_mflops",
    "compute_profile_error",
)
OBSOLETE_RESULT_FIELDS = (
    *REMOVED_LEGACY_COMPUTE_FIELDS,
    "gpu_profile_report_ncu",
)


class BackfillComputeProfileTests(unittest.TestCase):
    def _write_csv(self, path, fieldnames, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_vendor_plan(self, path):
        plan = {
            "profiles": {
                "cpu": {
                    "intel_advisor": {
                        "tool": "intel_advisor",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "model_mflop_per_request": 200.0,
                                "error": "",
                            }
                        ],
                    }
                },
                "gpu": {
                    "ncu": {
                        "tool": "ncu",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "gpu_executed_mflop_per_request_ncu": 100.0,
                                "error": "",
                            }
                        ],
                    }
                },
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f)

    def _write_dual_tool_plan(self, path):
        plan = {
            "profiles": {
                "cpu": {
                    "torch_profiler_eager": {
                        "tool": "torch_profiler_eager",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "model_logical_mflop_per_request_torch_profiler_eager": 200.0,
                                "error": "",
                            }
                        ],
                    }
                },
                "gpu": {
                    "torch_profiler_eager": {
                        "tool": "torch_profiler_eager",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "model_logical_mflop_per_request_torch_profiler_eager": 300.0,
                                "error": "",
                            }
                        ],
                    },
                    "ncu": {
                        "tool": "ncu",
                        "entries": [
                            {
                                "input_scale": 64.0,
                                "gpu_executed_mflop_per_request_ncu": 100.0,
                                "gpu_executed_tensor_mflop_per_request_ncu": 90.0,
                                "gpu_executed_scalar_mflop_per_request_ncu": 10.0,
                                "gpu_executed_tensor_share_pct_ncu": 90.0,
                                "gpu_kernel_launch_count_per_request_ncu": 5.0,
                                "gpu_kernel_time_sum_ms_per_request_ncu": 1.0,
                                "error": "",
                            }
                        ],
                    },
                },
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f)

    def test_entry_error_does_not_inherit_another_scale_failure(self):
        plan = {
            "profiles": {
                "gpu": {
                    "torch_profiler_eager": {
                        "tool": "torch_profiler_eager",
                        "error": "scale_2_failed",
                        "entries": [
                            {
                                "input_scale": 1,
                                "model_logical_mflop_per_request_torch_profiler_eager": 10,
                                "error": "",
                            },
                            {
                                "input_scale": 2,
                                "model_logical_mflop_per_request_torch_profiler_eager": None,
                                "error": "scale_2_failed",
                            },
                        ],
                    },
                    "ncu": {
                        "tool": "ncu",
                        "error": "",
                        "entries": [
                            {
                                "input_scale": 1,
                                "gpu_executed_mflop_per_request_ncu": 20,
                                "error": "",
                            }
                        ],
                    },
                }
            },
        }

        profile = find_compute_profile_entry(plan, "on", 1)

        self.assertEqual(profile["compute_profile_error_torch_profiler_eager"], "")
        self.assertEqual(profile["compute_profile_error_ncu"], "")
        self.assertEqual(
            profile["model_logical_mflop_per_request_torch_profiler_eager"],
            10.0,
        )

    def test_vendor_cpu_is_not_mislabeled_as_torch_profile(self):
        plan = {
            "compute_profile_tool_mode": "vendor",
            "profiles": {
                "cpu": {
                    "intel_advisor": {
                        "tool": "intel_advisor",
                        "entries": [
                            {
                                "input_scale": 64,
                                "model_mflop_per_request": 321,
                                "error": "",
                            }
                        ],
                    }
                }
            },
        }

        profile = find_compute_profile_entry(plan, "off", 64)

        self.assertNotIn("tool", profile)
        self.assertNotIn("model_mflop_per_request", profile)
        self.assertNotIn("error", profile)
        logical = profile[
            "model_logical_mflop_per_request_torch_profiler_eager"
        ]
        self.assertNotEqual(logical, logical)
        self.assertEqual(
            profile["compute_profile_error_torch_profiler_eager"],
            "",
        )

    def test_backfills_cpu_and_gpu_and_preserves_other_values_and_order(self):
        fields = [
            "row_marker",
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
            *REMOVED_LEGACY_COMPUTE_FIELDS,
            *COMPUTE_PROFILE_FIELDS,
            "gpu_profile_report_ncu",
            "status",
        ]
        input_rows = [
            {
                "row_marker": "first",
                "gpu_mode": "off",
                "input_scale": "64.0000005",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
                "compute_profile_tool": "old",
                "model_mflop_per_request": "1",
                "compute_mflops_app": "2",
                "compute_mflops": "3",
                "compute_profile_error": "old error",
                "gpu_profile_report_ncu": "legacy/ncu.csv",
                "status": "ok",
            },
            {
                "row_marker": "second",
                "gpu_mode": "on",
                "input_scale": "64",
                "latency_app_s": "0.4",
                "latency_s": "nan",
                "compute_profile_tool": "old",
                "model_mflop_per_request": "1",
                "compute_mflops_app": "2",
                "compute_mflops": "3",
                "compute_profile_error": "old error",
                "gpu_profile_report_ncu": "legacy/ncu.csv",
                "status": "ok",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, input_rows)
            self._write_vendor_plan(plan_path)
            with open(input_csv, "rb") as f:
                original_input = f.read()

            summary = backfill_compute_profile_csv(
                input_csv,
                plan_path,
                output_csv,
            )

            with open(output_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                output_fields = reader.fieldnames
                output_rows = list(reader)
            with open(input_csv, "rb") as f:
                unchanged_input = f.read()

        self.assertEqual(summary.row_count, 2)
        self.assertEqual(summary.diagnostic_count, 0)
        self.assertEqual(
            output_fields,
            [field for field in fields if field not in OBSOLETE_RESULT_FIELDS],
        )
        self.assertTrue(
            all(
                field not in row
                for row in output_rows
                for field in OBSOLETE_RESULT_FIELDS
            )
        )
        self.assertEqual(unchanged_input, original_input)
        self.assertEqual(
            [(row["row_marker"], row["status"]) for row in output_rows],
            [("first", "ok"), ("second", "ok")],
        )
        self.assertEqual(
            output_rows[0][
                "model_logical_mflop_per_request_torch_profiler_eager"
            ],
            "nan",
        )
        self.assertEqual(
            output_rows[1]["gpu_executed_mflop_per_request_ncu"],
            "100.000000",
        )
        self.assertEqual(
            output_rows[1]["gpu_executed_mflops_app_ncu"],
            "250.000000",
        )
        self.assertEqual(
            output_rows[1]["gpu_executed_mflops_packet_ncu"],
            "250.000000",
        )

    def test_dual_tool_plan_backfills_only_explicit_metrics(self):
        fields = [
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
            "status",
        ]
        rows = [
            {
                "gpu_mode": "off",
                "input_scale": "64",
                "latency_app_s": "0.5",
                "latency_s": "nan",
                "status": "ok",
            },
            {
                "gpu_mode": "on",
                "input_scale": "64",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
                "status": "ok",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_dual_tool_plan(plan_path)

            summary = backfill_compute_profile_csv(
                input_csv,
                plan_path,
                output_csv,
            )
            with open(output_csv, "r", encoding="utf-8", newline="") as f:
                output_rows = list(csv.DictReader(f))

        self.assertEqual(summary.diagnostic_count, 0)
        cpu_row, gpu_row = output_rows
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            self.assertNotIn(field, cpu_row)
            self.assertNotIn(field, gpu_row)
        self.assertEqual(
            cpu_row["model_logical_mflops_packet_torch_profiler_eager"],
            "400.000000",
        )
        self.assertEqual(
            cpu_row["gpu_executed_mflop_per_request_ncu"],
            "nan",
        )
        self.assertEqual(cpu_row["compute_profile_error_ncu"], "")

        self.assertEqual(
            gpu_row["model_logical_mflops_packet_torch_profiler_eager"],
            "1200.000000",
        )
        self.assertEqual(
            gpu_row["gpu_executed_mflop_per_request_ncu"],
            "100.000000",
        )
        self.assertEqual(gpu_row["gpu_executed_mflops_app_ncu"], "200.000000")
        self.assertEqual(
            gpu_row["gpu_executed_mflops_packet_ncu"],
            "400.000000",
        )

    def test_ncu_summary_overrides_gpu_rows_only_with_numeric_scale_matching(self):
        fields = [
            "row_marker",
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
        ]
        rows = [
            {
                "row_marker": "cpu",
                "gpu_mode": "off",
                "input_scale": "64",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
            },
            {
                "row_marker": "gpu",
                "gpu_mode": "on",
                "input_scale": "64",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
            },
        ]
        ncu_fields = [
            "input_scale",
            "gpu_hw_mflop_per_request_ncu",
            "gpu_hw_tensor_mflop_per_request_ncu",
            "gpu_hw_scalar_mflop_per_request_ncu",
            "gpu_hw_tensor_share_pct_ncu",
            "ncu_kernel_count",
            "ncu_kernel_time_sum_ms",
            "ncu_report",
        ]
        ncu_rows = [
            {
                "input_scale": "64.0000005",
                "gpu_hw_mflop_per_request_ncu": "120",
                "gpu_hw_tensor_mflop_per_request_ncu": "100",
                "gpu_hw_scalar_mflop_per_request_ncu": "20",
                "gpu_hw_tensor_share_pct_ncu": "83.333333",
                "ncu_kernel_count": "7",
                "ncu_kernel_time_sum_ms": "2.5",
                "ncu_report": "results/model/compute_profiles_tensor/ncu_scale_64.csv",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            ncu_summary = os.path.join(tmp, "gpu_hardware_flops_by_scale.csv")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_dual_tool_plan(plan_path)
            self._write_csv(ncu_summary, ncu_fields, ncu_rows)

            summary = backfill_compute_profile_csv(
                input_csv,
                plan_path,
                output_csv,
                ncu_summary=ncu_summary,
            )
            with open(output_csv, "r", encoding="utf-8", newline="") as f:
                output_rows = list(csv.DictReader(f))

        self.assertEqual(summary.row_count, 2)
        self.assertEqual(summary.diagnostic_count, 0)
        cpu_row, gpu_row = output_rows
        self.assertEqual(cpu_row["row_marker"], "cpu")
        self.assertEqual(cpu_row["gpu_executed_mflop_per_request_ncu"], "nan")
        self.assertEqual(cpu_row["compute_profile_error_ncu"], "")

        self.assertEqual(gpu_row["row_marker"], "gpu")
        self.assertEqual(
            gpu_row["gpu_executed_mflop_per_request_ncu"],
            "120.000000",
        )
        self.assertEqual(
            gpu_row["gpu_executed_tensor_mflop_per_request_ncu"],
            "100.000000",
        )
        self.assertEqual(
            gpu_row["gpu_executed_scalar_mflop_per_request_ncu"],
            "20.000000",
        )
        self.assertEqual(
            gpu_row["gpu_executed_tensor_share_pct_ncu"],
            "83.333333",
        )
        self.assertEqual(gpu_row["gpu_executed_mflops_app_ncu"], "240.000000")
        self.assertEqual(
            gpu_row["gpu_executed_mflops_packet_ncu"],
            "480.000000",
        )
        self.assertEqual(
            gpu_row["gpu_kernel_launch_count_per_request_ncu"],
            "7.000000",
        )
        self.assertEqual(
            gpu_row["gpu_kernel_time_sum_ms_per_request_ncu"],
            "2.500000",
        )
        self.assertNotIn("gpu_profile_report_ncu", gpu_row)

    def test_missing_scale_writes_nan_metrics_and_diagnostic(self):
        fields = [
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
            "note",
        ]
        rows = [
            {
                "gpu_mode": "off",
                "input_scale": "65",
                "latency_app_s": "0.5",
                "latency_s": "0.25",
                "note": "keep me",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_dual_tool_plan(plan_path)

            summary = backfill_compute_profile_csv(
                input_csv,
                plan_path,
                output_csv,
            )
            with open(output_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                output_fields = reader.fieldnames or []
                output_row = next(reader)

        self.assertEqual(summary.diagnostic_count, 1)
        self.assertEqual(output_fields, fields + list(COMPUTE_PROFILE_FIELDS))
        self.assertEqual(output_row["note"], "keep me")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            self.assertNotIn(field, output_row)
        self.assertEqual(
            output_row[
                "model_logical_mflop_per_request_torch_profiler_eager"
            ],
            "nan",
        )
        self.assertEqual(
            output_row["compute_profile_error_torch_profiler_eager"],
            "compute_profile_missing_scale:65",
        )

    def test_existing_output_is_not_overwritten_without_explicit_permission(self):
        fields = ["gpu_mode", "input_scale", "latency_app_s"]
        rows = [{"gpu_mode": "off", "input_scale": "64", "latency_app_s": "0.5"}]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "existing.csv")
            self._write_csv(input_csv, fields, rows)
            self._write_vendor_plan(plan_path)
            with open(output_csv, "w", encoding="utf-8") as f:
                f.write("do not replace")

            with self.assertRaises(FileExistsError):
                backfill_compute_profile_csv(
                    input_csv,
                    plan_path,
                    output_csv,
                )
            with open(output_csv, "r", encoding="utf-8") as f:
                output_contents = f.read()

        self.assertEqual(output_contents, "do not replace")

    def test_explicit_overwrite_can_atomically_replace_input_path(self):
        fields = ["gpu_mode", "input_scale", "latency_app_s"]
        rows = [{"gpu_mode": "off", "input_scale": "64", "latency_app_s": "0.5"}]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            self._write_csv(input_csv, fields, rows)
            self._write_dual_tool_plan(plan_path)

            backfill_compute_profile_csv(
                input_csv,
                plan_path,
                input_csv,
                overwrite=True,
            )
            with open(input_csv, "r", encoding="utf-8", newline="") as f:
                output_row = next(csv.DictReader(f))

        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            self.assertNotIn(field, output_row)
        self.assertEqual(
            output_row[
                "model_logical_mflop_per_request_torch_profiler_eager"
            ],
            "200.000000",
        )
        self.assertEqual(
            output_row[
                "model_logical_mflops_app_torch_profiler_eager"
            ],
            "400.000000",
        )
        self.assertEqual(
            output_row[
                "model_logical_mflops_packet_torch_profiler_eager"
            ],
            "400.000000",
        )


if __name__ == "__main__":
    unittest.main()

import csv
import json
import os
import tempfile
import unittest

from acprof.cli.backfill_compute import (
    COMPUTE_PROFILE_FIELDS,
    backfill_compute_profile_csv,
)


class BackfillComputeProfileTests(unittest.TestCase):
    def _write_csv(self, path, fieldnames, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_plan(self, path):
        plan = {
            "profiles": {
                "cpu": {
                    "tool": "intel_advisor",
                    "entries": [
                        {
                            "input_scale": 64.0,
                            "model_mflop_per_request": 200.0,
                            "error": "",
                        }
                    ],
                },
                "gpu": {
                    "tool": "ncu",
                    "entries": [
                        {
                            "input_scale": 64.0,
                            "model_mflop_per_request": 100.0,
                            "error": "",
                        }
                    ],
                },
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f)

    def test_backfills_cpu_and_gpu_and_preserves_other_values_and_order(self):
        fields = [
            "row_marker",
            "gpu_mode",
            "input_scale",
            "latency_app_s",
            "latency_s",
            *COMPUTE_PROFILE_FIELDS,
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
                "status": "ok",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_csv = os.path.join(tmp, "result_all.csv")
            plan_path = os.path.join(tmp, "compute_profile_plan.json")
            output_csv = os.path.join(tmp, "result_all.with_compute.csv")
            self._write_csv(input_csv, fields, input_rows)
            self._write_plan(plan_path)
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
        self.assertEqual(output_fields, fields)
        self.assertEqual(unchanged_input, original_input)
        self.assertEqual(
            [(row["row_marker"], row["status"]) for row in output_rows],
            [("first", "ok"), ("second", "ok")],
        )
        self.assertEqual(output_rows[0]["compute_profile_tool"], "intel_advisor")
        self.assertEqual(output_rows[0]["model_mflop_per_request"], "200.000000")
        self.assertEqual(output_rows[0]["compute_mflops_app"], "400.000000")
        self.assertEqual(output_rows[0]["compute_mflops"], "800.000000")
        self.assertEqual(output_rows[0]["compute_profile_error"], "")
        self.assertEqual(output_rows[1]["compute_profile_tool"], "ncu")
        self.assertEqual(output_rows[1]["model_mflop_per_request"], "100.000000")
        self.assertEqual(output_rows[1]["compute_mflops_app"], "250.000000")
        self.assertEqual(output_rows[1]["compute_mflops"], "250.000000")
        self.assertEqual(output_rows[1]["compute_profile_error"], "")

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
            self._write_plan(plan_path)

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
        self.assertEqual(output_row["compute_profile_tool"], "intel_advisor")
        self.assertEqual(output_row["model_mflop_per_request"], "nan")
        self.assertEqual(output_row["compute_mflops_app"], "nan")
        self.assertEqual(output_row["compute_mflops"], "nan")
        self.assertEqual(
            output_row["compute_profile_error"],
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
            self._write_plan(plan_path)
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
            self._write_plan(plan_path)

            backfill_compute_profile_csv(
                input_csv,
                plan_path,
                input_csv,
                overwrite=True,
            )
            with open(input_csv, "r", encoding="utf-8", newline="") as f:
                output_row = next(csv.DictReader(f))

        self.assertEqual(output_row["model_mflop_per_request"], "200.000000")
        self.assertEqual(output_row["compute_mflops_app"], "400.000000")
        self.assertEqual(output_row["compute_mflops"], "400.000000")


if __name__ == "__main__":
    unittest.main()

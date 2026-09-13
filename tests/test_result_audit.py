"""审计真实 CSV/JSON 的故障与历史兼容边界。"""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from acprof.analysis.audit import audit_result
from acprof.config import CSV_FIELDS


class ResultAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "result_all.csv"

    def row(self, **changes):
        return {**dict.fromkeys(CSV_FIELDS, "nan"), "cpu_cores": "1", "mem_cap_gb": "4",
                "gpu_mode": "off", "input_scale": "64", "repeat_idx": "0", "warmup": "0",
                "status": "ok", "error": "", "task_param": "{}", **changes}

    def write(self, *rows, fields=CSV_FIELDS):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_zero_is_not_missing_and_gpu_off_is_inapplicable(self):
        self.write(self.row(container_io_read_bytes_per_request="0"))
        before = self.path.read_bytes()
        report = audit_result(self.root)
        self.assertTrue(report["valid"])
        self.assertNotIn("container_io_read_bytes_per_request", report["missing_metrics"])
        self.assertEqual(report["missing_metrics"]["gpu_energy_eff_j"], {"not_applicable": 1})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_formal_filter_excludes_warmup_warn_and_error(self):
        self.write(self.row(), self.row(warmup="1"), self.row(repeat_idx="1", status="warn", error="idle drift"),
                   self.row(repeat_idx="2", status="error", error="timeout"))
        report = audit_result(self.root)
        self.assertEqual(report["counts"], {"rows": 4, "formal_ok": 1, "warmup": 1, "warn": 1, "error": 1})
        self.assertEqual(report["completion"], "unknown")

    def test_duplicate_and_invalid_status_are_reported(self):
        for rows, code in (([self.row(), self.row()], "invalid_csv"),
                           ([self.row(status="finished")], "invalid_status")):
            self.write(*rows)
            report = audit_result(self.root)
            self.assertFalse(report["valid"])
            self.assertIn(code, [issue["code"] for issue in report["issues"]])

    def test_invalid_number_is_not_explained_as_hardware_unavailable(self):
        self.write(self.row(latency_app_s="inf"))
        report = audit_result(self.root)
        self.assertFalse(report["valid"])
        self.assertIn("invalid_number", [issue["code"] for issue in report["issues"]])

    def test_plan_hash_and_row_coverage_are_checked(self):
        self.write(self.row())
        (self.root / "static_meta.json").write_text(json.dumps({"input_scale_plan_sha256": "0" * 64}))
        (self.root / "input_scale_plan.json").write_text('{}')
        (self.root / "run_state.json").write_text(json.dumps({
            "schema_version": 1, "status": "complete",
            "options": {"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 2},
            "runtime": {"planned": {"scales": [64]}},
        }))
        report = audit_result(self.root)
        self.assertFalse(report["valid"])
        codes = {issue["code"] for issue in report["issues"]}
        self.assertTrue({"input_plan_hash", "plan_coverage"} <= codes)
        self.assertEqual(report["coverage"]["missing"], 1)

    def test_historical_missing_columns_are_unknown_and_preserved(self):
        self.write(self.row(), fields=[field for field in CSV_FIELDS if field != "input_pixels_per_request"])
        report = audit_result(self.root)
        self.assertTrue(report["valid"])
        self.assertEqual(report["missing_metrics"]["input_pixels_per_request"], {"not_recorded": 1})

    def test_negative_effective_energy_is_valid_but_invalid_derived_value_is_not(self):
        self.write(self.row(vcpu_energy_eff_j="-0.1"))
        self.assertTrue(audit_result(self.root)["valid"])
        self.write(self.row(vcpu_energy_eff_j="2", container_attributed_energy_eff_j="20"))
        report = audit_result(self.root)
        self.assertFalse(report["valid"])
        self.assertIn("formula_mismatch", [issue["code"] for issue in report["issues"]])


if __name__ == "__main__":
    unittest.main()

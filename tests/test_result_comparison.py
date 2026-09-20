"""跨后端比较条件与续跑身份独立；审计只读取已有产物。"""
import contextlib
import copy
import csv
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from acprof.cli.audit import main
from acprof.cli.run_args import build_parser
from acprof.config import CSV_FIELDS


class ResultComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.left, self.right = self.root / "torch", self.root / "onnx"
        self.contract = {
            "schema_version": 1, "task": "tabular-regression", "batch_size": 1,
            "scenario": {"type": "serial"},
            "input": {"planned_scale": 2, "actual_scale": 2, "rows": 2, "feature_dim": 2},
            "output": {"type": "regression", "shape": [2, 1], "count": 2},
        }
        for directory, backend in ((self.left, "torch"), (self.right, "onnxruntime")):
            self.write_experiment(directory, backend)

    def write_json(self, directory, filename, payload):
        (directory / filename).write_text(json.dumps(payload))

    def change_json(self, directory, filename, update):
        path = directory / filename
        payload = json.loads(path.read_text())
        update(payload)
        path.write_text(json.dumps(payload))

    def write_experiment(self, directory, backend):
        directory.mkdir()
        options = vars(build_parser().parse_args([
            "--model", "example/source", "--profiling-mode", "basic", "--cpus", "1", "--mems", "4",
            "--gpus", "off", "--warmup", "0", "--repeat", "1", "--backend", backend,
        ]))
        options["measurement_environment"] = {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ACPROF_RUNTIME_THREADS": "1",
        }
        self.write_json(directory, "run_state.json", {
            "schema_version": 1, "run_id": backend, "status": "complete", "options": options,
            "host": {"source_sha256": backend, "packages_sha256": backend},
            "runtime": {"planned": {"scales": [2]}},
        })
        self.write_json(directory, "input_scale_plan.json", {
            "schema_version": 2, "model_id": f"example/{backend}", "pipeline_tag": "tabular-regression",
            "task_family": "structured", "scenario": {"type": "serial"},
            "quality_constraints": {"reference_id": "known-affine-v1", "atol": 1e-6, "rtol": 1e-5},
            "entries": [{"input_scale": 2, "payload": {"features": [[1, 2], [3, 4]], "batch_size": 1}}],
        })
        self.write_json(directory, "static_meta.json", {
            "schema_version": 7, "model_name": f"example/{backend}", "model_revision": backend,
            "runtime_backend": backend, "image_id": f"sha256:{backend}",
            "runtime_environment": {"environment_id": backend},
            "runtime_validation": {"devices": {"off": {"status": "ok", "runtime_parameters": {
                "effective": {"threads": 1},
            }}}},
            "cgroup_version": "2", "cgroup_collection_mode": "v2",
        })
        self.write_rows(directory, self.contract)

    def write_rows(self, directory, contract, *, count=3):
        row = {**dict.fromkeys(CSV_FIELDS, "nan"), "cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off",
               "input_scale": 2, "warmup": 0, "repeat_idx": 0, "status": "ok", "error": "",
               "workload_contract": json.dumps({"schema_version": 1, "request_count": count,
                    "variants": [{"count": count, "contract": contract}]})}
        with (directory / "result_all.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)

    def compare(self):
        module = importlib.import_module("acprof.analysis.comparison")
        return module.compare_results(self.left, self.right)

    def test_expected_backend_identity_changes_do_not_prevent_comparison(self):
        report = self.compare()
        self.assertEqual(report["status"], "compatible")
        self.assertEqual(report["conditions"]["planned_inputs"]["status"], "compatible")
        self.assertIn("runtime_backend", report["expected_differences"])
        self.assertIn("image_id", report["expected_differences"])
        self.assertNotEqual(report["experiments"]["left"]["run_id"], report["experiments"]["right"]["run_id"])

    def test_input_order_change_is_detected_without_requiring_whole_plan_hash(self):
        self.change_json(self.right, "input_scale_plan.json", lambda plan: plan["entries"][0]["payload"]["features"].reverse())
        report = self.compare()
        self.assertEqual(report["status"], "incompatible")
        self.assertEqual(report["conditions"]["planned_inputs"]["status"], "incompatible")

    def test_resource_and_measurement_protocol_changes_are_detected(self):
        self.change_json(self.right, "run_state.json", lambda state: state["options"].update(cpus="2", profiling_mode="full"))
        report = self.compare()
        self.assertEqual(report["conditions"]["resources"]["status"], "incompatible")
        self.assertEqual(report["conditions"]["measurement_protocol"]["status"], "incompatible")

    def test_quality_threshold_change_is_not_hidden_by_protocol_validation(self):
        self.change_json(self.right, "input_scale_plan.json", lambda plan: plan["quality_constraints"].update(atol=0.1))
        report = self.compare()
        self.assertEqual(report["conditions"]["quality_constraints"]["status"], "incompatible")

    def test_missing_historical_fields_remain_unknown(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "input_scale_plan.json", lambda plan: plan.pop("quality_constraints"))
        report = self.compare()
        self.assertEqual(report["status"], "unknown")
        self.assertEqual(report["conditions"]["quality_constraints"]["status"], "unknown")
        self.assertEqual(report["conditions"]["planned_inputs"]["status"], "compatible")

    def test_actual_change_does_not_change_preexecution_identity(self):
        before = (self.right / "run_state.json").read_bytes()
        contract = copy.deepcopy(self.contract)
        contract["input"]["actual_scale"] = 1
        self.write_rows(self.right, contract)
        report = self.compare()
        self.assertEqual(report["conditions"]["planned_inputs"]["status"], "compatible")
        self.assertEqual(report["conditions"]["actual_workload"]["status"], "incompatible")
        self.assertEqual((self.right / "run_state.json").read_bytes(), before)

    def test_auto_window_request_counts_do_not_change_per_request_work(self):
        self.write_rows(self.right, self.contract, count=9)
        self.assertEqual(self.compare()["conditions"]["actual_workload"]["status"], "compatible")

    def test_equivalent_csv_numeric_spelling_uses_existing_measurement_keys(self):
        path = self.right / "result_all.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0].update(cpu_cores="1.0", input_scale="2.00", warmup="0.0")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(self.compare()["conditions"]["actual_workload"]["status"], "compatible")

    def test_known_mismatch_is_reported_even_when_another_legacy_field_is_missing(self):
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata.pop("cgroup_collection_mode"))
        self.change_json(self.right, "run_state.json", lambda state: state["options"].update(batch_size=2))
        report = self.compare()
        self.assertEqual(report["conditions"]["resources"]["status"], "incompatible")

    def test_effective_threads_are_compared_instead_of_backend_variable_names(self):
        self.change_json(self.left, "run_state.json", lambda state: state["options"]["measurement_environment"].pop("ACPROF_RUNTIME_THREADS"))
        self.change_json(self.left, "run_state.json", lambda state: state["options"]["measurement_environment"].update(TORCH_NUM_THREADS="1"))
        self.change_json(self.right, "run_state.json", lambda state: state["options"]["measurement_environment"].update(ACPROF_RUNTIME_THREADS="1"))
        report = self.compare()
        self.assertEqual(report["status"], "compatible")
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata["runtime_validation"]["devices"]["off"]["runtime_parameters"]["effective"].update(threads=2))
        self.assertEqual(self.compare()["conditions"]["runtime_threads"]["status"], "incompatible")

    def test_probe_threads_do_not_establish_default_server_threads(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "run_state.json", lambda state: state["options"]["measurement_environment"].pop("ACPROF_RUNTIME_THREADS"))
        # Independent validation injects a quota-derived thread count. The
        # ordinary server does not, so equal probe counts do not prove equality.
        self.assertEqual(self.compare()["conditions"]["runtime_threads"]["status"], "unknown")

    def test_zero_thread_request_retains_unknown_runtime_default(self):
        self.change_json(self.left, "run_state.json", lambda state: state["options"]["measurement_environment"].update(ACPROF_RUNTIME_THREADS="0"))
        self.assertEqual(self.compare()["conditions"]["runtime_threads"]["status"], "unknown")

    def test_invalid_effective_thread_value_is_not_comparable(self):
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata["runtime_validation"]["devices"]["off"]["runtime_parameters"]["effective"].update(threads=True))
        self.assertEqual(self.compare()["conditions"]["runtime_threads"]["status"], "unknown")

    def test_missing_effective_threads_are_unknown_even_if_request_was_recorded(self):
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata.pop("runtime_validation"))
        self.assertEqual(self.compare()["conditions"]["runtime_threads"]["status"], "unknown")

    def test_malformed_metadata_produces_failed_audit_instead_of_crashing(self):
        self.change_json(self.right, "run_state.json", lambda state: state.update(options=[]))
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata.update(runtime_validation=[1]))
        report = self.compare()
        self.assertFalse(report["valid"])
        self.assertEqual(report["status"], "incompatible")
        self.assertTrue(report["experiments"]["right"]["issues"])

    def test_cli_exposes_comparison_and_unknown_requires_explicit_strict_mode(self):
        self.change_json(self.right, "input_scale_plan.json", lambda plan: plan.pop("quality_constraints"))
        snapshots = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main([str(self.left), "--compare", str(self.right), "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["comparison"]["status"], "unknown")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.left), "--compare", str(self.right), "--require-comparable"]), 1)
        self.assertEqual({path: path.read_bytes() for path in snapshots}, snapshots)


if __name__ == "__main__":
    unittest.main()

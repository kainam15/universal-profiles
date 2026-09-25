import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.cli.run_args import build_parser


class CapabilityTests(unittest.TestCase):
    def test_optional_dram_is_verified_without_becoming_a_full_prerequisite(self):
        caps = self.capabilities()
        report = caps.measurement_report('full', gpu_modes=['off'], dram_energy='auto')
        self.assertNotIn('dram_energy', report.requested)
        rows = [{'status': 'ok', 'gpu_mode': 'off', 'latency_app_s': '0.1',
                 'throughput_samples_per_s': '10', 'container_cpu_util_avg_pct': '1',
                 'container_mem_usage_avg_bytes': '1024', 'latency_s': '.09',
                 'cpu_energy_total_j': '1', 'vcpu_energy_total_j': '.2',
                 'cpu_instructions_per_request': '1', 'dram_energy_status': 'unavailable'}]
        caps.apply_collection_result(report, rows)
        self.assertTrue(report.to_dict()['full_profile_complete'])
        required = caps.measurement_report('full', gpu_modes=['off'], dram_energy='required')
        caps.apply_collection_result(required, rows)
        self.assertEqual(caps.missing_required_measurements(required, rows), ['dram_energy'])
        self.assertFalse(required.to_dict()['full_profile_complete'])
        rows[0].update(dram_window_energy_j='0', dram_energy_per_request_j='0',
                       dram_window_effective_energy_j='0', dram_energy_status='verified')
        caps.apply_collection_result(report, rows)
        self.assertEqual(report.measurement['dram_energy'].status.value, 'verified')

    def test_dram_off_and_basic_do_not_claim_measurement_and_required_basic_is_rejected(self):
        caps = self.capabilities()
        for mode, selection in [('full', 'off'), ('basic', 'auto')]:
            report = caps.measurement_report(mode, dram_energy=selection)
            self.assertEqual(report.measurement['dram_energy'].status.value, 'not_requested')
        with self.assertRaisesRegex(ValueError, 'full'):
            caps.measurement_report('basic', dram_energy='required')

    def test_dram_permission_evidence_is_preserved(self):
        caps = self.capabilities()
        report = caps.measurement_report('full')
        caps.apply_collection_result(report, [{'status': 'ok', 'dram_energy_status': 'permission_denied',
                                               'dram_energy_error': 'energy_uj permission denied'}])
        self.assertEqual(report.measurement['dram_energy'].status.value, 'permission_denied')

    def capabilities(self):
        self.assertIsNotNone(importlib.util.find_spec("acprof.capabilities"),
                             "a shared capability contract is required")
        return importlib.import_module("acprof.capabilities")

    def test_cli_preserves_full_default_and_exposes_basic(self):
        parser = build_parser()
        self.assertEqual(getattr(parser.parse_args(["--model", "test"]), "profiling_mode", None), "full")
        self.assertEqual(parser.parse_args(["--model", "test", "--profiling-mode", "basic"]).profiling_mode, "basic")

    def test_statuses_survive_json_without_boolean_coercion(self):
        caps = self.capabilities()
        states = ("available", "verified", "unsupported", "permission_denied", "not_requested", "unavailable", "error")
        report = caps.CapabilityReport(profiling_mode="basic")
        for status in states:
            item = caps.Capability(status, source="test")
            report.measurement[status] = item
            with self.assertRaises(TypeError):
                bool(item)
        payload = json.loads(json.dumps(report.to_dict()))
        self.assertEqual([payload["measurement"][state]["status"] for state in states], list(states))
        with self.assertRaises(ValueError):
            caps.Capability(False)

    def test_ncu_permissions_are_distinct_from_unsupported_missing_and_error(self):
        caps = self.capabilities()
        for detail, expected in (
            ("ncu_failed: ERR_NVGPUCTRPERM Permission to access GPU Performance Counters", "permission_denied"),
            ("this GPU is not supported", "unsupported"),
            ("ncu_not_found", "unavailable"),
            ("ncu failed: malformed CSV", "error"),
        ):
            self.assertEqual(caps.capability_from_error(detail, source="ncu").status.value, expected)

    def test_basic_policy_never_claims_packet_or_energy_measurements(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        for name in ("latency", "throughput", "container_cpu", "container_memory"):
            self.assertEqual(report.measurement[name].status.value, "available")
        for name in ("packet_latency", "cpu_energy", "cpu_instructions", "gpu_power", "gpu_flops"):
            self.assertEqual(report.measurement[name].status.value, "not_requested")
        self.assertEqual(report.to_dict()["profiling_mode"], "basic")
        self.assertFalse(report.to_dict()["full_profile_complete"])

    def test_preflight_availability_is_not_completed_measurement_evidence(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        payload = report.to_dict()
        self.assertFalse(payload["requested_measurements_complete"])
        self.assertTrue(payload["requested_measurements_available"])
        self.assertFalse(payload["collection_finished"])
        self.assertFalse(payload["collection_succeeded"])

    def test_failed_and_not_attempted_rows_are_finished_but_not_successful(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        rows = [
            {"status": "error", "error": "container_oom_killed during startup"},
            {"status": "error", "error": "client_request_timeout: planned_request_attempted=true"},
            {"status": "error", "error": "not_measured_after_timeout: planned_request_attempted=false"},
        ]
        caps.apply_collection_result(report, rows)
        payload = report.to_dict()
        self.assertTrue(payload["collection_finished"])
        self.assertFalse(payload["collection_succeeded"])
        self.assertFalse(payload["collection_complete"])
        self.assertFalse(payload["requested_measurements_complete"])
        self.assertEqual(payload["row_counts"], {
            "total": 3, "succeeded": 0, "failed": 2, "not_measured": 1, "unfinished": 0,
        })

    def test_running_or_undiagnosed_error_rows_cannot_claim_finished(self):
        caps = self.capabilities()
        for row in ({"status": "running"}, {"status": "error", "error": ""}):
            report = caps.measurement_report("basic", gpu_modes=["off"])
            caps.apply_collection_result(report, [row])
            self.assertFalse(report.to_dict()["collection_finished"])
            self.assertEqual(report.to_dict()["row_counts"]["unfinished"], 1)

    def test_old_report_unknown_completion_is_not_invented(self):
        caps = self.capabilities()
        payload = caps.CapabilityReport.from_dict({
            "schema_version": 1, "profiling_mode": "basic", "collection_complete": False,
        }).to_dict()
        self.assertIsNone(payload["collection_finished"])
        self.assertIsNone(payload["collection_succeeded"])
        self.assertIsNone(payload["row_counts"])

    def test_report_reader_accepts_known_versions_and_legacy_missing_version(self):
        caps = self.capabilities()
        for version in ({}, {"schema_version": 1}, {"schema_version": 2}):
            with self.subTest(version=version):
                restored = caps.CapabilityReport.from_dict({
                    **version, "profiling_mode": "basic", "collection_complete": False,
                }).to_dict()
                self.assertEqual(restored["schema_version"], 2)
                self.assertIs(restored["collection_complete"], False)
                self.assertIsNone(restored["collection_finished"])
                self.assertIsNone(restored["collection_succeeded"])
                self.assertIsNone(restored["row_counts"])

    def test_report_reader_rejects_unknown_and_non_integer_versions(self):
        caps = self.capabilities()
        for version in (0, 3, 999, True, False, 1.0, "2", None):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    caps.CapabilityReport.from_dict({"schema_version": version})

    def test_report_reader_rejects_non_boolean_collection_states(self):
        caps = self.capabilities()
        for field in ("collection_complete", "collection_finished", "collection_succeeded"):
            for value in ("false", "true", 0, 1, [], {}):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        caps.CapabilityReport.from_dict({"schema_version": 2, field: value})
        with self.assertRaisesRegex(ValueError, "collection_complete"):
            caps.CapabilityReport.from_dict({"schema_version": 2, "collection_complete": None})

    def test_report_reader_preserves_nullable_states_and_legacy_success(self):
        caps = self.capabilities()
        legacy = caps.CapabilityReport.from_dict({"collection_complete": True}).to_dict()
        self.assertIs(legacy["collection_finished"], True)
        self.assertIs(legacy["collection_succeeded"], True)
        self.assertIsNone(legacy["row_counts"])
        for value in (True, False, None):
            with self.subTest(value=value):
                restored = caps.CapabilityReport.from_dict({
                    "schema_version": 2, "collection_finished": value,
                    "collection_succeeded": value,
                }).to_dict()
                self.assertIs(restored["collection_finished"], value)
                self.assertIs(restored["collection_succeeded"], value)

    def test_empty_collection_has_no_finished_or_measurement_evidence(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        caps.apply_collection_result(report, [])
        payload = report.to_dict()
        for field in ("collection_finished", "collection_succeeded", "collection_complete",
                      "requested_measurements_complete"):
            self.assertIs(payload[field], False)
        self.assertEqual(payload["row_counts"]["total"], 0)

    def test_collection_status_roundtrip_preserves_terminal_failure(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        caps.apply_collection_result(report, [{"status": "error", "error": "timeout"}])
        payload = json.loads(json.dumps(report.to_dict()))
        restored = caps.CapabilityReport.from_dict(payload).to_dict()
        self.assertTrue(restored["collection_finished"])
        self.assertFalse(restored["collection_succeeded"])
        self.assertEqual(restored["row_counts"]["failed"], 1)

    def test_failed_requested_profiler_cannot_claim_complete_full_profile(self):
        caps = self.capabilities()
        report = caps.measurement_report("full", gpu_modes=["on"], compute_tool="ncu")
        caps.apply_profiler_plan(report, {
            "profiles": {"gpu": {"ncu": {"error": "ERR_NVGPUCTRPERM", "entries": []}}}
        }, source="compute_profile_plan")
        self.assertEqual(report.measurement["gpu_flops"].status.value, "permission_denied")
        self.assertFalse(report.to_dict()["requested_measurements_complete"])
        self.assertFalse(report.to_dict()["full_profile_complete"])

    def test_validation_evidence_only_verifies_successful_device(self):
        caps = self.capabilities()
        report = caps.CapabilityReport(profiling_mode="full")
        caps.apply_runtime_validation(report, {
            "devices": {"off": {"status": "ok", "validation": {
                "protocol": {"status": "verified"}, "task": {"status": "verified"},
            }}, "on": {"status": "resource_limit"}},
        }, environment_id="env-test")
        self.assertEqual(report.execution["cpu"].status.value, "verified")
        self.assertEqual(report.execution["cuda"].status.value, "unavailable")
        self.assertEqual(report.execution["cpu"].evidence["environment_id"], "env-test")

    def test_runtime_ok_without_complete_validation_cannot_claim_verified(self):
        caps = self.capabilities()
        for validation in (None, {}, {"protocol": {"status": "verified"}},
                           {"protocol": {"status": "verified"}, "task": {"status": "available"}}):
            report = caps.CapabilityReport(profiling_mode="basic")
            caps.apply_runtime_validation(report, {
                "devices": {"off": {"status": "ok", "validation": validation}},
            })
            self.assertEqual(report.execution["cpu"].status.value, "available")
            self.assertIn("validation", report.execution["cpu"].detail)

    def test_collection_verifies_actual_csv_fields_and_preserves_measured_zero(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        rows = [{"status": "ok", "gpu_mode": "off", "latency_app_s": "0.1",
                 "throughput_samples_per_s": "10", "container_cpu_util_avg_pct": "0",
                 "container_mem_usage_avg_bytes": "1024"}]
        caps.apply_collection_result(report, rows)
        self.assertEqual(report.measurement["container_memory"].status.value, "verified")
        self.assertEqual(report.measurement["container_cpu"].status.value, "verified")
        self.assertTrue(report.to_dict()["requested_measurements_complete"])
        self.assertEqual(rows[0]["container_cpu_util_avg_pct"], "0")
        self.assertFalse(report.to_dict()["full_profile_complete"])

    def test_empty_error_without_finite_profiler_metric_is_not_verified(self):
        caps = self.capabilities()
        report = caps.measurement_report("full", gpu_modes=["on"], compute_tool="ncu")
        caps.apply_profiler_plan(report, {"profiles": {"gpu": {"ncu": {
            "entries": [{"gpu_executed_mflop_per_request_ncu": None, "error": ""}],
        }}}}, source="test")
        self.assertEqual(report.measurement["gpu_flops"].status.value, "unavailable")

    def test_collection_permission_error_keeps_diagnostic_status(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        caps.apply_collection_result(report, [{"status": "error", "error": "PermissionError: cgroup memory.current"}])
        self.assertEqual(report.measurement["container_memory"].status.value, "permission_denied")

    def test_required_actual_measurements_fail_after_successful_preflight(self):
        caps = self.capabilities()
        for mode in ("full", "basic"):
            report = caps.measurement_report(mode, gpu_modes=["off"])
            rows = [{"status": "ok", "latency_app_s": "0.1", "throughput_samples_per_s": "10",
                     "container_cpu_util_avg_pct": "0", "container_mem_usage_avg_bytes": "nan",
                     "latency_s": "0.1", "cpu_energy_total_j": "1", "vcpu_energy_total_j": "0",
                     "cpu_instructions_per_request": "0"}]
            caps.apply_collection_result(report, rows)
            self.assertEqual(caps.missing_required_measurements(report, rows), ["container_memory"])
            self.assertEqual(report.measurement["container_cpu"].status.value, "verified")

    def test_partial_missing_measurements_do_not_hide_behind_finite_rows(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        row = {"status": "ok", "latency_app_s": "0.1", "throughput_samples_per_s": "10",
               "container_cpu_util_avg_pct": "0", "container_mem_usage_avg_bytes": "123"}
        rows = [row, {**row, "container_mem_usage_avg_bytes": ""}]
        caps.apply_collection_result(report, rows)
        self.assertEqual(caps.missing_required_measurements(report, rows), ["container_memory"])

    def test_explicit_oom_rows_do_not_invent_measurement_success_or_block_valid_rows(self):
        caps = self.capabilities()
        report = caps.measurement_report("basic", gpu_modes=["off"])
        rows = [{"status": "error", "error": "container_oom_killed"}, {
            "status": "ok", "latency_app_s": "0.1", "throughput_samples_per_s": "10",
            "container_cpu_util_avg_pct": "0", "container_mem_usage_avg_bytes": "123"}]
        caps.apply_collection_result(report, rows)
        self.assertEqual(caps.missing_required_measurements(report, rows), [])
        self.assertFalse(report.collection_complete)

    def test_unsupported_profiler_writes_missing_values_without_running_tool(self):
        from acprof.host import compute_profile
        from acprof.host.detect import TaskInfo
        task = TaskInfo("test/onnx", "tabular-regression", "structured", "onnxruntime", "onnxruntime", "main", "unit")
        with tempfile.TemporaryDirectory() as root:
            plan = Path(root) / "input.json"
            plan.write_text(json.dumps({"schema_version": 2, "entries": [{"input_scale": 2, "payload": {"rows": [[1, 2], [3, 4]]}}]}))
            with patch.object(compute_profile, "_profile_torch_entries", side_effect=AssertionError("unsupported Torch must not launch")), patch.object(
                compute_profile, "_profile_gpu_entries", side_effect=AssertionError("unsupported NCU must not launch")
            ), patch.object(compute_profile, "_find_executable", side_effect=AssertionError("unsupported tool must not be discovered")):
                path = compute_profile.collect_compute_profile_plan(
                    task_info=task, image_tag="test", cpu_list=[1], mem_list=[1], gpu_list=["off", "on"],
                    output_dir=root, input_scale_plan_file=str(plan), compute_profile_tool="both",
                    advisor_root=None, ncu_root=None, advisor_repeat=1, ncu_repeat=1, keep_profiles=True,
                )
            payload = json.loads(Path(path).read_text())
        torch_entry = payload["profiles"]["cpu"]["torch_profiler_eager"]["entries"][0]
        ncu_entry = payload["profiles"]["gpu"]["ncu"]["entries"][0]
        self.assertIn("unsupported", torch_entry["error"])
        self.assertIsNone(torch_entry["model_logical_mflop_per_request_torch_profiler_eager"])
        self.assertIn("unsupported", ncu_entry["error"])
        self.assertIsNone(ncu_entry["gpu_executed_mflop_per_request_ncu"])


if __name__ == "__main__":
    unittest.main()

import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import client
from acprof.monitors import energy_cpu
from acprof.config import CSV_FIELDS


REMOVED_LEGACY_COMPUTE_FIELDS = (
    "compute_profile_tool",
    "model_mflop_per_request",
    "compute_mflops_app",
    "compute_mflops",
    "compute_profile_error",
)


class EffectiveEnergyWarningTests(unittest.TestCase):
    def test_default_idle_diag_path_uses_dedicated_debug_directory(self) -> None:
        with patch.object(client, "IDLE_DIAG_PATH", ""):
            self.assertEqual(
                client._idle_diag_path("results/model/result_case.csv"),
                os.path.join(
                    "results",
                    "model",
                    "debug_idle_diag",
                    "result_case.csv.idle_diag.jsonl",
                ),
            )

    def test_matched_control_starts_all_monitors_before_wait_and_applies_baselines(self) -> None:
        events = []

        class FakeEnergyMonitor:
            def __init__(self, name, avg_field):
                self.name = name
                self.avg_field = avg_field
                self.applied = None

            def start(self):
                events.append(f"start:{self.name}")

            def stop(self):
                events.append(f"stop:{self.name}")
                result = SimpleNamespace(**{self.avg_field: 7.0})
                if self.name == "gpu":
                    return result, "GPU", "", [(0.0, 7.0), (1.0, 7.0)]
                return result, "", [SimpleNamespace(timestamp=0.0), SimpleNamespace(timestamp=1.0)]

            def apply_control_baseline(self, result, samples, trace=False):
                events.append(f"apply:{self.name}:{trace}")
                self.applied = (result, samples)

        class FakeResourceMonitor:
            def start(self):
                events.append("start:resource")

            def stop(self):
                events.append("stop:resource")
                return None, "", []

        class FakeMIPSMonitor:
            def start(self):
                events.append("start:mips")

            def stop(self, repeat_in_window, latency_app_s):
                events.append(f"stop:mips:{repeat_in_window}:{latency_app_s}")

        gpu_monitor = FakeEnergyMonitor("gpu", "avg_power_total_w")
        cpu_monitor = FakeEnergyMonitor("cpu", "cpu_avg_power_total_w")

        with patch.object(client, "IDLE_SECONDS", 2.0), patch.object(
            client, "IDLE_DEBUG", True
        ), patch.object(
            client.time,
            "sleep",
            side_effect=lambda seconds: events.append(f"sleep:{seconds}"),
        ):
            client._run_matched_control_window(
                gpu_monitor,
                cpu_monitor,
                FakeResourceMonitor(),
                FakeMIPSMonitor(),
            )

        self.assertEqual(
            events,
            [
                "start:gpu",
                "start:cpu",
                "start:resource",
                "start:mips",
                "sleep:2.0",
                "stop:mips:1:2.0",
                "stop:resource",
                "stop:gpu",
                "stop:cpu",
                "apply:gpu:True",
                "apply:cpu:True",
            ],
        )
        self.assertIsNotNone(gpu_monitor.applied)
        self.assertIsNotNone(cpu_monitor.applied)

    def test_csv_schema_uses_gpu_idle_power_w_field(self) -> None:
        self.assertIn("gpu_idle_power_w", CSV_FIELDS)
        self.assertIn("gpu_idle_measured_at", CSV_FIELDS)
        self.assertIn("gpu_idle_rel_range_so_far", CSV_FIELDS)
        self.assertNotIn("idle_power_w", CSV_FIELDS)
        gpu_idle_index = CSV_FIELDS.index("gpu_idle_power_w")
        self.assertEqual(CSV_FIELDS[gpu_idle_index + 1], "gpu_idle_measured_at")
        self.assertEqual(CSV_FIELDS[gpu_idle_index + 2], "gpu_idle_rel_range_so_far")

    def test_csv_schema_includes_idle_debug_fields_after_cpu_idle_power(self) -> None:
        self.assertIn("cpu_idle_measured_at", CSV_FIELDS)
        self.assertIn("cpu_idle_rel_range_so_far", CSV_FIELDS)
        self.assertNotIn("idle_measured_at", CSV_FIELDS)
        cpu_idle_index = CSV_FIELDS.index("cpu_idle_power_w")
        self.assertEqual(CSV_FIELDS[cpu_idle_index + 1], "cpu_idle_measured_at")
        self.assertEqual(CSV_FIELDS[cpu_idle_index + 2], "cpu_idle_rel_range_so_far")

    def test_csv_schema_prefixes_gpu_energy_fields(self) -> None:
        expected = [
            "gpu_energy_iters",
            "gpu_avg_power_total_w",
            "gpu_peak_power_total_w",
            "gpu_energy_total_j",
            "gpu_avg_power_eff_w",
            "gpu_peak_power_eff_w",
            "gpu_energy_eff_j",
        ]
        old_names = [
            "energy_iters",
            "avg_power_total_w",
            "peak_power_total_w",
            "energy_total_j",
            "avg_power_eff_w",
            "peak_power_eff_w",
            "energy_eff_j",
        ]

        for field in expected:
            self.assertIn(field, CSV_FIELDS)
        for field in old_names:
            self.assertNotIn(field, CSV_FIELDS)

    def test_csv_schema_distinguishes_torch_logical_and_ncu_executed_flops(self) -> None:
        torch_fields = [
            "model_logical_mflop_per_request_torch_profiler_eager",
            "model_logical_mflops_app_torch_profiler_eager",
            "model_logical_mflops_packet_torch_profiler_eager",
            "compute_profile_error_torch_profiler_eager",
        ]
        ncu_fields = [
            "gpu_executed_mflop_per_request_ncu",
            "gpu_executed_tensor_mflop_per_request_ncu",
            "gpu_executed_scalar_mflop_per_request_ncu",
            "gpu_executed_tensor_share_pct_ncu",
            "gpu_executed_mflops_app_ncu",
            "gpu_executed_mflops_packet_ncu",
            "gpu_kernel_launch_count_per_request_ncu",
            "gpu_kernel_time_sum_ms_per_request_ncu",
            "compute_profile_error_ncu",
        ]

        for field in [*torch_fields, *ncu_fields]:
            self.assertIn(field, CSV_FIELDS)
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            self.assertNotIn(field, CSV_FIELDS)
        self.assertNotIn("gpu_profile_report_ncu", CSV_FIELDS)
        self.assertLess(
            CSV_FIELDS.index(torch_fields[0]),
            CSV_FIELDS.index(ncu_fields[0]),
        )

    def test_csv_schema_includes_cpu_memory_behavior_metrics(self) -> None:
        fields = [
            "cpu_cache_references_per_request",
            "cpu_cache_misses_per_request",
            "cpu_cache_miss_rate_pct",
            "cpu_dtlb_loads_per_request",
            "cpu_dtlb_load_misses_per_request",
            "cpu_dtlb_load_miss_rate_pct",
        ]

        for field in fields:
            self.assertIn(field, CSV_FIELDS)
        self.assertEqual(
            CSV_FIELDS[
                CSV_FIELDS.index("cpu_perf_elapsed_s") + 1:
                CSV_FIELDS.index("cpu_perf_elapsed_s") + 1 + len(fields)
            ],
            fields,
        )

    def test_auto_repeat_window_prepares_each_scale_with_warmup_only(self) -> None:
        request_ids = []

        def fake_one_request(scale_value, req_id, payload_override=None):
            request_ids.append(req_id)
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with patch.object(
            client, "CASE_NAME", "case"
        ), patch.object(
            client, "REPEAT_IN_WINDOW", 0
        ), patch.object(
            client, "REPEAT_WINDOW_SECONDS", 0.05, create=True
        ), patch.object(
            client, "AUTO_WARMUP_REQUESTS", 2, create=True
        ), patch.object(
            client,
            "_one_request",
            side_effect=fake_one_request,
        ):
            repeat_counts = [
                client._prepare_repeat_window(1.0, "seq1", {}),
                client._prepare_repeat_window(2.0, "seq2", {}),
            ]

        self.assertEqual(
            repeat_counts,
            [1, 1],
        )
        self.assertEqual(
            request_ids,
            [
                "case_seq1_auto_warmup0",
                "case_seq1_auto_warmup1",
                "case_seq2_auto_warmup0",
                "case_seq2_auto_warmup1",
            ],
        )

    def test_auto_repeat_window_continues_until_target_duration_when_requests_get_faster(self) -> None:
        measurement_req_ids = []

        def fake_one_request(scale_value, req_id, payload_override=None):
            if "_auto_warmup" not in req_id:
                measurement_req_ids.append(req_id)
            latency = 0.2
            return {
                "latency_app_s": latency,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CASE_NAME", "case"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 0
            ), patch.object(
                client, "REPEAT_WINDOW_SECONDS", 1.0, create=True
            ), patch.object(
                client, "AUTO_WARMUP_REQUESTS", 0, create=True
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                side_effect=fake_one_request,
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["repeat_in_window"], "5")
        self.assertEqual(
            measurement_req_ids,
            [
                "case_seq1_r0:0",
                "case_seq1_r0:1",
                "case_seq1_r0:2",
                "case_seq1_r0:3",
                "case_seq1_r0:4",
            ],
        )

    def test_latency_app_distribution_fields_are_written_per_window(self) -> None:
        latencies = iter([0.01, 0.02, 0.10, 0.20, 0.30])

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": next(latencies),
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CASE_NAME", "case"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 5
            ), patch.object(
                client, "SLOW_LATENCY_THRESHOLD_S", 0.06, create=True
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                side_effect=fake_one_request,
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["latency_app_s"], "0.126000")
        self.assertEqual(rows[0]["latency_app_p50_s"], "0.100000")
        self.assertEqual(rows[0]["latency_app_p90_s"], "0.300000")
        self.assertEqual(rows[0]["latency_app_p95_s"], "0.300000")
        self.assertEqual(rows[0]["latency_app_slow_ratio"], "0.600000")

    def test_manual_repeat_in_window_skips_auto_warmup(self) -> None:
        def fake_one_request(scale_value, req_id, payload_override=None):
            self.assertNotIn("_auto_warmup", req_id)
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CASE_NAME", "case"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 20
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                side_effect=fake_one_request,
            ) as one_request:
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["repeat_in_window"], "20")
        self.assertEqual(one_request.call_count, 20)

    def test_gpu_energy_uses_single_idle_measurement_per_workload_window(self) -> None:
        sleep_calls = []

        class FakeGpuMonitor:
            measure_idle_calls = 0

            def __init__(self, *args, **kwargs):
                self.idle_power_w = float("nan")

            def measure_idle(self):
                type(self).measure_idle_calls += 1
                self.idle_power_w = 10.0
                return 10.0

            def start(self):
                pass

            def stop(self):
                return (
                    SimpleNamespace(
                        idle_power_w=self.idle_power_w,
                        energy_iters=10,
                        avg_power_total_w=12.0,
                        peak_power_total_w=13.0,
                        energy_total_j=1.0,
                        avg_power_eff_w=2.0,
                        peak_power_eff_w=3.0,
                        energy_eff_j=0.2,
                    ),
                    "Test GPU",
                    "",
                    [],
                )

            def close(self):
                pass

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CASE_NAME", "case"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 2
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", True
            ), patch.object(
                client,
                "energy_mod",
                SimpleNamespace(GPUEnergyMonitor=FakeGpuMonitor),
            ), patch.object(
                client,
                "IDLE_COOLDOWN_SECONDS",
                2.5,
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.time,
                "sleep",
                side_effect=lambda seconds: sleep_calls.append(seconds),
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                side_effect=fake_one_request,
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    rows = list(reader)

        self.assertEqual(FakeGpuMonitor.measure_idle_calls, 2)
        self.assertEqual(sleep_calls, [2.5, 2.5])
        self.assertIn("gpu_idle_power_w", fieldnames)
        self.assertNotIn("idle_power_w", fieldnames)
        self.assertIn("gpu_energy_iters", fieldnames)
        self.assertNotIn("energy_iters", fieldnames)
        self.assertIn("gpu_energy_eff_j", fieldnames)
        self.assertNotIn("energy_eff_j", fieldnames)
        self.assertEqual(rows[0]["gpu_idle_power_w"], "10.000000")
        self.assertEqual(rows[0]["gpu_energy_eff_j"], "0.200000")

    def test_idle_cooldown_applies_before_cpu_idle_without_gpu(self) -> None:
        sleep_calls = []

        class FakeCPUMonitor:
            measure_idle_calls = 0

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")

            def measure_idle(self, trace_interval_s=None):
                type(self).measure_idle_calls += 1
                self.idle_power_w = 5.0
                return self.idle_power_w

            def start(self):
                pass

            def stop(self):
                return SimpleNamespace(
                    cpu_energy_iters=2,
                    cpu_idle_power_w=self.idle_power_w,
                    cpu_avg_power_total_w=6.0,
                    cpu_peak_power_total_w=7.0,
                    cpu_energy_total_j=1.0,
                    cpu_avg_power_eff_w=1.0,
                    cpu_peak_power_eff_w=2.0,
                    cpu_energy_eff_j=0.5,
                    vcpu_cpu_share=0.5,
                    vcpu_cpu_time_s=0.1,
                    vcpu_avg_power_total_w=3.0,
                    vcpu_peak_power_total_w=4.0,
                    vcpu_energy_total_j=0.3,
                    vcpu_avg_power_eff_w=0.4,
                    vcpu_peak_power_eff_w=0.5,
                    vcpu_energy_eff_j=0.2,
                ), "", []

            def close(self):
                pass

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CASE_NAME", "case"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 2
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client,
                "IDLE_COOLDOWN_SECONDS",
                2.5,
                create=True,
            ), patch.object(
                client,
                "energy_mod",
                None,
            ), patch.object(
                client,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeCPUMonitor(**kwargs)),
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.time,
                "sleep",
                side_effect=lambda seconds: sleep_calls.append(seconds),
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                side_effect=fake_one_request,
            ):
                client.main()

        self.assertEqual(FakeCPUMonitor.measure_idle_calls, 2)
        self.assertEqual(sleep_calls, [2.5, 2.5])

    def test_client_entrypoint_prints_friendly_energy_abort_without_traceback(self) -> None:
        stderr = io.StringIO()
        with patch.object(
            client,
            "main",
            side_effect=client.EnergyAbort("gpu_idle_power_w failed"),
        ), self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            client.run_cli()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("[energy][ERROR] gpu_idle_power_w failed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_sniff_group_id_is_hidden_from_csv_but_kept_for_packet_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CASE_NAME", "case"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fieldnames = reader.fieldnames or []

            with open(f"{out_csv}.sniff_groups.jsonl", "r", encoding="utf-8") as f:
                sidecar_rows = [json.loads(line) for line in f if line.strip()]

        self.assertNotIn("sniff_group_id", fieldnames)
        self.assertNotIn("sniff_group_id", rows[0])
        self.assertEqual(sidecar_rows, [{"sniff_group_id": "case_seq1_r0"}])

    def test_negative_effective_metrics_are_reported_per_field(self) -> None:
        warnings = client._eff_negative_warnings(
            avg_power_eff_w=-0.1,
            peak_power_eff_w=0.0,
            energy_eff_j=-0.001,
        )

        self.assertEqual(warnings, ["gpu_avg_power_eff_w<0", "gpu_energy_eff_j<0"])

    def test_cpu_vcpu_negative_effective_metrics_keep_full_field_names(self) -> None:
        warnings = client._named_negative_warnings({
            "cpu_energy_eff_j": -0.001,
            "vcpu_avg_power_eff_w": -0.1,
            "vcpu_energy_total_j": 1.0,
        })

        self.assertEqual(warnings, ["cpu_energy_eff_j<0", "vcpu_avg_power_eff_w<0"])

    def test_cpu_monitor_unavailable_keeps_successful_row_ok(self) -> None:
        class FakeUnavailableCPUMonitor:
            def measure_idle(self):
                return float("nan")

            def start(self):
                return None

            def stop(self):
                return energy_cpu._nan_result(), "RAPL unavailable", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeUnavailableCPUMonitor()),
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["error"], "")
        self.assertEqual(rows[0]["cpu_energy_total_j"], "nan")

    def test_idle_debug_writes_csv_fields_and_diagnostic_jsonl(self) -> None:
        class FakeCPUMonitor:
            idle_values = iter([5.0, 5.5])
            trace_intervals = []

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")
                self.idle_trace = {}

            def measure_idle(self, trace_interval_s=None):
                type(self).trace_intervals.append(trace_interval_s)
                self.idle_power_w = next(type(self).idle_values)
                self.idle_trace = {
                    "idle_trace_schema": "cpu_rapl_idle_v1",
                    "actual_idle_duration_s": 3.0,
                    "rapl_trace": {
                        "interval_s": trace_interval_s,
                        "power_windows": [{"t0_s": 0.0, "t1_s": 0.1, "power_w": 6.0}],
                    },
                    "idle_proc_cpu_top": [{"pid": 123, "comm": "python", "cpu_time_ms": 10.0}],
                    "idle_container_cpu_delta_s": 0.01,
                }
                return self.idle_power_w

            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    cpu_energy_iters=2,
                    cpu_idle_power_w=self.idle_power_w,
                    cpu_avg_power_total_w=6.0,
                    cpu_peak_power_total_w=7.0,
                    cpu_energy_total_j=1.0,
                    cpu_avg_power_eff_w=1.0,
                    cpu_peak_power_eff_w=2.0,
                    cpu_energy_eff_j=0.5,
                    vcpu_cpu_share=0.5,
                    vcpu_cpu_time_s=0.1,
                    vcpu_avg_power_total_w=3.0,
                    vcpu_peak_power_total_w=4.0,
                    vcpu_energy_total_j=0.3,
                    vcpu_avg_power_eff_w=0.4,
                    vcpu_peak_power_eff_w=0.5,
                    vcpu_energy_eff_j=0.2,
                ), "", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = os.path.join(
                tmp_dir,
                "debug_idle_diag",
                "result.csv.idle_diag.jsonl",
            )
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CASE_NAME", "case"
            ), patch.object(
                client, "CPU_CORES", "1"
            ), patch.object(
                client, "MEM_CAP_GB", "2"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 2
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "IDLE_DEBUG", True, create=True
            ), patch.object(
                client, "IDLE_DIAG_PATH", "", create=True
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeCPUMonitor(**kwargs)),
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ), patch.object(
                client,
                "_collect_idle_debug_snapshot",
                return_value={
                    "snapshot_scope": "after_idle",
                    "loadavg": [0.1, 0.2, 0.3],
                    "top_cpu_processes": [{"pid": 123, "comm": "python", "cpu_pct": 4.5}],
                    "docker_containers": [{"name": "case"}],
                    "docker_stats": [{"name": "case", "cpu_perc": "0.1%"}],
                },
                create=True,
            ), patch.object(
                client,
                "_now_iso",
                side_effect=[
                    "2026-05-02T10:00:00+08:00",
                    "2026-05-02T10:00:01+08:00",
                ],
                create=True,
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag_rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(rows[0]["cpu_idle_measured_at"], "2026-05-02T10:00:00+08:00")
        self.assertEqual(rows[0]["gpu_idle_measured_at"], "nan")
        self.assertEqual(rows[0]["gpu_idle_rel_range_so_far"], "nan")
        self.assertEqual(rows[0]["cpu_idle_rel_range_so_far"], "0.000000")
        self.assertEqual(rows[1]["cpu_idle_measured_at"], "2026-05-02T10:00:01+08:00")
        self.assertEqual(rows[1]["gpu_idle_measured_at"], "nan")
        self.assertEqual(rows[1]["gpu_idle_rel_range_so_far"], "nan")
        self.assertEqual(rows[1]["cpu_idle_rel_range_so_far"], "0.095238")
        self.assertEqual(FakeCPUMonitor.trace_intervals, [0.1, 0.1])
        self.assertEqual(len(diag_rows), 2)
        self.assertEqual(diag_rows[1]["sniff_group_id"], "case_seq1_r1")
        self.assertEqual(diag_rows[1]["cpu_idle_measured_at"], "2026-05-02T10:00:01+08:00")
        self.assertEqual(diag_rows[1]["idle_trace_schema"], "cpu_rapl_idle_v1")
        self.assertEqual(diag_rows[1]["actual_idle_duration_s"], 3.0)
        self.assertEqual(diag_rows[1]["rapl_trace"]["interval_s"], 0.1)
        self.assertEqual(diag_rows[1]["idle_proc_cpu_top"][0]["comm"], "python")
        self.assertEqual(diag_rows[1]["idle_container_cpu_delta_s"], 0.01)
        self.assertEqual(diag_rows[1]["cpu_idle_valid_count"], 2)
        self.assertAlmostEqual(diag_rows[1]["cpu_idle_mean_w"], 5.25)
        self.assertEqual(diag_rows[1]["snapshot_scope"], "after_idle")
        self.assertEqual(diag_rows[1]["loadavg"], [0.1, 0.2, 0.3])
        self.assertEqual(diag_rows[1]["top_cpu_processes"][0]["comm"], "python")
        self.assertEqual(diag_rows[1]["docker_containers"][0]["name"], "case")
        self.assertEqual(diag_rows[1]["docker_stats"][0]["cpu_perc"], "0.1%")

    def test_idle_debug_writes_gpu_idle_fields_and_diagnostics(self) -> None:
        class FakeGpuMonitor:
            idle_values = iter([10.0, 11.0])
            trace_args = []

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")
                self.idle_trace = {}

            def measure_idle(self, trace=False):
                type(self).trace_args.append(trace)
                self.idle_power_w = next(type(self).idle_values)
                self.idle_trace = {
                    "gpu_idle_trace_schema": "nvml_gpu_idle_v1",
                    "gpu_idle_sample_count": 2,
                    "gpu_idle_power_samples": [{"t_s": 0.0, "power_w": self.idle_power_w}],
                }
                return self.idle_power_w

            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    energy_iters=2,
                    idle_power_w=self.idle_power_w,
                    avg_power_total_w=self.idle_power_w + 1.0,
                    peak_power_total_w=self.idle_power_w + 2.0,
                    energy_total_j=1.0,
                    avg_power_eff_w=1.0,
                    peak_power_eff_w=2.0,
                    energy_eff_j=0.5,
                ), "Fake GPU", "", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = f"{out_csv}.idle_diag.jsonl"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CASE_NAME", "case"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 2
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", True
            ), patch.object(
                client, "IDLE_DEBUG", True, create=True
            ), patch.object(
                client, "IDLE_DIAG_PATH", diag_path, create=True
            ), patch.object(
                client,
                "energy_mod",
                SimpleNamespace(GPUEnergyMonitor=lambda **kwargs: FakeGpuMonitor(**kwargs)),
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ), patch.object(
                client,
                "_collect_gpu_idle_debug_snapshot",
                return_value={
                    "gpu_snapshot_scope": "after_gpu_idle",
                    "nvidia_smi_gpu": {"pstate": "P0", "clocks_sm_mhz": 1200.0},
                    "nvidia_smi_pmon": [{"pid": 123, "type": "G", "command": "Xorg"}],
                },
                create=True,
            ), patch.object(
                client,
                "_now_iso",
                side_effect=[
                    "2026-05-02T10:00:00+08:00",
                    "2026-05-02T10:00:01+08:00",
                ],
                create=True,
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag_rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(FakeGpuMonitor.trace_args, [True, True])
        self.assertEqual(rows[0]["gpu_idle_measured_at"], "2026-05-02T10:00:00+08:00")
        self.assertEqual(rows[0]["gpu_idle_rel_range_so_far"], "0.000000")
        self.assertEqual(rows[1]["gpu_idle_measured_at"], "2026-05-02T10:00:01+08:00")
        self.assertEqual(rows[1]["gpu_idle_rel_range_so_far"], "0.095238")
        self.assertEqual(rows[1]["cpu_idle_measured_at"], "nan")
        self.assertEqual(rows[1]["cpu_idle_rel_range_so_far"], "nan")
        self.assertEqual(diag_rows[1]["gpu_idle_trace_schema"], "nvml_gpu_idle_v1")
        self.assertEqual(diag_rows[1]["cpu_idle_measured_at"], "nan")
        self.assertEqual(diag_rows[1]["gpu_idle_sample_count"], 2)
        self.assertEqual(diag_rows[1]["gpu_idle_valid_count"], 2)
        self.assertEqual(diag_rows[1]["gpu_idle_mean_w"], 10.5)
        self.assertEqual(diag_rows[1]["gpu_snapshot_scope"], "after_gpu_idle")
        self.assertEqual(diag_rows[1]["nvidia_smi_gpu"]["pstate"], "P0")
        self.assertEqual(diag_rows[1]["nvidia_smi_pmon"][0]["command"], "Xorg")

    def test_idle_debug_disabled_fills_nan_and_does_not_write_diag_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = f"{out_csv}.idle_diag.jsonl"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "IDLE_DEBUG", False, create=True
            ), patch.object(
                client, "IDLE_DIAG_PATH", diag_path, create=True
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["cpu_idle_measured_at"], "nan")
        self.assertEqual(rows[0]["gpu_idle_measured_at"], "nan")
        self.assertEqual(rows[0]["gpu_idle_rel_range_so_far"], "nan")
        self.assertEqual(rows[0]["cpu_idle_rel_range_so_far"], "nan")
        self.assertFalse(os.path.exists(diag_path))

    def test_resource_usage_metrics_are_written_to_successful_row(self) -> None:
        class FakeResourceUsageMonitor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    resource_usage_iters=2,
                    container_cpu_util_avg_pct=25.0,
                    container_cpu_util_peak_pct=50.0,
                    cpu_freq_avg_hz=3_000_000_000.0,
                    cpu_freq_peak_hz=3_200_000_000.0,
                    container_mem_usage_avg_bytes=1024.0,
                    container_mem_usage_peak_bytes=2048.0,
                    container_mem_util_avg_pct=1.0,
                    container_mem_util_peak_pct=2.0,
                    gpu_util_avg_pct=30.0,
                    gpu_util_peak_pct=40.0,
                    gpu_mem_used_avg_bytes=4096.0,
                    gpu_mem_used_peak_bytes=8192.0,
                    gpu_mem_util_avg_pct=3.0,
                    gpu_mem_util_peak_pct=4.0,
                    gpu_mem_total_bytes=100000.0,
                ), "", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "CPU_CORES", "1"
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                SimpleNamespace(ResourceUsageMonitor=lambda **kwargs: FakeResourceUsageMonitor(**kwargs)),
                create=True,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["resource_usage_iters"], "2.000000")
        self.assertEqual(rows[0]["container_cpu_util_avg_pct"], "25.000000")
        self.assertEqual(rows[0]["cpu_freq_avg_hz"], "3000000000.000000")
        self.assertEqual(rows[0]["cpu_freq_peak_hz"], "3200000000.000000")
        self.assertEqual(rows[0]["cpu_cycles_est_app"], "375000000.000000")
        self.assertEqual(rows[0]["cpu_cycles_est_packet"], "nan")
        self.assertNotIn("gpu_mem_total_bytes", rows[0])

    def test_mips_metrics_are_written_to_successful_row(self) -> None:
        class FakeMIPSMonitor:
            def start(self):
                return None

            def stop(self, repeat_in_window: int, latency_app_s: float):
                return SimpleNamespace(
                    instructions_total=1_000_000.0,
                    instructions_per_request=500_000.0,
                    perf_elapsed_s=0.25,
                    cpu_mips_app=1.0,
                    cache_references_per_request=10_000.0,
                    cache_misses_per_request=500.0,
                    cache_miss_rate_pct=5.0,
                    dtlb_loads_per_request=2_000.0,
                    dtlb_load_misses_per_request=20.0,
                    dtlb_load_miss_rate_pct=1.0,
                )

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 2
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "USE_MIPS", True, create=True
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "perf_mips_mod",
                SimpleNamespace(PerfMIPSMonitor=lambda container_name: FakeMIPSMonitor()),
                create=True,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.25, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["cpu_instructions_per_request"], "500000.000000")
        self.assertEqual(rows[0]["cpu_mips_app"], "1.000000")
        self.assertEqual(rows[0]["cpu_mips_packet"], "nan")
        self.assertEqual(rows[0]["cpu_perf_elapsed_s"], "0.250000")
        self.assertEqual(
            rows[0]["cpu_cache_references_per_request"],
            "10000.000000",
        )
        self.assertEqual(rows[0]["cpu_cache_misses_per_request"], "500.000000")
        self.assertEqual(rows[0]["cpu_cache_miss_rate_pct"], "5.000000")
        self.assertEqual(rows[0]["cpu_dtlb_loads_per_request"], "2000.000000")
        self.assertEqual(
            rows[0]["cpu_dtlb_load_misses_per_request"],
            "20.000000",
        )
        self.assertEqual(rows[0]["cpu_dtlb_load_miss_rate_pct"], "1.000000")

    def test_resource_usage_unavailable_keeps_successful_row_ok(self) -> None:
        class FakeUnavailableResourceUsageMonitor:
            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    resource_usage_iters=0,
                    container_cpu_util_avg_pct=float("nan"),
                    container_cpu_util_peak_pct=float("nan"),
                    cpu_freq_avg_hz=float("nan"),
                    cpu_freq_peak_hz=float("nan"),
                    container_mem_usage_avg_bytes=float("nan"),
                    container_mem_usage_peak_bytes=float("nan"),
                    container_mem_util_avg_pct=float("nan"),
                    container_mem_util_peak_pct=float("nan"),
                    gpu_util_avg_pct=float("nan"),
                    gpu_util_peak_pct=float("nan"),
                    gpu_mem_used_avg_bytes=float("nan"),
                    gpu_mem_used_peak_bytes=float("nan"),
                    gpu_mem_util_avg_pct=float("nan"),
                    gpu_mem_util_peak_pct=float("nan"),
                    gpu_mem_total_bytes=float("nan"),
                ), "resource usage unavailable", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                SimpleNamespace(ResourceUsageMonitor=lambda **kwargs: FakeUnavailableResourceUsageMonitor()),
                create=True,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["error"], "")
        self.assertEqual(rows[0]["container_cpu_util_avg_pct"], "nan")
        self.assertEqual(rows[0]["gpu_util_avg_pct"], "nan")

    def test_legacy_advisor_plan_does_not_emit_removed_generic_columns(self) -> None:
        plan = {
            "profiles": {
                "cpu": {
                    "tool": "intel_advisor",
                    "entries": [
                        {
                            "input_scale": 1.0,
                            "model_mflop_per_request": 200.0,
                            "error": "",
                        }
                    ],
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            plan_path = f"{tmp_dir}/compute_profile_plan.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f)

            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "GPU_MODE", "off"
            ), patch.object(
                client, "COMPUTE_PROFILE_PLAN_FILE", plan_path, create=True
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["status"], "ok")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            self.assertNotIn(field, rows[0])
        self.assertEqual(
            rows[0]["model_logical_mflop_per_request_torch_profiler_eager"],
            "nan",
        )

    def test_v2_compute_plan_writes_independent_torch_and_ncu_metrics(self) -> None:
        plan = {
            "compute_profile_schema_version": 2,
            "profiles": {
                "gpu": {
                    "torch_profiler_eager": {
                        "tool": "torch_profiler_eager",
                        "entries": [
                            {
                                "input_scale": 1.0,
                                "model_logical_mflop_per_request_torch_profiler_eager": 200.0,
                                "error": "",
                            }
                        ],
                    },
                    "ncu": {
                        "tool": "ncu",
                        "entries": [
                            {
                                "input_scale": 1.0,
                                "gpu_executed_mflop_per_request_ncu": 100.0,
                                "gpu_executed_tensor_mflop_per_request_ncu": 80.0,
                                "gpu_executed_scalar_mflop_per_request_ncu": 20.0,
                                "gpu_executed_tensor_share_pct_ncu": 80.0,
                                "gpu_kernel_launch_count_per_request_ncu": 12.0,
                                "gpu_kernel_time_sum_ms_per_request_ncu": 1.5,
                                "error": "",
                            }
                        ],
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            plan_path = f"{tmp_dir}/compute_profile_plan.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f)

            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "GPU_MODE", "on"
            ), patch.object(
                client, "COMPUTE_PROFILE_PLAN_FILE", plan_path, create=True
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client, "cpu_energy_mod", None
            ), patch.object(
                client, "resource_usage_mod", None
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    row = next(csv.DictReader(f))

        self.assertEqual(row["status"], "ok")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            self.assertNotIn(field, row)
        self.assertEqual(
            row["model_logical_mflop_per_request_torch_profiler_eager"],
            "200.000000",
        )
        self.assertEqual(
            row["model_logical_mflops_app_torch_profiler_eager"],
            "400.000000",
        )
        self.assertEqual(
            row["model_logical_mflops_packet_torch_profiler_eager"],
            "400.000000",
        )
        self.assertEqual(
            row["gpu_executed_mflop_per_request_ncu"],
            "100.000000",
        )
        self.assertEqual(row["gpu_executed_mflops_app_ncu"], "200.000000")
        self.assertEqual(
            row["gpu_executed_mflops_packet_ncu"],
            "200.000000",
        )
        self.assertNotIn("gpu_profile_report_ncu", row)
        self.assertEqual(row["compute_profile_error_torch_profiler_eager"], "")
        self.assertEqual(row["compute_profile_error_ncu"], "")

    def test_v2_compute_profile_failures_are_isolated(self) -> None:
        profile = {
            "tool": "torch_profiler_eager",
            "model_mflop_per_request": float("nan"),
            "error": "torch_failed",
            "model_logical_mflop_per_request_torch_profiler_eager": float("nan"),
            "compute_profile_error_torch_profiler_eager": "torch_failed",
            "gpu_executed_mflop_per_request_ncu": 100.0,
            "gpu_executed_tensor_mflop_per_request_ncu": 90.0,
            "gpu_executed_scalar_mflop_per_request_ncu": 10.0,
            "gpu_executed_tensor_share_pct_ncu": 90.0,
            "gpu_kernel_launch_count_per_request_ncu": 4.0,
            "gpu_kernel_time_sum_ms_per_request_ncu": 0.75,
            "compute_profile_error_ncu": "",
        }

        row_metrics = client._compute_profile_row_metrics(profile, 0.5)

        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            self.assertNotIn(field, row_metrics)
        self.assertEqual(
            row_metrics["compute_profile_error_torch_profiler_eager"],
            "torch_failed",
        )
        self.assertEqual(
            row_metrics["gpu_executed_mflop_per_request_ncu"],
            "100.000000",
        )
        self.assertEqual(
            row_metrics["gpu_executed_mflops_app_ncu"],
            "200.000000",
        )
        self.assertEqual(row_metrics["compute_profile_error_ncu"], "")

    def test_missing_compute_profile_keeps_successful_row_ok_with_nan_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            missing_plan_path = f"{tmp_dir}/missing_compute_profile_plan.json"

            with patch.object(
                client, "OUT_CSV", out_csv
            ), patch.object(
                client, "WARMUP", 0
            ), patch.object(
                client, "REPEAT", 1
            ), patch.object(
                client, "REPEAT_IN_WINDOW", 1
            ), patch.object(
                client, "USE_ENERGY", False
            ), patch.object(
                client, "GPU_MODE", "off"
            ), patch.object(
                client, "COMPUTE_PROFILE_PLAN_FILE", missing_plan_path, create=True
            ), patch.object(
                client, "energy_mod", None
            ), patch.object(
                client,
                "cpu_energy_mod",
                None,
            ), patch.object(
                client,
                "resource_usage_mod",
                None,
            ), patch.object(
                client,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch.object(
                client,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                client.main()
                with open(out_csv, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["status"], "ok")
        for field in REMOVED_LEGACY_COMPUTE_FIELDS:
            self.assertNotIn(field, rows[0])
        self.assertEqual(
            rows[0]["model_logical_mflop_per_request_torch_profiler_eager"],
            "nan",
        )
        self.assertIn(
            "compute_profile_plan_not_found",
            rows[0]["compute_profile_error_torch_profiler_eager"],
        )


if __name__ == "__main__":
    unittest.main()

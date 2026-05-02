import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import client
import energy_cpu
from config import CSV_FIELDS


class EffectiveEnergyWarningTests(unittest.TestCase):
    def test_csv_schema_uses_gpu_idle_power_w_field(self) -> None:
        self.assertIn("gpu_idle_power_w", CSV_FIELDS)
        self.assertNotIn("idle_power_w", CSV_FIELDS)

    def test_csv_schema_includes_idle_debug_fields_after_cpu_idle_power(self) -> None:
        self.assertIn("idle_measured_at", CSV_FIELDS)
        self.assertIn("cpu_idle_rel_range_so_far", CSV_FIELDS)
        cpu_idle_index = CSV_FIELDS.index("cpu_idle_power_w")
        self.assertEqual(CSV_FIELDS[cpu_idle_index + 1], "idle_measured_at")
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

    def test_auto_repeat_in_window_calibrates_per_input_scale(self) -> None:
        def fake_one_request(scale_value, req_id, payload_override=None):
            if "_calib" in req_id:
                latency = 0.01 if float(scale_value) == 1.0 else 0.02
            else:
                latency = 0.5
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
                client, "REPEAT_WINDOW_SECONDS", 10.0, create=True
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
                [
                    {"input_scale": 1.0, "scale_label": "seq1", "payload": {}},
                    {"input_scale": 2.0, "scale_label": "seq2", "payload": {}},
                ],
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

        self.assertEqual([row["repeat_in_window"] for row in rows], ["1000", "500"])
        self.assertEqual(one_request.call_count, 3 + 1000 + 3 + 500)

    def test_manual_repeat_in_window_skips_calibration(self) -> None:
        def fake_one_request(scale_value, req_id, payload_override=None):
            self.assertNotIn("_calib", req_id)
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
                "COOLDOWN_SECONDS",
                0,
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
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    rows = list(reader)

        self.assertEqual(FakeGpuMonitor.measure_idle_calls, 2)
        self.assertIn("gpu_idle_power_w", fieldnames)
        self.assertNotIn("idle_power_w", fieldnames)
        self.assertIn("gpu_energy_iters", fieldnames)
        self.assertNotIn("energy_iters", fieldnames)
        self.assertIn("gpu_energy_eff_j", fieldnames)
        self.assertNotIn("energy_eff_j", fieldnames)
        self.assertEqual(rows[0]["gpu_idle_power_w"], "10.000000")
        self.assertEqual(rows[0]["gpu_energy_eff_j"], "0.200000")

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
            diag_path = f"{out_csv}.idle_diag.jsonl"
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
                client, "IDLE_DIAG_PATH", diag_path, create=True
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

        self.assertEqual(rows[0]["idle_measured_at"], "2026-05-02T10:00:00+08:00")
        self.assertEqual(rows[0]["cpu_idle_rel_range_so_far"], "0.000000")
        self.assertEqual(rows[1]["idle_measured_at"], "2026-05-02T10:00:01+08:00")
        self.assertEqual(rows[1]["cpu_idle_rel_range_so_far"], "0.095238")
        self.assertEqual(FakeCPUMonitor.trace_intervals, [0.1, 0.1])
        self.assertEqual(len(diag_rows), 2)
        self.assertEqual(diag_rows[1]["sniff_group_id"], "case_seq1_r1")
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

        self.assertEqual(rows[0]["idle_measured_at"], "nan")
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
        self.assertNotIn("gpu_mem_total_bytes", rows[0])

    def test_resource_usage_unavailable_keeps_successful_row_ok(self) -> None:
        class FakeUnavailableResourceUsageMonitor:
            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    resource_usage_iters=0,
                    container_cpu_util_avg_pct=float("nan"),
                    container_cpu_util_peak_pct=float("nan"),
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

    def test_compute_profile_metrics_are_written_from_plan(self) -> None:
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
        self.assertEqual(rows[0]["compute_profile_tool"], "intel_advisor")
        self.assertEqual(rows[0]["model_mflop_per_request"], "200.000000")
        self.assertEqual(rows[0]["compute_mflops_app"], "400.000000")
        self.assertEqual(rows[0]["compute_mflops"], "400.000000")
        self.assertEqual(rows[0]["compute_profile_error"], "")

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
        self.assertEqual(rows[0]["model_mflop_per_request"], "nan")
        self.assertEqual(rows[0]["compute_mflops"], "nan")
        self.assertIn("compute_profile_plan_not_found", rows[0]["compute_profile_error"])


if __name__ == "__main__":
    unittest.main()

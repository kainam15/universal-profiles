import csv
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import client
import energy_cpu


class EffectiveEnergyWarningTests(unittest.TestCase):
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

        self.assertEqual(warnings, ["avg_power_eff_w<0", "energy_eff_j<0"])

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
        self.assertEqual(rows[0]["gpu_mem_total_bytes"], "100000.000000")

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

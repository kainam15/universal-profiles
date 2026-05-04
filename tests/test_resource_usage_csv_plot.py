import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import plot
from config import CSV_FIELDS


class ResourceUsageCsvPlotTests(unittest.TestCase):
    def test_resource_usage_fields_are_before_cold_start(self) -> None:
        expected = [
            "resource_usage_iters",
            "container_cpu_util_avg_pct",
            "container_cpu_util_peak_pct",
            "cpu_freq_avg_hz",
            "cpu_freq_peak_hz",
            "cpu_cycles_est_app",
            "cpu_cycles_est_packet",
            "cpu_instructions_per_request",
            "cpu_mips_app",
            "cpu_mips_packet",
            "cpu_perf_elapsed_s",
            "container_mem_usage_avg_bytes",
            "container_mem_usage_peak_bytes",
            "container_mem_util_avg_pct",
            "container_mem_util_peak_pct",
            "gpu_util_avg_pct",
            "gpu_util_peak_pct",
            "gpu_mem_used_avg_bytes",
            "gpu_mem_used_peak_bytes",
            "gpu_mem_util_avg_pct",
            "gpu_mem_util_peak_pct",
        ]

        cold_start_index = CSV_FIELDS.index("cold_start_s")
        self.assertEqual(
            CSV_FIELDS[cold_start_index - len(expected):cold_start_index],
            expected,
        )

    def test_prepare_df_converts_resource_usage_fields_and_derives_gib_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "warmup",
                "status",
                "container_cpu_util_avg_pct",
                "cpu_freq_avg_hz",
                "cpu_cycles_est_app",
                "cpu_instructions_per_request",
                "cpu_mips_app",
                "container_mem_usage_avg_bytes",
                "gpu_mem_used_avg_bytes",
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "cpu_cores": "1",
                    "mem_cap_gb": "4",
                    "gpu_mode": "on",
                    "input_scale": "64",
                    "warmup": "0",
                    "status": "ok",
                    "container_cpu_util_avg_pct": "25.5",
                    "cpu_freq_avg_hz": "3000000000",
                    "cpu_cycles_est_app": "765000000",
                    "cpu_instructions_per_request": "500000",
                    "cpu_mips_app": "2.5",
                    "container_mem_usage_avg_bytes": str(2 * 1024 ** 3),
                    "gpu_mem_used_avg_bytes": str(3 * 1024 ** 3),
                })
            static_meta_path = os.path.join(tmp, "static_meta.csv")
            with open(static_meta_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["gpu_mem_total_bytes"])
                writer.writeheader()
                writer.writerow({"gpu_mem_total_bytes": str(8 * 1024 ** 3)})

            df = plot.prepare_df(csv_path)

        self.assertEqual(float(df["container_cpu_util_avg_pct"].iloc[0]), 25.5)
        self.assertEqual(float(df["cpu_freq_avg_hz"].iloc[0]), 3_000_000_000.0)
        self.assertEqual(float(df["cpu_cycles_est_app"].iloc[0]), 765_000_000.0)
        self.assertEqual(float(df["cpu_instructions_per_request"].iloc[0]), 500_000.0)
        self.assertEqual(float(df["cpu_mips_app"].iloc[0]), 2.5)
        self.assertEqual(float(df["container_mem_usage_avg_gib"].iloc[0]), 2.0)
        self.assertEqual(float(df["gpu_mem_used_avg_gib"].iloc[0]), 3.0)
        self.assertEqual(float(df["gpu_mem_total_gib"].iloc[0]), 8.0)

    def test_prepare_df_accepts_padded_legacy_csv_headers_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(
                    "cpu_cores, mem_cap_gb, gpu_mode, input_scale, warmup, status, "
                    "gpu_mem_used_avg_bytes\n"
                )
                f.write(f"1, 4, on     , 64, 0, ok    , {3 * 1024 ** 3}\n")
            static_meta_path = os.path.join(tmp, "static_meta.csv")
            with open(static_meta_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["gpu_mem_total_bytes"])
                writer.writeheader()
                writer.writerow({"gpu_mem_total_bytes": str(8 * 1024 ** 3)})

            df = plot.prepare_df(csv_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df["gpu_mode"].iloc[0], "on")
        self.assertEqual(float(df["gpu_mem_total_gib"].iloc[0]), 8.0)

    def test_prepare_df_converts_compute_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "warmup",
                "status",
                "model_mflop_per_request",
                "compute_mflops_app",
                "compute_mflops",
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "cpu_cores": "1",
                    "mem_cap_gb": "4",
                    "gpu_mode": "off",
                    "input_scale": "64",
                    "warmup": "0",
                    "status": "ok",
                    "model_mflop_per_request": "200.5",
                    "compute_mflops_app": "401.0",
                    "compute_mflops": "802.0",
                })

            df = plot.prepare_df(csv_path)

        self.assertEqual(float(df["model_mflop_per_request"].iloc[0]), 200.5)
        self.assertEqual(float(df["compute_mflops_app"].iloc[0]), 401.0)
        self.assertEqual(float(df["compute_mflops"].iloc[0]), 802.0)

    def test_prepare_df_aliases_legacy_gpu_energy_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "warmup",
                "status",
                "avg_power_eff_w",
                "energy_eff_j",
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "cpu_cores": "1",
                    "mem_cap_gb": "4",
                    "gpu_mode": "on",
                    "input_scale": "64",
                    "warmup": "0",
                    "status": "ok",
                    "avg_power_eff_w": "2.5",
                    "energy_eff_j": "0.25",
                })

            df = plot.prepare_df(csv_path)

        self.assertEqual(float(df["gpu_avg_power_eff_w"].iloc[0]), 2.5)
        self.assertEqual(float(df["gpu_energy_eff_j"].iloc[0]), 0.25)

    def test_main_names_gpu_power_and_energy_plots_with_gpu_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "cpu_cores",
                        "mem_cap_gb",
                        "gpu_mode",
                        "input_scale",
                        "warmup",
                        "status",
                        "gpu_avg_power_eff_w",
                        "gpu_energy_eff_j",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "cpu_cores": "1",
                    "mem_cap_gb": "4",
                    "gpu_mode": "on",
                    "input_scale": "64",
                    "warmup": "0",
                    "status": "ok",
                    "gpu_avg_power_eff_w": "2.5",
                    "gpu_energy_eff_j": "0.25",
                })

            out_pngs = []

            def capture_plot_metric(*_args, **kwargs):
                out_pngs.append(os.path.basename(kwargs["out_png"]))

            with patch.object(sys, "argv", ["plot.py", csv_path]), patch.object(
                plot, "plot_metric", side_effect=capture_plot_metric
            ), patch.object(plot, "plot_cold_start_bar"):
                plot.main()

        self.assertIn("gpu_avg_power_vs_scale.png", out_pngs)
        self.assertIn("gpu_energy_vs_scale.png", out_pngs)
        self.assertNotIn("avg_power_vs_scale.png", out_pngs)
        self.assertNotIn("energy_vs_scale.png", out_pngs)


if __name__ == "__main__":
    unittest.main()

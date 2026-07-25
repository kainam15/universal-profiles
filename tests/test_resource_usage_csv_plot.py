import csv
import colorsys
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from acprof.cli import plot
from acprof.config import CSV_FIELDS


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

    def test_container_mem_util_plot_uses_fixed_mem_colors(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "cpu_cores": 1,
                    "mem_cap_gb": mem,
                    "gpu_mode": gpu_mode,
                    "input_scale": 64,
                    "container_mem_util_avg_pct": 10.0 + mem,
                }
                for mem, gpu_mode in [(2, "on"), (4, "off"), (8, "on"), (16, "off")]
            ]
        )
        colors_by_mem = {}

        def capture_plot(*_args, **kwargs):
            label = kwargs["label"]
            mem = int(label.rsplit("Mem", 1)[1])
            colors_by_mem[mem] = kwargs["color"]

        with patch.object(plot.plt, "plot", side_effect=capture_plot), patch.object(
            plot.plt, "legend"
        ):
            plot.plot_metric(
                df,
                metric="container_mem_util_avg_pct",
                title="Container Memory Utilization vs. Input Scale",
                ylabel="Memory Utilization (%)",
                xlabel="input_scale",
                out_png=None,
            )

        self.assertEqual(
            colors_by_mem,
            {
                2: (0.12, 0.47, 0.71),
                4: (0.93, 0.69, 0.13),
                8: (0.84, 0.15, 0.16),
                16: (0.58, 0.40, 0.74),
            },
        )

    def test_container_mem_util_plot_gets_darker_as_cpu_cores_increase(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "cpu_cores": cpu,
                    "mem_cap_gb": 4,
                    "gpu_mode": "on",
                    "input_scale": 64,
                    "container_mem_util_avg_pct": 20.0,
                }
                for cpu in (1, 2, 4, 8)
            ]
        )
        colors_by_cpu = {}

        def capture_plot(*_args, **kwargs):
            label = kwargs["label"]
            cpu = int(label.split("+CPU", 1)[1].split("+", 1)[0])
            colors_by_cpu[cpu] = kwargs["color"]

        with patch.object(plot.plt, "plot", side_effect=capture_plot), patch.object(
            plot.plt, "legend"
        ):
            plot.plot_metric(
                df,
                metric="container_mem_util_avg_pct",
                title="Container Memory Utilization vs. Input Scale",
                ylabel="Memory Utilization (%)",
                xlabel="input_scale",
                out_png=None,
            )

        lightness_by_cpu = {
            cpu: colorsys.rgb_to_hls(*color)[1]
            for cpu, color in colors_by_cpu.items()
        }
        self.assertEqual(set(lightness_by_cpu), {1, 2, 4, 8})
        self.assertGreater(lightness_by_cpu[1], lightness_by_cpu[2])
        self.assertGreater(lightness_by_cpu[2], lightness_by_cpu[4])
        self.assertGreater(lightness_by_cpu[4], lightness_by_cpu[8])

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

    def test_main_splits_plots_into_cpu_gpu_and_combined_directories(self) -> None:
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
                        "latency_s",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "off",
                        "input_scale": "64",
                        "warmup": "0",
                        "status": "ok",
                        "latency_s": "0.2",
                    },
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "on",
                        "input_scale": "64",
                        "warmup": "0",
                        "status": "ok",
                        "latency_s": "0.1",
                    },
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "unknown",
                        "input_scale": "64",
                        "warmup": "0",
                        "status": "ok",
                        "latency_s": "0.3",
                    },
                ])

            modes_by_directory = {}

            def capture_plot_metric(group_df, *_args, **kwargs):
                if kwargs["metric"] != "latency_s":
                    return
                output_directory = os.path.basename(
                    os.path.dirname(kwargs["out_png"])
                )
                modes_by_directory[output_directory] = set(group_df["gpu_mode"])

            with patch.object(
                plot, "plot_metric", side_effect=capture_plot_metric
            ), patch.object(plot, "plot_cold_start_bar"), patch.object(
                plot, "write_latency_model_report"
            ):
                plot.main([csv_path])

            self.assertEqual(
                modes_by_directory,
                {
                    "cpu": {"off"},
                    "gpu": {"on"},
                    "cpu+gpu": {"off", "on"},
                },
            )
            for directory in ("cpu", "gpu", "cpu+gpu"):
                self.assertTrue(os.path.isdir(os.path.join(tmp, directory)))

    def test_main_plots_packet_cpu_metrics_vs_input_scale(self) -> None:
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
                        "cpu_mips_packet",
                        "cpu_instructions_per_request",
                        "cpu_cycles_est_packet",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "cpu_cores": "1",
                    "mem_cap_gb": "4",
                    "gpu_mode": "off",
                    "input_scale": "64",
                    "warmup": "0",
                    "status": "ok",
                    "cpu_mips_packet": "2.5",
                    "cpu_instructions_per_request": "500000",
                    "cpu_cycles_est_packet": "750000000",
                })

            out_pngs = []

            def capture_plot_metric(*_args, **kwargs):
                out_pngs.append(os.path.basename(kwargs["out_png"]))

            with patch.object(sys, "argv", ["plot.py", csv_path]), patch.object(
                plot, "plot_metric", side_effect=capture_plot_metric
            ), patch.object(plot, "plot_cold_start_bar"):
                plot.main()

        self.assertIn("cpu_mips_packet_vs_scale.png", out_pngs)
        self.assertIn("cpu_instructions_per_request_vs_scale.png", out_pngs)
        self.assertIn("cpu_cycles_est_packet_vs_scale.png", out_pngs)


if __name__ == "__main__":
    unittest.main()

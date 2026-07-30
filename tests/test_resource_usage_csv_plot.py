import csv
import colorsys
import json
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
            "cpu_cache_references_per_request",
            "cpu_cache_misses_per_request",
            "cpu_cache_miss_rate_pct",
            "cpu_dtlb_loads_per_request",
            "cpu_dtlb_load_misses_per_request",
            "cpu_dtlb_load_miss_rate_pct",
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
            static_meta_path = os.path.join(tmp, "static_meta.json")
            with open(static_meta_path, "w", encoding="utf-8") as f:
                json.dump({"gpu_mem_total_bytes": 8 * 1024 ** 3}, f)

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
        self.assertEqual(
            float(
                df[
                    "model_logical_mflop_per_request_torch_profiler_eager"
                ].iloc[0]
            ),
            200.5,
        )
        self.assertEqual(
            float(
                df[
                    "model_logical_mflops_app_torch_profiler_eager"
                ].iloc[0]
            ),
            401.0,
        )
        self.assertEqual(
            float(
                df[
                    "model_logical_mflops_packet_torch_profiler_eager"
                ].iloc[0]
            ),
            802.0,
        )

    def test_prepare_df_converts_execution_profile_fields_and_derives_gib(
        self,
    ) -> None:
        execution_values = {
            "cpu_heap_peak_bytes_massif": str(2 * 1024 ** 3),
            "cpu_heap_extra_peak_bytes_massif": "4096",
            "cpu_stack_peak_bytes_massif": "8192",
            "cpu_heap_peak_total_bytes_massif": str(3 * 1024 ** 3),
            "cpu_heap_peak_at_ms_massif": "1234.5",
            "host_inference_wall_time_ms_per_request_nsys": "20.5",
            "cuda_api_time_sum_ms_per_request_nsys": "7.25",
            "cuda_api_call_count_per_request_nsys": "12",
            "gpu_kernel_time_sum_ms_per_request_nsys": "5.5",
            "gpu_kernel_launch_count_per_request_nsys": "8",
            "gpu_memcpy_time_sum_ms_per_request_nsys": "1.25",
            "gpu_memcpy_count_per_request_nsys": "3",
            "gpu_memcpy_bytes_per_request_nsys": "1048576",
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "warmup",
                "status",
                *execution_values,
                "compute_profile_error_massif",
                "compute_profile_error_nsys",
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
                    **execution_values,
                    "compute_profile_error_massif": "massif diagnostic",
                    "compute_profile_error_nsys": "nsys diagnostic",
                })

            df = plot.prepare_df(csv_path)

        for field, expected in execution_values.items():
            self.assertTrue(pd.api.types.is_numeric_dtype(df[field]), field)
            self.assertEqual(float(df[field].iloc[0]), float(expected), field)
        self.assertEqual(
            float(df["cpu_heap_peak_total_gib_massif"].iloc[0]),
            3.0,
        )
        self.assertEqual(
            df["compute_profile_error_massif"].iloc[0],
            "massif diagnostic",
        )
        self.assertEqual(
            df["compute_profile_error_nsys"].iloc[0],
            "nsys diagnostic",
        )

    def test_prepare_df_prefers_new_compute_fields_and_parses_ncu_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            ncu_values = {
                "gpu_executed_mflop_per_request_ncu": "300.5",
                "gpu_executed_tensor_mflop_per_request_ncu": "250.25",
                "gpu_executed_scalar_mflop_per_request_ncu": "50.25",
                "gpu_executed_tensor_share_pct_ncu": "83.27787",
                "gpu_executed_mflops_app_ncu": "601.0",
                "gpu_executed_mflops_packet_ncu": "1202.0",
                "gpu_kernel_launch_count_per_request_ncu": "156",
                "gpu_kernel_time_sum_ms_per_request_ncu": "12.5",
            }
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
                "model_logical_mflop_per_request_torch_profiler_eager",
                "model_logical_mflops_app_torch_profiler_eager",
                "model_logical_mflops_packet_torch_profiler_eager",
                *ncu_values,
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
                    "model_mflop_per_request": "200.5",
                    "compute_mflops_app": "401.0",
                    "compute_mflops": "802.0",
                    (
                        "model_logical_mflop_per_request_"
                        "torch_profiler_eager"
                    ): "999.5",
                    (
                        "model_logical_mflops_app_"
                        "torch_profiler_eager"
                    ): "1999.0",
                    (
                        "model_logical_mflops_packet_"
                        "torch_profiler_eager"
                    ): "not-a-number",
                    **ncu_values,
                })

            df = plot.prepare_df(csv_path)

        self.assertEqual(
            float(
                df[
                    "model_logical_mflop_per_request_torch_profiler_eager"
                ].iloc[0]
            ),
            999.5,
        )
        self.assertEqual(
            float(
                df[
                    "model_logical_mflops_app_torch_profiler_eager"
                ].iloc[0]
            ),
            1999.0,
        )
        self.assertEqual(
            float(
                df[
                    "model_logical_mflops_packet_torch_profiler_eager"
                ].iloc[0]
            ),
            802.0,
        )
        for column, value in ncu_values.items():
            self.assertEqual(float(df[column].iloc[0]), float(value))

    def test_prepare_df_routes_legacy_compute_values_by_profiler_tool(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "compute_profile_tool",
                "model_mflop_per_request",
                "compute_mflops_app",
                "compute_mflops",
                "warmup",
                "status",
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "off",
                        "input_scale": "64",
                        "compute_profile_tool": "torch_profiler",
                        "model_mflop_per_request": "100",
                        "compute_mflops_app": "200",
                        "compute_mflops": "250",
                        "warmup": "0",
                        "status": "ok",
                    },
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "on",
                        "input_scale": "64",
                        "compute_profile_tool": "ncu",
                        "model_mflop_per_request": "300",
                        "compute_mflops_app": "600",
                        "compute_mflops": "750",
                        "warmup": "0",
                        "status": "ok",
                    },
                ])

            prepared = plot.prepare_df(csv_path)

        torch_row = prepared.iloc[0]
        ncu_row = prepared.iloc[1]
        self.assertEqual(
            torch_row[
                "model_logical_mflop_per_request_torch_profiler_eager"
            ],
            100.0,
        )
        self.assertTrue(
            pd.isna(torch_row["gpu_executed_mflop_per_request_ncu"])
        )
        self.assertTrue(
            pd.isna(
                ncu_row[
                    "model_logical_mflop_per_request_torch_profiler_eager"
                ]
            )
        )
        self.assertEqual(
            ncu_row["gpu_executed_mflop_per_request_ncu"],
            300.0,
        )
        self.assertEqual(ncu_row["gpu_executed_mflops_app_ncu"], 600.0)
        self.assertEqual(ncu_row["gpu_executed_mflops_packet_ncu"], 750.0)

    def test_compute_plot_metrics_separate_torch_eager_from_ncu(self) -> None:
        metrics = {
            metric: (title, ylabel, filename)
            for metric, title, ylabel, filename in plot.PLOT_METRICS
        }

        self.assertNotIn("compute_mflops", metrics)
        self.assertEqual(
            metrics["model_logical_mflops_packet_torch_profiler_eager"],
            (
                (
                    "Torch Profiler Eager Logical Compute Throughput "
                    "(Packet Latency) vs. Input Scale"
                ),
                "Logical MFLOPS (packet latency)",
                "torch_profiler_eager_logical_mflops_packet_vs_scale.png",
            ),
        )
        self.assertEqual(
            metrics["gpu_executed_mflops_packet_ncu"],
            (
                (
                    "NCU GPU-Executed Compute Throughput "
                    "(Packet Latency) vs. Input Scale"
                ),
                "GPU-executed MFLOPS (packet latency)",
                "ncu_gpu_executed_mflops_packet_vs_scale.png",
            ),
        )
        self.assertIn(
            "gpu_executed_tensor_mflop_per_request_ncu",
            metrics,
        )
        self.assertIn(
            "gpu_executed_scalar_mflop_per_request_ncu",
            metrics,
        )
        self.assertIn("gpu_executed_tensor_share_pct_ncu", metrics)
        self.assertIn(
            "gpu_kernel_launch_count_per_request_ncu",
            metrics,
        )
        self.assertIn(
            "gpu_kernel_time_sum_ms_per_request_ncu",
            metrics,
        )

    def test_execution_profile_plot_metrics_are_registered(self) -> None:
        metrics = {
            metric: (title, ylabel, filename)
            for metric, title, ylabel, filename in plot.PLOT_METRICS
        }

        self.assertEqual(
            metrics["cpu_heap_peak_total_gib_massif"],
            (
                "Massif Process-Lifetime Peak Memory vs. Input Scale",
                "Peak heap + extra + stack (GiB)",
                "massif_cpu_heap_peak_total_vs_scale.png",
            ),
        )
        self.assertEqual(
            metrics["host_inference_wall_time_ms_per_request_nsys"],
            (
                "Nsight Systems Host Inference Wall Time vs. Input Scale",
                "Host wall time (ms/request)",
                "nsys_host_inference_wall_time_per_request_vs_scale.png",
            ),
        )
        self.assertEqual(
            metrics["cuda_api_time_sum_ms_per_request_nsys"][2],
            "nsys_cuda_api_time_sum_per_request_vs_scale.png",
        )
        self.assertEqual(
            metrics["gpu_kernel_time_sum_ms_per_request_nsys"][2],
            "nsys_gpu_kernel_time_sum_per_request_vs_scale.png",
        )
        self.assertEqual(
            metrics["gpu_memcpy_time_sum_ms_per_request_nsys"][2],
            "nsys_gpu_memcpy_time_sum_per_request_vs_scale.png",
        )

    def test_execution_profile_plot_skips_all_nan_column(self) -> None:
        df = pd.DataFrame([{
            "cpu_cores": 1,
            "mem_cap_gb": 4,
            "gpu_mode": "on",
            "input_scale": 64,
            "gpu_kernel_time_sum_ms_per_request_nsys": float("nan"),
        }])

        with patch.object(plot.plt, "figure") as figure, patch.object(
            plot.plt,
            "savefig",
        ) as savefig:
            plot.plot_metric(
                df,
                metric="gpu_kernel_time_sum_ms_per_request_nsys",
                title="Nsight Systems GPU Kernel Time vs. Input Scale",
                ylabel="Summed GPU kernel time (ms/request)",
                xlabel="input_scale",
                out_png="unused.png",
            )

        figure.assert_not_called()
        savefig.assert_not_called()

    def test_bandwidth_behavior_plot_metrics_are_registered(self) -> None:
        metrics = {
            metric: (title, ylabel, filename)
            for metric, title, ylabel, filename in plot.PLOT_METRICS
        }

        self.assertEqual(
            metrics["cpu_cache_misses_per_request"],
            (
                "CPU Cache Misses per Request vs. Input Scale",
                "Cache misses/request",
                "cpu_cache_misses_per_request_vs_scale.png",
            ),
        )
        self.assertEqual(
            metrics["cpu_cache_miss_rate_pct"],
            (
                "CPU Cache Miss Rate vs. Input Scale",
                "Cache miss rate (%)",
                "cpu_cache_miss_rate_vs_scale.png",
            ),
        )
        self.assertEqual(
            metrics["cpu_dtlb_load_misses_per_request"],
            (
                "CPU dTLB Load Misses per Request vs. Input Scale",
                "dTLB load misses/request",
                "cpu_dtlb_load_misses_per_request_vs_scale.png",
            ),
        )
        self.assertEqual(
            metrics["cpu_dtlb_load_miss_rate_pct"],
            (
                "CPU dTLB Load Miss Rate vs. Input Scale",
                "dTLB load miss rate (%)",
                "cpu_dtlb_load_miss_rate_vs_scale.png",
            ),
        )

    def test_prepare_df_converts_bandwidth_behavior_fields_to_numeric(self) -> None:
        bandwidth_fields = [
            "cpu_cache_references_per_request",
            "cpu_cache_misses_per_request",
            "cpu_cache_miss_rate_pct",
            "cpu_dtlb_loads_per_request",
            "cpu_dtlb_load_misses_per_request",
            "cpu_dtlb_load_miss_rate_pct",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "warmup",
                "status",
                *bandwidth_fields,
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
                    "cpu_cache_references_per_request": "1000",
                    "cpu_cache_misses_per_request": "125",
                    "cpu_cache_miss_rate_pct": "12.5",
                    "cpu_dtlb_loads_per_request": "800",
                    "cpu_dtlb_load_misses_per_request": "20",
                    "cpu_dtlb_load_miss_rate_pct": "2.5",
                })

            df = plot.prepare_df(csv_path)

        for field in bandwidth_fields:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[field]), field)
        self.assertEqual(float(df["cpu_cache_misses_per_request"].iloc[0]), 125.0)
        self.assertEqual(float(df["cpu_dtlb_load_miss_rate_pct"].iloc[0]), 2.5)

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
                    "gpu+cpu": {"off", "on"},
                },
            )
            for directory in ("cpu", "gpu", "gpu+cpu"):
                self.assertTrue(os.path.isdir(os.path.join(tmp, directory)))

    def test_main_plots_packet_and_bandwidth_cpu_metrics_vs_input_scale(self) -> None:
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
                        "cpu_cache_misses_per_request",
                        "cpu_cache_miss_rate_pct",
                        "cpu_dtlb_load_misses_per_request",
                        "cpu_dtlb_load_miss_rate_pct",
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
                    "cpu_cache_misses_per_request": "125",
                    "cpu_cache_miss_rate_pct": "12.5",
                    "cpu_dtlb_load_misses_per_request": "20",
                    "cpu_dtlb_load_miss_rate_pct": "2.5",
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
        self.assertIn("cpu_cache_misses_per_request_vs_scale.png", out_pngs)
        self.assertIn("cpu_cache_miss_rate_vs_scale.png", out_pngs)
        self.assertIn("cpu_dtlb_load_misses_per_request_vs_scale.png", out_pngs)
        self.assertIn("cpu_dtlb_load_miss_rate_vs_scale.png", out_pngs)


if __name__ == "__main__":
    unittest.main()

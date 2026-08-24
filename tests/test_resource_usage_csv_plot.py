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
    @staticmethod
    def _overview_spec(filename: str):
        return next(
            spec
            for spec in plot.METRIC_OVERVIEW_PLOTS
            if spec[1] == filename
        )

    @staticmethod
    def _overview_metric_names(spec) -> tuple[str, ...]:
        return tuple(
            panel[0]
            for panel in spec[4]
            if panel is not None
        )

    def test_resource_usage_fields_are_before_cold_start(self) -> None:
        expected = [
            "resource_usage_iters",
            "container_cpu_util_avg_pct",
            "container_cpu_util_peak_pct",
            "container_cpu_nr_periods_delta",
            "container_cpu_nr_throttled_delta",
            "container_cpu_throttled_period_ratio_pct",
            "container_cpu_throttled_time_s_per_request",
            "container_cpu_pressure_some_stall_pct",
            "container_cpu_pressure_full_stall_pct",
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
            "container_mem_peak_cgroup_bytes",
            "container_mem_anon_bytes_end",
            "container_mem_file_bytes_end",
            "container_mem_slab_bytes_end",
            "container_mem_pgfault_delta",
            "container_mem_pgmajfault_delta",
            "container_mem_workingset_refault_delta",
            "container_mem_high_events_delta",
            "container_mem_max_events_delta",
            "container_mem_oom_events_delta",
            "container_mem_oom_kill_events_delta",
            "container_mem_pressure_some_stall_pct",
            "container_mem_pressure_full_stall_pct",
            "container_swap_limit_bytes",
            "container_swap_usage_avg_bytes",
            "container_swap_usage_peak_bytes",
            "container_io_read_bytes_per_request",
            "container_io_write_bytes_per_request",
            "container_io_read_ops_per_request",
            "container_io_write_ops_per_request",
            "container_io_pressure_some_stall_pct",
            "container_io_pressure_full_stall_pct",
            "container_pids_current_end",
            "container_pids_peak_cgroup",
            "container_pids_max_events_delta",
            "gpu_sm_clock_mhz",
            "gpu_memory_clock_mhz",
            "gpu_pstate",
            "gpu_temp_c",
            "gpu_util_avg_pct",
            "gpu_util_peak_pct",
            "gpu_mem_used_avg_bytes",
            "gpu_mem_used_peak_bytes",
            "gpu_mem_util_avg_pct",
            "gpu_mem_util_peak_pct",
        ]

        cold_start_index = CSV_FIELDS.index("cold_start_started_at")
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
                "container_swap_limit_bytes",
                "container_swap_usage_avg_bytes",
                "container_swap_usage_peak_bytes",
                "container_io_read_bytes_per_request",
                "container_io_write_bytes_per_request",
                "gpu_sm_clock_mhz",
                "gpu_memory_clock_mhz",
                "gpu_pstate",
                "gpu_temp_c",
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
                    "container_swap_limit_bytes": str(4 * 1024 ** 3),
                    "container_swap_usage_avg_bytes": str(512 * 1024 ** 2),
                    "container_swap_usage_peak_bytes": str(1024 ** 3),
                    "container_io_read_bytes_per_request": "4096",
                    "container_io_write_bytes_per_request": "2048",
                    "gpu_sm_clock_mhz": "1800",
                    "gpu_memory_clock_mhz": "7500",
                    "gpu_pstate": "P0",
                    "gpu_temp_c": "67",
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
        self.assertEqual(float(df["gpu_sm_clock_mhz"].iloc[0]), 1800.0)
        self.assertEqual(float(df["gpu_memory_clock_mhz"].iloc[0]), 7500.0)
        self.assertEqual(df["gpu_pstate"].iloc[0], "P0")
        self.assertEqual(float(df["gpu_temp_c"].iloc[0]), 67.0)
        self.assertEqual(float(df["container_mem_usage_avg_gib"].iloc[0]), 2.0)
        self.assertEqual(float(df["container_swap_limit_gib"].iloc[0]), 4.0)
        self.assertEqual(
            float(df["container_swap_usage_avg_gib"].iloc[0]),
            0.5,
        )
        self.assertEqual(
            float(df["container_swap_usage_peak_gib"].iloc[0]),
            1.0,
        )
        self.assertEqual(
            float(df["container_io_read_bytes_per_request"].iloc[0]),
            4096.0,
        )
        self.assertEqual(
            float(df["container_io_write_bytes_per_request"].iloc[0]),
            2048.0,
        )
        self.assertEqual(float(df["gpu_mem_used_avg_gib"].iloc[0]), 3.0)
        self.assertEqual(float(df["gpu_mem_total_gib"].iloc[0]), 8.0)

    def test_prepare_df_converts_low_collection_cost_metrics(self) -> None:
        metric_fields = [
            "input_units_per_request",
            "packet_total_wire_bytes_per_request",
            "packet_protocol_overhead_ratio",
            "latency_s_per_input_unit",
            "latency_request_count",
            "latency_std_s",
            "latency_cv",
            "latency_iqr_s",
            "latency_max_s",
            "latency_app_request_count",
            "latency_app_std_s",
            "latency_app_cv",
            "latency_app_iqr_s",
            "latency_app_max_s",
            "latency_app_s_per_input_unit",
            "throughput_samples_per_s_per_cpu_core",
            "container_attributed_energy_eff_j",
            "container_attributed_samples_per_j",
            "container_attributed_edp_app_js",
            "output_tokens_per_s_app",
            "container_attributed_j_per_output_token",
            "container_attributed_j_per_input_unit",
            "container_cpu_nr_periods_delta",
            "container_cpu_nr_throttled_delta",
            "container_cpu_throttled_period_ratio_pct",
            "container_cpu_throttled_time_s_per_request",
            "container_cpu_pressure_some_stall_pct",
            "container_cpu_pressure_full_stall_pct",
            "container_mem_high_events_delta",
            "container_mem_max_events_delta",
            "container_mem_oom_events_delta",
            "container_mem_oom_kill_events_delta",
            "container_mem_pressure_some_stall_pct",
            "container_mem_pressure_full_stall_pct",
            "container_mem_peak_cgroup_bytes",
            "container_mem_anon_bytes_end",
            "container_mem_file_bytes_end",
            "container_mem_slab_bytes_end",
            "container_mem_pgfault_delta",
            "container_mem_pgmajfault_delta",
            "container_mem_workingset_refault_delta",
            "container_io_read_ops_per_request",
            "container_io_write_ops_per_request",
            "container_io_pressure_some_stall_pct",
            "container_io_pressure_full_stall_pct",
            "container_pids_current_end",
            "container_pids_peak_cgroup",
            "container_pids_max_events_delta",
            "cold_start_container_launch_s",
            "cold_start_server_setup_s",
            "cold_start_cuda_init_s",
            "cold_start_model_load_s",
            "cold_start_ready_wait_s",
            "cold_start_first_predict_app_s",
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
                *metric_fields,
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
                    **{field: "1.5" for field in metric_fields},
                })

            df = plot.prepare_df(csv_path)

        for field in metric_fields:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[field]), field)
            self.assertEqual(float(df[field].iloc[0]), 1.5, field)

    def test_prepare_df_excludes_error_rows_even_when_they_have_numeric_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "warmup",
                "status",
                "latency_s",
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "8",
                        "gpu_mode": "off",
                        "input_scale": "1",
                        "warmup": "0",
                        "status": "ok",
                        "latency_s": "0.25",
                    },
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "2",
                        "gpu_mode": "off",
                        "input_scale": "1",
                        "warmup": "0",
                        "status": " ERROR ",
                        "latency_s": "999",
                    },
                ])

            df = plot.prepare_df(csv_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df["status"].iloc[0], "ok")
        self.assertEqual(float(df["latency_s"].iloc[0]), 0.25)

    def test_prepare_df_returns_empty_frame_when_all_rows_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_case.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "warmup",
                "status",
                "latency_s",
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "cpu_cores": "1",
                    "mem_cap_gb": "2",
                    "gpu_mode": "off",
                    "input_scale": "1",
                    "warmup": "0",
                    "status": "error",
                    "latency_s": "999",
                })

            df = plot.prepare_df(csv_path)

        self.assertTrue(df.empty)
        self.assertIn("config", df.columns)

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

    def test_compute_overviews_separate_torch_eager_from_ncu(self) -> None:
        torch_spec = self._overview_spec(
            "torch_compute_overview_vs_scale.png"
        )
        ncu_arithmetic_spec = self._overview_spec(
            "ncu_arithmetic_overview_vs_scale.png"
        )
        ncu_runtime_spec = self._overview_spec(
            "ncu_runtime_overview_vs_scale.png"
        )

        self.assertEqual(
            self._overview_metric_names(torch_spec),
            (
                "model_logical_mflop_per_request_torch_profiler_eager",
                "model_logical_mflops_app_torch_profiler_eager",
                "model_logical_mflops_packet_torch_profiler_eager",
            ),
        )
        self.assertEqual(
            self._overview_metric_names(ncu_arithmetic_spec),
            (
                "gpu_executed_mflop_per_request_ncu",
                "gpu_executed_tensor_mflop_per_request_ncu",
                "gpu_executed_scalar_mflop_per_request_ncu",
                "gpu_executed_tensor_share_pct_ncu",
            ),
        )
        self.assertEqual(
            self._overview_metric_names(ncu_runtime_spec),
            (
                "gpu_executed_mflops_app_ncu",
                "gpu_executed_mflops_packet_ncu",
                "gpu_kernel_launch_count_per_request_ncu",
                "gpu_kernel_time_sum_ms_per_request_ncu",
            ),
        )
        overview_metrics = {
            metric
            for spec in plot.METRIC_OVERVIEW_PLOTS
            for metric in self._overview_metric_names(spec)
        }
        self.assertNotIn("compute_mflops", overview_metrics)

    def test_thirteen_overviews_and_massif_cover_all_legacy_metrics_once(
        self,
    ) -> None:
        grouped_metrics = [
            metric
            for spec in plot.METRIC_OVERVIEW_PLOTS
            for metric in self._overview_metric_names(spec)
        ]
        standalone_metrics = [spec[0] for spec in plot.PLOT_METRICS]
        legacy_metrics = [spec[0] for spec in plot.LEGACY_PLOT_METRICS]

        self.assertEqual(len(plot.METRIC_OVERVIEW_PLOTS), 13)
        self.assertEqual(standalone_metrics, ["cpu_heap_peak_total_gib_massif"])
        self.assertEqual(len(grouped_metrics), 49)
        self.assertEqual(len(set(grouped_metrics + standalone_metrics)), 50)
        self.assertEqual(
            set(grouped_metrics + standalone_metrics),
            set(legacy_metrics),
        )
        self.assertTrue(
            all(
                rows * columns <= 6
                for _title, _filename, rows, columns, _panels, _shared
                in plot.METRIC_OVERVIEW_PLOTS
            )
        )
        self.assertEqual(
            {spec[1] for spec in plot.METRIC_OVERVIEW_PLOTS},
            {
                "latency_overview_vs_scale.png",
                "service_efficiency_overview_vs_scale.png",
                "packet_overview_vs_scale.png",
                "torch_compute_overview_vs_scale.png",
                "ncu_arithmetic_overview_vs_scale.png",
                "ncu_runtime_overview_vs_scale.png",
                "nsys_timing_overview_vs_scale.png",
                "container_cpu_overview_vs_scale.png",
                "cpu_execution_overview_vs_scale.png",
                "cpu_memory_behavior_overview_vs_scale.png",
                "container_memory_process_overview_vs_scale.png",
                "container_io_overview_vs_scale.png",
                "gpu_resource_overview_vs_scale.png",
            },
        )

    def test_metric_overview_handles_partial_data_and_empty_slot(self) -> None:
        df = pd.DataFrame([
            {
                "cpu_cores": 2,
                "mem_cap_gb": 4,
                "gpu_mode": "off",
                "input_scale": input_scale,
                "throughput_samples_per_s": throughput,
            }
            for input_scale, throughput in ((64, 5.0), (128, 7.0))
        ])
        title, _filename, rows, columns, panels, shared_y_groups = (
            self._overview_spec("service_efficiency_overview_vs_scale.png")
        )

        with patch.object(plot.plt, "close"):
            plot.plot_metric_overview(
                df,
                panels=panels,
                rows=rows,
                columns=columns,
                shared_y_groups=shared_y_groups,
                title=title,
                xlabel="input_scale",
                out_png=None,
            )
            figure = plot.plt.gcf()

        try:
            self.assertEqual(len(figure.axes), 6)
            self.assertEqual(
                list(figure.axes[0].lines[0].get_ydata()),
                [5.0, 7.0],
            )
            for axis in figure.axes[1:5]:
                self.assertEqual([text.get_text() for text in axis.texts], ["No data"])
            self.assertFalse(figure.axes[5].get_visible())
            self.assertEqual(figure.axes[4].get_xlabel(), "input_scale")
            self.assertEqual(figure.axes[3].get_xlabel(), "input_scale")
        finally:
            plot.plt.close(figure)

    def test_energy_power_overview_specs_are_registered(self) -> None:
        self.assertEqual(
            plot.ENERGY_POWER_OVERVIEW_PLOTS,
            [
                (
                    (
                        "gpu_energy_eff_j",
                        "gpu_avg_power_eff_w",
                        "gpu_peak_power_eff_w",
                    ),
                    (
                        "gpu_energy_total_j",
                        "gpu_avg_power_total_w",
                        "gpu_peak_power_total_w",
                    ),
                    "GPU Board Energy and Power Overview vs. Input Scale",
                    "gpu_energy_power_overview_vs_scale.png",
                ),
                (
                    (
                        "cpu_energy_eff_j",
                        "cpu_avg_power_eff_w",
                        "cpu_peak_power_eff_w",
                    ),
                    (
                        "cpu_energy_total_j",
                        "cpu_avg_power_total_w",
                        "cpu_peak_power_total_w",
                    ),
                    "CPU Package Energy and Power Overview vs. Input Scale",
                    "cpu_package_energy_power_overview_vs_scale.png",
                ),
                (
                    (
                        "vcpu_energy_eff_j",
                        "vcpu_avg_power_eff_w",
                        "vcpu_peak_power_eff_w",
                    ),
                    (
                        "vcpu_energy_total_j",
                        "vcpu_avg_power_total_w",
                        "vcpu_peak_power_total_w",
                    ),
                    (
                        "Estimated vCPU-Attributed Energy and Power Overview "
                        "vs. Input Scale"
                    ),
                    "vcpu_estimated_energy_power_overview_vs_scale.png",
                ),
            ],
        )
        single_metrics = {metric for metric, *_rest in plot.PLOT_METRICS}
        for effective_metrics, total_metrics, *_rest in (
            plot.ENERGY_POWER_OVERVIEW_PLOTS
        ):
            for metric in (*effective_metrics, *total_metrics):
                self.assertNotIn(metric, single_metrics)

    def test_energy_power_overview_contains_six_comparable_panels(self) -> None:
        df = pd.DataFrame([
            {
                "cpu_cores": 2,
                "mem_cap_gb": 4,
                "gpu_mode": "on",
                "input_scale": input_scale,
                "gpu_energy_eff_j": energy_eff,
                "gpu_avg_power_eff_w": avg_power_eff,
                "gpu_peak_power_eff_w": peak_power_eff,
                "gpu_energy_total_j": energy_total,
                "gpu_avg_power_total_w": avg_power_total,
                "gpu_peak_power_total_w": peak_power_total,
            }
            for (
                input_scale,
                energy_eff,
                avg_power_eff,
                peak_power_eff,
                energy_total,
                avg_power_total,
                peak_power_total,
            ) in (
                (64, 0.8, 20.0, 38.0, 1.0, 30.0, 48.0),
                (128, 1.2, 25.0, 45.0, 1.5, 35.0, 55.0),
            )
        ])

        with patch.object(plot.plt, "close"):
            plot.plot_energy_power_overview(
                df,
                effective_metrics=(
                    "gpu_energy_eff_j",
                    "gpu_avg_power_eff_w",
                    "gpu_peak_power_eff_w",
                ),
                total_metrics=(
                    "gpu_energy_total_j",
                    "gpu_avg_power_total_w",
                    "gpu_peak_power_total_w",
                ),
                title="GPU Board Energy and Power Overview vs. Input Scale",
                xlabel="input_scale",
                out_png=None,
            )
            figure = plot.plt.gcf()

        try:
            self.assertEqual(len(figure.axes), 6)
            self.assertEqual(
                [axis.get_title() for axis in figure.axes],
                [
                    "Effective Energy per Request",
                    "Total Energy per Request",
                    "Average Effective Power",
                    "Average Total Power",
                    "Peak Effective Power",
                    "Peak Total Power",
                ],
            )
            self.assertEqual(
                [list(axis.lines[0].get_ydata()) for axis in figure.axes],
                [
                    [0.8, 1.2],
                    [1.0, 1.5],
                    [20.0, 25.0],
                    [30.0, 35.0],
                    [38.0, 45.0],
                    [48.0, 55.0],
                ],
            )
            self.assertEqual(
                [axis.lines[0].get_marker() for axis in figure.axes],
                ["s", "s", "o", "o", "^", "^"],
            )
            self.assertEqual(
                len({axis.lines[0].get_color() for axis in figure.axes}),
                1,
            )
            for left_index, right_index in ((0, 1), (2, 3), (4, 5)):
                shared_y = figure.axes[left_index].get_shared_y_axes()
                self.assertTrue(
                    shared_y.joined(
                        figure.axes[left_index],
                        figure.axes[right_index],
                    )
                )
            self.assertEqual(
                [axis.get_xlabel() for axis in figure.axes[-2:]],
                ["input_scale", "input_scale"],
            )
            self.assertEqual(len(figure.legends), 1)
        finally:
            plot.plt.close(figure)

    def test_execution_profile_overview_and_massif_are_registered(self) -> None:
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
        self.assertEqual(len(metrics), 1)
        nsys_spec = self._overview_spec("nsys_timing_overview_vs_scale.png")
        self.assertEqual(
            self._overview_metric_names(nsys_spec),
            (
                "host_inference_wall_time_ms_per_request_nsys",
                "cuda_api_time_sum_ms_per_request_nsys",
                "gpu_kernel_time_sum_ms_per_request_nsys",
                "gpu_memcpy_time_sum_ms_per_request_nsys",
            ),
        )

    def test_execution_profile_overview_skips_when_all_panels_lack_data(self) -> None:
        df = pd.DataFrame([{
            "cpu_cores": 1,
            "mem_cap_gb": 4,
            "gpu_mode": "on",
            "input_scale": 64,
            "gpu_kernel_time_sum_ms_per_request_nsys": float("nan"),
        }])
        title, _filename, rows, columns, panels, shared_y_groups = (
            self._overview_spec("nsys_timing_overview_vs_scale.png")
        )

        with patch.object(plot.plt, "subplots") as subplots:
            plot.plot_metric_overview(
                df,
                panels=panels,
                rows=rows,
                columns=columns,
                shared_y_groups=shared_y_groups,
                title=title,
                xlabel="input_scale",
                out_png="unused.png",
            )

        subplots.assert_not_called()

    def test_bandwidth_behavior_overview_is_registered(self) -> None:
        spec = self._overview_spec(
            "cpu_memory_behavior_overview_vs_scale.png"
        )
        self.assertEqual(
            self._overview_metric_names(spec),
            (
                "cpu_cache_misses_per_request",
                "cpu_cache_miss_rate_pct",
                "cpu_dtlb_load_misses_per_request",
                "cpu_dtlb_load_miss_rate_pct",
            ),
        )
        self.assertEqual(spec[2:4], (2, 2))
        self.assertEqual(spec[5], ((0, 2), (1, 3)))

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

    def test_prepare_df_converts_total_energy_and_power_fields(self) -> None:
        total_fields = [
            "gpu_energy_total_j",
            "gpu_avg_power_total_w",
            "gpu_peak_power_total_w",
            "cpu_energy_total_j",
            "cpu_avg_power_total_w",
            "cpu_peak_power_total_w",
            "vcpu_energy_total_j",
            "vcpu_avg_power_total_w",
            "vcpu_peak_power_total_w",
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
                *total_fields,
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "on",
                        "input_scale": "64",
                        "warmup": "0",
                        "status": "ok",
                        **{field: "12.5" for field in total_fields},
                    },
                    {
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "on",
                        "input_scale": "128",
                        "warmup": "0",
                        "status": "ok",
                        **{field: "not-available" for field in total_fields},
                    },
                ])

            df = plot.prepare_df(csv_path)

        for field in total_fields:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[field]), field)
            self.assertEqual(float(df[field].iloc[0]), 12.5)
            self.assertTrue(pd.isna(df[field].iloc[1]))

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

    def test_main_generates_one_gpu_energy_power_overview(self) -> None:
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
                        "gpu_peak_power_eff_w",
                        "gpu_energy_eff_j",
                        "gpu_avg_power_total_w",
                        "gpu_peak_power_total_w",
                        "gpu_energy_total_j",
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
                    "gpu_peak_power_eff_w": "3.5",
                    "gpu_energy_eff_j": "0.25",
                    "gpu_avg_power_total_w": "12.5",
                    "gpu_peak_power_total_w": "13.5",
                    "gpu_energy_total_j": "1.25",
                })

            out_pngs = []

            def capture_plot_metric(*_args, **kwargs):
                out_pngs.append(os.path.basename(kwargs["out_png"]))

            with patch.object(sys, "argv", ["plot.py", csv_path]), patch.object(
                plot, "plot_metric", side_effect=capture_plot_metric
            ), patch.object(
                plot,
                "plot_metric_overview",
                side_effect=capture_plot_metric,
            ), patch.object(
                plot,
                "plot_energy_power_overview",
                side_effect=capture_plot_metric,
            ), patch.object(plot, "plot_cold_start_bar"):
                plot.main()

        self.assertIn("gpu_energy_power_overview_vs_scale.png", out_pngs)
        self.assertNotIn("gpu_effective_energy_power_vs_scale.png", out_pngs)
        self.assertNotIn("gpu_total_energy_power_vs_scale.png", out_pngs)
        self.assertNotIn("gpu_avg_power_vs_scale.png", out_pngs)
        self.assertNotIn("gpu_energy_vs_scale.png", out_pngs)
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

            def capture_plot_overview(group_df, *_args, **kwargs):
                metrics = {
                    panel[0]
                    for panel in kwargs["panels"]
                    if panel is not None
                }
                if "latency_s" not in metrics:
                    return
                output_directory = os.path.basename(
                    os.path.dirname(kwargs["out_png"])
                )
                modes_by_directory[output_directory] = set(group_df["gpu_mode"])

            with patch.object(
                plot,
                "plot_metric_overview",
                side_effect=capture_plot_overview,
            ), patch.object(
                plot, "plot_metric"
            ), patch.object(
                plot, "plot_energy_power_overview"
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

    def test_main_consolidates_cpu_execution_and_memory_behavior_metrics(
        self,
    ) -> None:
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
                plot,
                "plot_metric_overview",
                side_effect=capture_plot_metric,
            ), patch.object(plot, "plot_metric"), patch.object(
                plot, "plot_energy_power_overview"
            ), patch.object(plot, "plot_cold_start_bar"):
                plot.main()

        self.assertIn("cpu_execution_overview_vs_scale.png", out_pngs)
        self.assertIn("cpu_memory_behavior_overview_vs_scale.png", out_pngs)
        for legacy_filename in (
            "cpu_mips_packet_vs_scale.png",
            "cpu_instructions_per_request_vs_scale.png",
            "cpu_cycles_est_packet_vs_scale.png",
            "cpu_cache_misses_per_request_vs_scale.png",
            "cpu_cache_miss_rate_vs_scale.png",
            "cpu_dtlb_load_misses_per_request_vs_scale.png",
            "cpu_dtlb_load_miss_rate_vs_scale.png",
        ):
            self.assertNotIn(legacy_filename, out_pngs)


if __name__ == "__main__":
    unittest.main()

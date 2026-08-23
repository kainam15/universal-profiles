"""AC-Prof Universal Profiler - Plotting tool adapted for generalized CSV schema."""

import csv
import colorsys
import inspect
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "results/result_all.csv"
EXCLUDE_WARMUP = True
ONLY_OK = True
AGG_FUNC = "mean"
SAVE_PNG = True
SHOW_PLOTS = False
GPU_GREEN = (0.18, 0.62, 0.28)
GPU_LIGHT_GREEN = (0.49, 0.78, 0.53)
GPU_MODE_ON_VALUES = {"on", "1", "1.0", "true", "yes", "gpu"}
GPU_MODE_OFF_VALUES = {"off", "0", "0.0", "false", "no", "cpu"}
BYTES_PER_GIB = 1024 ** 3
CPU_FIXED_COLORS = {
    1: (0.12, 0.47, 0.71),
    2: (0.93, 0.69, 0.13),
    4: (0.84, 0.15, 0.16),
    8: (0.58, 0.40, 0.74),
}
MEM_FIXED_COLORS = {
    2: (0.12, 0.47, 0.71),
    4: (0.93, 0.69, 0.13),
    8: (0.84, 0.15, 0.16),
    16: (0.58, 0.40, 0.74),
}
MEM_COLORED_METRICS = {"container_mem_util_avg_pct"}
GPU_METRIC_ALIASES = {
    "energy_iters": "gpu_energy_iters",
    "avg_power_total_w": "gpu_avg_power_total_w",
    "peak_power_total_w": "gpu_peak_power_total_w",
    "energy_total_j": "gpu_energy_total_j",
    "avg_power_eff_w": "gpu_avg_power_eff_w",
    "peak_power_eff_w": "gpu_peak_power_eff_w",
    "energy_eff_j": "gpu_energy_eff_j",
}
TORCH_EAGER_COMPUTE_LEGACY_FALLBACKS = {
    "model_logical_mflop_per_request_torch_profiler_eager": (
        "model_mflop_per_request"
    ),
    "model_logical_mflops_app_torch_profiler_eager": "compute_mflops_app",
    "model_logical_mflops_packet_torch_profiler_eager": "compute_mflops",
}
NCU_COMPUTE_LEGACY_FALLBACKS = {
    "gpu_executed_mflop_per_request_ncu": "model_mflop_per_request",
    "gpu_executed_mflops_app_ncu": "compute_mflops_app",
    "gpu_executed_mflops_packet_ncu": "compute_mflops",
}
COMPUTE_NUMERIC_COLUMNS = [
    "model_mflop_per_request",
    "compute_mflops_app",
    "compute_mflops",
    *TORCH_EAGER_COMPUTE_LEGACY_FALLBACKS,
    "gpu_executed_mflop_per_request_ncu",
    "gpu_executed_tensor_mflop_per_request_ncu",
    "gpu_executed_scalar_mflop_per_request_ncu",
    "gpu_executed_tensor_share_pct_ncu",
    "gpu_executed_mflops_app_ncu",
    "gpu_executed_mflops_packet_ncu",
    "gpu_kernel_launch_count_per_request_ncu",
    "gpu_kernel_time_sum_ms_per_request_ncu",
]
EXECUTION_PROFILE_NUMERIC_COLUMNS = [
    "cpu_heap_peak_bytes_massif",
    "cpu_heap_extra_peak_bytes_massif",
    "cpu_stack_peak_bytes_massif",
    "cpu_heap_peak_total_bytes_massif",
    "cpu_heap_peak_at_ms_massif",
    "host_inference_wall_time_ms_per_request_nsys",
    "cuda_api_time_sum_ms_per_request_nsys",
    "cuda_api_call_count_per_request_nsys",
    "gpu_kernel_time_sum_ms_per_request_nsys",
    "gpu_kernel_launch_count_per_request_nsys",
    "gpu_memcpy_time_sum_ms_per_request_nsys",
    "gpu_memcpy_count_per_request_nsys",
    "gpu_memcpy_bytes_per_request_nsys",
]
PLOT_METRICS = [
    (
        "latency_s",
        "Latency vs. Input Scale",
        "Latency (s)",
        "latency_vs_scale.png",
    ),
    (
        "latency_app_s",
        "App-Level Latency vs. Input Scale",
        "Latency (s)",
        "latency_app_vs_scale.png",
    ),
    (
        "latency_cv",
        "Packet Latency Variability vs. Input Scale",
        "Coefficient of variation",
        "latency_cv_vs_scale.png",
    ),
    (
        "latency_app_cv",
        "App-Level Latency Variability vs. Input Scale",
        "Coefficient of variation",
        "latency_app_cv_vs_scale.png",
    ),
    (
        "gpu_avg_power_eff_w",
        "GPU Average Effective Power vs. Input Scale",
        "Power (W)",
        "gpu_avg_power_vs_scale.png",
    ),
    (
        "gpu_energy_eff_j",
        "GPU Effective Energy vs. Input Scale",
        "Energy (J)",
        "gpu_energy_vs_scale.png",
    ),
    (
        "cpu_avg_power_eff_w",
        "CPU Package Average Effective Power vs. Input Scale",
        "Power (W)",
        "cpu_avg_power_vs_scale.png",
    ),
    (
        "cpu_energy_eff_j",
        "CPU Package Effective Energy vs. Input Scale",
        "Energy (J)",
        "cpu_energy_vs_scale.png",
    ),
    (
        "vcpu_avg_power_eff_w",
        "Estimated vCPU Average Effective Power vs. Input Scale",
        "Power (W)",
        "vcpu_avg_power_vs_scale.png",
    ),
    (
        "vcpu_energy_eff_j",
        "Estimated vCPU Effective Energy vs. Input Scale",
        "Energy (J)",
        "vcpu_energy_vs_scale.png",
    ),
    (
        "container_attributed_energy_eff_j",
        "Container-Attributed Effective Energy vs. Input Scale",
        "Estimated energy (J/request)",
        "container_attributed_energy_vs_scale.png",
    ),
    (
        "container_attributed_samples_per_j",
        "Container-Attributed Energy Efficiency vs. Input Scale",
        "Samples/J",
        "container_attributed_samples_per_j_vs_scale.png",
    ),
    (
        "throughput_samples_per_s",
        "Throughput vs. Input Scale",
        "Samples/s",
        "throughput_vs_scale.png",
    ),
    (
        "model_logical_mflop_per_request_torch_profiler_eager",
        "Torch Profiler Eager Logical FLOP per Request vs. Input Scale",
        "Logical MFLOP/request",
        "torch_profiler_eager_logical_mflop_per_request_vs_scale.png",
    ),
    (
        "model_logical_mflops_app_torch_profiler_eager",
        (
            "Torch Profiler Eager Logical Compute Throughput "
            "(Application Latency) vs. Input Scale"
        ),
        "Logical MFLOPS (application latency)",
        "torch_profiler_eager_logical_mflops_app_vs_scale.png",
    ),
    (
        "model_logical_mflops_packet_torch_profiler_eager",
        (
            "Torch Profiler Eager Logical Compute Throughput "
            "(Packet Latency) vs. Input Scale"
        ),
        "Logical MFLOPS (packet latency)",
        "torch_profiler_eager_logical_mflops_packet_vs_scale.png",
    ),
    (
        "gpu_executed_mflop_per_request_ncu",
        "NCU GPU-Executed FLOP per Request vs. Input Scale",
        "GPU-executed MFLOP/request",
        "ncu_gpu_executed_mflop_per_request_vs_scale.png",
    ),
    (
        "gpu_executed_tensor_mflop_per_request_ncu",
        "NCU GPU-Executed Tensor FLOP per Request vs. Input Scale",
        "GPU-executed tensor MFLOP/request",
        "ncu_gpu_executed_tensor_mflop_per_request_vs_scale.png",
    ),
    (
        "gpu_executed_scalar_mflop_per_request_ncu",
        "NCU GPU-Executed Scalar FLOP per Request vs. Input Scale",
        "GPU-executed scalar MFLOP/request",
        "ncu_gpu_executed_scalar_mflop_per_request_vs_scale.png",
    ),
    (
        "gpu_executed_tensor_share_pct_ncu",
        "NCU GPU-Executed Tensor FLOP Share vs. Input Scale",
        "Tensor FLOP share (%)",
        "ncu_gpu_executed_tensor_share_vs_scale.png",
    ),
    (
        "gpu_executed_mflops_app_ncu",
        (
            "NCU GPU-Executed Compute Throughput "
            "(Application Latency) vs. Input Scale"
        ),
        "GPU-executed MFLOPS (application latency)",
        "ncu_gpu_executed_mflops_app_vs_scale.png",
    ),
    (
        "gpu_executed_mflops_packet_ncu",
        (
            "NCU GPU-Executed Compute Throughput "
            "(Packet Latency) vs. Input Scale"
        ),
        "GPU-executed MFLOPS (packet latency)",
        "ncu_gpu_executed_mflops_packet_vs_scale.png",
    ),
    (
        "gpu_kernel_launch_count_per_request_ncu",
        "NCU GPU Kernel Launches per Request vs. Input Scale",
        "Kernel launches/request",
        "ncu_gpu_kernel_launch_count_per_request_vs_scale.png",
    ),
    (
        "gpu_kernel_time_sum_ms_per_request_ncu",
        "NCU GPU Kernel Time per Request vs. Input Scale",
        "Summed kernel time (ms/request)",
        "ncu_gpu_kernel_time_sum_ms_per_request_vs_scale.png",
    ),
    (
        "cpu_heap_peak_total_gib_massif",
        "Massif Process-Lifetime Peak Memory vs. Input Scale",
        "Peak heap + extra + stack (GiB)",
        "massif_cpu_heap_peak_total_vs_scale.png",
    ),
    (
        "host_inference_wall_time_ms_per_request_nsys",
        "Nsight Systems Host Inference Wall Time vs. Input Scale",
        "Host wall time (ms/request)",
        "nsys_host_inference_wall_time_per_request_vs_scale.png",
    ),
    (
        "cuda_api_time_sum_ms_per_request_nsys",
        "Nsight Systems CUDA API Time vs. Input Scale",
        "Summed CUDA API time (ms/request)",
        "nsys_cuda_api_time_sum_per_request_vs_scale.png",
    ),
    (
        "gpu_kernel_time_sum_ms_per_request_nsys",
        "Nsight Systems GPU Kernel Time vs. Input Scale",
        "Summed GPU kernel time (ms/request)",
        "nsys_gpu_kernel_time_sum_per_request_vs_scale.png",
    ),
    (
        "gpu_memcpy_time_sum_ms_per_request_nsys",
        "Nsight Systems GPU Memcpy Time vs. Input Scale",
        "Summed GPU memcpy time (ms/request)",
        "nsys_gpu_memcpy_time_sum_per_request_vs_scale.png",
    ),
    (
        "container_cpu_util_avg_pct",
        "Container CPU Utilization vs. Input Scale",
        "CPU Utilization (%)",
        "container_cpu_util_vs_scale.png",
    ),
    (
        "container_cpu_throttled_period_ratio_pct",
        "Container CPU Throttled Period Ratio vs. Input Scale",
        "Throttled periods (%)",
        "container_cpu_throttled_period_ratio_vs_scale.png",
    ),
    (
        "container_cpu_pressure_some_stall_pct",
        "Container CPU Pressure Stall vs. Input Scale",
        "PSI some stall (%)",
        "container_cpu_pressure_some_vs_scale.png",
    ),
    (
        "cpu_mips_packet",
        "Packet-Latency CPU MIPS vs. Input Scale",
        "MIPS",
        "cpu_mips_packet_vs_scale.png",
    ),
    (
        "cpu_instructions_per_request",
        "CPU Instructions per Request vs. Input Scale",
        "Instructions/request",
        "cpu_instructions_per_request_vs_scale.png",
    ),
    (
        "cpu_cache_misses_per_request",
        "CPU Cache Misses per Request vs. Input Scale",
        "Cache misses/request",
        "cpu_cache_misses_per_request_vs_scale.png",
    ),
    (
        "cpu_cache_miss_rate_pct",
        "CPU Cache Miss Rate vs. Input Scale",
        "Cache miss rate (%)",
        "cpu_cache_miss_rate_vs_scale.png",
    ),
    (
        "cpu_dtlb_load_misses_per_request",
        "CPU dTLB Load Misses per Request vs. Input Scale",
        "dTLB load misses/request",
        "cpu_dtlb_load_misses_per_request_vs_scale.png",
    ),
    (
        "cpu_dtlb_load_miss_rate_pct",
        "CPU dTLB Load Miss Rate vs. Input Scale",
        "dTLB load miss rate (%)",
        "cpu_dtlb_load_miss_rate_vs_scale.png",
    ),
    (
        "cpu_cycles_est_packet",
        "Packet-Latency Estimated CPU Cycles vs. Input Scale",
        "Estimated CPU cycles/request",
        "cpu_cycles_est_packet_vs_scale.png",
    ),
    (
        "container_mem_util_avg_pct",
        "Container Memory Utilization vs. Input Scale",
        "Memory Utilization (%)",
        "container_mem_util_vs_scale.png",
    ),
    (
        "container_mem_pressure_full_stall_pct",
        "Container Memory Full-Pressure Stall vs. Input Scale",
        "PSI full stall (%)",
        "container_mem_pressure_full_vs_scale.png",
    ),
    (
        "container_mem_usage_avg_gib",
        "Container Memory Usage vs. Input Scale",
        "Memory Usage (GiB)",
        "container_mem_usage_vs_scale.png",
    ),
    (
        "gpu_util_avg_pct",
        "GPU Utilization vs. Input Scale",
        "GPU Utilization (%)",
        "gpu_util_vs_scale.png",
    ),
    (
        "gpu_mem_util_avg_pct",
        "GPU Memory Utilization vs. Input Scale",
        "GPU Memory Utilization (%)",
        "gpu_mem_util_vs_scale.png",
    ),
    (
        "gpu_mem_used_avg_gib",
        "GPU Memory Used vs. Input Scale",
        "GPU Memory Used (GiB)",
        "gpu_mem_used_vs_scale.png",
    ),
]
PLOT_OUTPUT_DIRS = ("cpu", "gpu", "gpu+cpu")
LATENCY_MODEL_DIR = "latency_model"
LATENCY_MODEL_REPORT = "latency_model_report.json"
LATENCY_MODEL_RESIDUALS = "latency_model_residuals.csv"
LATENCY_MODEL_RESIDUAL_PLOT = "latency_model_residuals.png"
LATENCY_MODEL_FIT_CURVES_PLOT = "latency_model_fit_curves.png"
LATENCY_MODEL_FEATURES = {
    "cpu": [
        "intercept",
        "log_input_scale",
        "log_input_scale_squared",
        "log_cpu_cores",
        "log_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_log_cpu_cores",
        "log_input_scale_x_log_mem_cap_gb",
        "log_cpu_cores_x_log_mem_cap_gb",
    ],
    "gpu": [
        "intercept",
        "log_input_scale",
        "inverse_cpu_cores",
        "inverse_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_inverse_cpu_cores",
        "log_input_scale_x_inverse_cpu_cores_squared",
    ],
}
LATENCY_MODEL_RESOURCE_FEATURES = {
    "cpu": {
        "log_cpu_cores",
        "log_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_log_cpu_cores",
        "log_input_scale_x_log_mem_cap_gb",
        "log_cpu_cores_x_log_mem_cap_gb",
    },
    "gpu": {
        "inverse_cpu_cores",
        "inverse_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_inverse_cpu_cores",
        "log_input_scale_x_inverse_cpu_cores_squared",
    },
}
LATENCY_MODEL_FORMULAS = {
    "cpu": (
        "latency_s = exp(intercept + log_input_scale "
        "+ log_input_scale_squared "
        "+ log_cpu_cores + log_cpu_cores_squared + log_mem_cap_gb "
        "+ log_input_scale_x_log_cpu_cores "
        "+ log_input_scale_x_log_mem_cap_gb "
        "+ log_cpu_cores_x_log_mem_cap_gb)"
    ),
    "gpu": (
        "within-range latency_s = exp(intercept + log_input_scale "
        "+ piecewise_linear_log_input_scale_hinges "
        "+ inverse_cpu_cores + inverse_cpu_cores_squared "
        "+ log_mem_cap_gb + log_input_scale_x_inverse_cpu_cores "
        "+ log_input_scale_x_inverse_cpu_cores_squared); when activated, "
        "upper-tail latency_s = boundary_spline_latency_s + "
        "affine_tail(input_scale) - affine_tail(boundary_input_scale)"
    ),
}
LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES = [
    "intercept",
    "input_scale",
    "inverse_cpu_cores",
    "inverse_cpu_cores_squared",
    "log_mem_cap_gb",
    "input_scale_x_inverse_cpu_cores",
    "input_scale_x_inverse_cpu_cores_squared",
]
# A GPU log-spline is excellent inside the profiled scale range, but extending
# one noisy boundary segment can be brittle.  When a nested one-step-forward
# check exceeds this MAPE, upper extrapolation switches to a continuous affine
# work tail fitted in latency space.
LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE = 0.05
LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO = 10.0
LATENCY_MODEL_MIN_VALIDATION_R2 = 0.80
LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE = 0.20
LATENCY_MODEL_MAX_VALIDATION_MAPE = 0.20
LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE = 0.30
LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE = 0.30
LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR = 0.30
# Compatibility alias retained for callers that used the original max-scale-only
# threshold name.
LATENCY_MODEL_MAX_SCALE_CASE_RELATIVE_ERROR = (
    LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR
)
LATENCY_MODEL_MIN_VALIDATION_POINTS = 4
LATENCY_MODEL_RESIDUAL_FIELDS = [
    "report_schema_version",
    "case_id",
    "split",
    "hardware_model",
    "cpu_cores",
    "mem_cap_gb",
    "gpu_mode",
    "input_scale",
    "repeat_count",
    "latency_s",
    "latency_mean_s",
    "latency_std_s",
    "predicted_latency_s",
    "residual_s",
    "fitted_predicted_latency_s",
    "fitted_residual_s",
    "resource_config_oof_predicted_latency_s",
    "resource_config_oof_residual_s",
    "max_scale_holdout_predicted_latency_s",
    "max_scale_holdout_residual_s",
]


def make_config_label(row) -> str:
    cpu = int(row["cpu_cores"])
    mem = int(row["mem_cap_gb"])
    if is_gpu_on(row["gpu_mode"]):
        return f"GPU+CPU{cpu}+Mem{mem}"
    else:
        return f"CPU+CPU{cpu}+Mem{mem}"


def is_gpu_on(gpu_mode: object) -> bool:
    return str(gpu_mode).strip().lower() in GPU_MODE_ON_VALUES


def is_gpu_off(gpu_mode: object) -> bool:
    return str(gpu_mode).strip().lower() in GPU_MODE_OFF_VALUES


def build_plot_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split plot inputs into CPU-only, GPU-only, and comparison groups."""
    gpu_on = df["gpu_mode"].map(is_gpu_on)
    gpu_off = df["gpu_mode"].map(is_gpu_off)
    cpu_df = df[gpu_off].copy()
    gpu_df = df[gpu_on].copy()
    combined_df = (
        df[gpu_off | gpu_on].copy()
        if not cpu_df.empty and not gpu_df.empty
        else df.iloc[0:0].copy()
    )
    return list(zip(PLOT_OUTPUT_DIRS, (cpu_df, gpu_df, combined_df)))


def prepare_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    static_meta = read_static_meta(csv_path)
    if "gpu_mem_total_bytes" not in df.columns and static_meta.get("gpu_mem_total_bytes"):
        df["gpu_mem_total_bytes"] = pd.to_numeric(
            static_meta["gpu_mem_total_bytes"],
            errors="coerce",
        )

    for old_name, new_name in GPU_METRIC_ALIASES.items():
        if new_name not in df.columns and old_name in df.columns:
            df[new_name] = df[old_name]

    num_cols = [
        "input_scale",
        "latency_s", "latency_request_count",
        "latency_p50_s", "latency_p90_s", "latency_p95_s",
        "latency_std_s", "latency_cv", "latency_iqr_s", "latency_max_s",
        "latency_slow_ratio",
        "latency_app_s", "latency_app_request_count",
        "latency_app_p50_s", "latency_app_p90_s", "latency_app_p95_s",
        "latency_app_std_s", "latency_app_cv", "latency_app_iqr_s",
        "latency_app_max_s", "latency_app_slow_ratio",
        "gpu_energy_iters",
        "gpu_avg_power_total_w", "gpu_peak_power_total_w", "gpu_energy_total_j",
        "gpu_avg_power_eff_w", "gpu_peak_power_eff_w", "gpu_energy_eff_j",
        "cpu_avg_power_eff_w", "cpu_peak_power_eff_w", "cpu_energy_eff_j",
        "vcpu_avg_power_eff_w", "vcpu_peak_power_eff_w", "vcpu_energy_eff_j",
        "vcpu_cpu_share", "vcpu_cpu_time_s",
        "container_attributed_energy_eff_j",
        "container_attributed_samples_per_j",
        "container_attributed_edp_app_js",
        "output_tokens_per_s_app",
        "container_attributed_j_per_output_token",
        "resource_usage_iters",
        "container_cpu_util_avg_pct", "container_cpu_util_peak_pct",
        "container_cpu_nr_periods_delta",
        "container_cpu_nr_throttled_delta",
        "container_cpu_throttled_period_ratio_pct",
        "container_cpu_throttled_time_s_per_request",
        "container_cpu_pressure_some_stall_pct",
        "container_cpu_pressure_full_stall_pct",
        "cpu_freq_avg_hz", "cpu_freq_peak_hz",
        "cpu_cycles_est_app", "cpu_cycles_est_packet",
        "cpu_instructions_per_request",
        "cpu_cache_references_per_request", "cpu_cache_misses_per_request",
        "cpu_cache_miss_rate_pct",
        "cpu_dtlb_loads_per_request", "cpu_dtlb_load_misses_per_request",
        "cpu_dtlb_load_miss_rate_pct",
        "cpu_mips_app", "cpu_mips_packet",
        "cpu_perf_elapsed_s",
        "container_mem_usage_avg_bytes", "container_mem_usage_peak_bytes",
        "container_mem_util_avg_pct", "container_mem_util_peak_pct",
        "container_mem_high_events_delta", "container_mem_max_events_delta",
        "container_mem_oom_events_delta",
        "container_mem_oom_kill_events_delta",
        "container_mem_pressure_some_stall_pct",
        "container_mem_pressure_full_stall_pct",
        "container_swap_limit_bytes",
        "container_swap_usage_avg_bytes", "container_swap_usage_peak_bytes",
        "container_io_read_bytes_per_request",
        "container_io_write_bytes_per_request",
        "container_io_pressure_some_stall_pct",
        "container_io_pressure_full_stall_pct",
        "gpu_sm_clock_mhz", "gpu_memory_clock_mhz",
        "gpu_temp_c",
        "gpu_util_avg_pct", "gpu_util_peak_pct",
        "gpu_mem_used_avg_bytes", "gpu_mem_used_peak_bytes",
        "gpu_mem_util_avg_pct", "gpu_mem_util_peak_pct",
        "gpu_mem_total_bytes",
        "cpu_cores", "mem_cap_gb", "warmup", "cold_start_s",
        "throughput_samples_per_s",
        *COMPUTE_NUMERIC_COLUMNS,
        *EXECUTION_PROFILE_NUMERIC_COLUMNS,
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "compute_profile_tool" in df.columns:
        profile_tools = (
            df["compute_profile_tool"].fillna("").astype(str).str.strip().str.lower()
        )
        torch_fallback_rows = profile_tools.str.contains("torch")
        ncu_fallback_rows = (
            profile_tools.eq("ncu") | profile_tools.str.contains("nsight")
        )
    else:
        # Very old CSVs did not identify the profiler. Preserve their previous
        # generic-as-logical fallback, but do not label the same data as NCU.
        torch_fallback_rows = pd.Series(True, index=df.index)
        ncu_fallback_rows = pd.Series(False, index=df.index)

    def _fill_profile_specific_fallbacks(fallbacks, eligible_rows):
        for new_name, legacy_name in fallbacks.items():
            if legacy_name not in df.columns:
                continue
            if new_name not in df.columns:
                df[new_name] = np.nan
            missing_rows = df[new_name].isna() & eligible_rows
            df.loc[missing_rows, new_name] = df.loc[missing_rows, legacy_name]

    _fill_profile_specific_fallbacks(
        TORCH_EAGER_COMPUTE_LEGACY_FALLBACKS,
        torch_fallback_rows,
    )
    _fill_profile_specific_fallbacks(
        NCU_COMPUTE_LEGACY_FALLBACKS,
        ncu_fallback_rows,
    )

    bytes_to_gib = {
        "container_mem_usage_avg_bytes": "container_mem_usage_avg_gib",
        "container_mem_usage_peak_bytes": "container_mem_usage_peak_gib",
        "container_swap_limit_bytes": "container_swap_limit_gib",
        "container_swap_usage_avg_bytes": "container_swap_usage_avg_gib",
        "container_swap_usage_peak_bytes": "container_swap_usage_peak_gib",
        "gpu_mem_used_avg_bytes": "gpu_mem_used_avg_gib",
        "gpu_mem_used_peak_bytes": "gpu_mem_used_peak_gib",
        "gpu_mem_total_bytes": "gpu_mem_total_gib",
        "cpu_heap_peak_total_bytes_massif": "cpu_heap_peak_total_gib_massif",
    }
    for source_col, target_col in bytes_to_gib.items():
        if source_col in df.columns:
            df[target_col] = df[source_col] / float(BYTES_PER_GIB)

    if ONLY_OK and "status" in df.columns:
        normalized_status = df["status"].astype(str).str.strip().str.lower()
        df = df[normalized_status == "ok"].copy()

    if EXCLUDE_WARMUP and "warmup" in df.columns:
        df = df[df["warmup"].fillna(0).astype(int) == 0].copy()

    need = {"input_scale", "cpu_cores", "mem_cap_gb", "gpu_mode"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if df.empty:
        df["config"] = pd.Series(index=df.index, dtype="object")
    else:
        df["config"] = df.apply(make_config_label, axis=1)
    df = df[df["input_scale"].notna() & (df["input_scale"] > 0)].copy()

    return df


def read_static_meta(csv_path: str) -> dict[str, object]:
    result_dir = os.path.dirname(csv_path) or "."
    static_meta_json = os.path.join(result_dir, "static_meta.json")
    if os.path.exists(static_meta_json):
        with open(static_meta_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(
                f"static metadata must be a JSON object: {static_meta_json}"
            )
        return payload

    # Historical result directories used a one-row CSV.
    legacy_csv = os.path.join(result_dir, "static_meta.csv")
    if not os.path.exists(legacy_csv):
        return {}
    with open(legacy_csv, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f), None)
    return row or {}


def _sort_key(config: tuple[int, int, bool]) -> tuple[int, int, int]:
    cpu, mem, gpu_on = config
    return 0 if gpu_on else 1, cpu, mem


def build_cpu_base_colors(cpu_values: list[int]) -> dict[int, tuple[float, float, float]]:
    color_map = plt.get_cmap("tab10")
    cpu_colors = {}
    for index, cpu in enumerate(cpu_values):
        cpu_colors[cpu] = CPU_FIXED_COLORS.get(cpu, color_map(index % color_map.N)[:3])
    return cpu_colors


def shade_for_mem(
    base_rgb: tuple[float, float, float],
    mem_rank: int,
    mem_count: int,
) -> tuple[float, float, float]:
    red, green, blue = base_rgb
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    if mem_count <= 1:
        target_lightness = lightness
    else:
        min_lightness = 0.30
        max_lightness = 0.78
        ratio = mem_rank / (mem_count - 1)
        target_lightness = max_lightness - ratio * (max_lightness - min_lightness)

    return colorsys.hls_to_rgb(hue, target_lightness, saturation)


def shade_for_cpu(
    base_rgb: tuple[float, float, float],
    cpu_rank: int,
    cpu_count: int,
) -> tuple[float, float, float]:
    """Keep the base hue while making larger CPU configurations darker."""
    if cpu_count <= 1:
        return base_rgb
    return shade_for_mem(base_rgb, cpu_rank, cpu_count)


def build_gpu_mixed_colors(
    configs: list[tuple[int, int, bool]],
) -> dict[tuple[int, int, bool], tuple[float, float, float]]:
    gpu_configs = [config for config in configs if config[2]]
    gpu_cpus = sorted({cpu for cpu, _, gpu_on in gpu_configs if gpu_on})
    if not gpu_configs:
        return {}

    cpu_rank_map = {cpu: index for index, cpu in enumerate(gpu_cpus)}
    return {
        config: shade_for_mem(GPU_GREEN, cpu_rank_map[config[0]], len(gpu_cpus))
        for config in gpu_configs
    }


def color_for_mem(mem: int, mem_rank: int) -> tuple[float, float, float]:
    if mem in MEM_FIXED_COLORS:
        return MEM_FIXED_COLORS[mem]

    color_map = plt.get_cmap("tab10")
    return color_map(mem_rank % color_map.N)[:3]


def aggregate_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]
    metric_df = df[group_cols + [metric]].copy()
    metric_df = metric_df[metric_df[metric].notna()].copy()
    if metric_df.empty:
        return metric_df

    if AGG_FUNC == "median":
        agg_df = metric_df.groupby(group_cols, as_index=False)[metric].median()
    else:
        agg_df = metric_df.groupby(group_cols, as_index=False)[metric].mean()

    agg_df["config"] = agg_df.apply(make_config_label, axis=1)
    agg_df["gpu_on"] = agg_df["gpu_mode"].map(is_gpu_on)
    return agg_df


def aggregate_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode"]
    metric_df = df[group_cols + ["cold_start_s"]].copy()
    metric_df = metric_df[metric_df["cold_start_s"].notna()].copy()
    if metric_df.empty:
        return metric_df

    if AGG_FUNC == "median":
        agg_df = metric_df.groupby(group_cols, as_index=False)["cold_start_s"].median()
    else:
        agg_df = metric_df.groupby(group_cols, as_index=False)["cold_start_s"].mean()

    agg_df["config"] = agg_df.apply(make_config_label, axis=1)
    agg_df["gpu_on"] = agg_df["gpu_mode"].map(is_gpu_on)
    return agg_df


def plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, xlabel: str, out_png: str | None):
    if metric not in df.columns:
        print(f"[skip] Column {metric} not in CSV")
        return

    agg_df = aggregate_metric(df, metric)
    if agg_df.empty:
        print(f"[skip] {metric} all NaN")
        return

    cpu_values = sorted(int(value) for value in agg_df["cpu_cores"].unique())
    mem_values = sorted(int(value) for value in agg_df["mem_cap_gb"].unique())
    cpu_colors = build_cpu_base_colors(cpu_values)
    cpu_rank_map = {cpu: index for index, cpu in enumerate(cpu_values)}
    mem_rank_map = {mem: index for index, mem in enumerate(mem_values)}
    color_by_mem = metric in MEM_COLORED_METRICS

    configs = sorted(
        {
            (int(row.cpu_cores), int(row.mem_cap_gb), bool(row.gpu_on))
            for row in agg_df.itertuples(index=False)
        },
        key=_sort_key,
    )
    has_cpu_series = any(not gpu_on for _, _, gpu_on in configs)
    gpu_mixed_colors = build_gpu_mixed_colors(configs) if has_cpu_series else {}

    plt.figure(figsize=(10, 6))
    for cpu, mem, gpu_on in configs:
        sub_df = agg_df[
            (agg_df["cpu_cores"] == cpu)
            & (agg_df["mem_cap_gb"] == mem)
            & (agg_df["gpu_on"] == gpu_on)
        ].sort_values("input_scale")

        if color_by_mem:
            base_color = color_for_mem(mem, mem_rank_map[mem])
            color = shade_for_cpu(base_color, cpu_rank_map[cpu], len(cpu_values))
        elif gpu_on and has_cpu_series:
            color = gpu_mixed_colors[(cpu, mem, gpu_on)]
        else:
            color = shade_for_mem(cpu_colors[cpu], mem_rank_map[mem], len(mem_values))

        label_prefix = "GPU" if gpu_on else "CPU"

        plt.plot(
            sub_df["input_scale"],
            sub_df[metric],
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"{label_prefix}+CPU{cpu}+Mem{mem}",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="-", alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")

    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_cold_start_bar(df: pd.DataFrame, title: str, ylabel: str, out_png: str | None):
    if "cold_start_s" not in df.columns:
        print("[skip] Column cold_start_s not in CSV")
        return

    agg_df = aggregate_cold_start(df)
    if agg_df.empty:
        print("[skip] cold_start_s all NaN")
        return

    cpu_values = sorted(int(value) for value in agg_df["cpu_cores"].unique())
    mem_values = sorted(int(value) for value in agg_df["mem_cap_gb"].unique())
    cpu_colors = build_cpu_base_colors(cpu_values)
    mem_rank_map = {mem: index for index, mem in enumerate(mem_values)}

    configs = sorted(
        [
            (int(row.cpu_cores), int(row.mem_cap_gb), bool(row.gpu_on), row.config, float(row.cold_start_s))
            for row in agg_df.itertuples(index=False)
        ],
        key=lambda item: _sort_key(item[:3]),
    )
    has_cpu_series = any(not gpu_on for _, _, gpu_on, _, _ in configs)
    gpu_mixed_colors = build_gpu_mixed_colors([item[:3] for item in configs]) if has_cpu_series else {}

    labels = []
    values = []
    colors = []
    for cpu, mem, gpu_on, label, cold_start_s in configs:
        if gpu_on and has_cpu_series:
            color = gpu_mixed_colors[(cpu, mem, gpu_on)]
        else:
            color = shade_for_mem(cpu_colors[cpu], mem_rank_map[mem], len(mem_values))

        labels.append(label)
        values.append(cold_start_s)
        colors.append(color)

    plt.figure(figsize=(max(10, len(labels) * 0.55), 6))
    plt.bar(labels, values, color=colors)
    plt.title(title)
    plt.xlabel("Configuration")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="-", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")

    if SHOW_PLOTS:
        plt.show()
    plt.close()


def _residual_plot_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[float | None, float, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    residual_values = actual_values - predicted_values
    centered_values = actual_values - float(np.mean(actual_values))
    total_sum_squares = float(np.dot(centered_values, centered_values))
    if total_sum_squares > 0.0:
        r2 = 1.0 - (
            float(np.dot(residual_values, residual_values)) / total_sum_squares
        )
    else:
        r2 = None

    mean_actual = float(np.mean(np.abs(actual_values)))
    relative_mae = (
        float(np.mean(np.abs(residual_values))) / mean_actual
        if mean_actual > 0.0
        else math.nan
    )
    mean_absolute_percentage_error = float(
        np.mean(np.abs(residual_values) / np.abs(actual_values))
    )
    return r2, relative_mae, mean_absolute_percentage_error


def plot_latency_model_residuals(
    residuals_path: str,
    out_png: str | None,
) -> bool:
    """Plot OOF latency residual diagnostics from a model residual artifact."""
    if not os.path.exists(residuals_path):
        print(f"[skip] Cannot find {residuals_path}")
        return False

    residual_df = pd.read_csv(residuals_path, skipinitialspace=True)
    required_columns = {
        "hardware_model",
        "input_scale",
        "latency_s",
        "predicted_latency_s",
        "residual_s",
    }
    missing_columns = sorted(required_columns.difference(residual_df.columns))
    if missing_columns:
        print(
            "[skip] Latency residual CSV missing required columns: "
            f"{missing_columns}"
        )
        return False
    if residual_df.empty:
        print(f"[skip] No latency residual rows in {residuals_path}")
        return False

    numeric_columns = [
        "input_scale",
        "latency_s",
        "predicted_latency_s",
        "residual_s",
        "max_scale_holdout_predicted_latency_s",
        "max_scale_holdout_residual_s",
    ]
    for column in numeric_columns:
        if column in residual_df.columns:
            residual_df[column] = pd.to_numeric(
                residual_df[column],
                errors="coerce",
            )

    residual_df["hardware_model"] = (
        residual_df["hardware_model"].astype(str).str.strip().str.lower()
    )
    valid_mask = (
        residual_df[
            [
                "input_scale",
                "latency_s",
                "predicted_latency_s",
                "residual_s",
            ]
        ]
        .apply(np.isfinite)
        .all(axis=1)
        & (residual_df["input_scale"] > 0.0)
        & (residual_df["latency_s"] > 0.0)
        & (residual_df["predicted_latency_s"] > 0.0)
    )
    residual_df = residual_df[valid_mask].copy()
    if residual_df.empty:
        print(f"[skip] No valid latency residual rows in {residuals_path}")
        return False

    residual_df["relative_residual_pct"] = (
        100.0 * residual_df["residual_s"] / residual_df["latency_s"]
    )
    hardware_order = [
        hardware_model
        for hardware_model in ("cpu", "gpu")
        if hardware_model in set(residual_df["hardware_model"])
    ]
    hardware_order.extend(
        sorted(set(residual_df["hardware_model"]).difference(hardware_order))
    )
    fallback_colors = plt.get_cmap("tab10")
    hardware_styles = {
        "cpu": {
            "color": CPU_FIXED_COLORS[1],
            "marker": "o",
            "label": "CPU",
        },
        "gpu": {
            "color": GPU_GREEN,
            "marker": "s",
            "label": "GPU",
        },
    }
    for index, hardware_model in enumerate(hardware_order):
        hardware_styles.setdefault(
            hardware_model,
            {
                "color": fallback_colors(index % fallback_colors.N),
                "marker": "^",
                "label": hardware_model.upper(),
            },
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    parity_axis, prediction_axis, distribution_axis, scale_axis = axes.flat

    for hardware_model in hardware_order:
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        style = hardware_styles[hardware_model]
        parity_axis.scatter(
            hardware_df["latency_s"],
            hardware_df["predicted_latency_s"],
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.8,
            s=38,
            label=style["label"],
        )
    parity_min = float(
        min(
            residual_df["latency_s"].min(),
            residual_df["predicted_latency_s"].min(),
        )
    )
    parity_max = float(
        max(
            residual_df["latency_s"].max(),
            residual_df["predicted_latency_s"].max(),
        )
    )
    parity_axis.plot(
        [parity_min, parity_max],
        [parity_min, parity_max],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Ideal",
    )
    parity_axis.set_xscale("log")
    parity_axis.set_yscale("log")
    parity_axis.set_xlim(parity_min * 0.82, parity_max * 1.22)
    parity_axis.set_ylim(parity_min * 0.82, parity_max * 1.22)
    parity_axis.set_title("Resource-config OOF: actual vs. predicted")
    parity_axis.set_xlabel("Actual latency (s)")
    parity_axis.set_ylabel("Predicted latency (s)")
    parity_axis.grid(True, which="both", linestyle="-", alpha=0.25)
    parity_axis.legend(fontsize=8)
    r2, relative_mae, mean_absolute_percentage_error = _residual_plot_metrics(
        residual_df["latency_s"],
        residual_df["predicted_latency_s"],
    )
    metric_lines = [f"n = {len(residual_df)}"]
    if r2 is not None:
        metric_lines.append(f"R² = {r2:.4f}")
    if math.isfinite(relative_mae):
        metric_lines.append(f"Relative MAE = {100.0 * relative_mae:.1f}%")
    if math.isfinite(mean_absolute_percentage_error):
        metric_lines.append(
            f"Overall MAPE = {100.0 * mean_absolute_percentage_error:.1f}%"
        )
    for hardware_model in hardware_order:
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        _, _, hardware_mape = _residual_plot_metrics(
            hardware_df["latency_s"],
            hardware_df["predicted_latency_s"],
        )
        if math.isfinite(hardware_mape):
            metric_lines.append(
                f"{hardware_styles[hardware_model]['label']} MAPE = "
                f"{100.0 * hardware_mape:.1f}%"
            )
    parity_axis.text(
        0.04,
        0.96,
        "\n".join(metric_lines),
        transform=parity_axis.transAxes,
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )

    for hardware_model in hardware_order:
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        style = hardware_styles[hardware_model]
        prediction_axis.scatter(
            hardware_df["predicted_latency_s"],
            hardware_df["relative_residual_pct"],
            color=style["color"],
            marker=style["marker"],
            alpha=0.65,
            s=32,
            label=style["label"],
        )
    prediction_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    prediction_axis.set_xscale("log")
    prediction_axis.set_title("OOF residual vs. predicted latency")
    prediction_axis.set_xlabel("Predicted latency (s)")
    prediction_axis.set_ylabel("Relative residual (%)")
    prediction_axis.grid(True, which="both", linestyle="-", alpha=0.25)

    distribution_values = []
    distribution_labels = []
    distribution_colors = []
    for hardware_model in hardware_order:
        values = residual_df.loc[
            residual_df["hardware_model"] == hardware_model,
            "relative_residual_pct",
        ].to_numpy(dtype=float)
        distribution_values.append(values)
        distribution_labels.append(hardware_styles[hardware_model]["label"])
        distribution_colors.append(hardware_styles[hardware_model]["color"])
    boxplot_label_argument = (
        {"tick_labels": distribution_labels}
        if "tick_labels"
        in inspect.signature(distribution_axis.boxplot).parameters
        else {"labels": distribution_labels}
    )
    boxplot = distribution_axis.boxplot(
        distribution_values,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        **boxplot_label_argument,
    )
    for patch, color in zip(boxplot["boxes"], distribution_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    random_generator = np.random.default_rng(0)
    for position, (values, color) in enumerate(
        zip(distribution_values, distribution_colors),
        start=1,
    ):
        jitter = random_generator.uniform(-0.13, 0.13, size=len(values))
        distribution_axis.scatter(
            np.full(len(values), position, dtype=float) + jitter,
            values,
            color=color,
            alpha=0.45,
            s=18,
        )
    distribution_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.1,
    )
    distribution_axis.set_title("OOF residual distribution")
    distribution_axis.set_xlabel("Hardware model")
    distribution_axis.set_ylabel("Relative residual (%)")
    distribution_axis.grid(True, axis="y", linestyle="-", alpha=0.25)

    for hardware_model in hardware_order:
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        style = hardware_styles[hardware_model]
        scale_axis.scatter(
            hardware_df["input_scale"],
            hardware_df["relative_residual_pct"],
            color=style["color"],
            marker=style["marker"],
            alpha=0.4,
            s=25,
        )
        median_by_scale = (
            hardware_df.groupby("input_scale", as_index=False)[
                "relative_residual_pct"
            ]
            .median()
            .sort_values("input_scale")
        )
        scale_axis.plot(
            median_by_scale["input_scale"],
            median_by_scale["relative_residual_pct"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=f"{style['label']} OOF median",
        )

        holdout_columns = {
            "max_scale_holdout_predicted_latency_s",
            "max_scale_holdout_residual_s",
        }
        if holdout_columns.issubset(residual_df.columns):
            holdout_df = hardware_df.dropna(subset=list(holdout_columns)).copy()
            holdout_df = holdout_df[
                np.isfinite(
                    holdout_df["max_scale_holdout_predicted_latency_s"]
                )
                & np.isfinite(holdout_df["max_scale_holdout_residual_s"])
                & (holdout_df["max_scale_holdout_predicted_latency_s"] > 0.0)
            ]
            if not holdout_df.empty:
                holdout_relative_residual = (
                    100.0
                    * holdout_df["max_scale_holdout_residual_s"]
                    / holdout_df["latency_s"]
                )
                scale_axis.scatter(
                    holdout_df["input_scale"],
                    holdout_relative_residual,
                    color=style["color"],
                    marker="*",
                    edgecolor="black",
                    linewidth=0.45,
                    alpha=0.85,
                    s=75,
                    label=f"{style['label']} max-scale holdout",
                )
    scale_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    scale_axis.set_title("Residual vs. input scale")
    scale_axis.set_xlabel("Input scale")
    scale_axis.set_ylabel("Relative residual (%)")
    scale_axis.grid(True, which="both", linestyle="-", alpha=0.25)
    scale_axis.legend(fontsize=8, ncol=2)

    fig.suptitle("Latency Model Residual Diagnostics", fontsize=15)
    fig.text(
        0.5,
        0.015,
        (
            "Relative residual = (actual - predicted) / actual × 100; "
            "positive values indicate underprediction."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))

    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return True


def _hardware_model_name(gpu_mode: object) -> str:
    if pd.isna(gpu_mode):
        raise ValueError("gpu_mode contains a missing value")
    normalized = str(gpu_mode).strip().lower()
    if normalized in GPU_MODE_ON_VALUES:
        return "gpu"
    if normalized in GPU_MODE_OFF_VALUES:
        return "cpu"
    raise ValueError(f"unsupported gpu_mode value: {gpu_mode!r}")


def _format_input_scale_knot(input_scale: float) -> str:
    return (
        format(float(input_scale), ".12g")
        .replace("-", "neg_")
        .replace(".", "p")
    )


def _model_feature_names(
    hardware_model: str,
    input_scale_spline_knots: list[float] | None = None,
) -> list[str]:
    if hardware_model == "cpu":
        return list(LATENCY_MODEL_FEATURES[hardware_model])
    if hardware_model == "gpu":
        knots = input_scale_spline_knots or []
        return [
            "intercept",
            "log_input_scale",
            *[
                f"log_input_scale_hinge_at_{_format_input_scale_knot(knot)}"
                for knot in knots
            ],
            *LATENCY_MODEL_FEATURES[hardware_model][2:],
        ]
    raise ValueError(f"unknown latency hardware model: {hardware_model}")


def _gpu_input_scale_spline_knots(rows: list) -> list[float]:
    """Use every interior observed scale as a shared log-scale spline knot."""
    input_scales = sorted({float(row.input_scale) for row in rows})
    return input_scales[1:-1]


def _model_features(
    row,
    hardware_model: str,
    input_scale_spline_knots: list[float] | None = None,
) -> list[float]:
    input_scale = float(row.input_scale)
    cpu_cores = float(row.cpu_cores)
    mem_cap_gb = float(row.mem_cap_gb)
    log_input_scale = math.log(input_scale)
    log_mem_cap_gb = math.log(mem_cap_gb)

    if hardware_model == "cpu":
        log_cpu_cores = math.log(cpu_cores)
        return [
            1.0,
            log_input_scale,
            log_input_scale**2,
            log_cpu_cores,
            log_cpu_cores**2,
            log_mem_cap_gb,
            log_input_scale * log_cpu_cores,
            log_input_scale * log_mem_cap_gb,
            log_cpu_cores * log_mem_cap_gb,
        ]
    if hardware_model == "gpu":
        inverse_cpu_cores = 1.0 / cpu_cores
        inverse_cpu_cores_squared = inverse_cpu_cores**2
        spline_knots = input_scale_spline_knots or []
        return [
            1.0,
            log_input_scale,
            *[
                max(0.0, log_input_scale - math.log(float(knot)))
                for knot in spline_knots
            ],
            inverse_cpu_cores,
            inverse_cpu_cores_squared,
            log_mem_cap_gb,
            log_input_scale * inverse_cpu_cores,
            log_input_scale * inverse_cpu_cores_squared,
        ]
    raise ValueError(f"unknown latency hardware model: {hardware_model}")


def _fit_rank_revealing_linear_model(
    feature_matrix: np.ndarray,
    target: np.ndarray,
) -> dict:
    """Solve a standardized linear model after removing dependent columns."""
    if feature_matrix.ndim != 2 or target.ndim != 1:
        raise ValueError("model matrix and target have invalid dimensions")
    if feature_matrix.shape[0] != target.shape[0]:
        raise ValueError("model matrix and target row counts differ")
    if feature_matrix.shape[1] < 2:
        raise ValueError("model matrix needs an intercept and a predictor")
    if not np.isfinite(feature_matrix).all() or not np.isfinite(target).all():
        raise ValueError("model inputs contain non-finite values")

    means = np.zeros(feature_matrix.shape[1], dtype=float)
    scales = np.ones(feature_matrix.shape[1], dtype=float)
    standardized = np.zeros_like(feature_matrix)
    standardized[:, 0] = 1.0
    for feature_idx in range(1, feature_matrix.shape[1]):
        column = feature_matrix[:, feature_idx]
        means[feature_idx] = float(column.mean())
        scales[feature_idx] = float(column.std())
        if scales[feature_idx] > 1e-12:
            standardized[:, feature_idx] = (
                column - means[feature_idx]
            ) / scales[feature_idx]

    selected_indices: list[int] = []
    for feature_idx in range(standardized.shape[1]):
        candidate_indices = selected_indices + [feature_idx]
        candidate = standardized[:, candidate_indices]
        if int(np.linalg.matrix_rank(candidate)) > len(selected_indices):
            selected_indices.append(feature_idx)

    if len(selected_indices) < 2:
        raise ValueError("model matrix has no usable varying predictor")

    selected_matrix = standardized[:, selected_indices]
    standardized_coefficients, _, rank, singular_values = np.linalg.lstsq(
        selected_matrix,
        target,
        rcond=None,
    )
    if int(rank) != len(selected_indices):
        raise ValueError(
            f"rank-deficient model matrix: rank={int(rank)}, "
            f"features={len(selected_indices)}"
        )

    coefficients = np.zeros(feature_matrix.shape[1], dtype=float)
    for position, feature_idx in enumerate(selected_indices):
        if feature_idx == 0:
            continue
        coefficients[feature_idx] = (
            standardized_coefficients[position] / scales[feature_idx]
        )
    intercept_position = selected_indices.index(0)
    coefficients[0] = standardized_coefficients[intercept_position] - sum(
        coefficients[feature_idx] * means[feature_idx]
        for feature_idx in selected_indices
        if feature_idx != 0
    )

    smallest_singular = float(singular_values[-1])
    condition_number = (
        None
        if smallest_singular <= 0.0
        else float(singular_values[0] / smallest_singular)
    )
    return {
        "coefficients": [float(value) for value in coefficients],
        "selected_indices": selected_indices,
        "rank": int(rank),
        "condition_number": condition_number,
    }


def _gpu_upper_tail_features(row) -> list[float]:
    """Features for a stable affine GPU latency tail above the fitted range."""
    input_scale = float(row.input_scale)
    inverse_cpu_cores = 1.0 / float(row.cpu_cores)
    inverse_cpu_cores_squared = inverse_cpu_cores**2
    return [
        1.0,
        input_scale,
        inverse_cpu_cores,
        inverse_cpu_cores_squared,
        math.log(float(row.mem_cap_gb)),
        input_scale * inverse_cpu_cores,
        input_scale * inverse_cpu_cores_squared,
    ]


def _fit_gpu_upper_tail_model(rows: list) -> dict:
    feature_matrix = np.asarray(
        [_gpu_upper_tail_features(row) for row in rows],
        dtype=float,
    )
    target = np.asarray([float(row.latency_s) for row in rows], dtype=float)
    solution = _fit_rank_revealing_linear_model(feature_matrix, target)
    selected_feature_names = [
        LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES[feature_idx]
        for feature_idx in solution["selected_indices"]
    ]
    scale_slope_features = {
        "input_scale",
        "input_scale_x_inverse_cpu_cores",
        "input_scale_x_inverse_cpu_cores_squared",
    }
    if not scale_slope_features.intersection(selected_feature_names):
        raise ValueError("GPU upper-tail model has no input-scale slope")

    coefficients = solution["coefficients"]
    training_configurations = sorted({
        (float(row.cpu_cores), float(row.mem_cap_gb)) for row in rows
    })
    fitted_slopes = []
    for cpu_cores, mem_cap_gb in training_configurations:
        at_zero = SimpleNamespace(
            input_scale=0.0,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
        )
        at_one = SimpleNamespace(
            input_scale=1.0,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
        )
        fitted_slopes.append(float(np.dot(
            coefficients,
            np.asarray(_gpu_upper_tail_features(at_one))
            - np.asarray(_gpu_upper_tail_features(at_zero)),
        )))

    return {
        "feature_names": list(LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES),
        "selected_indices": solution["selected_indices"],
        "selected_feature_names": selected_feature_names,
        "dropped_feature_names": [
            feature_name
            for feature_name in LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES
            if feature_name not in selected_feature_names
        ],
        "coefficients": coefficients,
        "rank": solution["rank"],
        "condition_number": solution["condition_number"],
        "minimum_fitted_slope_s_per_scale": min(fitted_slopes),
    }


def _fit_log_latency_model(
    rows: list,
    hardware_model: str,
    *,
    calibrate_gpu_upper_tail: bool = True,
) -> dict:
    """Fit log(latency) with rank-revealing least squares.

    Regressors are standardized before solving. Constant or linearly dependent
    columns are removed so CPU-only, GPU-only, and reduced resource matrices do
    not fail merely because a feature is constant.
    """
    if len(rows) < 3:
        raise ValueError(f"need at least 3 case rows, got {len(rows)}")

    input_scale_spline_knots = (
        _gpu_input_scale_spline_knots(rows)
        if hardware_model == "gpu"
        else []
    )
    feature_names = _model_feature_names(
        hardware_model,
        input_scale_spline_knots,
    )
    feature_matrix = np.asarray(
        [
            _model_features(
                row,
                hardware_model,
                input_scale_spline_knots,
            )
            for row in rows
        ],
        dtype=float,
    )
    target = np.log(np.asarray([float(row.latency_s) for row in rows], dtype=float))
    solution = _fit_rank_revealing_linear_model(feature_matrix, target)
    selected_indices = solution["selected_indices"]
    model = {
        "hardware_model": hardware_model,
        "input_scale_spline_knots": input_scale_spline_knots,
        "input_scale_min": min(float(row.input_scale) for row in rows),
        "input_scale_max": max(float(row.input_scale) for row in rows),
        "feature_names": feature_names,
        "selected_indices": selected_indices,
        "selected_feature_names": [
            feature_names[feature_idx] for feature_idx in selected_indices
        ],
        "dropped_feature_names": [
            feature_names[feature_idx]
            for feature_idx in range(len(feature_names))
            if feature_idx not in selected_indices
        ],
        "coefficients": solution["coefficients"],
        "rank": solution["rank"],
        "condition_number": solution["condition_number"],
        "fit_case_rows": len(rows),
    }

    if hardware_model == "gpu" and calibrate_gpu_upper_tail:
        input_scales = sorted({float(row.input_scale) for row in rows})
        input_scale_span_ratio = input_scales[-1] / input_scales[0]
        calibration = {
            "available": False,
            "held_out_input_scale": None,
            "mean_absolute_percentage_error": None,
            "activation_threshold_mape": (
                LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
            ),
        }
        if len(input_scales) >= 3:
            calibration_scale = input_scales[-1]
            calibration_rows = [
                row for row in rows
                if float(row.input_scale) < calibration_scale
            ]
            calibration_test_rows = [
                row for row in rows
                if float(row.input_scale) == calibration_scale
            ]
            try:
                calibration_model = _fit_log_latency_model(
                    calibration_rows,
                    hardware_model,
                    calibrate_gpu_upper_tail=False,
                )
                calibration_predictions = np.asarray([
                    _predict_latency(row, calibration_model)
                    for row in calibration_test_rows
                ])
                calibration_actuals = np.asarray([
                    float(row.latency_s) for row in calibration_test_rows
                ])
                calibration_mape = float(np.mean(
                    np.abs(calibration_actuals - calibration_predictions)
                    / calibration_actuals
                ))
                calibration = {
                    "available": True,
                    "held_out_input_scale": calibration_scale,
                    "train_input_scale_max": max(
                        float(row.input_scale) for row in calibration_rows
                    ),
                    "mean_absolute_percentage_error": calibration_mape,
                    "activation_threshold_mape": (
                        LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
                    ),
                }
            except ValueError as exc:
                calibration["reason"] = str(exc)

        try:
            upper_tail_model = _fit_gpu_upper_tail_model(rows)
            upper_tail_model["calibration"] = calibration
            upper_tail_model["training_input_scale_span_ratio"] = (
                input_scale_span_ratio
            )
            upper_tail_model["enabled"] = bool(
                calibration["available"]
                and calibration["mean_absolute_percentage_error"]
                > LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
                and input_scale_span_ratio
                >= LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO
                and upper_tail_model["minimum_fitted_slope_s_per_scale"] > 0.0
            )
            model["gpu_upper_tail"] = upper_tail_model
        except ValueError as exc:
            model["gpu_upper_tail"] = {
                "enabled": False,
                "reason": str(exc),
                "calibration": calibration,
            }

    return model


def _predict_primary_latency(row, model: dict) -> float:
    """Evaluate the positive log-link model without tail extrapolation."""
    linear_prediction = sum(
        coefficient * feature
        for coefficient, feature in zip(
            model["coefficients"],
            _model_features(
                row,
                model["hardware_model"],
                model.get("input_scale_spline_knots", []),
            ),
        )
    )
    # The exponential link makes predictions positive. Clipping only protects
    # artifact generation from floating-point overflow on extreme extrapolation.
    bounded_prediction = min(max(linear_prediction, -700.0), 700.0)
    return float(math.exp(bounded_prediction))


def _predict_latency(row, model: dict) -> float:
    primary_prediction = _predict_primary_latency(row, model)
    upper_tail_model = model.get("gpu_upper_tail", {})
    input_scale_max = model.get("input_scale_max")
    if not (
        model.get("hardware_model") == "gpu"
        and upper_tail_model.get("enabled")
        and input_scale_max is not None
        and float(row.input_scale) > float(input_scale_max)
    ):
        return primary_prediction

    boundary_row = SimpleNamespace(
        input_scale=float(input_scale_max),
        cpu_cores=float(row.cpu_cores),
        mem_cap_gb=float(row.mem_cap_gb),
    )
    boundary_prediction = _predict_primary_latency(boundary_row, model)
    feature_delta = (
        np.asarray(_gpu_upper_tail_features(row), dtype=float)
        - np.asarray(_gpu_upper_tail_features(boundary_row), dtype=float)
    )
    tail_delta = float(np.dot(
        np.asarray(upper_tail_model["coefficients"], dtype=float),
        feature_delta,
    ))
    tail_prediction = boundary_prediction + tail_delta
    if math.isfinite(tail_prediction) and tail_prediction > 0.0:
        return tail_prediction
    return primary_prediction


def plot_latency_model_fit_curves(
    residuals_path: str,
    report_path: str,
    out_png: str | None,
) -> bool:
    """Plot full-fit latency curves for every CPU/memory configuration."""
    if not os.path.exists(residuals_path):
        print(f"[skip] Cannot find {residuals_path}")
        return False
    if not os.path.exists(report_path):
        print(f"[skip] Cannot find {report_path}")
        return False

    residual_df = pd.read_csv(residuals_path, skipinitialspace=True)
    required_columns = {
        "hardware_model",
        "cpu_cores",
        "mem_cap_gb",
        "input_scale",
        "latency_s",
    }
    missing_columns = sorted(required_columns.difference(residual_df.columns))
    if missing_columns:
        print(
            "[skip] Latency residual CSV missing fit-curve columns: "
            f"{missing_columns}"
        )
        return False
    if residual_df.empty:
        print(f"[skip] No latency fit rows in {residuals_path}")
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    for column in ("cpu_cores", "mem_cap_gb", "input_scale", "latency_s"):
        residual_df[column] = pd.to_numeric(residual_df[column], errors="coerce")
    residual_df["hardware_model"] = (
        residual_df["hardware_model"].astype(str).str.strip().str.lower()
    )
    valid_mask = (
        residual_df[
            ["cpu_cores", "mem_cap_gb", "input_scale", "latency_s"]
        ]
        .apply(np.isfinite)
        .all(axis=1)
        & (residual_df["cpu_cores"] > 0.0)
        & (residual_df["mem_cap_gb"] > 0.0)
        & (residual_df["input_scale"] > 0.0)
        & (residual_df["latency_s"] > 0.0)
    )
    residual_df = residual_df[valid_mask].copy()

    fitted_models = {}
    report_models = report.get("models", {})
    for hardware_model in ("cpu", "gpu"):
        model_report = report_models.get(hardware_model, {})
        coefficient_map = model_report.get("coefficients", {})
        input_scale_basis = model_report.get("input_scale_basis", {})
        input_scale_spline_knots = input_scale_basis.get("knots", [])
        if not isinstance(input_scale_spline_knots, list):
            continue
        try:
            input_scale_spline_knots = [
                float(knot) for knot in input_scale_spline_knots
            ]
        except (TypeError, ValueError):
            continue
        feature_names = model_report.get(
            "feature_columns",
            LATENCY_MODEL_FEATURES[hardware_model],
        )
        if feature_names != _model_feature_names(
            hardware_model,
            input_scale_spline_knots,
        ):
            continue
        try:
            coefficients = [
                float(coefficient_map[feature_name])
                for feature_name in feature_names
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(coefficients).all():
            continue
        fitted_models[hardware_model] = {
            "hardware_model": hardware_model,
            "input_scale_spline_knots": input_scale_spline_knots,
            "coefficients": coefficients,
        }

    hardware_order = [
        hardware_model
        for hardware_model in ("cpu", "gpu")
        if hardware_model in fitted_models
        and hardware_model in set(residual_df["hardware_model"])
    ]
    if not hardware_order:
        print("[skip] No fitted CPU/GPU latency models available for curve plot")
        return False

    cpu_values = sorted(int(value) for value in residual_df["cpu_cores"].unique())
    mem_values = sorted(int(value) for value in residual_df["mem_cap_gb"].unique())
    cpu_colors = build_cpu_base_colors(cpu_values)
    mem_rank_map = {mem: index for index, mem in enumerate(mem_values)}
    line_styles = ("-", "--", "-.", ":")

    fig, axes = plt.subplots(
        1,
        len(hardware_order),
        figsize=(7.5 * len(hardware_order), 8.5),
        squeeze=False,
    )
    axes = list(axes.flat)
    legend_entries = {}
    hardware_titles = {
        "cpu": "CPU-off model",
        "gpu": "GPU-on model",
    }

    for axis, hardware_model in zip(axes, hardware_order):
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        model = fitted_models[hardware_model]
        configurations = sorted(
            {
                (int(row.cpu_cores), int(row.mem_cap_gb))
                for row in hardware_df.itertuples(index=False)
            }
        )

        for cpu_cores, mem_cap_gb in configurations:
            configuration_df = hardware_df[
                (hardware_df["cpu_cores"] == cpu_cores)
                & (hardware_df["mem_cap_gb"] == mem_cap_gb)
            ].sort_values("input_scale")
            min_scale = float(configuration_df["input_scale"].min())
            max_scale = float(configuration_df["input_scale"].max())
            if min_scale == max_scale:
                curve_scales = np.asarray([min_scale], dtype=float)
            else:
                curve_scales = np.linspace(min_scale, max_scale, 240)
            curve_predictions = [
                _predict_latency(
                    SimpleNamespace(
                        input_scale=input_scale,
                        cpu_cores=cpu_cores,
                        mem_cap_gb=mem_cap_gb,
                    ),
                    model,
                )
                for input_scale in curve_scales
            ]

            color = cpu_colors[cpu_cores]
            line_style = line_styles[
                mem_rank_map[mem_cap_gb] % len(line_styles)
            ]
            label = f"{cpu_cores} CPU / {mem_cap_gb} GiB"
            line, = axis.plot(
                curve_scales,
                curve_predictions,
                color=color,
                linestyle=line_style,
                linewidth=1.8,
                alpha=0.92,
                label=label,
            )
            axis.scatter(
                configuration_df["input_scale"],
                configuration_df["latency_s"],
                facecolor="white",
                edgecolor=color,
                marker="o",
                linewidth=1.1,
                s=30,
                alpha=0.95,
                zorder=3,
            )
            legend_entries.setdefault((cpu_cores, mem_cap_gb), line)

        axis.set_title(hardware_titles[hardware_model])
        axis.set_xlabel("Input scale")
        axis.set_ylabel("Latency (s, log scale)")
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="-", alpha=0.25)

    input_scale_type = str(report.get("input_scale_type", "")).strip()
    if input_scale_type and input_scale_type != "input_scale":
        for axis in axes:
            axis.set_xlabel(f"Input scale ({input_scale_type})")

    ordered_legend_entries = sorted(legend_entries.items())
    fig.legend(
        [line for _, line in ordered_legend_entries],
        [
            f"{cpu_cores} CPU / {mem_cap_gb} GiB"
            for (cpu_cores, mem_cap_gb), _ in ordered_legend_entries
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, len(ordered_legend_entries)),
        fontsize=8,
        title="Resource configuration",
    )
    fig.suptitle("Latency Model Full-Fit Curves", fontsize=15)
    fig.text(
        0.5,
        0.935,
        (
            "Curves: full-data least-squares fit; "
            "hollow markers: measured case medians"
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.18, 1.0, 0.91))

    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return True


def _regression_metrics(
    actuals: list[float],
    predictions: list[float],
) -> dict[str, float | int | None]:
    prediction_count = len(predictions)
    nonfinite_prediction_count = sum(
        not math.isfinite(float(prediction)) for prediction in predictions
    )
    nonpositive_prediction_count = sum(
        math.isfinite(float(prediction)) and float(prediction) <= 0.0
        for prediction in predictions
    )
    valid_pairs = [
        (float(actual), float(prediction))
        for actual, prediction in zip(actuals, predictions)
        if math.isfinite(float(actual)) and math.isfinite(float(prediction))
    ]
    if not valid_pairs:
        return {
            "r2": None,
            "mae": None,
            "rmse": None,
            "relative_mae": None,
            "mean_absolute_percentage_error": None,
            "maximum_absolute_percentage_error": None,
            "smape": None,
            "p95_absolute_error": None,
            "mean_actual": None,
            "prediction_count": prediction_count,
            "nonfinite_prediction_count": nonfinite_prediction_count,
            "nonpositive_prediction_count": nonpositive_prediction_count,
            "min_prediction": None,
            "max_prediction": None,
        }

    valid_actuals = [pair[0] for pair in valid_pairs]
    valid_predictions = [pair[1] for pair in valid_pairs]
    errors = [
        actual - predicted
        for actual, predicted in zip(valid_actuals, valid_predictions)
    ]
    absolute_errors = [abs(error) for error in errors]
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_actual = sum(valid_actuals) / len(valid_actuals)
    ss_tot = sum((actual - mean_actual) ** 2 for actual in valid_actuals)
    ss_res = sum(error * error for error in errors)
    r2 = None if ss_tot <= 1e-24 else 1.0 - (ss_res / ss_tot)
    relative_mae = None if mean_actual <= 0.0 else mae / mean_actual
    absolute_percentage_errors = [
        abs(actual - predicted) / abs(actual)
        for actual, predicted in zip(valid_actuals, valid_predictions)
        if abs(actual) > 0.0
    ]
    mean_absolute_percentage_error = (
        None
        if not absolute_percentage_errors
        else sum(absolute_percentage_errors) / len(absolute_percentage_errors)
    )
    maximum_absolute_percentage_error = (
        None
        if not absolute_percentage_errors
        else max(absolute_percentage_errors)
    )
    smape_terms = [
        2.0 * abs(actual - predicted) / (abs(actual) + abs(predicted))
        for actual, predicted in zip(valid_actuals, valid_predictions)
        if abs(actual) + abs(predicted) > 0.0
    ]
    smape = None if not smape_terms else sum(smape_terms) / len(smape_terms)
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "relative_mae": relative_mae,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "maximum_absolute_percentage_error": maximum_absolute_percentage_error,
        "smape": smape,
        "p95_absolute_error": float(np.percentile(absolute_errors, 95)),
        "mean_actual": mean_actual,
        "prediction_count": prediction_count,
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "nonpositive_prediction_count": nonpositive_prediction_count,
        "min_prediction": min(valid_predictions),
        "max_prediction": max(valid_predictions),
    }


def _clean_json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json_value(item) for item in value]
    return value


def _latency_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "latency_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required modeling columns: {sorted(missing)}")

    model_df = df[list(required)].copy()
    for col in ("cpu_cores", "mem_cap_gb", "input_scale", "latency_s"):
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df[
        model_df["cpu_cores"].notna()
        & model_df["mem_cap_gb"].notna()
        & model_df["input_scale"].notna()
        & model_df["latency_s"].notna()
        & (model_df["cpu_cores"] > 0)
        & (model_df["mem_cap_gb"] > 0)
        & (model_df["input_scale"] > 0)
        & (model_df["latency_s"] > 0)
    ].copy()
    model_df["source_row"] = model_df.index
    model_df["hardware_model"] = model_df["gpu_mode"].map(_hardware_model_name)
    return model_df.sort_values(
        ["hardware_model", "cpu_cores", "mem_cap_gb", "input_scale", "source_row"]
    ).reset_index(drop=True)


def _aggregate_latency_model_points(model_df: pd.DataFrame) -> list:
    """Collapse repeated windows before any fit or validation split."""
    point_df = model_df.groupby(
        ["hardware_model", "cpu_cores", "mem_cap_gb", "input_scale"],
        as_index=False,
        sort=True,
    ).agg(
        latency_s=("latency_s", "median"),
        latency_mean_s=("latency_s", "mean"),
        latency_std_s=("latency_s", lambda values: values.std(ddof=0)),
        repeat_count=("latency_s", "size"),
    )
    point_df["latency_std_s"] = point_df["latency_std_s"].fillna(0.0)
    return list(point_df.itertuples(index=False))


def _point_key(row) -> tuple[str, float, float, float]:
    return (
        str(row.hardware_model),
        float(row.cpu_cores),
        float(row.mem_cap_gb),
        float(row.input_scale),
    )


def _resource_configuration_validation(
    rows: list,
    hardware_model: str,
) -> tuple[dict, dict[tuple[str, float, float, float], float]]:
    """Leave one complete (CPU, memory) configuration out per fold."""
    configurations = sorted({
        (float(row.cpu_cores), float(row.mem_cap_gb)) for row in rows
    })
    if len(configurations) < 3:
        return {
            "available": False,
            "reason": (
                "need at least 3 resource configurations so each held-out fold "
                "retains at least 2 training configurations"
            ),
            "folds": 0,
            "completed_folds": 0,
            "failed_folds": [],
            "r2_quality_gate_applicable": True,
            "train_test_group_overlap_count": 0,
            "metrics": _regression_metrics([], []),
        }, {}

    predictions: dict[tuple[str, float, float, float], float] = {}
    failed_folds = []
    fold_details = []
    train_test_group_overlap_count = 0
    completed_folds = 0
    for cpu_cores, mem_cap_gb in configurations:
        test_rows = [
            row
            for row in rows
            if (
                float(row.cpu_cores) == cpu_cores
                and float(row.mem_cap_gb) == mem_cap_gb
            )
        ]
        train_rows = [
            row
            for row in rows
            if not (
                float(row.cpu_cores) == cpu_cores
                and float(row.mem_cap_gb) == mem_cap_gb
            )
        ]
        train_groups = {
            (float(row.cpu_cores), float(row.mem_cap_gb)) for row in train_rows
        }
        test_groups = {
            (float(row.cpu_cores), float(row.mem_cap_gb)) for row in test_rows
        }
        fold_overlap_count = len(train_groups & test_groups)
        train_test_group_overlap_count += fold_overlap_count
        fold_detail = {
            "held_out_cpu_cores": cpu_cores,
            "held_out_mem_cap_gb": mem_cap_gb,
            "train_case_rows": len(train_rows),
            "test_case_rows": len(test_rows),
            "train_test_group_overlap_count": fold_overlap_count,
        }
        try:
            fold_model = _fit_log_latency_model(train_rows, hardware_model)
            if not (
                LATENCY_MODEL_RESOURCE_FEATURES[hardware_model]
                & set(fold_model["selected_feature_names"])
            ):
                raise ValueError(
                    "held-out fold has no identifiable CPU or memory predictor"
                )
        except ValueError as exc:
            fold_detail["status"] = "failed"
            fold_detail["reason"] = str(exc)
            fold_details.append(fold_detail)
            failed_folds.append({
                "cpu_cores": cpu_cores,
                "mem_cap_gb": mem_cap_gb,
                "reason": str(exc),
            })
            continue

        fold_predictions = [
            _predict_latency(row, fold_model) for row in test_rows
        ]
        for row, prediction in zip(test_rows, fold_predictions):
            predictions[_point_key(row)] = prediction
        fold_detail["status"] = "ok"
        fold_detail["metrics"] = _regression_metrics(
            [float(row.latency_s) for row in test_rows],
            fold_predictions,
        )
        fold_details.append(fold_detail)
        completed_folds += 1

    predicted_rows = [row for row in rows if _point_key(row) in predictions]
    actuals = [float(row.latency_s) for row in predicted_rows]
    predicted = [predictions[_point_key(row)] for row in predicted_rows]
    case_relative_errors = [
        abs(actual - prediction) / actual
        for actual, prediction in zip(actuals, predicted)
    ]
    worst_case_idx = (
        None
        if not predicted_rows
        else max(
            range(len(predicted_rows)),
            key=lambda idx: case_relative_errors[idx],
        )
    )
    worst_case = (
        None if worst_case_idx is None else predicted_rows[worst_case_idx]
    )
    completed_fold_details = [
        fold_detail
        for fold_detail in fold_details
        if fold_detail.get("status") == "ok"
    ]
    worst_relative_mae_fold = (
        None
        if not completed_fold_details
        else max(
            completed_fold_details,
            key=lambda fold_detail: (
                math.inf
                if fold_detail["metrics"].get("relative_mae") is None
                else float(fold_detail["metrics"]["relative_mae"])
            ),
        )
    )
    return {
        "available": not failed_folds and len(predicted_rows) == len(rows),
        "method": "leave-one-(cpu_cores,mem_cap_gb)-configuration-out",
        "r2_quality_gate_applicable": True,
        "group_columns": ["cpu_cores", "mem_cap_gb"],
        "all_input_scales_move_with_held_out_configuration": True,
        "folds": len(configurations),
        "completed_folds": completed_folds,
        "failed_folds": failed_folds,
        "fold_details": fold_details,
        "train_test_group_overlap_count": train_test_group_overlap_count,
        "test_case_rows": len(predicted_rows),
        "worst_fold_relative_mae": (
            None
            if worst_relative_mae_fold is None
            else worst_relative_mae_fold["metrics"]["relative_mae"]
        ),
        "worst_fold_configuration": (
            None
            if worst_relative_mae_fold is None
            else {
                "cpu_cores": worst_relative_mae_fold["held_out_cpu_cores"],
                "mem_cap_gb": worst_relative_mae_fold["held_out_mem_cap_gb"],
            }
        ),
        "worst_case_relative_error": (
            None
            if worst_case_idx is None
            else case_relative_errors[worst_case_idx]
        ),
        "worst_case_configuration": (
            None
            if worst_case is None
            else {
                "cpu_cores": float(worst_case.cpu_cores),
                "mem_cap_gb": float(worst_case.mem_cap_gb),
                "input_scale": float(worst_case.input_scale),
            }
        ),
        "metrics": _regression_metrics(actuals, predicted),
    }, predictions


def _input_scale_validation(
    rows: list,
    hardware_model: str,
) -> tuple[dict, dict[tuple[str, float, float, float], float]]:
    """Train on smaller scales and hold out the largest scale for extrapolation."""
    input_scales = sorted({float(row.input_scale) for row in rows})
    if len(input_scales) < 3:
        return {
            "available": False,
            "reason": (
                "need at least 3 input scales so the maximum can be held out "
                "while at least 2 distinct scales remain for training"
            ),
            "held_out_input_scale": None,
            "group_columns": ["input_scale"],
            "r2_quality_gate_applicable": False,
            "r2_quality_gate_note": (
                "A single held-out scale tests level extrapolation; R-squared "
                "within that scale only measures resource-configuration "
                "variation and is not used as an extrapolation gate."
            ),
            "train_test_group_overlap_count": 0,
            "metrics": _regression_metrics([], []),
        }, {}

    held_out_scale = input_scales[-1]
    train_rows = [row for row in rows if float(row.input_scale) < held_out_scale]
    test_rows = [row for row in rows if float(row.input_scale) == held_out_scale]
    train_scale_groups = {float(row.input_scale) for row in train_rows}
    test_scale_groups = {float(row.input_scale) for row in test_rows}
    train_test_group_overlap_count = len(train_scale_groups & test_scale_groups)
    try:
        scale_model = _fit_log_latency_model(train_rows, hardware_model)
        if "log_input_scale" not in scale_model["selected_feature_names"]:
            raise ValueError(
                "maximum-scale training fold has no identifiable input-scale predictor"
            )
    except ValueError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "held_out_input_scale": held_out_scale,
            "group_columns": ["input_scale"],
            "r2_quality_gate_applicable": False,
            "r2_quality_gate_note": (
                "A single held-out scale tests level extrapolation; R-squared "
                "within that scale only measures resource-configuration "
                "variation and is not used as an extrapolation gate."
            ),
            "train_input_scale_max": max(
                (float(row.input_scale) for row in train_rows),
                default=None,
            ),
            "train_test_group_overlap_count": train_test_group_overlap_count,
            "metrics": _regression_metrics([], []),
        }, {}

    predictions = {
        _point_key(row): _predict_latency(row, scale_model) for row in test_rows
    }
    actuals = [float(row.latency_s) for row in test_rows]
    predicted = [predictions[_point_key(row)] for row in test_rows]
    case_relative_errors = [
        abs(actual - prediction) / actual
        for actual, prediction in zip(actuals, predicted)
    ]
    worst_case_idx = max(
        range(len(test_rows)),
        key=lambda idx: case_relative_errors[idx],
    )
    worst_case = test_rows[worst_case_idx]
    return {
        "available": True,
        "method": "forward holdout of maximum input_scale",
        "group_columns": ["input_scale"],
        "r2_quality_gate_applicable": False,
        "r2_quality_gate_note": (
            "A single held-out scale tests level extrapolation; R-squared "
            "within that scale only measures resource-configuration variation "
            "and is reported for context, not used as an extrapolation gate."
        ),
        "held_out_input_scale": held_out_scale,
        "train_input_scale_max": max(float(row.input_scale) for row in train_rows),
        "test_input_scale_min": min(float(row.input_scale) for row in test_rows),
        "strict_extrapolation": (
            max(float(row.input_scale) for row in train_rows)
            < min(float(row.input_scale) for row in test_rows)
        ),
        "train_test_group_overlap_count": train_test_group_overlap_count,
        "train_case_rows": len(train_rows),
        "test_case_rows": len(test_rows),
        "worst_case_relative_error": case_relative_errors[worst_case_idx],
        "worst_case_configuration": {
            "cpu_cores": float(worst_case.cpu_cores),
            "mem_cap_gb": float(worst_case.mem_cap_gb),
            "input_scale": float(worst_case.input_scale),
        },
        "metrics": _regression_metrics(actuals, predicted),
    }, predictions


def _validation_quality_failures(validation_name: str, validation: dict) -> list[str]:
    if not validation.get("available"):
        return [
            f"{validation_name} unavailable: "
            f"{validation.get('reason', 'incomplete validation folds')}"
        ]

    metrics = validation["metrics"]
    failures = []
    if int(metrics.get("prediction_count") or 0) < LATENCY_MODEL_MIN_VALIDATION_POINTS:
        failures.append(
            f"{validation_name} has fewer than "
            f"{LATENCY_MODEL_MIN_VALIDATION_POINTS} predictions"
        )
    if int(validation.get("train_test_group_overlap_count") or 0) > 0:
        failures.append(f"{validation_name} has train/test group leakage")
    failed_quality_folds = []
    for fold_detail in validation.get("fold_details", []):
        if fold_detail.get("status") != "ok":
            continue
        fold_metrics = fold_detail["metrics"]
        fold_relative_mae = fold_metrics.get("relative_mae")
        fold_mape = fold_metrics.get("mean_absolute_percentage_error")
        if (
            fold_relative_mae is None
            or not math.isfinite(float(fold_relative_mae))
            or float(fold_relative_mae)
            > LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE
            or fold_mape is None
            or not math.isfinite(float(fold_mape))
            or float(fold_mape) > LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE
            or int(fold_metrics.get("nonfinite_prediction_count") or 0) > 0
            or int(fold_metrics.get("nonpositive_prediction_count") or 0) > 0
        ):
            failed_quality_folds.append(fold_detail)
    if failed_quality_folds:
        worst_fold = max(
            failed_quality_folds,
            key=lambda fold_detail: max(
                math.inf
                if (
                    fold_detail["metrics"].get("relative_mae") is None
                    or not math.isfinite(
                        float(fold_detail["metrics"]["relative_mae"])
                    )
                )
                else (
                    float(fold_detail["metrics"]["relative_mae"])
                    / LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE
                ),
                math.inf
                if (
                    fold_detail["metrics"].get(
                        "mean_absolute_percentage_error"
                    )
                    is None
                    or not math.isfinite(
                        float(
                            fold_detail["metrics"][
                                "mean_absolute_percentage_error"
                            ]
                        )
                    )
                )
                else (
                    float(
                        fold_detail["metrics"][
                            "mean_absolute_percentage_error"
                        ]
                    )
                    / LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE
                ),
            ),
        )
        failures.append(
            f"{validation_name} has {len(failed_quality_folds)} held-out "
            "configuration fold(s) above the relative-MAE/MAPE/validity "
            "threshold; "
            f"worst=(cpu_cores={worst_fold['held_out_cpu_cores']},"
            f"mem_cap_gb={worst_fold['held_out_mem_cap_gb']}), "
            f"relative_mae={worst_fold['metrics'].get('relative_mae')!r}, "
            "mean_absolute_percentage_error="
            f"{worst_fold['metrics'].get('mean_absolute_percentage_error')!r}"
        )
    worst_case_relative_error = validation.get("worst_case_relative_error")
    if (
        worst_case_relative_error is not None
        and (
            not math.isfinite(float(worst_case_relative_error))
            or float(worst_case_relative_error)
            > LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR
        )
    ):
        failures.append(
            f"{validation_name} worst case relative_error="
            f"{worst_case_relative_error!r} exceeds "
            f"{LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR}; "
            f"configuration={validation.get('worst_case_configuration')!r}"
        )
    if validation.get("r2_quality_gate_applicable", True):
        r2 = metrics.get("r2")
        if (
            r2 is None
            or not math.isfinite(float(r2))
            or float(r2) < LATENCY_MODEL_MIN_VALIDATION_R2
        ):
            failures.append(
                f"{validation_name} r2={r2!r} is below "
                f"{LATENCY_MODEL_MIN_VALIDATION_R2}"
            )
    relative_mae = metrics.get("relative_mae")
    if (
        relative_mae is None
        or not math.isfinite(float(relative_mae))
        or float(relative_mae) > LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE
    ):
        failures.append(
            f"{validation_name} relative_mae={relative_mae!r} exceeds "
            f"{LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE}"
        )
    mean_absolute_percentage_error = metrics.get(
        "mean_absolute_percentage_error"
    )
    if (
        mean_absolute_percentage_error is None
        or not math.isfinite(float(mean_absolute_percentage_error))
        or float(mean_absolute_percentage_error) > LATENCY_MODEL_MAX_VALIDATION_MAPE
    ):
        failures.append(
            f"{validation_name} mean_absolute_percentage_error="
            f"{mean_absolute_percentage_error!r} exceeds "
            f"{LATENCY_MODEL_MAX_VALIDATION_MAPE}"
        )
    if int(metrics.get("nonfinite_prediction_count") or 0) > 0:
        failures.append(f"{validation_name} contains non-finite predictions")
    if int(metrics.get("nonpositive_prediction_count") or 0) > 0:
        failures.append(f"{validation_name} contains non-positive predictions")
    return failures


def _validation_is_evaluable(validation: dict) -> bool:
    if not validation.get("available"):
        return False
    metrics = validation.get("metrics", {})
    metric_names = ["relative_mae", "mean_absolute_percentage_error"]
    if validation.get("r2_quality_gate_applicable", True):
        metric_names.append("r2")
    return bool(
        int(metrics.get("prediction_count") or 0)
        >= LATENCY_MODEL_MIN_VALIDATION_POINTS
        and all(
            metrics.get(metric_name) is not None
            and math.isfinite(float(metrics[metric_name]))
            for metric_name in metric_names
        )
    )


def _training_range(rows: list) -> dict[str, dict[str, float]]:
    return {
        "cpu_cores": {
            "min": min(float(row.cpu_cores) for row in rows),
            "max": max(float(row.cpu_cores) for row in rows),
        },
        "mem_cap_gb": {
            "min": min(float(row.mem_cap_gb) for row in rows),
            "max": max(float(row.mem_cap_gb) for row in rows),
        },
        "input_scale": {
            "min": min(float(row.input_scale) for row in rows),
            "max": max(float(row.input_scale) for row in rows),
        },
    }


def _format_optional_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.9f}"


def _write_skipped_latency_model_report(
    output_dir: str,
    static_meta: dict[str, object],
    reason: str,
) -> None:
    report_path = os.path.join(output_dir, LATENCY_MODEL_REPORT)
    residuals_path = os.path.join(output_dir, LATENCY_MODEL_RESIDUALS)
    # Always replace any previous residual artifact so a skipped rerun cannot
    # leave stale predictions that appear to belong to the new report.
    with open(residuals_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LATENCY_MODEL_RESIDUAL_FIELDS)
        writer.writeheader()
    report = {
        "report_schema_version": 2,
        "status": "skipped",
        "prediction_ready": False,
        "reason": reason,
        "target_metric": "latency_s",
        "model_name": static_meta.get("model_name", ""),
        "task_family": static_meta.get("task_family", ""),
        "residuals_csv": LATENCY_MODEL_RESIDUALS,
        "residuals_granularity": "header only because model generation was skipped",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
        f.write("\n")
    print(f"[saved] {report_path}")


def write_latency_model_report(
    df: pd.DataFrame,
    static_meta: dict[str, object],
    output_dir: str,
) -> None:
    """Fit validated positive latency models and write report/residual artifacts."""
    model_output_dir = os.path.join(output_dir, LATENCY_MODEL_DIR)
    os.makedirs(model_output_dir, exist_ok=True)

    if "latency_s" not in df.columns:
        _write_skipped_latency_model_report(
            model_output_dir,
            static_meta,
            "latency_s column missing",
        )
        return

    try:
        model_df = _latency_model_frame(df)
    except ValueError as exc:
        _write_skipped_latency_model_report(
            model_output_dir,
            static_meta,
            str(exc),
        )
        return

    if len(model_df) < 3:
        _write_skipped_latency_model_report(
            model_output_dir,
            static_meta,
            f"need at least 3 valid raw rows, got {len(model_df)}",
        )
        return

    points = _aggregate_latency_model_points(model_df)
    if len(points) < 3:
        _write_skipped_latency_model_report(
            model_output_dir,
            static_meta,
            f"need at least 3 unique configuration-scale cases, got {len(points)}",
        )
        return

    model_reports = {}
    fit_predictions: dict[tuple[str, float, float, float], float] = {}
    configuration_predictions: dict[
        tuple[str, float, float, float],
        float,
    ] = {}
    scale_predictions: dict[tuple[str, float, float, float], float] = {}
    top_level_failures = []

    for hardware_model in ("cpu", "gpu"):
        hardware_rows = [
            row for row in points if str(row.hardware_model) == hardware_model
        ]
        if not hardware_rows:
            continue

        try:
            fitted_model = _fit_log_latency_model(hardware_rows, hardware_model)
        except ValueError as exc:
            reason = str(exc)
            model_reports[hardware_model] = {
                "status": "skipped",
                "prediction_ready": False,
                "reason": reason,
                "raw_rows": sum(int(row.repeat_count) for row in hardware_rows),
                "case_rows": len(hardware_rows),
            }
            top_level_failures.append(f"{hardware_model}: fit failed: {reason}")
            continue

        hardware_fit_predictions = {
            _point_key(row): _predict_latency(row, fitted_model)
            for row in hardware_rows
        }
        fit_predictions.update(hardware_fit_predictions)
        fit_metrics = _regression_metrics(
            [float(row.latency_s) for row in hardware_rows],
            [hardware_fit_predictions[_point_key(row)] for row in hardware_rows],
        )

        configuration_validation, hardware_configuration_predictions = (
            _resource_configuration_validation(hardware_rows, hardware_model)
        )
        input_scale_validation, hardware_scale_predictions = (
            _input_scale_validation(hardware_rows, hardware_model)
        )
        configuration_predictions.update(hardware_configuration_predictions)
        scale_predictions.update(hardware_scale_predictions)

        quality_failures = []
        quality_failures.extend(_validation_quality_failures(
            "resource_configuration_holdout",
            configuration_validation,
        ))
        quality_failures.extend(_validation_quality_failures(
            "input_scale_holdout",
            input_scale_validation,
        ))
        if int(fit_metrics["nonfinite_prediction_count"] or 0) > 0:
            quality_failures.append("full fit contains non-finite predictions")
        if int(fit_metrics["nonpositive_prediction_count"] or 0) > 0:
            quality_failures.append("full fit contains non-positive predictions")

        validation_evaluable = bool(
            _validation_is_evaluable(configuration_validation)
            and _validation_is_evaluable(input_scale_validation)
        )
        if quality_failures:
            model_status = "poor_fit" if validation_evaluable else "unvalidated"
        else:
            model_status = "ok"

        coefficients = {
            name: value
            for name, value in zip(
                fitted_model["feature_names"],
                fitted_model["coefficients"],
            )
        }
        if hardware_model == "gpu":
            upper_tail_model = fitted_model.get("gpu_upper_tail", {})
            upper_tail_feature_names = upper_tail_model.get(
                "feature_names",
                [],
            )
            upper_tail_coefficients = upper_tail_model.get(
                "coefficients",
                [],
            )
            upper_tail_report = {
                "type": "continuous_affine_latency_tail",
                "enabled": bool(upper_tail_model.get("enabled")),
                "activation_rule": (
                    "enable when nested one-step-forward spline MAPE exceeds "
                    f"{LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE}, the "
                    "training scale span is at least "
                    f"{LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO}x, "
                    "and all fitted training-configuration slopes are positive"
                ),
                "continuity_rule": (
                    "spline prediction at the upper training boundary plus "
                    "the affine tail's change beyond that boundary"
                ),
                "feature_columns": upper_tail_feature_names,
                "selected_feature_columns": upper_tail_model.get(
                    "selected_feature_names",
                    [],
                ),
                "dropped_feature_columns": upper_tail_model.get(
                    "dropped_feature_names",
                    [],
                ),
                "coefficients": dict(zip(
                    upper_tail_feature_names,
                    upper_tail_coefficients,
                )),
                "minimum_fitted_slope_s_per_scale": upper_tail_model.get(
                    "minimum_fitted_slope_s_per_scale"
                ),
                "training_input_scale_span_ratio": upper_tail_model.get(
                    "training_input_scale_span_ratio"
                ),
                "calibration": upper_tail_model.get("calibration", {}),
            }
            if upper_tail_model.get("reason"):
                upper_tail_report["reason"] = upper_tail_model["reason"]
            input_scale_basis = {
                "type": "continuous_piecewise_linear_spline_in_log_space",
                "knots": fitted_model["input_scale_spline_knots"],
                "knot_rule": "all interior observed training input scales",
                "interpolation": (
                    "piecewise linear in log(input_scale) and log(latency_s)"
                ),
                "lower_extrapolation": (
                    "continue the nearest boundary segment in log-log space"
                ),
                "upper_extrapolation": upper_tail_report,
            }
        else:
            input_scale_basis = {
                "type": "quadratic_response_surface_in_log_space",
                "knots": [],
                "interactions": [
                    "log_input_scale_x_log_cpu_cores",
                    "log_input_scale_x_log_mem_cap_gb",
                    "log_cpu_cores_x_log_mem_cap_gb",
                ],
            }
        model_reports[hardware_model] = {
            "status": model_status,
            "prediction_ready": model_status == "ok",
            "formula": LATENCY_MODEL_FORMULAS[hardware_model],
            "feature_columns": fitted_model["feature_names"],
            "selected_feature_columns": fitted_model["selected_feature_names"],
            "dropped_feature_columns": fitted_model["dropped_feature_names"],
            "coefficients": coefficients,
            "input_scale_basis": input_scale_basis,
            "prediction_estimand": (
                "positive latency prediction for the median-aggregated case"
            ),
            "smearing_correction_applied": False,
            "raw_rows": sum(int(row.repeat_count) for row in hardware_rows),
            "case_rows": len(hardware_rows),
            "training_range": _training_range(hardware_rows),
            "numerical_diagnostics": {
                "solver": "numpy.linalg.lstsq",
                "standardized_before_solve": True,
                "rank": fitted_model["rank"],
                "condition_number": fitted_model["condition_number"],
                **(
                    {
                        "upper_tail_rank": fitted_model.get(
                            "gpu_upper_tail",
                            {},
                        ).get("rank"),
                        "upper_tail_condition_number": fitted_model.get(
                            "gpu_upper_tail",
                            {},
                        ).get("condition_number"),
                    }
                    if hardware_model == "gpu"
                    else {}
                ),
            },
            "metrics": {
                "fit": fit_metrics,
                "resource_configuration_holdout": configuration_validation["metrics"],
                "input_scale_holdout": input_scale_validation["metrics"],
            },
            "validation": {
                "resource_configuration_holdout": configuration_validation,
                "input_scale_holdout": input_scale_validation,
            },
            "quality_gate": {
                "passed": not quality_failures,
                "failures": quality_failures,
            },
        }
        top_level_failures.extend(
            f"{hardware_model}: {failure}" for failure in quality_failures
        )

    if not fit_predictions:
        status = "skipped"
    elif any(
        model_report.get("status") == "poor_fit"
        for model_report in model_reports.values()
    ):
        status = "poor_fit"
    elif any(
        model_report.get("status") != "ok"
        for model_report in model_reports.values()
    ):
        status = "unvalidated"
    else:
        status = "ok"

    def metrics_for_predictions(predictions: dict) -> dict:
        predicted_rows = [row for row in points if _point_key(row) in predictions]
        return _regression_metrics(
            [float(row.latency_s) for row in predicted_rows],
            [predictions[_point_key(row)] for row in predicted_rows],
        )

    residuals_path = os.path.join(
        model_output_dir,
        LATENCY_MODEL_RESIDUALS,
    )
    with open(residuals_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LATENCY_MODEL_RESIDUAL_FIELDS)
        writer.writeheader()
        for case_id, row in enumerate(points):
            key = _point_key(row)
            actual = float(row.latency_s)
            fitted_prediction = fit_predictions.get(key)
            configuration_prediction = configuration_predictions.get(key)
            scale_prediction = scale_predictions.get(key)
            writer.writerow({
                "report_schema_version": 2,
                "case_id": case_id,
                "split": (
                    "out_of_fold_test"
                    if configuration_prediction is not None
                    else "validation_unavailable"
                ),
                "hardware_model": row.hardware_model,
                "cpu_cores": int(row.cpu_cores),
                "mem_cap_gb": int(row.mem_cap_gb),
                "gpu_mode": "on" if row.hardware_model == "gpu" else "off",
                "input_scale": f"{float(row.input_scale):.6f}",
                "repeat_count": int(row.repeat_count),
                "latency_s": f"{actual:.9f}",
                "latency_mean_s": f"{float(row.latency_mean_s):.9f}",
                "latency_std_s": f"{float(row.latency_std_s):.9f}",
                # Backward-compatible aliases now point to the honest
                # configuration-out-of-fold prediction, not an in-cell repeat.
                "predicted_latency_s": _format_optional_float(
                    configuration_prediction
                ),
                "residual_s": _format_optional_float(
                    None
                    if configuration_prediction is None
                    else actual - configuration_prediction
                ),
                "fitted_predicted_latency_s": _format_optional_float(
                    fitted_prediction
                ),
                "fitted_residual_s": _format_optional_float(
                    None
                    if fitted_prediction is None
                    else actual - fitted_prediction
                ),
                "resource_config_oof_predicted_latency_s": _format_optional_float(
                    configuration_prediction
                ),
                "resource_config_oof_residual_s": _format_optional_float(
                    None
                    if configuration_prediction is None
                    else actual - configuration_prediction
                ),
                "max_scale_holdout_predicted_latency_s": _format_optional_float(
                    scale_prediction
                ),
                "max_scale_holdout_residual_s": _format_optional_float(
                    None if scale_prediction is None else actual - scale_prediction
                ),
            })

    fit_metrics = metrics_for_predictions(fit_predictions)
    configuration_metrics = metrics_for_predictions(configuration_predictions)
    input_scale_metrics = metrics_for_predictions(scale_predictions)
    report = {
        "report_schema_version": 2,
        "status": status,
        "prediction_ready": status == "ok",
        "target_metric": "latency_s",
        "model_name": static_meta.get("model_name", ""),
        "task_family": static_meta.get("task_family", ""),
        "input_scale_type": static_meta.get("input_scale_type", "input_scale"),
        "model_type": (
            "separate_cpu_log_response_surface_gpu_log_spline_affine_tail"
        ),
        "positive_prediction_form": True,
        "formula": LATENCY_MODEL_FORMULAS,
        "feature_columns": {
            hardware_model: model_reports.get(hardware_model, {}).get(
                "feature_columns",
                LATENCY_MODEL_FEATURES[hardware_model],
            )
            for hardware_model in ("cpu", "gpu")
        },
        "separation_note": (
            "CPU-off and GPU-on use independent coefficients. Their separate "
            "input-scale/CPU interaction terms are the split-model equivalent "
            "of input_scale x CPU x GPU interactions in one joint model."
        ),
        "aggregation": {
            "group_columns": [
                "hardware_model",
                "cpu_cores",
                "mem_cap_gb",
                "input_scale",
            ],
            "target_statistic": "median",
            "repetitions_split_across_train_and_test": False,
        },
        "rows": len(model_df),
        "raw_rows": len(model_df),
        "case_rows": len(points),
        "fit_case_rows": len(fit_predictions),
        "resource_configuration_oof_test_case_rows": len(
            configuration_predictions
        ),
        "input_scale_holdout_test_case_rows": len(scale_predictions),
        # Kept for report-v1 readers. This is cross-validation, so these are
        # accounting aliases rather than one fixed pair of disjoint row sets.
        "train_rows": len(points),
        "test_rows": len(configuration_predictions),
        "legacy_row_count_note": (
            "train_rows is the number of cases in the final full fit; test_rows "
            "is the number receiving one resource-configuration OOF prediction. "
            "They are not a single fixed mutually exclusive split."
        ),
        "split_rule": (
            "resource configuration: leave one complete (cpu_cores,mem_cap_gb) "
            "out per fold; input scale: train below maximum and hold out maximum"
        ),
        "metrics": {
            "fit": fit_metrics,
            "resource_configuration_holdout": configuration_metrics,
            "input_scale_holdout": input_scale_metrics,
            # Compatibility aliases: "test" is now true configuration OOF.
            "train": fit_metrics,
            "test": configuration_metrics,
        },
        "models": model_reports,
        "quality_gate": {
            "passed": status == "ok",
            "thresholds": {
                # Compatibility key retained; the scope field below makes clear
                # that a fixed-scale extrapolation fold is gated by relative
                # errors rather than configuration-variation R-squared.
                "minimum_validation_r2": LATENCY_MODEL_MIN_VALIDATION_R2,
                "minimum_resource_configuration_validation_r2": (
                    LATENCY_MODEL_MIN_VALIDATION_R2
                ),
                "r2_quality_gate_scope": (
                    "resource_configuration_holdout only"
                ),
                "maximum_validation_relative_mae": (
                    LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE
                ),
                "maximum_validation_mean_absolute_percentage_error": (
                    LATENCY_MODEL_MAX_VALIDATION_MAPE
                ),
                "maximum_single_configuration_fold_relative_mae": (
                    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE
                ),
                "maximum_single_configuration_fold_mean_absolute_percentage_error": (
                    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE
                ),
                "maximum_single_validation_case_relative_error": (
                    LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR
                ),
                "maximum_single_scale_holdout_case_relative_error": (
                    LATENCY_MODEL_MAX_SCALE_CASE_RELATIVE_ERROR
                ),
                "minimum_validation_predictions": (
                    LATENCY_MODEL_MIN_VALIDATION_POINTS
                ),
                "require_finite_positive_predictions": True,
            },
            "failures": top_level_failures,
        },
        "prediction_scope": {
            "supported": (
                "interpolation within each hardware model's training_range; "
                "GPU input-scale interpolation uses a continuous log-log spline; "
                "GPU upper extrapolation can use a calibrated continuous affine "
                "latency tail; the maximum observed input scale is separately "
                "forward-validated"
            ),
            "warning": (
                "Predictions outside profiled CPU, memory, or input-scale ranges "
                "are unvalidated extrapolations."
            ),
        },
        "residuals_csv": LATENCY_MODEL_RESIDUALS,
        "residuals_granularity": (
            "one median-aggregated hardware/cpu/memory/input-scale case per row"
        ),
    }
    report_path = os.path.join(
        model_output_dir,
        LATENCY_MODEL_REPORT,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_clean_json_value(report), f, ensure_ascii=True, indent=2)
        f.write("\n")

    print(f"[saved] {report_path}")
    print(f"[saved] {residuals_path}")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help"}:
        print("usage: plot.py [result_csv]")
        print()
        print("Plot AC-Prof result CSV files.")
        print()
        print("positional arguments:")
        print("  result_csv  CSV path (default: result_all.csv)")
        return

    csv_path = args[0] if args else CSV_PATH
    df = prepare_df(csv_path)
    static_meta = read_static_meta(csv_path)

    scale_type = "input_scale"
    if static_meta.get("input_scale_type"):
        scale_type = static_meta["input_scale_type"]
    elif "input_scale_type" in df.columns:
        types = df["input_scale_type"].dropna().unique()
        if len(types) == 1:
            scale_type = types[0]

    xlabel = scale_type

    output_dir = os.path.dirname(csv_path) or "."

    for group_name, group_df in build_plot_groups(df):
        group_output_dir = os.path.join(output_dir, group_name)
        if SAVE_PNG:
            os.makedirs(group_output_dir, exist_ok=True)
        if group_df.empty:
            print(f"[skip] No data available for {group_name} plots")
            continue

        for metric, title, ylabel, filename in PLOT_METRICS:
            plot_metric(
                group_df,
                metric=metric,
                title=title,
                ylabel=ylabel,
                xlabel=xlabel,
                out_png=(
                    os.path.join(group_output_dir, filename)
                    if SAVE_PNG
                    else None
                ),
            )

        plot_cold_start_bar(
            group_df,
            title="Cold Start by Configuration",
            ylabel="Cold Start (s)",
            out_png=(
                os.path.join(group_output_dir, "cold_start_bar.png")
                if SAVE_PNG
                else None
            ),
        )
    write_latency_model_report(df, static_meta, output_dir)
    model_output_dir = os.path.join(output_dir, LATENCY_MODEL_DIR)
    residuals_path = os.path.join(
        model_output_dir,
        LATENCY_MODEL_RESIDUALS,
    )
    report_path = os.path.join(
        model_output_dir,
        LATENCY_MODEL_REPORT,
    )
    if os.path.exists(residuals_path):
        plot_latency_model_residuals(
            residuals_path,
            (
                os.path.join(
                    model_output_dir,
                    LATENCY_MODEL_RESIDUAL_PLOT,
                )
                if SAVE_PNG
                else None
            ),
        )
        plot_latency_model_fit_curves(
            residuals_path,
            report_path,
            (
                os.path.join(
                    model_output_dir,
                    LATENCY_MODEL_FIT_CURVES_PLOT,
                )
                if SAVE_PNG
                else None
            ),
        )


if __name__ == "__main__":
    main()

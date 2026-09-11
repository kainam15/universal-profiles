"""结果 CSV 整理、分组和基础聚合。"""

import csv
import json
import os

import numpy as np
import pandas as pd

from acprof.analysis.latency_model import (
    GPU_MODE_OFF_VALUES,
    GPU_MODE_ON_VALUES,
)

from acprof.plotting.config import (
    BYTES_PER_GIB,
    COMPUTE_NUMERIC_COLUMNS,
    EXECUTION_PROFILE_NUMERIC_COLUMNS,
    GPU_METRIC_ALIASES,
    NCU_COMPUTE_LEGACY_FALLBACKS,
    PLOT_OUTPUT_DIRS,
    TORCH_EAGER_COMPUTE_LEGACY_FALLBACKS,
)


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


def prepare_df(
    csv_path: str,
    *,
    only_ok: bool = True,
    exclude_warmup: bool = True,
) -> pd.DataFrame:
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
        "input_scale", "input_units_per_request", "input_num_samples",
        "request_payload_bytes",
        "packet_request_wire_bytes_per_request",
        "packet_response_wire_bytes_per_request",
        "packet_total_wire_bytes_per_request",
        "packet_tcp_payload_bytes_per_request",
        "packet_protocol_overhead_bytes_per_request",
        "packet_protocol_overhead_ratio",
        "latency_s", "latency_s_per_input_unit", "latency_request_count",
        "latency_p50_s", "latency_p90_s", "latency_p95_s",
        "latency_std_s", "latency_cv", "latency_iqr_s", "latency_max_s",
        "latency_slow_ratio",
        "latency_app_s", "latency_app_s_per_input_unit",
        "latency_app_request_count",
        "latency_app_p50_s", "latency_app_p90_s", "latency_app_p95_s",
        "latency_app_std_s", "latency_app_cv", "latency_app_iqr_s",
        "latency_app_max_s", "latency_app_slow_ratio",
        "gpu_idle_power_w", "gpu_energy_iters",
        "gpu_avg_power_total_w", "gpu_peak_power_total_w", "gpu_energy_total_j",
        "gpu_avg_power_eff_w", "gpu_peak_power_eff_w", "gpu_energy_eff_j",
        "cpu_idle_power_w", "cpu_energy_iters",
        "cpu_avg_power_total_w", "cpu_peak_power_total_w", "cpu_energy_total_j",
        "cpu_avg_power_eff_w", "cpu_peak_power_eff_w", "cpu_energy_eff_j",
        "vcpu_avg_power_total_w", "vcpu_peak_power_total_w", "vcpu_energy_total_j",
        "vcpu_avg_power_eff_w", "vcpu_peak_power_eff_w", "vcpu_energy_eff_j",
        "vcpu_cpu_share", "vcpu_cpu_time_s",
        "container_attributed_energy_eff_j",
        "container_attributed_samples_per_j",
        "container_attributed_edp_app_js",
        "output_tokens_per_s_app",
        "container_attributed_j_per_output_token",
        "container_attributed_j_per_input_unit",
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
        "container_mem_peak_cgroup_bytes",
        "container_mem_anon_bytes_end", "container_mem_file_bytes_end",
        "container_mem_slab_bytes_end",
        "container_mem_pgfault_delta", "container_mem_pgmajfault_delta",
        "container_mem_workingset_refault_delta",
        "container_mem_high_events_delta", "container_mem_max_events_delta",
        "container_mem_oom_events_delta",
        "container_mem_oom_kill_events_delta",
        "container_mem_pressure_some_stall_pct",
        "container_mem_pressure_full_stall_pct",
        "container_swap_limit_bytes",
        "container_swap_usage_avg_bytes", "container_swap_usage_peak_bytes",
        "container_io_read_bytes_per_request",
        "container_io_write_bytes_per_request",
        "container_io_read_ops_per_request",
        "container_io_write_ops_per_request",
        "container_io_pressure_some_stall_pct",
        "container_io_pressure_full_stall_pct",
        "container_pids_current_end", "container_pids_peak_cgroup",
        "container_pids_max_events_delta",
        "gpu_sm_clock_mhz", "gpu_memory_clock_mhz",
        "gpu_temp_c",
        "gpu_util_avg_pct", "gpu_util_peak_pct",
        "gpu_mem_used_avg_bytes", "gpu_mem_used_peak_bytes",
        "gpu_mem_util_avg_pct", "gpu_mem_util_peak_pct",
        "gpu_mem_total_bytes",
        "cpu_cores", "mem_cap_gb", "warmup",
        "cold_start_container_launch_s", "cold_start_server_setup_s",
        "cold_start_cuda_init_s", "cold_start_model_load_s",
        "cold_start_ready_wait_s", "cold_start_first_predict_app_s",
        "cold_start_s", "throughput_samples_per_s",
        "throughput_samples_per_s_per_cpu_core",
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
        "container_mem_peak_cgroup_bytes": "container_mem_peak_cgroup_gib",
        "container_mem_anon_bytes_end": "container_mem_anon_end_gib",
        "container_mem_file_bytes_end": "container_mem_file_end_gib",
        "container_mem_slab_bytes_end": "container_mem_slab_end_gib",
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

    # The compatibility conversions above can fragment wide historical
    # frames; consolidate once before adding cross-column derived metrics.
    df = df.copy()

    # Current CSVs write the container-attributed value directly. Historical
    # results predate that derived column but contain the same two source
    # measurements, so reconstruct only missing values with the documented
    # CPU/GPU attribution semantics.
    if "vcpu_energy_eff_j" in df.columns and "gpu_mode" in df.columns:
        if "container_attributed_energy_eff_j" not in df.columns:
            df["container_attributed_energy_eff_j"] = np.nan
        attributed_energy = df["container_attributed_energy_eff_j"].copy()
        gpu_on = df["gpu_mode"].map(is_gpu_on)
        gpu_off = df["gpu_mode"].map(is_gpu_off)
        vcpu_energy = pd.to_numeric(df["vcpu_energy_eff_j"], errors="coerce")

        cpu_candidates = gpu_off & attributed_energy.isna() & vcpu_energy.ge(0)
        attributed_energy.loc[cpu_candidates] = vcpu_energy.loc[cpu_candidates]

        if "gpu_energy_eff_j" in df.columns:
            gpu_energy = pd.to_numeric(df["gpu_energy_eff_j"], errors="coerce")
            gpu_candidates = (
                gpu_on
                & attributed_energy.isna()
                & vcpu_energy.ge(0)
                & gpu_energy.ge(0)
            )
            attributed_energy.loc[gpu_candidates] = (
                vcpu_energy.loc[gpu_candidates]
                + gpu_energy.loc[gpu_candidates]
            )
        df["container_attributed_energy_eff_j"] = attributed_energy

    if (
        "container_attributed_energy_eff_j" in df.columns
        and "latency_app_s" in df.columns
    ):
        if "container_attributed_edp_app_js" not in df.columns:
            df["container_attributed_edp_app_js"] = np.nan
        missing_edp = df["container_attributed_edp_app_js"].isna()
        energy = pd.to_numeric(
            df["container_attributed_energy_eff_j"],
            errors="coerce",
        )
        latency = pd.to_numeric(df["latency_app_s"], errors="coerce")
        valid_edp = missing_edp & energy.ge(0) & latency.ge(0)
        df.loc[valid_edp, "container_attributed_edp_app_js"] = (
            energy.loc[valid_edp] * latency.loc[valid_edp]
        )

    if only_ok and "status" in df.columns:
        normalized_status = df["status"].astype(str).str.strip().str.lower()
        df = df[normalized_status == "ok"].copy()

    if exclude_warmup and "warmup" in df.columns:
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


def aggregate_metric(df: pd.DataFrame, metric: str, *, agg_func: str = "mean") -> pd.DataFrame:
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]
    metric_df = df[group_cols + [metric]].copy()
    metric_df = metric_df[metric_df[metric].notna()].copy()
    if metric_df.empty:
        return metric_df

    if agg_func == "median":
        agg_df = metric_df.groupby(group_cols, as_index=False)[metric].median()
    else:
        agg_df = metric_df.groupby(group_cols, as_index=False)[metric].mean()

    agg_df["config"] = agg_df.apply(make_config_label, axis=1)
    agg_df["gpu_on"] = agg_df["gpu_mode"].map(is_gpu_on)
    return agg_df


def aggregate_cold_start(df: pd.DataFrame, *, agg_func: str = "mean") -> pd.DataFrame:
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode"]
    metric_df = df[group_cols + ["cold_start_s"]].copy()
    metric_df = metric_df[metric_df["cold_start_s"].notna()].copy()
    if metric_df.empty:
        return metric_df

    if agg_func == "median":
        agg_df = metric_df.groupby(group_cols, as_index=False)["cold_start_s"].median()
    else:
        agg_df = metric_df.groupby(group_cols, as_index=False)["cold_start_s"].mean()

    agg_df["config"] = agg_df.apply(make_config_label, axis=1)
    agg_df["gpu_on"] = agg_df["gpu_mode"].map(is_gpu_on)
    return agg_df

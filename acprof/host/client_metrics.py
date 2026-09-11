"""测量结果的纯计算与格式化，不初始化客户端或监测器。"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from acprof.host.compute_profile_plan import (
    NCU_ERROR_FIELD,
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    NCU_TOTAL_MFLOP_FIELD,
    TORCH_ERROR_FIELD,
    TORCH_LOGICAL_MFLOP_FIELD,
    compute_mflops as _compute_mflops,
)


def _to_float_or_nan(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _fmt_float(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.6f}"


def _finite_positive(value: Any) -> bool:
    number = _to_float_or_nan(value)
    return math.isfinite(number) and number > 0.0


def _input_units_per_request(input_scale: Any, batch_size: Any) -> float:
    scale = _to_float_or_nan(input_scale)
    batch = _to_float_or_nan(batch_size)
    if math.isfinite(scale) and scale > 0.0 and math.isfinite(batch) and batch > 0.0:
        return scale * batch
    return float("nan")


def _per_positive_denominator(value: Any, denominator: Any) -> float:
    numerator = _to_float_or_nan(value)
    divisor = _to_float_or_nan(denominator)
    if math.isfinite(numerator) and math.isfinite(divisor) and divisor > 0.0:
        return numerator / divisor
    return float("nan")


def _compute_profile_row_metrics(
    profile: Dict[str, Any],
    latency_app_s: Any,
) -> Dict[str, str]:
    """Format independent Torch-eager and NCU metrics for one live row."""
    logical_mflop = _to_float_or_nan(profile.get(TORCH_LOGICAL_MFLOP_FIELD))
    logical_mflops_app = _compute_mflops(logical_mflop, latency_app_s)

    ncu_total_mflop = _to_float_or_nan(profile.get(NCU_TOTAL_MFLOP_FIELD))
    ncu_mflops_app = _compute_mflops(ncu_total_mflop, latency_app_s)
    return {
        # Explicit Torch eager logical FLOPs.
        TORCH_LOGICAL_MFLOP_FIELD: _fmt_float(logical_mflop),
        "model_logical_mflops_app_torch_profiler_eager": _fmt_float(
            logical_mflops_app
        ),
        # Packet latency is not known until merge_packet_latency runs, so use
        # the application-denominator value until the packet merge completes.
        "model_logical_mflops_packet_torch_profiler_eager": _fmt_float(
            logical_mflops_app
        ),
        TORCH_ERROR_FIELD: str(profile.get(TORCH_ERROR_FIELD) or ""),
        # Explicit NCU GPU-executed FLOPs and launch metadata.
        NCU_TOTAL_MFLOP_FIELD: _fmt_float(ncu_total_mflop),
        NCU_TENSOR_MFLOP_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_TENSOR_MFLOP_FIELD))
        ),
        NCU_SCALAR_MFLOP_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_SCALAR_MFLOP_FIELD))
        ),
        NCU_TENSOR_SHARE_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_TENSOR_SHARE_FIELD))
        ),
        "gpu_executed_mflops_app_ncu": _fmt_float(ncu_mflops_app),
        "gpu_executed_mflops_packet_ncu": _fmt_float(ncu_mflops_app),
        NCU_KERNEL_COUNT_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_KERNEL_COUNT_FIELD))
        ),
        NCU_KERNEL_TIME_FIELD: _fmt_float(
            _to_float_or_nan(profile.get(NCU_KERNEL_TIME_FIELD))
        ),
        NCU_ERROR_FIELD: str(profile.get(NCU_ERROR_FIELD) or ""),
    }


EXECUTION_PROFILE_NUMERIC_FIELDS = (
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
)


EXECUTION_PROFILE_ERROR_FIELDS = (
    "compute_profile_error_massif",
    "compute_profile_error_nsys",
)


def _execution_profile_row_metrics(
    profile: Dict[str, Any],
) -> Dict[str, str]:
    """Format intrusive profiler summaries without changing their semantics."""
    result = {
        field: _fmt_float(_to_float_or_nan(profile.get(field)))
        for field in EXECUTION_PROFILE_NUMERIC_FIELDS
    }
    result.update({
        field: str(profile.get(field) or "")
        for field in EXECUTION_PROFILE_ERROR_FIELDS
    })
    return result


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _mean_finite(xs: List[float]) -> float:
    values = [
        value
        for value in (_to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    ]
    return _mean(values)


def _prepared_body_size_bytes(response: Any) -> float:
    """Return the encoded body size from requests' PreparedRequest."""
    prepared = getattr(response, "request", None)
    body = getattr(prepared, "body", None)
    if isinstance(body, str):
        return float(len(body.encode("utf-8")))
    if isinstance(body, (bytes, bytearray, memoryview)):
        return float(len(body))
    return float("nan")


def _input_num_samples(input_metadata: Any) -> float:
    if not isinstance(input_metadata, dict):
        return float("nan")
    for key in ("input_num_samples", "num_samples", "actual_num_samples"):
        value = _to_float_or_nan(input_metadata.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def _percentile_nearest_rank(xs: List[float], percentile: float) -> float:
    values = sorted(
        value
        for value in (_to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    )
    if not values:
        return float("nan")
    rank = int(math.ceil((float(percentile) / 100.0) * len(values)))
    index = min(max(rank - 1, 0), len(values) - 1)
    return values[index]


def _slow_ratio(xs: List[float], *, slow_latency_threshold_s: float) -> float:
    values = [
        value
        for value in (_to_float_or_nan(item) for item in xs)
        if math.isfinite(value)
    ]
    if not values:
        return float("nan")
    return sum(value > slow_latency_threshold_s for value in values) / float(len(values))


def _latency_distribution_metrics(
    prefix: str,
    latencies: List[float],
    *,
    slow_latency_threshold_s: float,
) -> Dict[str, float]:
    values = [
        value
        for value in (_to_float_or_nan(item) for item in latencies)
        if math.isfinite(value)
    ]
    mean = _mean(values)
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        if len(values) >= 2
        else float("nan")
    )
    return {
        f"{prefix}_request_count": float(len(values)),
        f"{prefix}_p50_s": _percentile_nearest_rank(values, 50.0),
        f"{prefix}_p90_s": _percentile_nearest_rank(values, 90.0),
        f"{prefix}_p95_s": _percentile_nearest_rank(values, 95.0),
        f"{prefix}_std_s": std,
        f"{prefix}_cv": (
            std / mean
            if math.isfinite(std) and math.isfinite(mean) and mean > 0.0
            else float("nan")
        ),
        f"{prefix}_iqr_s": (
            _percentile_nearest_rank(values, 75.0)
            - _percentile_nearest_rank(values, 25.0)
            if len(values) >= 2
            else float("nan")
        ),
        f"{prefix}_max_s": max(values) if values else float("nan"),
        f"{prefix}_slow_ratio": _slow_ratio(
            values, slow_latency_threshold_s=slow_latency_threshold_s
        ),
    }


def _eff_negative_warnings(
    *,
    avg_power_eff_w: float,
    peak_power_eff_w: float,
    energy_eff_j: float,
) -> List[str]:
    metrics = [
        ("gpu_avg_power_eff_w", avg_power_eff_w),
        ("gpu_peak_power_eff_w", peak_power_eff_w),
        ("gpu_energy_eff_j", energy_eff_j),
    ]
    return [f"{name}<0" for name, value in metrics if value == value and value < 0.0]


def _named_negative_warnings(metrics: Dict[str, float]) -> List[str]:
    return [f"{name}<0" for name, value in metrics.items() if value == value and value < 0.0]


def _estimate_cpu_cycles(
    latency_s: float,
    cpu_freq_avg_hz: float,
    cpu_cores: float,
    container_cpu_util_avg_pct: float,
) -> float:
    latency = _to_float_or_nan(latency_s)
    freq = _to_float_or_nan(cpu_freq_avg_hz)
    cores = _to_float_or_nan(cpu_cores)
    cpu_util_pct = _to_float_or_nan(container_cpu_util_avg_pct)
    if (
        latency == latency
        and freq == freq
        and cores == cores
        and cpu_util_pct == cpu_util_pct
        and latency > 0.0
        and freq > 0.0
        and cores > 0.0
        and cpu_util_pct >= 0.0
    ):
        return latency * freq * cores * (cpu_util_pct / 100.0)
    return float("nan")


GPU_METRIC_FIELDS = [
    "gpu_idle_power_w",
    "gpu_energy_iters",
    "gpu_avg_power_total_w",
    "gpu_peak_power_total_w",
    "gpu_energy_total_j",
    "gpu_avg_power_eff_w",
    "gpu_peak_power_eff_w",
    "gpu_energy_eff_j",
]


CPU_METRIC_FIELDS = [
    "cpu_idle_power_w",
    "cpu_energy_iters",
    "cpu_avg_power_total_w",
    "cpu_peak_power_total_w",
    "cpu_energy_total_j",
    "cpu_avg_power_eff_w",
    "cpu_peak_power_eff_w",
    "cpu_energy_eff_j",
    "vcpu_cpu_share",
    "vcpu_cpu_time_s",
    "vcpu_avg_power_total_w",
    "vcpu_peak_power_total_w",
    "vcpu_energy_total_j",
    "vcpu_avg_power_eff_w",
    "vcpu_peak_power_eff_w",
    "vcpu_energy_eff_j",
]


RESOURCE_USAGE_METRIC_FIELDS = [
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
    "gpu_util_avg_pct",
    "gpu_util_peak_pct",
    "gpu_mem_used_avg_bytes",
    "gpu_mem_used_peak_bytes",
    "gpu_mem_util_avg_pct",
    "gpu_mem_util_peak_pct",
]


LATENCY_PACKET_DISTRIBUTION_FIELDS = [
    "latency_request_count",
    "latency_p50_s",
    "latency_p90_s",
    "latency_p95_s",
    "latency_std_s",
    "latency_cv",
    "latency_iqr_s",
    "latency_max_s",
    "latency_slow_ratio",
]


LATENCY_APP_DISTRIBUTION_FIELDS = [
    "latency_app_request_count",
    "latency_app_p50_s",
    "latency_app_p90_s",
    "latency_app_p95_s",
    "latency_app_std_s",
    "latency_app_cv",
    "latency_app_iqr_s",
    "latency_app_max_s",
    "latency_app_slow_ratio",
]


EFFICIENCY_METRIC_FIELDS = [
    "container_attributed_energy_eff_j",
    "container_attributed_samples_per_j",
    "container_attributed_edp_app_js",
    "output_tokens_per_s_app",
    "container_attributed_j_per_output_token",
    "container_attributed_j_per_input_unit",
]


MIPS_METRIC_FIELDS = [
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
]


def _nan_metrics(fields: List[str]) -> Dict[str, float]:
    return {field: float("nan") for field in fields}


def _divide_if_number(value: float, divisor: float) -> float:
    value = _to_float_or_nan(value)
    if value == value:
        return value / divisor
    return value


def _gpu_metrics_from_result(result: Any, repeat_in_window: int) -> Dict[str, float]:
    return {
        "gpu_idle_power_w": _to_float_or_nan(result.idle_power_w),
        "gpu_energy_iters": float(result.energy_iters),
        "gpu_avg_power_total_w": _to_float_or_nan(result.avg_power_total_w),
        "gpu_peak_power_total_w": _to_float_or_nan(result.peak_power_total_w),
        "gpu_energy_total_j": _divide_if_number(result.energy_total_j, float(repeat_in_window)),
        "gpu_avg_power_eff_w": _to_float_or_nan(result.avg_power_eff_w),
        "gpu_peak_power_eff_w": _to_float_or_nan(result.peak_power_eff_w),
        "gpu_energy_eff_j": _divide_if_number(result.energy_eff_j, float(repeat_in_window)),
    }


def _cpu_metrics_from_result(result: Any, repeat_in_window: int) -> Dict[str, float]:
    return {
        "cpu_idle_power_w": _to_float_or_nan(result.cpu_idle_power_w),
        "cpu_energy_iters": float(result.cpu_energy_iters),
        "cpu_avg_power_total_w": _to_float_or_nan(result.cpu_avg_power_total_w),
        "cpu_peak_power_total_w": _to_float_or_nan(result.cpu_peak_power_total_w),
        "cpu_energy_total_j": _divide_if_number(result.cpu_energy_total_j, float(repeat_in_window)),
        "cpu_avg_power_eff_w": _to_float_or_nan(result.cpu_avg_power_eff_w),
        "cpu_peak_power_eff_w": _to_float_or_nan(result.cpu_peak_power_eff_w),
        "cpu_energy_eff_j": _divide_if_number(result.cpu_energy_eff_j, float(repeat_in_window)),
        "vcpu_cpu_share": _to_float_or_nan(result.vcpu_cpu_share),
        "vcpu_cpu_time_s": _divide_if_number(result.vcpu_cpu_time_s, float(repeat_in_window)),
        "vcpu_avg_power_total_w": _to_float_or_nan(result.vcpu_avg_power_total_w),
        "vcpu_peak_power_total_w": _to_float_or_nan(result.vcpu_peak_power_total_w),
        "vcpu_energy_total_j": _divide_if_number(result.vcpu_energy_total_j, float(repeat_in_window)),
        "vcpu_avg_power_eff_w": _to_float_or_nan(result.vcpu_avg_power_eff_w),
        "vcpu_peak_power_eff_w": _to_float_or_nan(result.vcpu_peak_power_eff_w),
        "vcpu_energy_eff_j": _divide_if_number(result.vcpu_energy_eff_j, float(repeat_in_window)),
    }


def _derived_efficiency_metrics(
    *,
    gpu_mode: str,
    batch_size: int,
    latency_app_s: float,
    output_token_count_avg: float,
    gpu_energy_eff_j: float,
    vcpu_energy_eff_j: float,
    input_units_per_request: float = float("nan"),
) -> Dict[str, float]:
    latency = _to_float_or_nan(latency_app_s)
    output_tokens = _to_float_or_nan(output_token_count_avg)
    gpu_energy = _to_float_or_nan(gpu_energy_eff_j)
    vcpu_energy = _to_float_or_nan(vcpu_energy_eff_j)

    required_energy = [vcpu_energy]
    if str(gpu_mode).strip().lower() == "on":
        required_energy.append(gpu_energy)
    attributed_energy = (
        sum(required_energy)
        if all(math.isfinite(value) and value >= 0.0 for value in required_energy)
        else float("nan")
    )

    samples_per_j = (
        float(batch_size) / attributed_energy
        if math.isfinite(attributed_energy)
        and attributed_energy > 0.0
        and batch_size > 0
        else float("nan")
    )
    edp_app = (
        attributed_energy * latency
        if math.isfinite(attributed_energy)
        and attributed_energy >= 0.0
        and math.isfinite(latency)
        and latency > 0.0
        else float("nan")
    )
    output_tokens_per_s = (
        output_tokens / latency
        if math.isfinite(output_tokens)
        and output_tokens > 0.0
        and math.isfinite(latency)
        and latency > 0.0
        else float("nan")
    )
    joules_per_output_token = (
        attributed_energy / output_tokens
        if math.isfinite(attributed_energy)
        and attributed_energy >= 0.0
        and math.isfinite(output_tokens)
        and output_tokens > 0.0
        else float("nan")
    )
    joules_per_input_unit = _per_positive_denominator(
        attributed_energy,
        input_units_per_request,
    )
    return {
        "container_attributed_energy_eff_j": attributed_energy,
        "container_attributed_samples_per_j": samples_per_j,
        "container_attributed_edp_app_js": edp_app,
        "output_tokens_per_s_app": output_tokens_per_s,
        "container_attributed_j_per_output_token": joules_per_output_token,
        "container_attributed_j_per_input_unit": joules_per_input_unit,
    }


def _resource_usage_metrics_from_result(
    result: Any,
    repeat_in_window: int,
) -> Dict[str, float]:
    io_read_bytes = _to_float_or_nan(
        getattr(result, "container_io_read_bytes", float("nan"))
    )
    io_write_bytes = _to_float_or_nan(
        getattr(result, "container_io_write_bytes", float("nan"))
    )
    io_read_ops = _to_float_or_nan(
        getattr(result, "container_io_read_ops", float("nan"))
    )
    io_write_ops = _to_float_or_nan(
        getattr(result, "container_io_write_ops", float("nan"))
    )
    if repeat_in_window > 0:
        io_read_bytes /= float(repeat_in_window)
        io_write_bytes /= float(repeat_in_window)
        io_read_ops /= float(repeat_in_window)
        io_write_ops /= float(repeat_in_window)
    else:
        io_read_bytes = float("nan")
        io_write_bytes = float("nan")
        io_read_ops = float("nan")
        io_write_ops = float("nan")
    return {
        "resource_usage_iters": float(result.resource_usage_iters),
        "container_cpu_util_avg_pct": _to_float_or_nan(result.container_cpu_util_avg_pct),
        "container_cpu_util_peak_pct": _to_float_or_nan(result.container_cpu_util_peak_pct),
        "container_cpu_nr_periods_delta": _to_float_or_nan(
            getattr(result, "container_cpu_nr_periods_delta", float("nan"))
        ),
        "container_cpu_nr_throttled_delta": _to_float_or_nan(
            getattr(result, "container_cpu_nr_throttled_delta", float("nan"))
        ),
        "container_cpu_throttled_period_ratio_pct": _to_float_or_nan(
            getattr(
                result,
                "container_cpu_throttled_period_ratio_pct",
                float("nan"),
            )
        ),
        "container_cpu_throttled_time_s_per_request": (
            _divide_if_number(
                getattr(result, "container_cpu_throttled_time_s", float("nan")),
                float(repeat_in_window),
            )
            if repeat_in_window > 0
            else float("nan")
        ),
        "container_cpu_pressure_some_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_cpu_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_cpu_pressure_full_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_cpu_pressure_full_stall_pct",
                float("nan"),
            )
        ),
        "cpu_freq_avg_hz": _to_float_or_nan(
            getattr(result, "cpu_freq_avg_hz", float("nan"))
        ),
        "cpu_freq_peak_hz": _to_float_or_nan(
            getattr(result, "cpu_freq_peak_hz", float("nan"))
        ),
        "container_mem_usage_avg_bytes": _to_float_or_nan(result.container_mem_usage_avg_bytes),
        "container_mem_usage_peak_bytes": _to_float_or_nan(result.container_mem_usage_peak_bytes),
        "container_mem_util_avg_pct": _to_float_or_nan(result.container_mem_util_avg_pct),
        "container_mem_util_peak_pct": _to_float_or_nan(result.container_mem_util_peak_pct),
        "container_mem_peak_cgroup_bytes": _to_float_or_nan(
            getattr(result, "container_mem_peak_cgroup_bytes", float("nan"))
        ),
        "container_mem_anon_bytes_end": _to_float_or_nan(
            getattr(result, "container_mem_anon_bytes_end", float("nan"))
        ),
        "container_mem_file_bytes_end": _to_float_or_nan(
            getattr(result, "container_mem_file_bytes_end", float("nan"))
        ),
        "container_mem_slab_bytes_end": _to_float_or_nan(
            getattr(result, "container_mem_slab_bytes_end", float("nan"))
        ),
        "container_mem_pgfault_delta": _to_float_or_nan(
            getattr(result, "container_mem_pgfault_delta", float("nan"))
        ),
        "container_mem_pgmajfault_delta": _to_float_or_nan(
            getattr(result, "container_mem_pgmajfault_delta", float("nan"))
        ),
        "container_mem_workingset_refault_delta": _to_float_or_nan(
            getattr(
                result,
                "container_mem_workingset_refault_delta",
                float("nan"),
            )
        ),
        "container_mem_high_events_delta": _to_float_or_nan(
            getattr(result, "container_mem_high_events_delta", float("nan"))
        ),
        "container_mem_max_events_delta": _to_float_or_nan(
            getattr(result, "container_mem_max_events_delta", float("nan"))
        ),
        "container_mem_oom_events_delta": _to_float_or_nan(
            getattr(result, "container_mem_oom_events_delta", float("nan"))
        ),
        "container_mem_oom_kill_events_delta": _to_float_or_nan(
            getattr(result, "container_mem_oom_kill_events_delta", float("nan"))
        ),
        "container_mem_pressure_some_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_mem_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_mem_pressure_full_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_mem_pressure_full_stall_pct",
                float("nan"),
            )
        ),
        "container_swap_limit_bytes": _to_float_or_nan(
            getattr(result, "container_swap_limit_bytes", float("nan"))
        ),
        "container_swap_usage_avg_bytes": _to_float_or_nan(
            getattr(result, "container_swap_usage_avg_bytes", float("nan"))
        ),
        "container_swap_usage_peak_bytes": _to_float_or_nan(
            getattr(result, "container_swap_usage_peak_bytes", float("nan"))
        ),
        "container_io_read_bytes_per_request": io_read_bytes,
        "container_io_write_bytes_per_request": io_write_bytes,
        "container_io_read_ops_per_request": io_read_ops,
        "container_io_write_ops_per_request": io_write_ops,
        "container_io_pressure_some_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_io_pressure_some_stall_pct",
                float("nan"),
            )
        ),
        "container_io_pressure_full_stall_pct": _to_float_or_nan(
            getattr(
                result,
                "container_io_pressure_full_stall_pct",
                float("nan"),
            )
        ),
        "container_pids_current_end": _to_float_or_nan(
            getattr(result, "container_pids_current_end", float("nan"))
        ),
        "container_pids_peak_cgroup": _to_float_or_nan(
            getattr(result, "container_pids_peak_cgroup", float("nan"))
        ),
        "container_pids_max_events_delta": _to_float_or_nan(
            getattr(result, "container_pids_max_events_delta", float("nan"))
        ),
        "gpu_util_avg_pct": _to_float_or_nan(result.gpu_util_avg_pct),
        "gpu_util_peak_pct": _to_float_or_nan(result.gpu_util_peak_pct),
        "gpu_mem_used_avg_bytes": _to_float_or_nan(result.gpu_mem_used_avg_bytes),
        "gpu_mem_used_peak_bytes": _to_float_or_nan(result.gpu_mem_used_peak_bytes),
        "gpu_mem_util_avg_pct": _to_float_or_nan(result.gpu_mem_util_avg_pct),
        "gpu_mem_util_peak_pct": _to_float_or_nan(result.gpu_mem_util_peak_pct),
    }


def _gpu_runtime_metrics_from_result(resource_usage_result: Any) -> Dict[str, Any]:
    pstate = str(
        getattr(resource_usage_result, "gpu_pstate", "nan")
        if resource_usage_result is not None
        else "nan"
    ).strip().upper()
    if not (
        pstate.startswith("P")
        and pstate[1:].isdigit()
        and 0 <= int(pstate[1:]) <= 15
    ):
        pstate = "nan"
    return {
        "gpu_sm_clock_mhz": _to_float_or_nan(
            getattr(resource_usage_result, "gpu_sm_clock_mhz", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
        "gpu_memory_clock_mhz": _to_float_or_nan(
            getattr(resource_usage_result, "gpu_memory_clock_mhz", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
        "gpu_pstate": pstate,
        "gpu_temp_c": _to_float_or_nan(
            getattr(resource_usage_result, "gpu_temp_c", float("nan"))
            if resource_usage_result is not None
            else float("nan")
        ),
    }


def _mips_metrics_from_result(result: Any) -> Dict[str, float]:
    return {
        "cpu_instructions_per_request": _to_float_or_nan(
            result.instructions_per_request
        ),
        "cpu_mips_app": _to_float_or_nan(result.cpu_mips_app),
        "cpu_mips_packet": float("nan"),
        "cpu_perf_elapsed_s": _to_float_or_nan(result.perf_elapsed_s),
        "cpu_cache_references_per_request": _to_float_or_nan(
            getattr(result, "cache_references_per_request", float("nan"))
        ),
        "cpu_cache_misses_per_request": _to_float_or_nan(
            getattr(result, "cache_misses_per_request", float("nan"))
        ),
        "cpu_cache_miss_rate_pct": _to_float_or_nan(
            getattr(result, "cache_miss_rate_pct", float("nan"))
        ),
        "cpu_dtlb_loads_per_request": _to_float_or_nan(
            getattr(result, "dtlb_loads_per_request", float("nan"))
        ),
        "cpu_dtlb_load_misses_per_request": _to_float_or_nan(
            getattr(result, "dtlb_load_misses_per_request", float("nan"))
        ),
        "cpu_dtlb_load_miss_rate_pct": _to_float_or_nan(
            getattr(result, "dtlb_load_miss_rate_pct", float("nan"))
        ),
    }


def _idle_power_debug_stats(values: List[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_idle_valid_count": 0,
            f"{prefix}_idle_min_w": float("nan"),
            f"{prefix}_idle_max_w": float("nan"),
            f"{prefix}_idle_mean_w": float("nan"),
            f"{prefix}_idle_rel_range_so_far": float("nan"),
        }

    mean_idle = sum(values) / len(values)
    relative_range = (
        (max(values) - min(values)) / mean_idle
        if mean_idle > 0.0
        else float("nan")
    )
    return {
        f"{prefix}_idle_valid_count": len(values),
        f"{prefix}_idle_min_w": min(values),
        f"{prefix}_idle_max_w": max(values),
        f"{prefix}_idle_mean_w": mean_idle,
        f"{prefix}_idle_rel_range_so_far": relative_range,
    }


def _idle_debug_stats(values: List[float]) -> Dict[str, float]:
    return _idle_power_debug_stats(values, "cpu")

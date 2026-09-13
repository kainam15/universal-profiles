"""当前图表配置、颜色和数值字段。"""

GPU_GREEN = (0.18, 0.62, 0.28)
GPU_LIGHT_GREEN = (0.49, 0.78, 0.53)
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
COMPUTE_NUMERIC_COLUMNS = [
    "model_logical_mflop_per_request_torch_profiler_eager",
    "model_logical_mflops_app_torch_profiler_eager",
    "model_logical_mflops_packet_torch_profiler_eager",
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
ENERGY_POWER_OVERVIEW_PLOTS = [
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
        "Estimated vCPU-Attributed Energy and Power Overview vs. Input Scale",
        "vcpu_estimated_energy_power_overview_vs_scale.png",
    ),
]
# Each overview is:
# (title, filename, rows, columns, row-major panels, shared-y index groups).
# A panel is (metric, panel title, y-axis label); None leaves an empty slot.
METRIC_OVERVIEW_PLOTS = [
    (
        "Latency Overview vs. Input Scale",
        "latency_overview_vs_scale.png",
        3,
        2,
        (
            ("latency_s", "Packet Latency", "Latency (s)"),
            ("latency_app_s", "Application Latency", ""),
            (
                "latency_s_per_input_unit",
                "Packet Latency per Input Unit",
                "Latency (s/input unit)",
            ),
            (
                "latency_app_s_per_input_unit",
                "Application Latency per Input Unit",
                "",
            ),
            ("latency_cv", "Packet Latency CV", "Coefficient of variation"),
            ("latency_app_cv", "Application Latency CV", ""),
        ),
        ((0, 1), (2, 3), (4, 5)),
    ),
    (
        "Service Efficiency Overview vs. Input Scale",
        "service_efficiency_overview_vs_scale.png",
        3,
        2,
        (
            ("throughput_samples_per_s", "Throughput", "Samples/s"),
            (
                "throughput_samples_per_s_per_cpu_core",
                "Throughput per CPU Core",
                "Samples/s/CPU core",
            ),
            (
                "container_attributed_energy_eff_j",
                "Container-Attributed Effective Energy",
                "Estimated energy (J/request)",
            ),
            (
                "container_attributed_j_per_input_unit",
                "Container-Attributed Energy per Input Unit",
                "Estimated energy (J/input unit)",
            ),
            (
                "container_attributed_samples_per_j",
                "Container-Attributed Energy Efficiency",
                "Samples/J",
            ),
            None,
        ),
        (),
    ),
    (
        "Packet Transport Overview vs. Input Scale",
        "packet_overview_vs_scale.png",
        1,
        2,
        (
            (
                "packet_total_wire_bytes_per_request",
                "Captured Wire Bytes per Request",
                "Wire bytes/request",
            ),
            (
                "packet_protocol_overhead_ratio",
                "Captured L2-L4 Protocol Overhead",
                "Protocol overhead ratio",
            ),
        ),
        (),
    ),
    (
        "Torch Profiler Eager Logical Compute Overview vs. Input Scale",
        "torch_compute_overview_vs_scale.png",
        3,
        1,
        (
            (
                "model_logical_mflop_per_request_torch_profiler_eager",
                "Logical FLOP per Request",
                "Logical MFLOP/request",
            ),
            (
                "model_logical_mflops_app_torch_profiler_eager",
                "Logical Throughput (Application Latency)",
                "Logical MFLOPS",
            ),
            (
                "model_logical_mflops_packet_torch_profiler_eager",
                "Logical Throughput (Packet Latency)",
                "Logical MFLOPS",
            ),
        ),
        (),
    ),
    (
        "NCU GPU-Executed Arithmetic Overview vs. Input Scale",
        "ncu_arithmetic_overview_vs_scale.png",
        2,
        2,
        (
            (
                "gpu_executed_mflop_per_request_ncu",
                "Total Executed FLOP per Request",
                "GPU-executed MFLOP/request",
            ),
            (
                "gpu_executed_tensor_mflop_per_request_ncu",
                "Tensor Executed FLOP per Request",
                "",
            ),
            (
                "gpu_executed_scalar_mflop_per_request_ncu",
                "Scalar Executed FLOP per Request",
                "GPU-executed scalar MFLOP/request",
            ),
            (
                "gpu_executed_tensor_share_pct_ncu",
                "Tensor FLOP Share",
                "Tensor FLOP share (%)",
            ),
        ),
        ((0, 1),),
    ),
    (
        "NCU Throughput and Kernel Overview vs. Input Scale",
        "ncu_runtime_overview_vs_scale.png",
        2,
        2,
        (
            (
                "gpu_executed_mflops_app_ncu",
                "Executed Throughput (Application Latency)",
                "GPU-executed MFLOPS",
            ),
            (
                "gpu_executed_mflops_packet_ncu",
                "Executed Throughput (Packet Latency)",
                "",
            ),
            (
                "gpu_kernel_launch_count_per_request_ncu",
                "Kernel Launches per Request",
                "Kernel launches/request",
            ),
            (
                "gpu_kernel_time_sum_ms_per_request_ncu",
                "Summed Kernel Time per Request",
                "Summed kernel time (ms/request)",
            ),
        ),
        ((0, 1),),
    ),
    (
        "Nsight Systems Timing Overview vs. Input Scale",
        "nsys_timing_overview_vs_scale.png",
        2,
        2,
        (
            (
                "host_inference_wall_time_ms_per_request_nsys",
                "Host Inference Wall Time",
                "Host wall time (ms/request)",
            ),
            (
                "cuda_api_time_sum_ms_per_request_nsys",
                "Summed CUDA API Time",
                "Summed CUDA API time (ms/request)",
            ),
            (
                "gpu_kernel_time_sum_ms_per_request_nsys",
                "Summed GPU Kernel Time",
                "Summed GPU kernel time (ms/request)",
            ),
            (
                "gpu_memcpy_time_sum_ms_per_request_nsys",
                "Summed GPU Memcpy Time",
                "Summed GPU memcpy time (ms/request)",
            ),
        ),
        (),
    ),
    (
        "Container CPU Overview vs. Input Scale",
        "container_cpu_overview_vs_scale.png",
        3,
        1,
        (
            (
                "container_cpu_util_avg_pct",
                "CPU Utilization",
                "CPU utilization (%)",
            ),
            (
                "container_cpu_throttled_period_ratio_pct",
                "CPU Throttled Period Ratio",
                "Throttled periods (%)",
            ),
            (
                "container_cpu_pressure_some_stall_pct",
                "CPU Pressure Some Stall",
                "PSI some stall (%)",
            ),
        ),
        (),
    ),
    (
        "CPU Execution Overview vs. Input Scale",
        "cpu_execution_overview_vs_scale.png",
        3,
        1,
        (
            ("cpu_mips_packet", "Packet-Latency CPU MIPS", "MIPS"),
            (
                "cpu_instructions_per_request",
                "CPU Instructions per Request",
                "Instructions/request",
            ),
            (
                "cpu_cycles_est_packet",
                "Packet-Latency Estimated CPU Cycles",
                "Estimated CPU cycles/request",
            ),
        ),
        (),
    ),
    (
        "CPU Cache and dTLB Behavior Overview vs. Input Scale",
        "cpu_memory_behavior_overview_vs_scale.png",
        2,
        2,
        (
            (
                "cpu_cache_misses_per_request",
                "Cache Misses per Request",
                "Misses/request",
            ),
            (
                "cpu_cache_miss_rate_pct",
                "Cache Miss Rate",
                "Miss rate (%)",
            ),
            (
                "cpu_dtlb_load_misses_per_request",
                "dTLB Load Misses per Request",
                "Misses/request",
            ),
            (
                "cpu_dtlb_load_miss_rate_pct",
                "dTLB Load Miss Rate",
                "Miss rate (%)",
            ),
        ),
        # Cache-miss and dTLB-miss counts use the same unit, but their
        # magnitudes can differ by orders of magnitude.  Give the count
        # panels independent scales so the smaller dTLB series stays legible;
        # the two percentage panels remain directly comparable.
        ((1, 3),),
    ),
    (
        "Container Memory and Process Overview vs. Input Scale",
        "container_memory_process_overview_vs_scale.png",
        3,
        2,
        (
            (
                "container_mem_usage_avg_gib",
                "Average Memory Usage",
                "Memory (GiB)",
            ),
            (
                "container_mem_peak_cgroup_gib",
                "cgroup Lifetime Memory Peak",
                "",
            ),
            (
                "container_mem_util_avg_pct",
                "Memory Utilization",
                "Memory utilization (%)",
            ),
            (
                "container_mem_pressure_full_stall_pct",
                "Memory Full-Pressure Stall",
                "PSI full stall (%)",
            ),
            (
                "container_mem_pgmajfault_delta",
                "Major Page Faults",
                "Major faults/window",
            ),
            (
                "container_pids_peak_cgroup",
                "cgroup Lifetime PID Peak",
                "Tasks",
            ),
        ),
        ((0, 1),),
    ),
    (
        "Container Block I/O Overview vs. Input Scale",
        "container_io_overview_vs_scale.png",
        1,
        2,
        (
            (
                "container_io_read_ops_per_request",
                "Block Read Operations",
                "Operations/request",
            ),
            (
                "container_io_write_ops_per_request",
                "Block Write Operations",
                "Operations/request",
            ),
        ),
        # Read and write operation counts can differ by orders of magnitude.
        # Scale each panel independently so small read/write values stay visible.
        (),
    ),
    (
        "GPU Resource Overview vs. Input Scale",
        "gpu_resource_overview_vs_scale.png",
        2,
        2,
        (
            ("gpu_util_avg_pct", "GPU Utilization", "Utilization (%)"),
            (
                "gpu_mem_util_avg_pct",
                "GPU Memory Capacity Utilization",
                "",
            ),
            (
                "gpu_mem_used_avg_gib",
                "GPU Memory Used",
                "GPU memory used (GiB)",
            ),
            None,
        ),
        ((0, 1),),
    ),
]
# Historical single-plot metadata is retained only as a migration map for old
# filenames; the main plotting workflow no longer iterates it directly.
# Massif remains standalone because it measures process-lifetime memory with a
# different collection scope from the cgroup-window memory overview.
PLOT_METRICS = [
    ('cpu_heap_peak_total_gib_massif', 'Massif Process-Lifetime Peak Memory vs. Input Scale', 'Peak heap + extra + stack (GiB)', 'massif_cpu_heap_peak_total_vs_scale.png'),
]
PLOT_OUTPUT_DIRS = ("cpu", "gpu", "gpu+cpu")
LATENCY_MODEL_RESIDUAL_PLOT = "latency_model_residuals.png"
LATENCY_MODEL_FIT_CURVES_PLOT = "latency_model_fit_curves.png"
RESOURCE_FEASIBILITY_PLOT = "resource_feasibility_heatmap.png"
TAIL_LATENCY_PLOT = "tail_latency_overview_vs_scale.png"
LATENCY_ENERGY_PARETO_PLOT = "latency_energy_pareto.png"
COLD_START_BREAKDOWN_PLOT = "cold_start_breakdown.png"
FEASIBILITY_STATE_SPECS = (
    ("ok", "OK", "#4daf4a", "OK"),
    ("warn", "Warning", "#ffd54f", "WARN"),
    ("mixed", "Partial failure", "#ff9800", "PART"),
    ("timeout", "Request timeout", "#7b1fa2", "TIME"),
    ("startup_oom", "Startup OOM", "#e53935", "OOM-S"),
    (
        "pruned_startup_oom",
        "Inferred startup OOM (not measured)",
        "#b71c1c",
        "P-OOM",
    ),
    ("runtime_oom", "Runtime OOM", "#8e0000", "OOM-R"),
    ("skipped", "Skipped after timeout", "#90a4ae", "SKIP"),
    ("error", "Other error", "#5d4037", "ERR"),
    ("unknown", "Unmeasured / unknown", "#eeeeee", "N/A"),
)
FEASIBILITY_STATE_INDEX = {
    state: index
    for index, (state, _label, _color, _short_label)
    in enumerate(FEASIBILITY_STATE_SPECS)
}
FEASIBILITY_TEXT_COLORS = {
    "timeout": "white",
    "startup_oom": "white",
    "pruned_startup_oom": "white",
    "runtime_oom": "white",
    "error": "white",
}
COLD_START_PHASE_SPECS = (
    ("cold_start_container_launch_s", "Container launch", "#4c78a8"),
    ("cold_start_server_setup_s", "Server setup", "#f58518"),
    ("cold_start_cuda_init_s", "CUDA init", "#eeca3b"),
    ("cold_start_model_load_s", "Model load", "#54a24b"),
    ("cold_start_ready_wait_s", "Ready wait", "#b279a2"),
)

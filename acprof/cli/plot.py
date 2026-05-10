"""AC-Prof Universal Profiler - Plotting tool adapted for generalized CSV schema."""

import csv
import colorsys
import json
import math
import os
import sys

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
        "throughput_samples_per_s",
        "Throughput vs. Input Scale",
        "Samples/s",
        "throughput_vs_scale.png",
    ),
    (
        "compute_mflops",
        "Compute Throughput vs. Input Scale",
        "MFLOPS",
        "compute_mflops_vs_scale.png",
    ),
    (
        "container_cpu_util_avg_pct",
        "Container CPU Utilization vs. Input Scale",
        "CPU Utilization (%)",
        "container_cpu_util_vs_scale.png",
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
LATENCY_MODEL_REPORT = "latency_model_report.json"
LATENCY_MODEL_RESIDUALS = "latency_model_residuals.csv"
LATENCY_MODEL_FEATURES = [
    "intercept",
    "input_scale",
    "inverse_cpu_cores",
    "mem_cap_gb",
    "gpu_on",
]


def make_config_label(row) -> str:
    gpu = str(row["gpu_mode"]).lower()
    cpu = int(row["cpu_cores"])
    mem = int(row["mem_cap_gb"])
    if gpu in ("on", "1", "true", "yes", "gpu"):
        return f"GPU+CPU{cpu}+Mem{mem}"
    else:
        return f"CPU+CPU{cpu}+Mem{mem}"


def is_gpu_on(gpu_mode: object) -> bool:
    return str(gpu_mode).lower() in {"on", "1", "true", "yes", "gpu"}


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
        "latency_s", "latency_p50_s", "latency_p90_s", "latency_p95_s", "latency_slow_ratio",
        "latency_app_s", "latency_app_p50_s", "latency_app_p90_s", "latency_app_p95_s", "latency_app_slow_ratio",
        "gpu_energy_iters",
        "gpu_avg_power_total_w", "gpu_peak_power_total_w", "gpu_energy_total_j",
        "gpu_avg_power_eff_w", "gpu_peak_power_eff_w", "gpu_energy_eff_j",
        "cpu_avg_power_eff_w", "cpu_peak_power_eff_w", "cpu_energy_eff_j",
        "vcpu_avg_power_eff_w", "vcpu_peak_power_eff_w", "vcpu_energy_eff_j",
        "vcpu_cpu_share", "vcpu_cpu_time_s",
        "resource_usage_iters",
        "container_cpu_util_avg_pct", "container_cpu_util_peak_pct",
        "cpu_freq_avg_hz", "cpu_freq_peak_hz",
        "cpu_cycles_est_app", "cpu_cycles_est_packet",
        "cpu_instructions_per_request", "cpu_mips_app", "cpu_mips_packet",
        "cpu_perf_elapsed_s",
        "container_mem_usage_avg_bytes", "container_mem_usage_peak_bytes",
        "container_mem_util_avg_pct", "container_mem_util_peak_pct",
        "gpu_util_avg_pct", "gpu_util_peak_pct",
        "gpu_mem_used_avg_bytes", "gpu_mem_used_peak_bytes",
        "gpu_mem_util_avg_pct", "gpu_mem_util_peak_pct",
        "gpu_mem_total_bytes",
        "cpu_cores", "mem_cap_gb", "warmup", "cold_start_s",
        "throughput_samples_per_s",
        "model_mflop_per_request", "compute_mflops_app", "compute_mflops",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bytes_to_gib = {
        "container_mem_usage_avg_bytes": "container_mem_usage_avg_gib",
        "container_mem_usage_peak_bytes": "container_mem_usage_peak_gib",
        "gpu_mem_used_avg_bytes": "gpu_mem_used_avg_gib",
        "gpu_mem_used_peak_bytes": "gpu_mem_used_peak_gib",
        "gpu_mem_total_bytes": "gpu_mem_total_gib",
    }
    for source_col, target_col in bytes_to_gib.items():
        if source_col in df.columns:
            df[target_col] = df[source_col] / float(BYTES_PER_GIB)

    if ONLY_OK and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    if EXCLUDE_WARMUP and "warmup" in df.columns:
        df = df[df["warmup"].fillna(0).astype(int) == 0].copy()

    need = {"input_scale", "cpu_cores", "mem_cap_gb", "gpu_mode"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["config"] = df.apply(make_config_label, axis=1)
    df = df[df["input_scale"].notna() & (df["input_scale"] > 0)].copy()

    return df


def read_static_meta(csv_path: str) -> dict[str, str]:
    static_meta_path = os.path.join(os.path.dirname(csv_path) or ".", "static_meta.csv")
    if not os.path.exists(static_meta_path):
        return {}

    with open(static_meta_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
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
            color = color_for_mem(mem, mem_rank_map[mem])
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


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[idx]] for idx, row in enumerate(matrix)]

    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        pivot_value = augmented[col][col]
        for idx in range(col, size + 1):
            augmented[col][idx] /= pivot_value

        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for idx in range(col, size + 1):
                augmented[row][idx] -= factor * augmented[col][idx]

    return [augmented[row][size] for row in range(size)]


def _model_features(row) -> list[float]:
    cpu_cores = float(row.cpu_cores)
    return [
        1.0,
        float(row.input_scale),
        1.0 / cpu_cores,
        float(row.mem_cap_gb),
        1.0 if is_gpu_on(row.gpu_mode) else 0.0,
    ]


def _fit_linear_model(rows: list) -> list[float]:
    feature_count = len(LATENCY_MODEL_FEATURES)
    xtx = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    xty = [0.0 for _ in range(feature_count)]

    for row in rows:
        features = _model_features(row)
        target = float(row.latency_s)
        for i in range(feature_count):
            xty[i] += features[i] * target
            for j in range(feature_count):
                xtx[i][j] += features[i] * features[j]

    return _solve_linear_system(xtx, xty)


def _predict_latency(row, coefficients: list[float]) -> float:
    return sum(
        coefficient * feature
        for coefficient, feature in zip(coefficients, _model_features(row))
    )


def _regression_metrics(actuals: list[float], predictions: list[float]) -> dict[str, float | None]:
    if not actuals:
        return {"r2": None, "mae": None, "rmse": None}

    errors = [actual - predicted for actual, predicted in zip(actuals, predictions)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_actual = sum(actuals) / len(actuals)
    ss_tot = sum((actual - mean_actual) ** 2 for actual in actuals)
    ss_res = sum(error * error for error in errors)
    r2 = None if ss_tot <= 0.0 else 1.0 - (ss_res / ss_tot)
    return {"r2": r2, "mae": mae, "rmse": rmse}


def _clean_json_value(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    return value


def _latency_model_rows(df: pd.DataFrame) -> list:
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
    model_df["gpu_on_sort"] = model_df["gpu_mode"].map(is_gpu_on)
    model_df = model_df.sort_values(
        ["gpu_on_sort", "cpu_cores", "mem_cap_gb", "input_scale", "source_row"]
    ).reset_index(drop=True)
    return list(model_df.itertuples(index=False))


def _split_latency_model_rows(rows: list) -> tuple[list, list]:
    test_rows = [row for idx, row in enumerate(rows) if (idx + 1) % 5 == 0]
    train_rows = [row for idx, row in enumerate(rows) if (idx + 1) % 5 != 0]

    if not test_rows and len(rows) > len(LATENCY_MODEL_FEATURES):
        test_rows = [rows[-1]]
        train_rows = rows[:-1]

    return train_rows, test_rows


def _write_skipped_latency_model_report(
    output_dir: str,
    static_meta: dict[str, str],
    reason: str,
) -> None:
    report_path = os.path.join(output_dir, LATENCY_MODEL_REPORT)
    report = {
        "status": "skipped",
        "reason": reason,
        "target_metric": "latency_s",
        "model_name": static_meta.get("model_name", ""),
        "task_family": static_meta.get("task_family", ""),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
        f.write("\n")
    print(f"[saved] {report_path}")


def write_latency_model_report(
    df: pd.DataFrame,
    static_meta: dict[str, str],
    output_dir: str,
) -> None:
    """Fit a small latency model and write report/residual artifacts."""
    if "latency_s" not in df.columns:
        _write_skipped_latency_model_report(output_dir, static_meta, "latency_s column missing")
        return

    try:
        rows = _latency_model_rows(df)
    except ValueError as exc:
        _write_skipped_latency_model_report(output_dir, static_meta, str(exc))
        return

    feature_count = len(LATENCY_MODEL_FEATURES)
    if len(rows) < feature_count + 2:
        _write_skipped_latency_model_report(
            output_dir,
            static_meta,
            f"need at least {feature_count + 2} valid rows, got {len(rows)}",
        )
        return

    train_rows, test_rows = _split_latency_model_rows(rows)
    if len(train_rows) < feature_count:
        _write_skipped_latency_model_report(
            output_dir,
            static_meta,
            f"need at least {feature_count} train rows, got {len(train_rows)}",
        )
        return

    try:
        coefficients = _fit_linear_model(train_rows)
    except ValueError as exc:
        _write_skipped_latency_model_report(output_dir, static_meta, str(exc))
        return

    train_actual = [float(row.latency_s) for row in train_rows]
    train_predictions = [_predict_latency(row, coefficients) for row in train_rows]
    test_actual = [float(row.latency_s) for row in test_rows]
    test_predictions = [_predict_latency(row, coefficients) for row in test_rows]

    residuals_path = os.path.join(output_dir, LATENCY_MODEL_RESIDUALS)
    split_by_source = {int(row.source_row): "train" for row in train_rows}
    split_by_source.update({int(row.source_row): "test" for row in test_rows})
    with open(residuals_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_row",
            "split",
            "cpu_cores",
            "mem_cap_gb",
            "gpu_mode",
            "input_scale",
            "latency_s",
            "predicted_latency_s",
            "residual_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            predicted = _predict_latency(row, coefficients)
            actual = float(row.latency_s)
            writer.writerow({
                "source_row": int(row.source_row),
                "split": split_by_source[int(row.source_row)],
                "cpu_cores": int(row.cpu_cores),
                "mem_cap_gb": int(row.mem_cap_gb),
                "gpu_mode": row.gpu_mode,
                "input_scale": f"{float(row.input_scale):.6f}",
                "latency_s": f"{actual:.9f}",
                "predicted_latency_s": f"{predicted:.9f}",
                "residual_s": f"{(actual - predicted):.9f}",
            })

    report = {
        "status": "ok",
        "target_metric": "latency_s",
        "model_name": static_meta.get("model_name", ""),
        "task_family": static_meta.get("task_family", ""),
        "input_scale_type": static_meta.get("input_scale_type", "input_scale"),
        "model_type": "ordinary_least_squares",
        "formula": "latency_s = intercept + input_scale + inverse_cpu_cores + mem_cap_gb + gpu_on",
        "feature_columns": LATENCY_MODEL_FEATURES,
        "coefficients": {
            name: value for name, value in zip(LATENCY_MODEL_FEATURES, coefficients)
        },
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "split_rule": "deterministic: every fifth sorted row is test",
        "metrics": {
            "train": _regression_metrics(train_actual, train_predictions),
            "test": _regression_metrics(test_actual, test_predictions),
        },
        "residuals_csv": LATENCY_MODEL_RESIDUALS,
    }
    report_path = os.path.join(output_dir, LATENCY_MODEL_REPORT)
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

    for metric, title, ylabel, filename in PLOT_METRICS:
        plot_metric(
            df, metric=metric,
            title=title,
            ylabel=ylabel, xlabel=xlabel,
            out_png=os.path.join(output_dir, filename) if SAVE_PNG else None,
        )

    plot_cold_start_bar(
        df,
        title="Cold Start by Configuration",
        ylabel="Cold Start (s)",
        out_png=os.path.join(output_dir, "cold_start_bar.png") if SAVE_PNG else None,
    )
    write_latency_model_report(df, static_meta, output_dir)


if __name__ == "__main__":
    main()

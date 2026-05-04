"""AC-Prof Universal Profiler - Plotting tool adapted for generalized CSV schema."""

import csv
import colorsys
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
GPU_METRIC_ALIASES = {
    "energy_iters": "gpu_energy_iters",
    "avg_power_total_w": "gpu_avg_power_total_w",
    "peak_power_total_w": "gpu_peak_power_total_w",
    "energy_total_j": "gpu_energy_total_j",
    "avg_power_eff_w": "gpu_avg_power_eff_w",
    "peak_power_eff_w": "gpu_peak_power_eff_w",
    "energy_eff_j": "gpu_energy_eff_j",
}


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
        "input_scale", "latency_s", "latency_app_s",
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

        if gpu_on and has_cpu_series:
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


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
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

    plot_metric(
        df, metric="latency_s",
        title="Latency vs. Input Scale",
        ylabel="Latency (s)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "latency_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="latency_app_s",
        title="App-Level Latency vs. Input Scale",
        ylabel="Latency (s)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "latency_app_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="gpu_avg_power_eff_w",
        title="GPU Average Effective Power vs. Input Scale",
        ylabel="Power (W)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "gpu_avg_power_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="gpu_energy_eff_j",
        title="GPU Effective Energy vs. Input Scale",
        ylabel="Energy (J)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "gpu_energy_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="cpu_avg_power_eff_w",
        title="CPU Package Average Effective Power vs. Input Scale",
        ylabel="Power (W)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "cpu_avg_power_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="cpu_energy_eff_j",
        title="CPU Package Effective Energy vs. Input Scale",
        ylabel="Energy (J)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "cpu_energy_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="vcpu_avg_power_eff_w",
        title="Estimated vCPU Average Effective Power vs. Input Scale",
        ylabel="Power (W)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "vcpu_avg_power_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="vcpu_energy_eff_j",
        title="Estimated vCPU Effective Energy vs. Input Scale",
        ylabel="Energy (J)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "vcpu_energy_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="throughput_samples_per_s",
        title="Throughput vs. Input Scale",
        ylabel="Samples/s", xlabel=xlabel,
        out_png=os.path.join(output_dir, "throughput_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="compute_mflops",
        title="Compute Throughput vs. Input Scale",
        ylabel="MFLOPS", xlabel=xlabel,
        out_png=os.path.join(output_dir, "compute_mflops_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="container_cpu_util_avg_pct",
        title="Container CPU Utilization vs. Input Scale",
        ylabel="CPU Utilization (%)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "container_cpu_util_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="container_mem_util_avg_pct",
        title="Container Memory Utilization vs. Input Scale",
        ylabel="Memory Utilization (%)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "container_mem_util_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="container_mem_usage_avg_gib",
        title="Container Memory Usage vs. Input Scale",
        ylabel="Memory Usage (GiB)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "container_mem_usage_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="gpu_util_avg_pct",
        title="GPU Utilization vs. Input Scale",
        ylabel="GPU Utilization (%)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "gpu_util_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="gpu_mem_util_avg_pct",
        title="GPU Memory Utilization vs. Input Scale",
        ylabel="GPU Memory Utilization (%)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "gpu_mem_util_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="gpu_mem_used_avg_gib",
        title="GPU Memory Used vs. Input Scale",
        ylabel="GPU Memory Used (GiB)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "gpu_mem_used_vs_scale.png") if SAVE_PNG else None,
    )

    plot_cold_start_bar(
        df,
        title="Cold Start by Configuration",
        ylabel="Cold Start (s)",
        out_png=os.path.join(output_dir, "cold_start_bar.png") if SAVE_PNG else None,
    )


if __name__ == "__main__":
    main()

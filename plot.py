"""AC-Prof Universal Profiler - Plotting tool adapted for generalized CSV schema."""

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
CPU_FIXED_COLORS = {
    1: (0.12, 0.47, 0.71),
    2: (0.93, 0.69, 0.13),
    4: (0.84, 0.15, 0.16),
    8: (0.58, 0.40, 0.74),
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

    df = pd.read_csv(csv_path)

    num_cols = [
        "input_scale", "latency_s", "latency_app_s",
        "avg_power_eff_w", "peak_power_eff_w", "energy_eff_j",
        "cpu_cores", "mem_cap_gb", "warmup",
        "throughput_samples_per_s",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    df = prepare_df(csv_path)

    # Determine x-axis label from input_scale_type
    scale_type = "input_scale"
    if "input_scale_type" in df.columns:
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
        df, metric="avg_power_eff_w",
        title="Average Effective Power vs. Input Scale",
        ylabel="Power (W)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "avg_power_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="energy_eff_j",
        title="Effective Energy vs. Input Scale",
        ylabel="Energy (J)", xlabel=xlabel,
        out_png=os.path.join(output_dir, "energy_vs_scale.png") if SAVE_PNG else None,
    )

    plot_metric(
        df, metric="throughput_samples_per_s",
        title="Throughput vs. Input Scale",
        ylabel="Samples/s", xlabel=xlabel,
        out_png=os.path.join(output_dir, "throughput_vs_scale.png") if SAVE_PNG else None,
    )


if __name__ == "__main__":
    main()

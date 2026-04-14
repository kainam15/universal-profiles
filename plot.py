"""AC-Prof Universal Profiler - Plotting tool adapted for generalized CSV schema."""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "results/result_all.csv"
EXCLUDE_WARMUP = True
ONLY_OK = True
AGG_FUNC = "mean"
SAVE_PNG = True


def make_config_label(row) -> str:
    gpu = str(row["gpu_mode"]).lower()
    cpu = int(row["cpu_cores"])
    mem = int(row["mem_cap_gb"])
    if gpu in ("on", "1", "true", "yes", "gpu"):
        return f"GPU+CPU{cpu}+Mem{mem}"
    else:
        return f"CPU{cpu}+Mem{mem}"


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


def _sort_key(cfg: str):
    gpu = 1 if cfg.startswith("GPU+") else 0
    cpu_n = mem_n = 0
    try:
        cpu_n = int(cfg.split("CPU")[1].split("+")[0])
        mem_n = int(cfg.split("Mem")[1])
    except Exception:
        pass
    return (-gpu, cpu_n, mem_n)


def plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, xlabel: str, out_png: str | None):
    if metric not in df.columns:
        print(f"[skip] Column {metric} not in CSV")
        return

    d = df[["config", "input_scale", metric]].copy()
    d = d[d[metric].notna()].copy()
    if d.empty:
        print(f"[skip] {metric} all NaN")
        return

    if AGG_FUNC == "median":
        g = d.groupby(["config", "input_scale"], as_index=False)[metric].median()
    else:
        g = d.groupby(["config", "input_scale"], as_index=False)[metric].mean()

    configs = sorted(g["config"].unique(), key=_sort_key)

    plt.figure(figsize=(10, 6))
    for cfg in configs:
        sub = g[g["config"] == cfg].sort_values("input_scale")
        plt.plot(sub["input_scale"], sub[metric], marker="o", linewidth=2, label=cfg)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="-", alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")

    plt.show()


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

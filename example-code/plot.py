import os
import math
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "result.csv"

# 你想把 warmup=1 的点排除掉（更像“稳态”）
EXCLUDE_WARMUP = True

# 只画 status=ok 的点（避免 error 行污染）
ONLY_OK = True

# 聚合方式：同一 config & context_length 可能有 repeat_idx 多次，取均值更平滑
AGG_FUNC = "mean"   # 可改成 "median"

# 是否把图保存为 png
SAVE_PNG = True


def make_config_label(row) -> str:
    # 生成类似 “GPU+CPU8+Mem16” / “CPU6+Mem12”
    gpu = str(row["gpu_mode"]).lower()
    cpu = int(row["cpu_cores"])
    mem = int(row["mem_cap_gb"])
    if gpu in ("on", "1", "true", "yes", "gpu"):
        return f"GPU+CPU{cpu}+Mem{mem}"
    else:
        return f"CPU{cpu}+Mem{mem}"


def prepare_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 {csv_path}，请确认脚本与 result.csv 在同一目录。")

    df = pd.read_csv(csv_path)

    # 基础清洗：把关键列转成数值（有 nan 也没关系）
    num_cols = ["context_length", "latency_s", "avg_power_eff_w", "peak_power_eff_w", "energy_eff_j",
                "cpu_cores", "mem_cap_gb", "warmup"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 过滤
    if ONLY_OK and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    if EXCLUDE_WARMUP and "warmup" in df.columns:
        # warmup==0 才保留
        df = df[df["warmup"].fillna(0).astype(int) == 0].copy()

    # 必要列检查
    need = {"context_length", "cpu_cores", "mem_cap_gb", "gpu_mode"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}")

    # 配置标签
    df["config"] = df.apply(make_config_label, axis=1)

    # context_length 必须是正数
    df = df[df["context_length"].notna() & (df["context_length"] > 0)].copy()

    return df


def plot_metric_vs_context_length(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_png: str | None):
    if metric not in df.columns:
        print(f"[跳过] CSV 里没有列 {metric}")
        return

    d = df[["config", "context_length", metric]].copy()
    d = d[d[metric].notna()].copy()
    if d.empty:
        print(f"[跳过] {metric} 全是 NaN，没法画。")
        return

    # 聚合同一 config & context_length
    if AGG_FUNC == "median":
        g = d.groupby(["config", "context_length"], as_index=False)[metric].median()
    else:
        g = d.groupby(["config", "context_length"], as_index=False)[metric].mean()

    # 为了让 legend 顺序更稳定：按 “gpu优先 / cpu / mem” 排个序
    def sort_key(cfg: str):
        # GPU+CPU8+Mem16 / CPU6+Mem12
        gpu = 1 if cfg.startswith("GPU+") else 0
        # 抠出 CPU / Mem 数字
        cpu_n = 0
        mem_n = 0
        try:
            if cfg.startswith("GPU+"):
                # GPU+CPU8+Mem16
                cpu_n = int(cfg.split("CPU")[1].split("+")[0])
                mem_n = int(cfg.split("Mem")[1])
            else:
                # CPU6+Mem12
                cpu_n = int(cfg.split("CPU")[1].split("+")[0])
                mem_n = int(cfg.split("Mem")[1])
        except Exception:
            pass
        return (-gpu, cpu_n, mem_n)

    configs = sorted(g["config"].unique(), key=sort_key)

    plt.figure()
    for cfg in configs:
        sub = g[g["config"] == cfg].sort_values("context_length")
        plt.plot(sub["context_length"], sub[metric], marker="o", linewidth=2, label=cfg)

    plt.title(title)
    plt.xlabel("context_length")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[保存] {out_png}")

    plt.show()


def main():
    df = prepare_df(CSV_PATH)

    plot_metric_vs_context_length(
        df,
        metric="latency_s",
        title="Latency vs. Context Length for Different Resource Configurations",
        ylabel="Latency (s)",
        out_png="latency_s_vs_context_length.png" if SAVE_PNG else None,
    )

    plot_metric_vs_context_length(
        df,
        metric="avg_power_eff_w",
        title="Average Power vs. Context Length for Different Resource Configurations",
        ylabel="Average Power (W)",
        out_png="avg_power_eff_w_vs_context_length.png" if SAVE_PNG else None,
    )

    plot_metric_vs_context_length(
        df,
        metric="peak_power_eff_w",
        title="Peak Power vs. Context Length for Different Resource Configurations",
        ylabel="Peak Power (W)",
        out_png="peak_power_eff_w_vs_context_length.png" if SAVE_PNG else None,
    )

    plot_metric_vs_context_length(
        df,
        metric="energy_eff_j",
        title="Energy vs. Context Length for Different Resource Configurations",
        ylabel="Energy (J)",
        out_png="energy_eff_j_vs_context_length.png" if SAVE_PNG else None,
    )


if __name__ == "__main__":
    main()


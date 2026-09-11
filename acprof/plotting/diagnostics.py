"""可行性、尾延迟、Pareto 和冷启动诊断图表。"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from acprof.plotting.config import (
    COLD_START_PHASE_SPECS,
    FEASIBILITY_STATE_INDEX,
    FEASIBILITY_STATE_SPECS,
    FEASIBILITY_TEXT_COLORS,
)

from acprof.plotting.data import (
    is_gpu_off,
    is_gpu_on,
    make_config_label,
)

from acprof.plotting.styles import (
    _configurations_with_colors,
    build_cpu_base_colors,
)


def _format_axis_value(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:g}"


def _classify_feasibility_rows(rows: pd.DataFrame) -> str:
    if "status" in rows.columns:
        statuses = (
            rows["status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        statuses = pd.Series("", index=rows.index, dtype="object")
    if "error" in rows.columns:
        errors = (
            rows["error"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        errors = pd.Series("", index=rows.index, dtype="object")

    has_ok = statuses.eq("ok").any()
    has_warn = statuses.eq("warn").any()
    error_mask = statuses.eq("error") | errors.ne("")
    has_error = error_mask.any()
    if has_ok:
        if has_error:
            return "mixed"
        if has_warn:
            return "warn"
        return "ok"
    if has_warn and not has_error:
        return "warn"
    if has_error:
        error_messages = errors[error_mask]
        nonempty_errors = error_messages[error_messages.ne("")]
        if (
            not nonempty_errors.empty
            and nonempty_errors.str.contains(
                "not_measured_after_startup_oom_pruning"
            ).all()
        ):
            return "pruned_startup_oom"
        if (
            not nonempty_errors.empty
            and nonempty_errors.str.contains("not_measured_after_timeout").all()
        ):
            return "skipped"
        combined_error = " ".join(nonempty_errors.tolist())
        startup_oom = (
            ("oom" in combined_error or "out of memory" in combined_error)
            and (
                "startup" in combined_error
                or "container_start_failed" in combined_error
            )
        )
        if startup_oom:
            return "startup_oom"
        if "oom" in combined_error or "out of memory" in combined_error:
            return "runtime_oom"
        if "timeout" in combined_error:
            return "timeout"
        return "error"
    return "unknown"


def summarize_resource_feasibility(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated rows into one categorical outcome per matrix cell."""
    required = {"cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=[*sorted(required), "state"])

    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]
    records = []
    for key, rows in df.groupby(group_cols, dropna=False, sort=True):
        record = dict(zip(group_cols, key))
        record["state"] = _classify_feasibility_rows(rows)
        records.append(record)
    summary = pd.DataFrame.from_records(records)
    if not summary.empty:
        summary["gpu_on"] = summary["gpu_mode"].map(is_gpu_on)
    return summary


def plot_resource_feasibility_heatmap(
    df: pd.DataFrame,
    *,
    xlabel: str,
    out_png: str | None,
    show_plots: bool = False,
) -> bool:
    """Plot success and failure boundaries without discarding error rows."""
    if "status" not in df.columns and "error" not in df.columns:
        print("[skip] No status/error columns for resource feasibility plot")
        return False

    summary = summarize_resource_feasibility(df)
    summary = summary[
        summary["gpu_mode"].map(lambda value: is_gpu_on(value) or is_gpu_off(value))
    ].copy()
    if summary.empty:
        print("[skip] No resource feasibility data")
        return False

    cpu_values = sorted(int(value) for value in summary["cpu_cores"].dropna().unique())
    mem_values = sorted(int(value) for value in summary["mem_cap_gb"].dropna().unique())
    scale_values = sorted(float(value) for value in summary["input_scale"].dropna().unique())
    mode_values = [
        mode
        for mode in (False, True)
        if (summary["gpu_on"] == mode).any()
    ]
    if not cpu_values or not mem_values or not scale_values or not mode_values:
        print("[skip] Incomplete resource feasibility dimensions")
        return False

    cmap = ListedColormap([spec[2] for spec in FEASIBILITY_STATE_SPECS])
    fig, axes = plt.subplots(
        len(cpu_values),
        len(mode_values),
        figsize=(max(8.5, 6.0 * len(mode_values)), max(4.5, 2.7 * len(cpu_values))),
        squeeze=False,
    )
    used_states = set(summary["state"])
    for row_index, cpu in enumerate(cpu_values):
        for column_index, gpu_on in enumerate(mode_values):
            axis = axes[row_index, column_index]
            panel_df = summary[
                (summary["cpu_cores"] == cpu)
                & (summary["gpu_on"] == gpu_on)
            ]
            matrix = np.full(
                (len(mem_values), len(scale_values)),
                FEASIBILITY_STATE_INDEX["unknown"],
                dtype=float,
            )
            for mem_index, mem in enumerate(mem_values):
                for scale_index, input_scale in enumerate(scale_values):
                    cell = panel_df[
                        (panel_df["mem_cap_gb"] == mem)
                        & np.isclose(panel_df["input_scale"], input_scale)
                    ]
                    state = "unknown" if cell.empty else str(cell.iloc[0]["state"])
                    if state == "unknown":
                        used_states.add(state)
                    matrix[mem_index, scale_index] = FEASIBILITY_STATE_INDEX[state]
                    short_label = FEASIBILITY_STATE_SPECS[
                        FEASIBILITY_STATE_INDEX[state]
                    ][3]
                    axis.text(
                        scale_index,
                        mem_index,
                        short_label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=FEASIBILITY_TEXT_COLORS.get(state, "black"),
                    )

            axis.imshow(
                matrix,
                cmap=cmap,
                vmin=-0.5,
                vmax=len(FEASIBILITY_STATE_SPECS) - 0.5,
                aspect="auto",
                interpolation="nearest",
            )
            axis.set_xticks(range(len(scale_values)))
            axis.set_xticklabels(
                [_format_axis_value(value) for value in scale_values],
                rotation=35,
                ha="right",
            )
            axis.set_yticks(range(len(mem_values)))
            axis.set_yticklabels([str(value) for value in mem_values])
            mode_label = "GPU" if gpu_on else "CPU-only"
            axis.set_title(f"{mode_label}, {cpu} CPU core{'s' if cpu != 1 else ''}")
            if row_index == len(cpu_values) - 1:
                axis.set_xlabel(xlabel)
            if column_index == 0:
                axis.set_ylabel("Memory cap (GiB)")

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for state, label, color, _short_label in FEASIBILITY_STATE_SPECS
        if state in used_states
    ]
    legend_columns = min(4, max(1, len(legend_handles)))
    fig.suptitle("Resource Feasibility and Failure Boundary", y=0.995)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=legend_columns,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))

    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")
    if show_plots:
        plt.show()
    plt.close(fig)
    return True


def _representative_max_memory_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Select one high-memory configuration per CPU count and hardware mode."""
    if df.empty:
        return df.copy()
    group_cols = ["gpu_mode", "cpu_cores"]
    max_memory = df.groupby(group_cols)["mem_cap_gb"].transform("max")
    return df[df["mem_cap_gb"] == max_memory].copy()


def _aggregate_tail_latency(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in (
            "latency_p50_s",
            "latency_p90_s",
            "latency_p95_s",
            "latency_app_p50_s",
            "latency_app_p90_s",
            "latency_app_p95_s",
        )
        if column in df.columns
    ]
    if not metric_columns:
        return pd.DataFrame()
    usable = df[
        ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", *metric_columns]
    ].copy()
    usable = usable[usable[metric_columns].notna().any(axis=1)]
    usable = _representative_max_memory_rows(usable)
    if usable.empty:
        return usable
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]
    aggregated = usable.groupby(group_cols, as_index=False)[metric_columns].median()
    aggregated["config"] = aggregated.apply(make_config_label, axis=1)
    aggregated["gpu_on"] = aggregated["gpu_mode"].map(is_gpu_on)
    return aggregated


def plot_tail_latency_overview(
    df: pd.DataFrame,
    *,
    xlabel: str,
    out_png: str | None,
    show_plots: bool = False,
) -> bool:
    """Show within-window P50/P90/P95 and P95/P50 tail inflation."""
    aggregated = _aggregate_tail_latency(df)
    if aggregated.empty:
        print("[skip] Tail latency percentile columns unavailable or all NaN")
        return False

    configs, config_colors = _configurations_with_colors(aggregated)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex="col", squeeze=False)
    families = (
        (
            "Packet latency percentiles",
            "latency_p50_s",
            "latency_p90_s",
            "latency_p95_s",
        ),
        (
            "Application latency percentiles",
            "latency_app_p50_s",
            "latency_app_p90_s",
            "latency_app_p95_s",
        ),
    )
    plotted_any = False
    for column_index, (title, p50_column, p90_column, p95_column) in enumerate(families):
        percentile_axis = axes[0, column_index]
        ratio_axis = axes[1, column_index]
        percentile_axis.set_title(title)
        ratio_axis.set_title("P95 / P50 tail inflation")
        family_available = {
            p50_column,
            p95_column,
        }.issubset(aggregated.columns)
        if not family_available:
            for axis in (percentile_axis, ratio_axis):
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            continue

        for cpu, mem, gpu_on in configs:
            sub_df = aggregated[
                (aggregated["cpu_cores"] == cpu)
                & (aggregated["mem_cap_gb"] == mem)
                & (aggregated["gpu_on"] == gpu_on)
            ].sort_values("input_scale")
            sub_df = sub_df.dropna(subset=[p50_column, p95_column])
            sub_df = sub_df[
                (sub_df[p50_column] > 0.0)
                & (sub_df[p95_column] >= sub_df[p50_column])
            ]
            if sub_df.empty:
                continue
            plotted_any = True
            color = config_colors[(cpu, mem, gpu_on)]
            x_values = sub_df["input_scale"].to_numpy(dtype=float)
            p50_values = sub_df[p50_column].to_numpy(dtype=float)
            p95_values = sub_df[p95_column].to_numpy(dtype=float)
            percentile_axis.fill_between(
                x_values,
                p50_values,
                p95_values,
                color=color,
                alpha=0.12,
            )
            percentile_axis.plot(
                x_values,
                p50_values,
                color=color,
                linestyle="-",
                linewidth=1.8,
            )
            if p90_column in sub_df.columns:
                p90_rows = sub_df[sub_df[p90_column].notna()]
                if not p90_rows.empty:
                    percentile_axis.plot(
                        p90_rows["input_scale"],
                        p90_rows[p90_column],
                        color=color,
                        linestyle="--",
                        linewidth=1.4,
                    )
            percentile_axis.plot(
                x_values,
                p95_values,
                color=color,
                linestyle=":",
                linewidth=1.8,
            )
            ratio_axis.plot(
                x_values,
                p95_values / p50_values,
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=5,
            )

        percentile_axis.set_ylabel("Latency (s)")
        ratio_axis.set_ylabel("P95 / P50")
        ratio_axis.set_xlabel(xlabel)
        for axis in (percentile_axis, ratio_axis):
            axis.grid(True, which="both", linestyle="-", alpha=0.35)

    if not plotted_any:
        plt.close(fig)
        print("[skip] Tail latency percentiles have no valid P50/P95 pairs")
        return False

    config_handles = [
        Line2D([0], [0], color=config_colors[config], linewidth=2)
        for config in configs
    ]
    config_labels = [
        f"{'GPU' if gpu_on else 'CPU'}+CPU{cpu}+Mem{mem}"
        for cpu, mem, gpu_on in configs
    ]
    percentile_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="P50"),
        Line2D([0], [0], color="black", linestyle="--", label="P90"),
        Line2D([0], [0], color="black", linestyle=":", label="P95"),
        Patch(facecolor="gray", alpha=0.18, label="P50–P95 band"),
    ]
    fig.suptitle(
        "Tail Latency at Representative Max-Memory Configurations",
        y=0.995,
    )
    fig.legend(
        handles=config_handles + percentile_handles,
        labels=config_labels + [handle.get_label() for handle in percentile_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(6, max(1, len(config_handles + percentile_handles))),
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")
    if show_plots:
        plt.show()
    plt.close(fig)
    return True


def _pareto_frontier_mask(latencies: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Return points not dominated while minimizing latency and energy."""
    frontier = np.ones(len(latencies), dtype=bool)
    for index in range(len(latencies)):
        dominated = (
            (latencies <= latencies[index])
            & (energies <= energies[index])
            & (
                (latencies < latencies[index])
                | (energies < energies[index])
            )
        )
        dominated[index] = False
        if dominated.any():
            frontier[index] = False
    return frontier


def _pareto_latency_column(df: pd.DataFrame) -> tuple[str, str] | None:
    candidates = (
        ("latency_app_p95_s", "Application P95 latency (s)"),
        ("latency_p95_s", "Packet P95 latency (s)"),
        ("latency_app_s", "Application mean latency (s)"),
        ("latency_s", "Packet mean latency (s)"),
    )
    for column, label in candidates:
        if column in df.columns and (df[column] > 0.0).any():
            return column, label
    return None


def plot_latency_energy_pareto(
    df: pd.DataFrame,
    *,
    xlabel: str,
    out_png: str | None,
    show_plots: bool = False,
) -> bool:
    energy_column = "container_attributed_energy_eff_j"
    latency_spec = _pareto_latency_column(df)
    if energy_column not in df.columns or latency_spec is None:
        print("[skip] Latency/attributed-energy columns unavailable for Pareto plot")
        return False
    latency_column, latency_label = latency_spec
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]
    pareto_df = df[group_cols + [latency_column, energy_column]].copy()
    pareto_df = pareto_df.dropna(subset=[latency_column, energy_column])
    pareto_df = pareto_df[
        (pareto_df[latency_column] > 0.0)
        & (pareto_df[energy_column] > 0.0)
    ]
    if pareto_df.empty:
        print("[skip] Latency/attributed-energy pairs all invalid")
        return False
    pareto_df = pareto_df.groupby(group_cols, as_index=False)[
        [latency_column, energy_column]
    ].median()
    pareto_df["gpu_on"] = pareto_df["gpu_mode"].map(is_gpu_on)

    scale_values = sorted(float(value) for value in pareto_df["input_scale"].unique())
    columns = min(3, len(scale_values))
    rows = math.ceil(len(scale_values) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.2 * columns, 5.0 * rows + 1.2),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    cpu_values = sorted(int(value) for value in pareto_df["cpu_cores"].unique())
    mem_values = sorted(int(value) for value in pareto_df["mem_cap_gb"].unique())
    cpu_colors = build_cpu_base_colors(cpu_values)
    for panel_index, input_scale in enumerate(scale_values):
        axis = flat_axes[panel_index]
        scale_df = pareto_df[
            np.isclose(pareto_df["input_scale"], input_scale)
        ].copy()
        latency_values = scale_df[latency_column].to_numpy(dtype=float)
        energy_values = scale_df[energy_column].to_numpy(dtype=float)
        frontier_mask = _pareto_frontier_mask(latency_values, energy_values)
        for point_index, point in enumerate(scale_df.itertuples(index=False)):
            cpu = int(point.cpu_cores)
            mem = int(point.mem_cap_gb)
            gpu_on = bool(point.gpu_on)
            marker = "o" if gpu_on else "s"
            axis.scatter(
                getattr(point, latency_column),
                getattr(point, energy_column),
                color=cpu_colors[cpu],
                marker=marker,
                s=45.0 + 5.0 * mem,
                alpha=0.82,
                edgecolor="black" if frontier_mask[point_index] else "white",
                linewidth=1.5 if frontier_mask[point_index] else 0.6,
                zorder=3 if frontier_mask[point_index] else 2,
            )
            if frontier_mask[point_index] and len(scale_df) <= 20:
                prefix = "G" if gpu_on else "C"
                axis.annotate(
                    f"{prefix}{cpu}/M{mem}",
                    (getattr(point, latency_column), getattr(point, energy_column)),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
        frontier_df = scale_df.loc[frontier_mask].sort_values(latency_column)
        if len(frontier_df) > 1:
            axis.plot(
                frontier_df[latency_column],
                frontier_df[energy_column],
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{xlabel} = {_format_axis_value(input_scale)}")
        axis.set_xlabel(latency_label)
        axis.set_ylabel("Container-attributed energy (J/request)")
        axis.grid(True, which="both", linestyle="-", alpha=0.25)

    for axis in flat_axes[len(scale_values):]:
        axis.set_visible(False)

    cpu_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=cpu_colors[cpu],
            label=f"{cpu} CPU core{'s' if cpu != 1 else ''}",
        )
        for cpu in cpu_values
    ]
    mode_handles = []
    if pareto_df["gpu_on"].eq(False).any():
        mode_handles.append(
            Line2D([0], [0], marker="s", linestyle="none", color="gray", label="CPU-only")
        )
    if pareto_df["gpu_on"].eq(True).any():
        mode_handles.append(
            Line2D([0], [0], marker="o", linestyle="none", color="gray", label="GPU")
        )
    memory_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="gray",
            markerfacecolor="none",
            markersize=math.sqrt(45.0 + 5.0 * mem),
            label=f"{mem} GiB",
        )
        for mem in mem_values
    ]
    frontier_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        label="Pareto frontier",
    )
    legend_handles = cpu_handles + mode_handles + memory_handles + [frontier_handle]
    fig.suptitle("Latency–Energy Pareto Frontiers by Input Scale", y=0.995)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(6, len(legend_handles)),
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")
    if show_plots:
        plt.show()
    plt.close(fig)
    return True


def plot_cold_start_breakdown(
    df: pd.DataFrame,
    *,
    out_png: str | None,
    show_plots: bool = False,
) -> bool:
    phase_columns = [spec[0] for spec in COLD_START_PHASE_SPECS]
    missing_columns = [column for column in phase_columns if column not in df.columns]
    if missing_columns:
        print(
            "[skip] Cold-start breakdown columns missing: "
            + ", ".join(missing_columns)
        )
        return False

    optional_columns = [
        column
        for column in ("cold_start_s", "cold_start_first_predict_app_s")
        if column in df.columns
    ]
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode"]
    cold_df = df[group_cols + phase_columns + optional_columns].copy()
    cold_df = cold_df.dropna(subset=phase_columns)
    cold_df = cold_df[(cold_df[phase_columns] >= 0.0).all(axis=1)]
    if cold_df.empty:
        print("[skip] Cold-start phase breakdown all NaN or incomplete")
        return False
    aggregated = cold_df.groupby(group_cols, as_index=False)[
        phase_columns + optional_columns
    ].median()
    aggregated = _representative_max_memory_rows(aggregated)
    aggregated["gpu_on"] = aggregated["gpu_mode"].map(is_gpu_on)
    aggregated["config"] = aggregated.apply(make_config_label, axis=1)
    aggregated = aggregated.sort_values(
        ["gpu_on", "cpu_cores", "mem_cap_gb"],
        ascending=[False, True, True],
    )

    labels = aggregated["config"].tolist()
    x_values = np.arange(len(labels), dtype=float)
    bottoms = np.zeros(len(labels), dtype=float)
    fig, (axis, first_predict_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(max(10.0, 1.1 * len(labels)), 9.0),
    )
    for column, label, color in COLD_START_PHASE_SPECS:
        values = aggregated[column].to_numpy(dtype=float)
        axis.bar(
            x_values,
            values,
            bottom=bottoms,
            color=color,
            label=label,
            width=0.72,
        )
        bottoms += values

    if "cold_start_s" in aggregated.columns:
        valid_total = aggregated["cold_start_s"].notna()
        axis.scatter(
            x_values[valid_total],
            aggregated.loc[valid_total, "cold_start_s"],
            marker="x",
            color="black",
            s=55,
            linewidth=1.5,
            label="Measured total to /ready",
            zorder=4,
        )
    first_predict = aggregated.get(
        "cold_start_first_predict_app_s",
        pd.Series(np.nan, index=aggregated.index),
    )
    valid_first = first_predict.notna()
    if valid_first.any():
        first_predict_axis.scatter(
            x_values[valid_first],
            first_predict.loc[valid_first],
            marker="D",
            facecolor="white",
            edgecolor="black",
            s=38,
            linewidth=1.0,
            label="First predict latency",
            zorder=4,
        )
    else:
        first_predict_axis.text(
            0.5,
            0.5,
            "No data",
            transform=first_predict_axis.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

    first_predict_axis.set_xticks(x_values)
    first_predict_axis.set_xticklabels(labels, rotation=40, ha="right")
    first_predict_axis.set_xlabel(
        "Representative max-memory configuration per CPU count"
    )
    first_predict_axis.set_ylabel("Latency (s)")
    first_predict_axis.set_title("First Predict Latency")
    first_predict_axis.grid(True, axis="y", linestyle="-", alpha=0.35)
    axis.set_ylabel("Time (s)")
    axis.set_title("Cold-Start Phase Breakdown")
    axis.grid(True, axis="y", linestyle="-", alpha=0.35)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=8)
    fig.tight_layout(h_pad=2.0)
    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")
    if show_plots:
        plt.show()
    plt.close(fig)
    return True

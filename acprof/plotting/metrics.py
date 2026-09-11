"""单指标、指标概览及能耗图表。"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from acprof.plotting.config import (
    GPU_GREEN,
    MEM_COLORED_METRICS,
)

from acprof.plotting.data import (
    aggregate_cold_start,
    aggregate_metric,
    is_gpu_off,
    is_gpu_on,
)

from acprof.plotting.styles import (
    _configurations_with_colors,
    _sort_key,
    build_cpu_base_colors,
    build_gpu_mixed_colors,
    color_for_mem,
    shade_for_cpu,
    shade_for_mem,
)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    xlabel: str,
    out_png: str | None,
    *,
    show_plots: bool = False,
    agg_func: str = "mean",
):
    if metric not in df.columns:
        print(f"[skip] Column {metric} not in CSV")
        return

    agg_df = aggregate_metric(df, metric, agg_func=agg_func)
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

    if show_plots:
        plt.show()
    plt.close()


def plot_metric_overview(
    df: pd.DataFrame,
    *,
    panels: tuple,
    rows: int,
    columns: int,
    shared_y_groups: tuple,
    title: str,
    xlabel: str,
    out_png: str | None,
    figure_note: str | None = None,
    show_plots: bool = False,
    agg_func: str = "mean",
) -> None:
    """Plot several related metrics with shared configuration colors."""
    if rows < 1 or columns < 1 or len(panels) != rows * columns:
        raise ValueError("Overview panel count must match rows * columns")

    aggregated_metrics = {}
    for panel in panels:
        if panel is None:
            continue
        metric = panel[0]
        if metric in aggregated_metrics:
            continue
        agg_df = None
        if metric not in df.columns:
            print(f"[skip] Column {metric} not in CSV")
        else:
            candidate_df = aggregate_metric(df, metric, agg_func=agg_func)
            if candidate_df.empty:
                print(f"[skip] {metric} all NaN")
            else:
                agg_df = candidate_df
        aggregated_metrics[metric] = agg_df

    available_frames = [
        agg_df
        for agg_df in aggregated_metrics.values()
        if agg_df is not None
    ]
    if not available_frames:
        return

    combined_df = pd.concat(available_frames, ignore_index=True)
    configs, config_colors = _configurations_with_colors(combined_df)
    if "container_mem_util_avg_pct" in aggregated_metrics:
        # The memory/process overview uses GPU memory caps for green shades
        # across all panels and the legend, including GPU-only figures.
        gpu_mems = sorted({mem for _, mem, gpu_on in configs if gpu_on})
        gpu_mem_colors = {
            mem: shade_for_mem(GPU_GREEN, rank, len(gpu_mems))
            for rank, mem in enumerate(gpu_mems)
        }
        config_colors.update({
            config: gpu_mem_colors[config[1]]
            for config in configs
            if config[2]
        })
    figure_width = 18 if columns > 1 else 11
    figure_height = {1: 8, 2: 11, 3: 14}.get(rows, 4 * rows + 2)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(figure_width, figure_height),
        sharex=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)

    for index_group in shared_y_groups:
        if len(index_group) < 2:
            continue
        if any(index < 0 or index >= len(flat_axes) for index in index_group):
            raise ValueError("Shared-y panel index is outside the overview grid")
        base_axis = flat_axes[index_group[0]]
        for index in index_group[1:]:
            flat_axes[index].sharey(base_axis)

    for panel_index, (axis, panel) in enumerate(zip(flat_axes, panels)):
        if panel is None:
            axis.set_visible(False)
            continue
        metric, panel_title, ylabel = panel[:3]
        agg_df = aggregated_metrics[metric]
        axis.set_title(panel_title)
        if ylabel:
            axis.set_ylabel(ylabel)
        axis.grid(True, which="both", linestyle="-", alpha=0.5)
        if agg_df is None:
            axis.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue

        marker = (
            panel[3]
            if len(panel) > 3
            else ("o", "s", "^", "D", "v", "P")[panel_index % 6]
        )
        for cpu, mem, gpu_on in configs:
            sub_df = agg_df[
                (agg_df["cpu_cores"] == cpu)
                & (agg_df["mem_cap_gb"] == mem)
                & (agg_df["gpu_on"] == gpu_on)
            ].sort_values("input_scale")
            if sub_df.empty:
                continue
            axis.plot(
                sub_df["input_scale"],
                sub_df[metric],
                color=config_colors[(cpu, mem, gpu_on)],
                linestyle="-",
                marker=marker,
                linewidth=2,
                markersize=6,
            )

    for column_index in range(columns):
        for row_index in range(rows - 1, -1, -1):
            panel_index = row_index * columns + column_index
            if panels[panel_index] is None:
                continue
            axis = axes[row_index, column_index]
            axis.set_xlabel(xlabel)
            axis.tick_params(axis="x", labelbottom=True)
            break

    config_handles = [
        Line2D(
            [0],
            [0],
            color=config_colors[config],
            linestyle="-",
            linewidth=2,
        )
        for config in configs
    ]
    config_labels = [
        f"{'GPU' if gpu_on else 'CPU'}+CPU{cpu}+Mem{mem}"
        for cpu, mem, gpu_on in configs
    ]
    legend_column_limit = 6 if columns > 1 else 4
    legend_columns = min(
        legend_column_limit,
        max(1, len(config_handles)),
    )
    legend_rows = math.ceil(len(config_handles) / legend_columns)
    note_lines = figure_note.count("\n") + 1 if figure_note else 0
    note_height_inches = 0.12 + 0.22 * note_lines if figure_note else 0.0
    top_margin_inches = 0.8 + 0.23 * legend_rows + note_height_inches
    layout_top = max(
        0.68,
        min(0.9, 1.0 - top_margin_inches / figure_height),
    )
    fig.suptitle(title, y=0.995)
    fig.legend(
        handles=config_handles,
        labels=config_labels,
        title="Configuration",
        fontsize=8,
        ncol=legend_columns,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    if figure_note:
        note_y = min(
            0.9,
            layout_top
            + note_height_inches / (2.0 * figure_height)
            + 0.012,
        )
        fig.text(
            0.5,
            note_y,
            figure_note,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#f2f2f2",
                "edgecolor": "#bdbdbd",
                "alpha": 0.95,
            },
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))

    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")

    if show_plots:
        plt.show()
    plt.close(fig)


def _finite_numeric_values(
    df: pd.DataFrame,
    column: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> pd.Series:
    """Return finite numeric values suitable for a figure-level summary."""
    if column not in df.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(df[column], errors="coerce")
    values = values[np.isfinite(values)]
    if positive:
        values = values[values > 0.0]
    elif nonnegative:
        values = values[values >= 0.0]
    return values.astype(float)


def _idle_mean_and_max_relative_range(
    df: pd.DataFrame,
    *,
    power_column: str,
    relative_range_column: str,
) -> tuple[float, float]:
    power_values = _finite_numeric_values(
        df,
        power_column,
        positive=True,
    )
    relative_ranges = _finite_numeric_values(
        df,
        relative_range_column,
        nonnegative=True,
    )
    mean_power = (
        float(power_values.mean())
        if not power_values.empty
        else float("nan")
    )
    max_relative_range = (
        float(relative_ranges.max())
        if not relative_ranges.empty
        else float("nan")
    )
    return mean_power, max_relative_range


def _format_idle_summary(
    *,
    mean_power_w: float,
    max_relative_range: float,
    range_label: str = "max within-case range/mean",
) -> str:
    mean_text = (
        f"{mean_power_w:.2f} W"
        if math.isfinite(mean_power_w)
        else "N/A"
    )
    range_text = (
        f"{max_relative_range * 100.0:.2f}%"
        if math.isfinite(max_relative_range)
        else "N/A"
    )
    return f"mean {mean_text} | {range_label} {range_text}"


def _energy_idle_annotation(
    df: pd.DataFrame,
    *,
    effective_metrics: tuple[str, str, str],
    total_metrics: tuple[str, str, str],
) -> str:
    """Describe the idle baseline and its worst within-case variation."""
    metric = effective_metrics[0]
    if metric.startswith("gpu_"):
        mean_power, max_range = _idle_mean_and_max_relative_range(
            df,
            power_column="gpu_idle_power_w",
            relative_range_column="gpu_idle_rel_range_so_far",
        )
        return "Measured GPU board idle: " + _format_idle_summary(
            mean_power_w=mean_power,
            max_relative_range=max_range,
        )

    mode_specs = (
        ("CPU-only", df["gpu_mode"].map(is_gpu_off)),
        ("GPU-enabled", df["gpu_mode"].map(is_gpu_on)),
    )
    lines = []
    if metric.startswith("cpu_"):
        for mode_label, mode_mask in mode_specs:
            mode_df = df[mode_mask]
            if mode_df.empty:
                continue
            mean_power, max_range = _idle_mean_and_max_relative_range(
                mode_df,
                power_column="cpu_idle_power_w",
                relative_range_column="cpu_idle_rel_range_so_far",
            )
            lines.append(
                f"{mode_label} measured package idle: "
                + _format_idle_summary(
                    mean_power_w=mean_power,
                    max_relative_range=max_range,
                )
            )
        return "\n".join(lines) or "Measured CPU package idle: N/A"

    if metric.startswith("vcpu_"):
        for mode_label, mode_mask in mode_specs:
            mode_df = df[mode_mask]
            if mode_df.empty:
                continue
            if (
                total_metrics[1] in mode_df.columns
                and effective_metrics[1] in mode_df.columns
            ):
                total_values = pd.to_numeric(
                    mode_df[total_metrics[1]],
                    errors="coerce",
                )
                effective_values = pd.to_numeric(
                    mode_df[effective_metrics[1]],
                    errors="coerce",
                )
                attributed_idle = total_values - effective_values
                attributed_idle = attributed_idle[
                    np.isfinite(attributed_idle)
                    & (attributed_idle >= 0.0)
                ]
            else:
                attributed_idle = pd.Series(dtype=float)
            mean_attributed_idle = (
                float(attributed_idle.mean())
                if not attributed_idle.empty
                else float("nan")
            )
            _source_mean, source_max_range = (
                _idle_mean_and_max_relative_range(
                    mode_df,
                    power_column="cpu_idle_power_w",
                    relative_range_column="cpu_idle_rel_range_so_far",
                )
            )
            lines.append(
                f"{mode_label} estimated attributed idle "
                "(package baseline x interval CPU share): "
                + _format_idle_summary(
                    mean_power_w=mean_attributed_idle,
                    max_relative_range=source_max_range,
                    range_label="source package max range/mean",
                )
            )
        return "\n".join(lines) or (
            "Estimated attributed idle "
            "(CPU package baseline x interval CPU share): N/A"
        )

    return "Idle baseline: N/A"


def plot_energy_power_overview(
    df: pd.DataFrame,
    *,
    effective_metrics: tuple[str, str, str],
    total_metrics: tuple[str, str, str],
    title: str,
    xlabel: str,
    out_png: str | None,
    show_plots: bool = False,
    agg_func: str = "mean",
) -> None:
    """Compare effective and total energy/power in a 3-by-2 figure."""
    panels = (
        (
            effective_metrics[0],
            "Effective Energy per Request",
            "Energy (J/request)",
            "s",
        ),
        (total_metrics[0], "Total Energy per Request", "", "s"),
        (
            effective_metrics[1],
            "Average Effective Power",
            "Average power (W)",
            "o",
        ),
        (total_metrics[1], "Average Total Power", "", "o"),
        (
            effective_metrics[2],
            "Peak Effective Power",
            "Peak power (W)",
            "^",
        ),
        (total_metrics[2], "Peak Total Power", "", "^"),
    )
    plot_metric_overview(
        df,
        panels=panels,
        rows=3,
        columns=2,
        shared_y_groups=((0, 1), (2, 3), (4, 5)),
        title=title,
        xlabel=xlabel,
        out_png=out_png,
        figure_note=_energy_idle_annotation(
            df,
            effective_metrics=effective_metrics,
            total_metrics=total_metrics,
        ),
        show_plots=show_plots,
        agg_func=agg_func,
    )


def plot_cold_start_bar(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    out_png: str | None,
    *,
    show_plots: bool = False,
    agg_func: str = "mean",
):
    if "cold_start_s" not in df.columns:
        print("[skip] Column cold_start_s not in CSV")
        return

    agg_df = aggregate_cold_start(df, agg_func=agg_func)
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

    if show_plots:
        plt.show()
    plt.close()

"""Plot CLI and compatibility exports for the analysis and plotting packages."""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from acprof.analysis.latency_model import (
    GPU_MODE_OFF_VALUES,
    GPU_MODE_ON_VALUES,
    LATENCY_MODEL_FEATURES,
    LATENCY_MODEL_FORMULAS,
    LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE,
    LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES,
    LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO,
    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE,
    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE,
    LATENCY_MODEL_MAX_SCALE_CASE_RELATIVE_ERROR,
    LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR,
    LATENCY_MODEL_MAX_VALIDATION_MAPE,
    LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE,
    LATENCY_MODEL_MIN_VALIDATION_POINTS,
    LATENCY_MODEL_MIN_VALIDATION_R2,
    LATENCY_MODEL_RESOURCE_FEATURES,
    _aggregate_latency_model_points,
    _fit_gpu_upper_tail_model,
    _fit_log_latency_model,
    _fit_rank_revealing_linear_model,
    _format_input_scale_knot,
    _gpu_input_scale_spline_knots,
    _gpu_upper_tail_features,
    _hardware_model_name,
    _input_scale_validation,
    _latency_model_frame,
    _model_feature_names,
    _model_features,
    _point_key,
    _predict_latency,
    _predict_primary_latency,
    _regression_metrics,
    _resource_configuration_validation,
    _training_range,
    _validation_is_evaluable,
    _validation_quality_failures,
)

from acprof.analysis.latency_report import (
    LATENCY_MODEL_DIR,
    LATENCY_MODEL_REPORT,
    LATENCY_MODEL_RESIDUALS,
    LATENCY_MODEL_RESIDUAL_FIELDS,
    _clean_json_value,
    _format_optional_float,
    _write_skipped_latency_model_report,
    write_latency_model_report,
)

from acprof.plotting.config import (
    BYTES_PER_GIB,
    COLD_START_BREAKDOWN_PLOT,
    COLD_START_PHASE_SPECS,
    COMPUTE_NUMERIC_COLUMNS,
    CPU_FIXED_COLORS,
    ENERGY_POWER_OVERVIEW_PLOTS,
    EXECUTION_PROFILE_NUMERIC_COLUMNS,
    FEASIBILITY_STATE_INDEX,
    FEASIBILITY_STATE_SPECS,
    FEASIBILITY_TEXT_COLORS,
    GPU_GREEN,
    GPU_LIGHT_GREEN,
    GPU_METRIC_ALIASES,
    LATENCY_ENERGY_PARETO_PLOT,
    LATENCY_MODEL_FIT_CURVES_PLOT,
    LATENCY_MODEL_RESIDUAL_PLOT,
    LEGACY_PLOT_METRICS,
    MEM_COLORED_METRICS,
    MEM_FIXED_COLORS,
    METRIC_OVERVIEW_PLOTS,
    NCU_COMPUTE_LEGACY_FALLBACKS,
    PLOT_METRICS,
    PLOT_OUTPUT_DIRS,
    RESOURCE_FEASIBILITY_PLOT,
    TAIL_LATENCY_PLOT,
    TORCH_EAGER_COMPUTE_LEGACY_FALLBACKS,
)

from acprof.plotting.data import (
    build_plot_groups,
    is_gpu_off,
    is_gpu_on,
    make_config_label,
    prepare_df,
    read_static_meta,
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

from acprof.plotting.metrics import (
    _energy_idle_annotation,
    _finite_numeric_values,
    _format_idle_summary,
    _idle_mean_and_max_relative_range,
)

from acprof.plotting.diagnostics import (
    _aggregate_tail_latency,
    _classify_feasibility_rows,
    _format_axis_value,
    _pareto_frontier_mask,
    _pareto_latency_column,
    _representative_max_memory_rows,
    summarize_resource_feasibility,
)

from acprof.plotting.latency import (
    _residual_plot_metrics,
)

import acprof.plotting.data as _plot_data

import acprof.plotting.metrics as _plot_metrics

import acprof.plotting.diagnostics as _plot_diagnostics

import acprof.plotting.latency as _plot_latency

CSV_PATH = "results/result_all.csv"
EXCLUDE_WARMUP = True
ONLY_OK = True
AGG_FUNC = "mean"
SAVE_PNG = True
SHOW_PLOTS = False


def aggregate_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return _plot_data.aggregate_metric(
        df=df,
        metric=metric,
        agg_func=AGG_FUNC,
    )


def aggregate_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    return _plot_data.aggregate_cold_start(
        df=df,
        agg_func=AGG_FUNC,
    )


def plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, xlabel: str, out_png: str | None):
    return _plot_metrics.plot_metric(
        df=df,
        metric=metric,
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
        agg_func=AGG_FUNC,
    )


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
) -> None:
    return _plot_metrics.plot_metric_overview(
        df=df,
        panels=panels,
        rows=rows,
        columns=columns,
        shared_y_groups=shared_y_groups,
        title=title,
        xlabel=xlabel,
        out_png=out_png,
        figure_note=figure_note,
        show_plots=SHOW_PLOTS,
        agg_func=AGG_FUNC,
    )


def plot_energy_power_overview(
    df: pd.DataFrame,
    *,
    effective_metrics: tuple[str, str, str],
    total_metrics: tuple[str, str, str],
    title: str,
    xlabel: str,
    out_png: str | None,
) -> None:
    return _plot_metrics.plot_energy_power_overview(
        df=df,
        effective_metrics=effective_metrics,
        total_metrics=total_metrics,
        title=title,
        xlabel=xlabel,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
        agg_func=AGG_FUNC,
    )


def plot_cold_start_bar(df: pd.DataFrame, title: str, ylabel: str, out_png: str | None):
    return _plot_metrics.plot_cold_start_bar(
        df=df,
        title=title,
        ylabel=ylabel,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
        agg_func=AGG_FUNC,
    )


def plot_resource_feasibility_heatmap(
    df: pd.DataFrame,
    *,
    xlabel: str,
    out_png: str | None,
) -> bool:
    return _plot_diagnostics.plot_resource_feasibility_heatmap(
        df=df,
        xlabel=xlabel,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
    )


def plot_tail_latency_overview(
    df: pd.DataFrame,
    *,
    xlabel: str,
    out_png: str | None,
) -> bool:
    return _plot_diagnostics.plot_tail_latency_overview(
        df=df,
        xlabel=xlabel,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
    )


def plot_latency_energy_pareto(
    df: pd.DataFrame,
    *,
    xlabel: str,
    out_png: str | None,
) -> bool:
    return _plot_diagnostics.plot_latency_energy_pareto(
        df=df,
        xlabel=xlabel,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
    )


def plot_cold_start_breakdown(
    df: pd.DataFrame,
    *,
    out_png: str | None,
) -> bool:
    return _plot_diagnostics.plot_cold_start_breakdown(
        df=df,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
    )


def plot_latency_model_residuals(
    residuals_path: str,
    out_png: str | None,
) -> bool:
    return _plot_latency.plot_latency_model_residuals(
        residuals_path=residuals_path,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
    )


def plot_latency_model_fit_curves(
    residuals_path: str,
    report_path: str,
    out_png: str | None,
) -> bool:
    return _plot_latency.plot_latency_model_fit_curves(
        residuals_path=residuals_path,
        report_path=report_path,
        out_png=out_png,
        show_plots=SHOW_PLOTS,
    )


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
    all_status_df = prepare_df(csv_path, only_ok=False)
    df = all_status_df.copy()
    if ONLY_OK and "status" in df.columns:
        normalized_status = df["status"].astype(str).str.strip().str.lower()
        df = df[normalized_status == "ok"].copy()
    static_meta = read_static_meta(csv_path)

    scale_type = "input_scale"
    if static_meta.get("input_scale_type"):
        scale_type = static_meta["input_scale_type"]
    else:
        metadata_df = df if not df.empty else all_status_df
        types = (
            metadata_df["input_scale_type"].dropna().unique()
            if "input_scale_type" in metadata_df.columns
            else []
        )
        if len(types) == 1:
            scale_type = types[0]

    xlabel = scale_type

    output_dir = os.path.dirname(csv_path) or "."

    performance_groups = dict(build_plot_groups(df))
    feasibility_groups = dict(build_plot_groups(all_status_df))
    for group_name in PLOT_OUTPUT_DIRS:
        group_df = performance_groups[group_name]
        feasibility_df = feasibility_groups[group_name]
        group_output_dir = os.path.join(output_dir, group_name)
        if group_df.empty and feasibility_df.empty:
            print(f"[skip] No data available for {group_name} plots")
            continue
        if SAVE_PNG:
            os.makedirs(group_output_dir, exist_ok=True)

        if not feasibility_df.empty:
            plot_resource_feasibility_heatmap(
                feasibility_df,
                xlabel=xlabel,
                out_png=(
                    os.path.join(group_output_dir, RESOURCE_FEASIBILITY_PLOT)
                    if SAVE_PNG
                    else None
                ),
            )
        if group_df.empty:
            print(f"[skip] No successful data available for {group_name} metric plots")
            continue

        plot_tail_latency_overview(
            group_df,
            xlabel=xlabel,
            out_png=(
                os.path.join(group_output_dir, TAIL_LATENCY_PLOT)
                if SAVE_PNG
                else None
            ),
        )
        plot_latency_energy_pareto(
            group_df,
            xlabel=xlabel,
            out_png=(
                os.path.join(group_output_dir, LATENCY_ENERGY_PARETO_PLOT)
                if SAVE_PNG
                else None
            ),
        )

        for (
            title,
            filename,
            rows,
            columns,
            panels,
            shared_y_groups,
        ) in METRIC_OVERVIEW_PLOTS:
            plot_metric_overview(
                group_df,
                panels=panels,
                rows=rows,
                columns=columns,
                shared_y_groups=shared_y_groups,
                title=title,
                xlabel=xlabel,
                out_png=(
                    os.path.join(group_output_dir, filename)
                    if SAVE_PNG
                    else None
                ),
            )

        for metric, title, ylabel, filename in PLOT_METRICS:
            plot_metric(
                group_df,
                metric=metric,
                title=title,
                ylabel=ylabel,
                xlabel=xlabel,
                out_png=(
                    os.path.join(group_output_dir, filename)
                    if SAVE_PNG
                    else None
                ),
            )

        for (
            effective_metrics,
            total_metrics,
            title,
            filename,
        ) in ENERGY_POWER_OVERVIEW_PLOTS:
            plot_energy_power_overview(
                group_df,
                effective_metrics=effective_metrics,
                total_metrics=total_metrics,
                title=title,
                xlabel=xlabel,
                out_png=(
                    os.path.join(group_output_dir, filename)
                    if SAVE_PNG
                    else None
                ),
            )

        plot_cold_start_bar(
            group_df,
            title="Cold Start by Configuration",
            ylabel="Cold Start (s)",
            out_png=(
                os.path.join(group_output_dir, "cold_start_bar.png")
                if SAVE_PNG
                else None
            ),
        )
        plot_cold_start_breakdown(
            group_df,
            out_png=(
                os.path.join(group_output_dir, COLD_START_BREAKDOWN_PLOT)
                if SAVE_PNG
                else None
            ),
        )
    write_latency_model_report(df, static_meta, output_dir)
    model_output_dir = os.path.join(output_dir, LATENCY_MODEL_DIR)
    residuals_path = os.path.join(
        model_output_dir,
        LATENCY_MODEL_RESIDUALS,
    )
    report_path = os.path.join(
        model_output_dir,
        LATENCY_MODEL_REPORT,
    )
    if os.path.exists(residuals_path):
        plot_latency_model_residuals(
            residuals_path,
            (
                os.path.join(
                    model_output_dir,
                    LATENCY_MODEL_RESIDUAL_PLOT,
                )
                if SAVE_PNG
                else None
            ),
        )
        plot_latency_model_fit_curves(
            residuals_path,
            report_path,
            (
                os.path.join(
                    model_output_dir,
                    LATENCY_MODEL_FIT_CURVES_PLOT,
                )
                if SAVE_PNG
                else None
            ),
        )


if __name__ == "__main__":
    main()

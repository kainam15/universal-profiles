"""Plot CLI: prepare current results and invoke plotting implementations."""

import os
import acprof.analysis.latency_report as analysis_latency_report
import acprof.plotting.config as plotting_config
import acprof.plotting.data as plotting_data
import acprof.plotting.diagnostics as plotting_diagnostics
import acprof.plotting.latency as plotting_latency
import acprof.plotting.metrics as plotting_metrics
import sys


CSV_PATH = "results/result_all.csv"
ONLY_OK = True
SAVE_PNG = True


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
    all_status_df = plotting_data.prepare_df(csv_path, only_ok=False)
    df = all_status_df.copy()
    if ONLY_OK and "status" in df.columns:
        normalized_status = df["status"].astype(str).str.strip().str.lower()
        df = df[normalized_status == "ok"].copy()
    static_meta = plotting_data.read_static_meta(csv_path)

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

    performance_groups = dict(plotting_data.build_plot_groups(df))
    feasibility_groups = dict(plotting_data.build_plot_groups(all_status_df))
    for group_name in plotting_config.PLOT_OUTPUT_DIRS:
        group_df = performance_groups[group_name]
        feasibility_df = feasibility_groups[group_name]
        group_output_dir = os.path.join(output_dir, group_name)
        if group_df.empty and feasibility_df.empty:
            print(f"[skip] No data available for {group_name} plots")
            continue
        if SAVE_PNG:
            os.makedirs(group_output_dir, exist_ok=True)

        if not feasibility_df.empty:
            plotting_diagnostics.plot_resource_feasibility_heatmap(
                feasibility_df,
                xlabel=xlabel,
                out_png=(
                    os.path.join(group_output_dir, plotting_config.RESOURCE_FEASIBILITY_PLOT)
                    if SAVE_PNG
                    else None
                ),
            )
        if group_df.empty:
            print(f"[skip] No successful data available for {group_name} metric plots")
            continue

        plotting_diagnostics.plot_tail_latency_overview(
            group_df,
            xlabel=xlabel,
            out_png=(
                os.path.join(group_output_dir, plotting_config.TAIL_LATENCY_PLOT)
                if SAVE_PNG
                else None
            ),
        )
        plotting_diagnostics.plot_latency_energy_pareto(
            group_df,
            xlabel=xlabel,
            out_png=(
                os.path.join(group_output_dir, plotting_config.LATENCY_ENERGY_PARETO_PLOT)
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
        ) in plotting_config.METRIC_OVERVIEW_PLOTS:
            plotting_metrics.plot_metric_overview(
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

        for metric, title, ylabel, filename in plotting_config.PLOT_METRICS:
            plotting_metrics.plot_metric(
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
        ) in plotting_config.ENERGY_POWER_OVERVIEW_PLOTS:
            plotting_metrics.plot_energy_power_overview(
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

        plotting_metrics.plot_cold_start_bar(
            group_df,
            title="Cold Start by Configuration",
            ylabel="Cold Start (s)",
            out_png=(
                os.path.join(group_output_dir, "cold_start_bar.png")
                if SAVE_PNG
                else None
            ),
        )
        plotting_diagnostics.plot_cold_start_breakdown(
            group_df,
            out_png=(
                os.path.join(group_output_dir, plotting_config.COLD_START_BREAKDOWN_PLOT)
                if SAVE_PNG
                else None
            ),
        )
    analysis_latency_report.write_latency_model_report(df, static_meta, output_dir)
    model_output_dir = os.path.join(output_dir, analysis_latency_report.LATENCY_MODEL_DIR)
    residuals_path = os.path.join(
        model_output_dir,
        analysis_latency_report.LATENCY_MODEL_RESIDUALS,
    )
    report_path = os.path.join(
        model_output_dir,
        analysis_latency_report.LATENCY_MODEL_REPORT,
    )
    if os.path.exists(residuals_path):
        plotting_latency.plot_latency_model_residuals(
            residuals_path,
            (
                os.path.join(
                    model_output_dir,
                    plotting_config.LATENCY_MODEL_RESIDUAL_PLOT,
                )
                if SAVE_PNG
                else None
            ),
        )
        plotting_latency.plot_latency_model_fit_curves(
            residuals_path,
            report_path,
            (
                os.path.join(
                    model_output_dir,
                    plotting_config.LATENCY_MODEL_FIT_CURVES_PLOT,
                )
                if SAVE_PNG
                else None
            ),
        )


if __name__ == "__main__":
    main()

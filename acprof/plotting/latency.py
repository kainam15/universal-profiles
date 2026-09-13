"""延迟模型残差和拟合曲线图表。"""

import inspect
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from acprof.analysis.latency_model import (
    LATENCY_MODEL_FEATURES,
    _model_feature_names,
    _predict_latency,
)

from acprof.plotting.config import (
    CPU_FIXED_COLORS,
    GPU_GREEN,
)

from acprof.plotting.styles import (
    build_cpu_base_colors,
)


def _residual_plot_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[float | None, float, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    residual_values = actual_values - predicted_values
    centered_values = actual_values - float(np.mean(actual_values))
    total_sum_squares = float(np.dot(centered_values, centered_values))
    if total_sum_squares > 0.0:
        r2 = 1.0 - (
            float(np.dot(residual_values, residual_values)) / total_sum_squares
        )
    else:
        r2 = None

    mean_actual = float(np.mean(np.abs(actual_values)))
    relative_mae = (
        float(np.mean(np.abs(residual_values))) / mean_actual
        if mean_actual > 0.0
        else math.nan
    )
    mean_absolute_percentage_error = float(
        np.mean(np.abs(residual_values) / np.abs(actual_values))
    )
    return r2, relative_mae, mean_absolute_percentage_error


def plot_latency_model_residuals(
    residuals_path: str,
    out_png: str | None,
    *,
    show_plots: bool = False,
) -> bool:
    """Plot OOF latency residual diagnostics from a model residual artifact."""
    if not os.path.exists(residuals_path):
        print(f"[skip] Cannot find {residuals_path}")
        return False

    residual_df = pd.read_csv(residuals_path, skipinitialspace=True)
    required_columns = {
        "hardware_model",
        "input_scale",
        "latency_s",
        "resource_config_oof_predicted_latency_s",
        "resource_config_oof_residual_s",
    }
    missing_columns = sorted(required_columns.difference(residual_df.columns))
    if missing_columns:
        print(
            "[skip] Latency residual CSV missing required columns: "
            f"{missing_columns}"
        )
        return False
    if residual_df.empty:
        print(f"[skip] No latency residual rows in {residuals_path}")
        return False

    numeric_columns = [
        "input_scale",
        "latency_s",
        "resource_config_oof_predicted_latency_s",
        "resource_config_oof_residual_s",
        "max_scale_holdout_predicted_latency_s",
        "max_scale_holdout_residual_s",
    ]
    for column in numeric_columns:
        if column in residual_df.columns:
            residual_df[column] = pd.to_numeric(
                residual_df[column],
                errors="coerce",
            )

    residual_df["hardware_model"] = (
        residual_df["hardware_model"].astype(str).str.strip().str.lower()
    )
    valid_mask = (
        residual_df[
            [
                "input_scale",
                "latency_s",
                "resource_config_oof_predicted_latency_s",
                "resource_config_oof_residual_s",
            ]
        ]
        .apply(np.isfinite)
        .all(axis=1)
        & (residual_df["input_scale"] > 0.0)
        & (residual_df["latency_s"] > 0.0)
        & (residual_df["resource_config_oof_predicted_latency_s"] > 0.0)
    )
    residual_df = residual_df[valid_mask].copy()
    if residual_df.empty:
        print(f"[skip] No valid latency residual rows in {residuals_path}")
        return False

    residual_df["relative_residual_pct"] = (
        100.0 * residual_df["resource_config_oof_residual_s"] / residual_df["latency_s"]
    )
    hardware_order = [
        hardware_model
        for hardware_model in ("cpu", "gpu")
        if hardware_model in set(residual_df["hardware_model"])
    ]
    hardware_order.extend(
        sorted(set(residual_df["hardware_model"]).difference(hardware_order))
    )
    fallback_colors = plt.get_cmap("tab10")
    hardware_styles = {
        "cpu": {
            "color": CPU_FIXED_COLORS[1],
            "marker": "o",
            "label": "CPU",
        },
        "gpu": {
            "color": GPU_GREEN,
            "marker": "s",
            "label": "GPU",
        },
    }
    for index, hardware_model in enumerate(hardware_order):
        hardware_styles.setdefault(
            hardware_model,
            {
                "color": fallback_colors(index % fallback_colors.N),
                "marker": "^",
                "label": hardware_model.upper(),
            },
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    parity_axis, prediction_axis, distribution_axis, scale_axis = axes.flat

    for hardware_model in hardware_order:
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        style = hardware_styles[hardware_model]
        parity_axis.scatter(
            hardware_df["latency_s"],
            hardware_df["resource_config_oof_predicted_latency_s"],
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.8,
            s=38,
            label=style["label"],
        )
    parity_min = float(
        min(
            residual_df["latency_s"].min(),
            residual_df["resource_config_oof_predicted_latency_s"].min(),
        )
    )
    parity_max = float(
        max(
            residual_df["latency_s"].max(),
            residual_df["resource_config_oof_predicted_latency_s"].max(),
        )
    )
    parity_axis.plot(
        [parity_min, parity_max],
        [parity_min, parity_max],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Ideal",
    )
    parity_axis.set_xscale("log")
    parity_axis.set_yscale("log")
    parity_axis.set_xlim(parity_min * 0.82, parity_max * 1.22)
    parity_axis.set_ylim(parity_min * 0.82, parity_max * 1.22)
    parity_axis.set_title("Resource-config OOF: actual vs. predicted")
    parity_axis.set_xlabel("Actual latency (s)")
    parity_axis.set_ylabel("Predicted latency (s)")
    parity_axis.grid(True, which="both", linestyle="-", alpha=0.25)
    parity_axis.legend(fontsize=8)
    r2, relative_mae, mean_absolute_percentage_error = _residual_plot_metrics(
        residual_df["latency_s"],
        residual_df["resource_config_oof_predicted_latency_s"],
    )
    metric_lines = [f"n = {len(residual_df)}"]
    if r2 is not None:
        metric_lines.append(f"R² = {r2:.4f}")
    if math.isfinite(relative_mae):
        metric_lines.append(f"Relative MAE = {100.0 * relative_mae:.1f}%")
    if math.isfinite(mean_absolute_percentage_error):
        metric_lines.append(
            f"Overall MAPE = {100.0 * mean_absolute_percentage_error:.1f}%"
        )
    for hardware_model in hardware_order:
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        _, _, hardware_mape = _residual_plot_metrics(
            hardware_df["latency_s"],
            hardware_df["resource_config_oof_predicted_latency_s"],
        )
        if math.isfinite(hardware_mape):
            metric_lines.append(
                f"{hardware_styles[hardware_model]['label']} MAPE = "
                f"{100.0 * hardware_mape:.1f}%"
            )
    parity_axis.text(
        0.04,
        0.96,
        "\n".join(metric_lines),
        transform=parity_axis.transAxes,
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )

    for hardware_model in hardware_order:
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        style = hardware_styles[hardware_model]
        prediction_axis.scatter(
            hardware_df["resource_config_oof_predicted_latency_s"],
            hardware_df["relative_residual_pct"],
            color=style["color"],
            marker=style["marker"],
            alpha=0.65,
            s=32,
            label=style["label"],
        )
    prediction_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    prediction_axis.set_xscale("log")
    prediction_axis.set_title("OOF residual vs. predicted latency")
    prediction_axis.set_xlabel("Predicted latency (s)")
    prediction_axis.set_ylabel("Relative residual (%)")
    prediction_axis.grid(True, which="both", linestyle="-", alpha=0.25)

    distribution_values = []
    distribution_labels = []
    distribution_colors = []
    for hardware_model in hardware_order:
        values = residual_df.loc[
            residual_df["hardware_model"] == hardware_model,
            "relative_residual_pct",
        ].to_numpy(dtype=float)
        distribution_values.append(values)
        distribution_labels.append(hardware_styles[hardware_model]["label"])
        distribution_colors.append(hardware_styles[hardware_model]["color"])
    boxplot_label_argument = (
        {"tick_labels": distribution_labels}
        if "tick_labels"
        in inspect.signature(distribution_axis.boxplot).parameters
        else {"labels": distribution_labels}
    )
    boxplot = distribution_axis.boxplot(
        distribution_values,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        **boxplot_label_argument,
    )
    for patch, color in zip(boxplot["boxes"], distribution_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    random_generator = np.random.default_rng(0)
    for position, (values, color) in enumerate(
        zip(distribution_values, distribution_colors),
        start=1,
    ):
        jitter = random_generator.uniform(-0.13, 0.13, size=len(values))
        distribution_axis.scatter(
            np.full(len(values), position, dtype=float) + jitter,
            values,
            color=color,
            alpha=0.45,
            s=18,
        )
    distribution_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.1,
    )
    distribution_axis.set_title("OOF residual distribution")
    distribution_axis.set_xlabel("Hardware model")
    distribution_axis.set_ylabel("Relative residual (%)")
    distribution_axis.grid(True, axis="y", linestyle="-", alpha=0.25)

    for hardware_model in hardware_order:
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        style = hardware_styles[hardware_model]
        scale_axis.scatter(
            hardware_df["input_scale"],
            hardware_df["relative_residual_pct"],
            color=style["color"],
            marker=style["marker"],
            alpha=0.4,
            s=25,
        )
        median_by_scale = (
            hardware_df.groupby("input_scale", as_index=False)[
                "relative_residual_pct"
            ]
            .median()
            .sort_values("input_scale")
        )
        scale_axis.plot(
            median_by_scale["input_scale"],
            median_by_scale["relative_residual_pct"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=f"{style['label']} OOF median",
        )

        holdout_columns = {
            "max_scale_holdout_predicted_latency_s",
            "max_scale_holdout_residual_s",
        }
        if holdout_columns.issubset(residual_df.columns):
            holdout_df = hardware_df.dropna(subset=list(holdout_columns)).copy()
            holdout_df = holdout_df[
                np.isfinite(
                    holdout_df["max_scale_holdout_predicted_latency_s"]
                )
                & np.isfinite(holdout_df["max_scale_holdout_residual_s"])
                & (holdout_df["max_scale_holdout_predicted_latency_s"] > 0.0)
            ]
            if not holdout_df.empty:
                holdout_relative_residual = (
                    100.0
                    * holdout_df["max_scale_holdout_residual_s"]
                    / holdout_df["latency_s"]
                )
                scale_axis.scatter(
                    holdout_df["input_scale"],
                    holdout_relative_residual,
                    color=style["color"],
                    marker="*",
                    edgecolor="black",
                    linewidth=0.45,
                    alpha=0.85,
                    s=75,
                    label=f"{style['label']} max-scale holdout",
                )
    scale_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    scale_axis.set_title("Residual vs. input scale")
    scale_axis.set_xlabel("Input scale")
    scale_axis.set_ylabel("Relative residual (%)")
    scale_axis.grid(True, which="both", linestyle="-", alpha=0.25)
    scale_axis.legend(fontsize=8, ncol=2)

    fig.suptitle("Latency Model Residual Diagnostics", fontsize=15)
    fig.text(
        0.5,
        0.015,
        (
            "Relative residual = (actual - predicted) / actual × 100; "
            "positive values indicate underprediction."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))

    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")

    if show_plots:
        plt.show()
    plt.close(fig)
    return True


def plot_latency_model_fit_curves(
    residuals_path: str,
    report_path: str,
    out_png: str | None,
    *,
    show_plots: bool = False,
) -> bool:
    """Plot full-fit latency curves for every CPU/memory configuration."""
    if not os.path.exists(residuals_path):
        print(f"[skip] Cannot find {residuals_path}")
        return False
    if not os.path.exists(report_path):
        print(f"[skip] Cannot find {report_path}")
        return False

    residual_df = pd.read_csv(residuals_path, skipinitialspace=True)
    required_columns = {
        "hardware_model",
        "cpu_cores",
        "mem_cap_gb",
        "input_scale",
        "latency_s",
    }
    missing_columns = sorted(required_columns.difference(residual_df.columns))
    if missing_columns:
        print(
            "[skip] Latency residual CSV missing fit-curve columns: "
            f"{missing_columns}"
        )
        return False
    if residual_df.empty:
        print(f"[skip] No latency fit rows in {residuals_path}")
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    for column in ("cpu_cores", "mem_cap_gb", "input_scale", "latency_s"):
        residual_df[column] = pd.to_numeric(residual_df[column], errors="coerce")
    residual_df["hardware_model"] = (
        residual_df["hardware_model"].astype(str).str.strip().str.lower()
    )
    valid_mask = (
        residual_df[
            ["cpu_cores", "mem_cap_gb", "input_scale", "latency_s"]
        ]
        .apply(np.isfinite)
        .all(axis=1)
        & (residual_df["cpu_cores"] > 0.0)
        & (residual_df["mem_cap_gb"] > 0.0)
        & (residual_df["input_scale"] > 0.0)
        & (residual_df["latency_s"] > 0.0)
    )
    residual_df = residual_df[valid_mask].copy()

    fitted_models = {}
    report_models = report.get("models", {})
    for hardware_model in ("cpu", "gpu"):
        model_report = report_models.get(hardware_model, {})
        coefficient_map = model_report.get("coefficients", {})
        input_scale_basis = model_report.get("input_scale_basis", {})
        input_scale_spline_knots = input_scale_basis.get("knots", [])
        if not isinstance(input_scale_spline_knots, list):
            continue
        try:
            input_scale_spline_knots = [
                float(knot) for knot in input_scale_spline_knots
            ]
        except (TypeError, ValueError):
            continue
        feature_names = model_report.get(
            "feature_columns",
            LATENCY_MODEL_FEATURES[hardware_model],
        )
        if feature_names != _model_feature_names(
            hardware_model,
            input_scale_spline_knots,
        ):
            continue
        try:
            coefficients = [
                float(coefficient_map[feature_name])
                for feature_name in feature_names
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(coefficients).all():
            continue
        fitted_models[hardware_model] = {
            "hardware_model": hardware_model,
            "input_scale_spline_knots": input_scale_spline_knots,
            "coefficients": coefficients,
        }

    hardware_order = [
        hardware_model
        for hardware_model in ("cpu", "gpu")
        if hardware_model in fitted_models
        and hardware_model in set(residual_df["hardware_model"])
    ]
    if not hardware_order:
        print("[skip] No fitted CPU/GPU latency models available for curve plot")
        return False

    cpu_values = sorted(int(value) for value in residual_df["cpu_cores"].unique())
    mem_values = sorted(int(value) for value in residual_df["mem_cap_gb"].unique())
    cpu_colors = build_cpu_base_colors(cpu_values)
    mem_rank_map = {mem: index for index, mem in enumerate(mem_values)}
    line_styles = ("-", "--", "-.", ":")

    fig, axes = plt.subplots(
        1,
        len(hardware_order),
        figsize=(7.5 * len(hardware_order), 8.5),
        squeeze=False,
    )
    axes = list(axes.flat)
    legend_entries = {}
    hardware_titles = {
        "cpu": "CPU-off model",
        "gpu": "GPU-on model",
    }

    for axis, hardware_model in zip(axes, hardware_order):
        hardware_df = residual_df[
            residual_df["hardware_model"] == hardware_model
        ]
        model = fitted_models[hardware_model]
        configurations = sorted(
            {
                (int(row.cpu_cores), int(row.mem_cap_gb))
                for row in hardware_df.itertuples(index=False)
            }
        )

        for cpu_cores, mem_cap_gb in configurations:
            configuration_df = hardware_df[
                (hardware_df["cpu_cores"] == cpu_cores)
                & (hardware_df["mem_cap_gb"] == mem_cap_gb)
            ].sort_values("input_scale")
            min_scale = float(configuration_df["input_scale"].min())
            max_scale = float(configuration_df["input_scale"].max())
            if min_scale == max_scale:
                curve_scales = np.asarray([min_scale], dtype=float)
            else:
                curve_scales = np.linspace(min_scale, max_scale, 240)
            curve_predictions = [
                _predict_latency(
                    SimpleNamespace(
                        input_scale=input_scale,
                        cpu_cores=cpu_cores,
                        mem_cap_gb=mem_cap_gb,
                    ),
                    model,
                )
                for input_scale in curve_scales
            ]

            color = cpu_colors[cpu_cores]
            line_style = line_styles[
                mem_rank_map[mem_cap_gb] % len(line_styles)
            ]
            label = f"{cpu_cores} CPU / {mem_cap_gb} GiB"
            line, = axis.plot(
                curve_scales,
                curve_predictions,
                color=color,
                linestyle=line_style,
                linewidth=1.8,
                alpha=0.92,
                label=label,
            )
            axis.scatter(
                configuration_df["input_scale"],
                configuration_df["latency_s"],
                facecolor="white",
                edgecolor=color,
                marker="o",
                linewidth=1.1,
                s=30,
                alpha=0.95,
                zorder=3,
            )
            legend_entries.setdefault((cpu_cores, mem_cap_gb), line)

        axis.set_title(hardware_titles[hardware_model])
        axis.set_xlabel("Input scale")
        axis.set_ylabel("Latency (s, log scale)")
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="-", alpha=0.25)

    input_scale_type = str(report.get("input_scale_type", "")).strip()
    if input_scale_type and input_scale_type != "input_scale":
        for axis in axes:
            axis.set_xlabel(f"Input scale ({input_scale_type})")

    ordered_legend_entries = sorted(legend_entries.items())
    fig.legend(
        [line for _, line in ordered_legend_entries],
        [
            f"{cpu_cores} CPU / {mem_cap_gb} GiB"
            for (cpu_cores, mem_cap_gb), _ in ordered_legend_entries
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, len(ordered_legend_entries)),
        fontsize=8,
        title="Resource configuration",
    )
    fig.suptitle("Latency Model Full-Fit Curves", fontsize=15)
    fig.text(
        0.5,
        0.935,
        (
            "Curves: full-data least-squares fit; "
            "hollow markers: measured case medians"
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.18, 1.0, 0.91))

    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")

    if show_plots:
        plt.show()
    plt.close(fig)
    return True

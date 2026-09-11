"""延迟模型报告和残差 CSV 产物写出。"""

import csv
import json
import math
import os

import numpy as np
import pandas as pd

from acprof.analysis.latency_model import (
    LATENCY_MODEL_FEATURES,
    LATENCY_MODEL_FORMULAS,
    LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE,
    LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO,
    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE,
    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE,
    LATENCY_MODEL_MAX_SCALE_CASE_RELATIVE_ERROR,
    LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR,
    LATENCY_MODEL_MAX_VALIDATION_MAPE,
    LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE,
    LATENCY_MODEL_MIN_VALIDATION_POINTS,
    LATENCY_MODEL_MIN_VALIDATION_R2,
    _aggregate_latency_model_points,
    _fit_log_latency_model,
    _input_scale_validation,
    _latency_model_frame,
    _point_key,
    _predict_latency,
    _regression_metrics,
    _resource_configuration_validation,
    _training_range,
    _validation_is_evaluable,
    _validation_quality_failures,
)

LATENCY_MODEL_DIR = "latency_model"
LATENCY_MODEL_REPORT = "latency_model_report.json"
LATENCY_MODEL_RESIDUALS = "latency_model_residuals.csv"
LATENCY_MODEL_RESIDUAL_FIELDS = [
    "report_schema_version",
    "case_id",
    "split",
    "hardware_model",
    "cpu_cores",
    "mem_cap_gb",
    "gpu_mode",
    "input_scale",
    "repeat_count",
    "latency_s",
    "latency_mean_s",
    "latency_std_s",
    "predicted_latency_s",
    "residual_s",
    "fitted_predicted_latency_s",
    "fitted_residual_s",
    "resource_config_oof_predicted_latency_s",
    "resource_config_oof_residual_s",
    "max_scale_holdout_predicted_latency_s",
    "max_scale_holdout_residual_s",
]


def _clean_json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json_value(item) for item in value]
    return value


def _format_optional_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.9f}"


def _write_skipped_latency_model_report(
    output_dir: str,
    static_meta: dict[str, object],
    reason: str,
) -> None:
    report_path = os.path.join(output_dir, LATENCY_MODEL_REPORT)
    residuals_path = os.path.join(output_dir, LATENCY_MODEL_RESIDUALS)
    # Always replace any previous residual artifact so a skipped rerun cannot
    # leave stale predictions that appear to belong to the new report.
    with open(residuals_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LATENCY_MODEL_RESIDUAL_FIELDS)
        writer.writeheader()
    report = {
        "report_schema_version": 2,
        "status": "skipped",
        "prediction_ready": False,
        "reason": reason,
        "target_metric": "latency_s",
        "model_name": static_meta.get("model_name", ""),
        "task_family": static_meta.get("task_family", ""),
        "residuals_csv": LATENCY_MODEL_RESIDUALS,
        "residuals_granularity": "header only because model generation was skipped",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
        f.write("\n")
    print(f"[saved] {report_path}")


def write_latency_model_report(
    df: pd.DataFrame,
    static_meta: dict[str, object],
    output_dir: str,
) -> None:
    """Fit validated positive latency models and write report/residual artifacts."""
    model_output_dir = os.path.join(output_dir, LATENCY_MODEL_DIR)
    os.makedirs(model_output_dir, exist_ok=True)

    if "latency_s" not in df.columns:
        _write_skipped_latency_model_report(
            model_output_dir,
            static_meta,
            "latency_s column missing",
        )
        return

    try:
        model_df = _latency_model_frame(df)
    except ValueError as exc:
        _write_skipped_latency_model_report(
            model_output_dir,
            static_meta,
            str(exc),
        )
        return

    if len(model_df) < 3:
        _write_skipped_latency_model_report(
            model_output_dir,
            static_meta,
            f"need at least 3 valid raw rows, got {len(model_df)}",
        )
        return

    points = _aggregate_latency_model_points(model_df)
    if len(points) < 3:
        _write_skipped_latency_model_report(
            model_output_dir,
            static_meta,
            f"need at least 3 unique configuration-scale cases, got {len(points)}",
        )
        return

    model_reports = {}
    fit_predictions: dict[tuple[str, float, float, float], float] = {}
    configuration_predictions: dict[
        tuple[str, float, float, float],
        float,
    ] = {}
    scale_predictions: dict[tuple[str, float, float, float], float] = {}
    top_level_failures = []

    for hardware_model in ("cpu", "gpu"):
        hardware_rows = [
            row for row in points if str(row.hardware_model) == hardware_model
        ]
        if not hardware_rows:
            continue

        try:
            fitted_model = _fit_log_latency_model(hardware_rows, hardware_model)
        except ValueError as exc:
            reason = str(exc)
            model_reports[hardware_model] = {
                "status": "skipped",
                "prediction_ready": False,
                "reason": reason,
                "raw_rows": sum(int(row.repeat_count) for row in hardware_rows),
                "case_rows": len(hardware_rows),
            }
            top_level_failures.append(f"{hardware_model}: fit failed: {reason}")
            continue

        hardware_fit_predictions = {
            _point_key(row): _predict_latency(row, fitted_model)
            for row in hardware_rows
        }
        fit_predictions.update(hardware_fit_predictions)
        fit_metrics = _regression_metrics(
            [float(row.latency_s) for row in hardware_rows],
            [hardware_fit_predictions[_point_key(row)] for row in hardware_rows],
        )

        configuration_validation, hardware_configuration_predictions = (
            _resource_configuration_validation(hardware_rows, hardware_model)
        )
        input_scale_validation, hardware_scale_predictions = (
            _input_scale_validation(hardware_rows, hardware_model)
        )
        configuration_predictions.update(hardware_configuration_predictions)
        scale_predictions.update(hardware_scale_predictions)

        quality_failures = []
        quality_failures.extend(_validation_quality_failures(
            "resource_configuration_holdout",
            configuration_validation,
        ))
        quality_failures.extend(_validation_quality_failures(
            "input_scale_holdout",
            input_scale_validation,
        ))
        if int(fit_metrics["nonfinite_prediction_count"] or 0) > 0:
            quality_failures.append("full fit contains non-finite predictions")
        if int(fit_metrics["nonpositive_prediction_count"] or 0) > 0:
            quality_failures.append("full fit contains non-positive predictions")

        validation_evaluable = bool(
            _validation_is_evaluable(configuration_validation)
            and _validation_is_evaluable(input_scale_validation)
        )
        if quality_failures:
            model_status = "poor_fit" if validation_evaluable else "unvalidated"
        else:
            model_status = "ok"

        coefficients = {
            name: value
            for name, value in zip(
                fitted_model["feature_names"],
                fitted_model["coefficients"],
            )
        }
        if hardware_model == "gpu":
            upper_tail_model = fitted_model.get("gpu_upper_tail", {})
            upper_tail_feature_names = upper_tail_model.get(
                "feature_names",
                [],
            )
            upper_tail_coefficients = upper_tail_model.get(
                "coefficients",
                [],
            )
            upper_tail_report = {
                "type": "continuous_affine_latency_tail",
                "enabled": bool(upper_tail_model.get("enabled")),
                "activation_rule": (
                    "enable when nested one-step-forward spline MAPE exceeds "
                    f"{LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE}, the "
                    "training scale span is at least "
                    f"{LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO}x, "
                    "and all fitted training-configuration slopes are positive"
                ),
                "continuity_rule": (
                    "spline prediction at the upper training boundary plus "
                    "the affine tail's change beyond that boundary"
                ),
                "feature_columns": upper_tail_feature_names,
                "selected_feature_columns": upper_tail_model.get(
                    "selected_feature_names",
                    [],
                ),
                "dropped_feature_columns": upper_tail_model.get(
                    "dropped_feature_names",
                    [],
                ),
                "coefficients": dict(zip(
                    upper_tail_feature_names,
                    upper_tail_coefficients,
                )),
                "minimum_fitted_slope_s_per_scale": upper_tail_model.get(
                    "minimum_fitted_slope_s_per_scale"
                ),
                "training_input_scale_span_ratio": upper_tail_model.get(
                    "training_input_scale_span_ratio"
                ),
                "calibration": upper_tail_model.get("calibration", {}),
            }
            if upper_tail_model.get("reason"):
                upper_tail_report["reason"] = upper_tail_model["reason"]
            input_scale_basis = {
                "type": "continuous_piecewise_linear_spline_in_log_space",
                "knots": fitted_model["input_scale_spline_knots"],
                "knot_rule": "all interior observed training input scales",
                "interpolation": (
                    "piecewise linear in log(input_scale) and log(latency_s)"
                ),
                "lower_extrapolation": (
                    "continue the nearest boundary segment in log-log space"
                ),
                "upper_extrapolation": upper_tail_report,
            }
        else:
            input_scale_basis = {
                "type": "quadratic_response_surface_in_log_space",
                "knots": [],
                "interactions": [
                    "log_input_scale_x_log_cpu_cores",
                    "log_input_scale_x_log_mem_cap_gb",
                    "log_cpu_cores_x_log_mem_cap_gb",
                ],
            }
        model_reports[hardware_model] = {
            "status": model_status,
            "prediction_ready": model_status == "ok",
            "formula": LATENCY_MODEL_FORMULAS[hardware_model],
            "feature_columns": fitted_model["feature_names"],
            "selected_feature_columns": fitted_model["selected_feature_names"],
            "dropped_feature_columns": fitted_model["dropped_feature_names"],
            "coefficients": coefficients,
            "input_scale_basis": input_scale_basis,
            "prediction_estimand": (
                "positive latency prediction for the median-aggregated case"
            ),
            "smearing_correction_applied": False,
            "raw_rows": sum(int(row.repeat_count) for row in hardware_rows),
            "case_rows": len(hardware_rows),
            "training_range": _training_range(hardware_rows),
            "numerical_diagnostics": {
                "solver": "numpy.linalg.lstsq",
                "standardized_before_solve": True,
                "rank": fitted_model["rank"],
                "condition_number": fitted_model["condition_number"],
                **(
                    {
                        "upper_tail_rank": fitted_model.get(
                            "gpu_upper_tail",
                            {},
                        ).get("rank"),
                        "upper_tail_condition_number": fitted_model.get(
                            "gpu_upper_tail",
                            {},
                        ).get("condition_number"),
                    }
                    if hardware_model == "gpu"
                    else {}
                ),
            },
            "metrics": {
                "fit": fit_metrics,
                "resource_configuration_holdout": configuration_validation["metrics"],
                "input_scale_holdout": input_scale_validation["metrics"],
            },
            "validation": {
                "resource_configuration_holdout": configuration_validation,
                "input_scale_holdout": input_scale_validation,
            },
            "quality_gate": {
                "passed": not quality_failures,
                "failures": quality_failures,
            },
        }
        top_level_failures.extend(
            f"{hardware_model}: {failure}" for failure in quality_failures
        )

    if not fit_predictions:
        status = "skipped"
    elif any(
        model_report.get("status") == "poor_fit"
        for model_report in model_reports.values()
    ):
        status = "poor_fit"
    elif any(
        model_report.get("status") != "ok"
        for model_report in model_reports.values()
    ):
        status = "unvalidated"
    else:
        status = "ok"

    def metrics_for_predictions(predictions: dict) -> dict:
        predicted_rows = [row for row in points if _point_key(row) in predictions]
        return _regression_metrics(
            [float(row.latency_s) for row in predicted_rows],
            [predictions[_point_key(row)] for row in predicted_rows],
        )

    residuals_path = os.path.join(
        model_output_dir,
        LATENCY_MODEL_RESIDUALS,
    )
    with open(residuals_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LATENCY_MODEL_RESIDUAL_FIELDS)
        writer.writeheader()
        for case_id, row in enumerate(points):
            key = _point_key(row)
            actual = float(row.latency_s)
            fitted_prediction = fit_predictions.get(key)
            configuration_prediction = configuration_predictions.get(key)
            scale_prediction = scale_predictions.get(key)
            writer.writerow({
                "report_schema_version": 2,
                "case_id": case_id,
                "split": (
                    "out_of_fold_test"
                    if configuration_prediction is not None
                    else "validation_unavailable"
                ),
                "hardware_model": row.hardware_model,
                "cpu_cores": int(row.cpu_cores),
                "mem_cap_gb": int(row.mem_cap_gb),
                "gpu_mode": "on" if row.hardware_model == "gpu" else "off",
                "input_scale": f"{float(row.input_scale):.6f}",
                "repeat_count": int(row.repeat_count),
                "latency_s": f"{actual:.9f}",
                "latency_mean_s": f"{float(row.latency_mean_s):.9f}",
                "latency_std_s": f"{float(row.latency_std_s):.9f}",
                # Backward-compatible aliases now point to the honest
                # configuration-out-of-fold prediction, not an in-cell repeat.
                "predicted_latency_s": _format_optional_float(
                    configuration_prediction
                ),
                "residual_s": _format_optional_float(
                    None
                    if configuration_prediction is None
                    else actual - configuration_prediction
                ),
                "fitted_predicted_latency_s": _format_optional_float(
                    fitted_prediction
                ),
                "fitted_residual_s": _format_optional_float(
                    None
                    if fitted_prediction is None
                    else actual - fitted_prediction
                ),
                "resource_config_oof_predicted_latency_s": _format_optional_float(
                    configuration_prediction
                ),
                "resource_config_oof_residual_s": _format_optional_float(
                    None
                    if configuration_prediction is None
                    else actual - configuration_prediction
                ),
                "max_scale_holdout_predicted_latency_s": _format_optional_float(
                    scale_prediction
                ),
                "max_scale_holdout_residual_s": _format_optional_float(
                    None if scale_prediction is None else actual - scale_prediction
                ),
            })

    fit_metrics = metrics_for_predictions(fit_predictions)
    configuration_metrics = metrics_for_predictions(configuration_predictions)
    input_scale_metrics = metrics_for_predictions(scale_predictions)
    report = {
        "report_schema_version": 2,
        "status": status,
        "prediction_ready": status == "ok",
        "target_metric": "latency_s",
        "model_name": static_meta.get("model_name", ""),
        "task_family": static_meta.get("task_family", ""),
        "input_scale_type": static_meta.get("input_scale_type", "input_scale"),
        "model_type": (
            "separate_cpu_log_response_surface_gpu_log_spline_affine_tail"
        ),
        "positive_prediction_form": True,
        "formula": LATENCY_MODEL_FORMULAS,
        "feature_columns": {
            hardware_model: model_reports.get(hardware_model, {}).get(
                "feature_columns",
                LATENCY_MODEL_FEATURES[hardware_model],
            )
            for hardware_model in ("cpu", "gpu")
        },
        "separation_note": (
            "CPU-off and GPU-on use independent coefficients. Their separate "
            "input-scale/CPU interaction terms are the split-model equivalent "
            "of input_scale x CPU x GPU interactions in one joint model."
        ),
        "aggregation": {
            "group_columns": [
                "hardware_model",
                "cpu_cores",
                "mem_cap_gb",
                "input_scale",
            ],
            "target_statistic": "median",
            "repetitions_split_across_train_and_test": False,
        },
        "rows": len(model_df),
        "raw_rows": len(model_df),
        "case_rows": len(points),
        "fit_case_rows": len(fit_predictions),
        "resource_configuration_oof_test_case_rows": len(
            configuration_predictions
        ),
        "input_scale_holdout_test_case_rows": len(scale_predictions),
        # Kept for report-v1 readers. This is cross-validation, so these are
        # accounting aliases rather than one fixed pair of disjoint row sets.
        "train_rows": len(points),
        "test_rows": len(configuration_predictions),
        "legacy_row_count_note": (
            "train_rows is the number of cases in the final full fit; test_rows "
            "is the number receiving one resource-configuration OOF prediction. "
            "They are not a single fixed mutually exclusive split."
        ),
        "split_rule": (
            "resource configuration: leave one complete (cpu_cores,mem_cap_gb) "
            "out per fold; input scale: train below maximum and hold out maximum"
        ),
        "metrics": {
            "fit": fit_metrics,
            "resource_configuration_holdout": configuration_metrics,
            "input_scale_holdout": input_scale_metrics,
            # Compatibility aliases: "test" is now true configuration OOF.
            "train": fit_metrics,
            "test": configuration_metrics,
        },
        "models": model_reports,
        "quality_gate": {
            "passed": status == "ok",
            "thresholds": {
                # Compatibility key retained; the scope field below makes clear
                # that a fixed-scale extrapolation fold is gated by relative
                # errors rather than configuration-variation R-squared.
                "minimum_validation_r2": LATENCY_MODEL_MIN_VALIDATION_R2,
                "minimum_resource_configuration_validation_r2": (
                    LATENCY_MODEL_MIN_VALIDATION_R2
                ),
                "r2_quality_gate_scope": (
                    "resource_configuration_holdout only"
                ),
                "maximum_validation_relative_mae": (
                    LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE
                ),
                "maximum_validation_mean_absolute_percentage_error": (
                    LATENCY_MODEL_MAX_VALIDATION_MAPE
                ),
                "maximum_single_configuration_fold_relative_mae": (
                    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE
                ),
                "maximum_single_configuration_fold_mean_absolute_percentage_error": (
                    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE
                ),
                "maximum_single_validation_case_relative_error": (
                    LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR
                ),
                "maximum_single_scale_holdout_case_relative_error": (
                    LATENCY_MODEL_MAX_SCALE_CASE_RELATIVE_ERROR
                ),
                "minimum_validation_predictions": (
                    LATENCY_MODEL_MIN_VALIDATION_POINTS
                ),
                "require_finite_positive_predictions": True,
            },
            "failures": top_level_failures,
        },
        "prediction_scope": {
            "supported": (
                "interpolation within each hardware model's training_range; "
                "GPU input-scale interpolation uses a continuous log-log spline; "
                "GPU upper extrapolation can use a calibrated continuous affine "
                "latency tail; the maximum observed input scale is separately "
                "forward-validated"
            ),
            "warning": (
                "Predictions outside profiled CPU, memory, or input-scale ranges "
                "are unvalidated extrapolations."
            ),
        },
        "residuals_csv": LATENCY_MODEL_RESIDUALS,
        "residuals_granularity": (
            "one median-aggregated hardware/cpu/memory/input-scale case per row"
        ),
    }
    report_path = os.path.join(
        model_output_dir,
        LATENCY_MODEL_REPORT,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_clean_json_value(report), f, ensure_ascii=True, indent=2)
        f.write("\n")

    print(f"[saved] {report_path}")
    print(f"[saved] {residuals_path}")

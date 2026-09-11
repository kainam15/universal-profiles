"""数值延迟模型拟合、预测及验证；不依赖绘图层。"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

GPU_MODE_ON_VALUES = {"on", "1", "1.0", "true", "yes", "gpu"}
GPU_MODE_OFF_VALUES = {"off", "0", "0.0", "false", "no", "cpu"}
LATENCY_MODEL_FEATURES = {
    "cpu": [
        "intercept",
        "log_input_scale",
        "log_input_scale_squared",
        "log_cpu_cores",
        "log_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_log_cpu_cores",
        "log_input_scale_x_log_mem_cap_gb",
        "log_cpu_cores_x_log_mem_cap_gb",
    ],
    "gpu": [
        "intercept",
        "log_input_scale",
        "inverse_cpu_cores",
        "inverse_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_inverse_cpu_cores",
        "log_input_scale_x_inverse_cpu_cores_squared",
    ],
}
LATENCY_MODEL_RESOURCE_FEATURES = {
    "cpu": {
        "log_cpu_cores",
        "log_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_log_cpu_cores",
        "log_input_scale_x_log_mem_cap_gb",
        "log_cpu_cores_x_log_mem_cap_gb",
    },
    "gpu": {
        "inverse_cpu_cores",
        "inverse_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_inverse_cpu_cores",
        "log_input_scale_x_inverse_cpu_cores_squared",
    },
}
LATENCY_MODEL_FORMULAS = {
    "cpu": (
        "latency_s = exp(intercept + log_input_scale "
        "+ log_input_scale_squared "
        "+ log_cpu_cores + log_cpu_cores_squared + log_mem_cap_gb "
        "+ log_input_scale_x_log_cpu_cores "
        "+ log_input_scale_x_log_mem_cap_gb "
        "+ log_cpu_cores_x_log_mem_cap_gb)"
    ),
    "gpu": (
        "within-range latency_s = exp(intercept + log_input_scale "
        "+ piecewise_linear_log_input_scale_hinges "
        "+ inverse_cpu_cores + inverse_cpu_cores_squared "
        "+ log_mem_cap_gb + log_input_scale_x_inverse_cpu_cores "
        "+ log_input_scale_x_inverse_cpu_cores_squared); when activated, "
        "upper-tail latency_s = boundary_spline_latency_s + "
        "affine_tail(input_scale) - affine_tail(boundary_input_scale)"
    ),
}
LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES = [
    "intercept",
    "input_scale",
    "inverse_cpu_cores",
    "inverse_cpu_cores_squared",
    "log_mem_cap_gb",
    "input_scale_x_inverse_cpu_cores",
    "input_scale_x_inverse_cpu_cores_squared",
]
# A GPU log-spline is excellent inside the profiled scale range, but extending
# one noisy boundary segment can be brittle.  When a nested one-step-forward
# check exceeds this MAPE, upper extrapolation switches to a continuous affine
# work tail fitted in latency space.
LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE = 0.05
LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO = 10.0
LATENCY_MODEL_MIN_VALIDATION_R2 = 0.80
LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE = 0.20
LATENCY_MODEL_MAX_VALIDATION_MAPE = 0.20
LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE = 0.30
LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE = 0.30
LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR = 0.30
# Compatibility alias retained for callers that used the original max-scale-only
# threshold name.
LATENCY_MODEL_MAX_SCALE_CASE_RELATIVE_ERROR = (
    LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR
)
LATENCY_MODEL_MIN_VALIDATION_POINTS = 4


def _hardware_model_name(gpu_mode: object) -> str:
    if pd.isna(gpu_mode):
        raise ValueError("gpu_mode contains a missing value")
    normalized = str(gpu_mode).strip().lower()
    if normalized in GPU_MODE_ON_VALUES:
        return "gpu"
    if normalized in GPU_MODE_OFF_VALUES:
        return "cpu"
    raise ValueError(f"unsupported gpu_mode value: {gpu_mode!r}")


def _format_input_scale_knot(input_scale: float) -> str:
    return (
        format(float(input_scale), ".12g")
        .replace("-", "neg_")
        .replace(".", "p")
    )


def _model_feature_names(
    hardware_model: str,
    input_scale_spline_knots: list[float] | None = None,
) -> list[str]:
    if hardware_model == "cpu":
        return list(LATENCY_MODEL_FEATURES[hardware_model])
    if hardware_model == "gpu":
        knots = input_scale_spline_knots or []
        return [
            "intercept",
            "log_input_scale",
            *[
                f"log_input_scale_hinge_at_{_format_input_scale_knot(knot)}"
                for knot in knots
            ],
            *LATENCY_MODEL_FEATURES[hardware_model][2:],
        ]
    raise ValueError(f"unknown latency hardware model: {hardware_model}")


def _gpu_input_scale_spline_knots(rows: list) -> list[float]:
    """Use every interior observed scale as a shared log-scale spline knot."""
    input_scales = sorted({float(row.input_scale) for row in rows})
    return input_scales[1:-1]


def _model_features(
    row,
    hardware_model: str,
    input_scale_spline_knots: list[float] | None = None,
) -> list[float]:
    input_scale = float(row.input_scale)
    cpu_cores = float(row.cpu_cores)
    mem_cap_gb = float(row.mem_cap_gb)
    log_input_scale = math.log(input_scale)
    log_mem_cap_gb = math.log(mem_cap_gb)

    if hardware_model == "cpu":
        log_cpu_cores = math.log(cpu_cores)
        return [
            1.0,
            log_input_scale,
            log_input_scale**2,
            log_cpu_cores,
            log_cpu_cores**2,
            log_mem_cap_gb,
            log_input_scale * log_cpu_cores,
            log_input_scale * log_mem_cap_gb,
            log_cpu_cores * log_mem_cap_gb,
        ]
    if hardware_model == "gpu":
        inverse_cpu_cores = 1.0 / cpu_cores
        inverse_cpu_cores_squared = inverse_cpu_cores**2
        spline_knots = input_scale_spline_knots or []
        return [
            1.0,
            log_input_scale,
            *[
                max(0.0, log_input_scale - math.log(float(knot)))
                for knot in spline_knots
            ],
            inverse_cpu_cores,
            inverse_cpu_cores_squared,
            log_mem_cap_gb,
            log_input_scale * inverse_cpu_cores,
            log_input_scale * inverse_cpu_cores_squared,
        ]
    raise ValueError(f"unknown latency hardware model: {hardware_model}")


def _fit_rank_revealing_linear_model(
    feature_matrix: np.ndarray,
    target: np.ndarray,
) -> dict:
    """Solve a standardized linear model after removing dependent columns."""
    if feature_matrix.ndim != 2 or target.ndim != 1:
        raise ValueError("model matrix and target have invalid dimensions")
    if feature_matrix.shape[0] != target.shape[0]:
        raise ValueError("model matrix and target row counts differ")
    if feature_matrix.shape[1] < 2:
        raise ValueError("model matrix needs an intercept and a predictor")
    if not np.isfinite(feature_matrix).all() or not np.isfinite(target).all():
        raise ValueError("model inputs contain non-finite values")

    means = np.zeros(feature_matrix.shape[1], dtype=float)
    scales = np.ones(feature_matrix.shape[1], dtype=float)
    standardized = np.zeros_like(feature_matrix)
    standardized[:, 0] = 1.0
    for feature_idx in range(1, feature_matrix.shape[1]):
        column = feature_matrix[:, feature_idx]
        means[feature_idx] = float(column.mean())
        scales[feature_idx] = float(column.std())
        if scales[feature_idx] > 1e-12:
            standardized[:, feature_idx] = (
                column - means[feature_idx]
            ) / scales[feature_idx]

    selected_indices: list[int] = []
    for feature_idx in range(standardized.shape[1]):
        candidate_indices = selected_indices + [feature_idx]
        candidate = standardized[:, candidate_indices]
        if int(np.linalg.matrix_rank(candidate)) > len(selected_indices):
            selected_indices.append(feature_idx)

    if len(selected_indices) < 2:
        raise ValueError("model matrix has no usable varying predictor")

    selected_matrix = standardized[:, selected_indices]
    standardized_coefficients, _, rank, singular_values = np.linalg.lstsq(
        selected_matrix,
        target,
        rcond=None,
    )
    if int(rank) != len(selected_indices):
        raise ValueError(
            f"rank-deficient model matrix: rank={int(rank)}, "
            f"features={len(selected_indices)}"
        )

    coefficients = np.zeros(feature_matrix.shape[1], dtype=float)
    for position, feature_idx in enumerate(selected_indices):
        if feature_idx == 0:
            continue
        coefficients[feature_idx] = (
            standardized_coefficients[position] / scales[feature_idx]
        )
    intercept_position = selected_indices.index(0)
    coefficients[0] = standardized_coefficients[intercept_position] - sum(
        coefficients[feature_idx] * means[feature_idx]
        for feature_idx in selected_indices
        if feature_idx != 0
    )

    smallest_singular = float(singular_values[-1])
    condition_number = (
        None
        if smallest_singular <= 0.0
        else float(singular_values[0] / smallest_singular)
    )
    return {
        "coefficients": [float(value) for value in coefficients],
        "selected_indices": selected_indices,
        "rank": int(rank),
        "condition_number": condition_number,
    }


def _gpu_upper_tail_features(row) -> list[float]:
    """Features for a stable affine GPU latency tail above the fitted range."""
    input_scale = float(row.input_scale)
    inverse_cpu_cores = 1.0 / float(row.cpu_cores)
    inverse_cpu_cores_squared = inverse_cpu_cores**2
    return [
        1.0,
        input_scale,
        inverse_cpu_cores,
        inverse_cpu_cores_squared,
        math.log(float(row.mem_cap_gb)),
        input_scale * inverse_cpu_cores,
        input_scale * inverse_cpu_cores_squared,
    ]


def _fit_gpu_upper_tail_model(rows: list) -> dict:
    feature_matrix = np.asarray(
        [_gpu_upper_tail_features(row) for row in rows],
        dtype=float,
    )
    target = np.asarray([float(row.latency_s) for row in rows], dtype=float)
    solution = _fit_rank_revealing_linear_model(feature_matrix, target)
    selected_feature_names = [
        LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES[feature_idx]
        for feature_idx in solution["selected_indices"]
    ]
    scale_slope_features = {
        "input_scale",
        "input_scale_x_inverse_cpu_cores",
        "input_scale_x_inverse_cpu_cores_squared",
    }
    if not scale_slope_features.intersection(selected_feature_names):
        raise ValueError("GPU upper-tail model has no input-scale slope")

    coefficients = solution["coefficients"]
    training_configurations = sorted({
        (float(row.cpu_cores), float(row.mem_cap_gb)) for row in rows
    })
    fitted_slopes = []
    for cpu_cores, mem_cap_gb in training_configurations:
        at_zero = SimpleNamespace(
            input_scale=0.0,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
        )
        at_one = SimpleNamespace(
            input_scale=1.0,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
        )
        fitted_slopes.append(float(np.dot(
            coefficients,
            np.asarray(_gpu_upper_tail_features(at_one))
            - np.asarray(_gpu_upper_tail_features(at_zero)),
        )))

    return {
        "feature_names": list(LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES),
        "selected_indices": solution["selected_indices"],
        "selected_feature_names": selected_feature_names,
        "dropped_feature_names": [
            feature_name
            for feature_name in LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES
            if feature_name not in selected_feature_names
        ],
        "coefficients": coefficients,
        "rank": solution["rank"],
        "condition_number": solution["condition_number"],
        "minimum_fitted_slope_s_per_scale": min(fitted_slopes),
    }


def _fit_log_latency_model(
    rows: list,
    hardware_model: str,
    *,
    calibrate_gpu_upper_tail: bool = True,
) -> dict:
    """Fit log(latency) with rank-revealing least squares.

    Regressors are standardized before solving. Constant or linearly dependent
    columns are removed so CPU-only, GPU-only, and reduced resource matrices do
    not fail merely because a feature is constant.
    """
    if len(rows) < 3:
        raise ValueError(f"need at least 3 case rows, got {len(rows)}")

    input_scale_spline_knots = (
        _gpu_input_scale_spline_knots(rows)
        if hardware_model == "gpu"
        else []
    )
    feature_names = _model_feature_names(
        hardware_model,
        input_scale_spline_knots,
    )
    feature_matrix = np.asarray(
        [
            _model_features(
                row,
                hardware_model,
                input_scale_spline_knots,
            )
            for row in rows
        ],
        dtype=float,
    )
    target = np.log(np.asarray([float(row.latency_s) for row in rows], dtype=float))
    solution = _fit_rank_revealing_linear_model(feature_matrix, target)
    selected_indices = solution["selected_indices"]
    model = {
        "hardware_model": hardware_model,
        "input_scale_spline_knots": input_scale_spline_knots,
        "input_scale_min": min(float(row.input_scale) for row in rows),
        "input_scale_max": max(float(row.input_scale) for row in rows),
        "feature_names": feature_names,
        "selected_indices": selected_indices,
        "selected_feature_names": [
            feature_names[feature_idx] for feature_idx in selected_indices
        ],
        "dropped_feature_names": [
            feature_names[feature_idx]
            for feature_idx in range(len(feature_names))
            if feature_idx not in selected_indices
        ],
        "coefficients": solution["coefficients"],
        "rank": solution["rank"],
        "condition_number": solution["condition_number"],
        "fit_case_rows": len(rows),
    }

    if hardware_model == "gpu" and calibrate_gpu_upper_tail:
        input_scales = sorted({float(row.input_scale) for row in rows})
        input_scale_span_ratio = input_scales[-1] / input_scales[0]
        calibration = {
            "available": False,
            "held_out_input_scale": None,
            "mean_absolute_percentage_error": None,
            "activation_threshold_mape": (
                LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
            ),
        }
        if len(input_scales) >= 3:
            calibration_scale = input_scales[-1]
            calibration_rows = [
                row for row in rows
                if float(row.input_scale) < calibration_scale
            ]
            calibration_test_rows = [
                row for row in rows
                if float(row.input_scale) == calibration_scale
            ]
            try:
                calibration_model = _fit_log_latency_model(
                    calibration_rows,
                    hardware_model,
                    calibrate_gpu_upper_tail=False,
                )
                calibration_predictions = np.asarray([
                    _predict_latency(row, calibration_model)
                    for row in calibration_test_rows
                ])
                calibration_actuals = np.asarray([
                    float(row.latency_s) for row in calibration_test_rows
                ])
                calibration_mape = float(np.mean(
                    np.abs(calibration_actuals - calibration_predictions)
                    / calibration_actuals
                ))
                calibration = {
                    "available": True,
                    "held_out_input_scale": calibration_scale,
                    "train_input_scale_max": max(
                        float(row.input_scale) for row in calibration_rows
                    ),
                    "mean_absolute_percentage_error": calibration_mape,
                    "activation_threshold_mape": (
                        LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
                    ),
                }
            except ValueError as exc:
                calibration["reason"] = str(exc)

        try:
            upper_tail_model = _fit_gpu_upper_tail_model(rows)
            upper_tail_model["calibration"] = calibration
            upper_tail_model["training_input_scale_span_ratio"] = (
                input_scale_span_ratio
            )
            upper_tail_model["enabled"] = bool(
                calibration["available"]
                and calibration["mean_absolute_percentage_error"]
                > LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
                and input_scale_span_ratio
                >= LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO
                and upper_tail_model["minimum_fitted_slope_s_per_scale"] > 0.0
            )
            model["gpu_upper_tail"] = upper_tail_model
        except ValueError as exc:
            model["gpu_upper_tail"] = {
                "enabled": False,
                "reason": str(exc),
                "calibration": calibration,
            }

    return model


def _predict_primary_latency(row, model: dict) -> float:
    """Evaluate the positive log-link model without tail extrapolation."""
    linear_prediction = sum(
        coefficient * feature
        for coefficient, feature in zip(
            model["coefficients"],
            _model_features(
                row,
                model["hardware_model"],
                model.get("input_scale_spline_knots", []),
            ),
        )
    )
    # The exponential link makes predictions positive. Clipping only protects
    # artifact generation from floating-point overflow on extreme extrapolation.
    bounded_prediction = min(max(linear_prediction, -700.0), 700.0)
    return float(math.exp(bounded_prediction))


def _predict_latency(row, model: dict) -> float:
    primary_prediction = _predict_primary_latency(row, model)
    upper_tail_model = model.get("gpu_upper_tail", {})
    input_scale_max = model.get("input_scale_max")
    if not (
        model.get("hardware_model") == "gpu"
        and upper_tail_model.get("enabled")
        and input_scale_max is not None
        and float(row.input_scale) > float(input_scale_max)
    ):
        return primary_prediction

    boundary_row = SimpleNamespace(
        input_scale=float(input_scale_max),
        cpu_cores=float(row.cpu_cores),
        mem_cap_gb=float(row.mem_cap_gb),
    )
    boundary_prediction = _predict_primary_latency(boundary_row, model)
    feature_delta = (
        np.asarray(_gpu_upper_tail_features(row), dtype=float)
        - np.asarray(_gpu_upper_tail_features(boundary_row), dtype=float)
    )
    tail_delta = float(np.dot(
        np.asarray(upper_tail_model["coefficients"], dtype=float),
        feature_delta,
    ))
    tail_prediction = boundary_prediction + tail_delta
    if math.isfinite(tail_prediction) and tail_prediction > 0.0:
        return tail_prediction
    return primary_prediction


def _regression_metrics(
    actuals: list[float],
    predictions: list[float],
) -> dict[str, float | int | None]:
    prediction_count = len(predictions)
    nonfinite_prediction_count = sum(
        not math.isfinite(float(prediction)) for prediction in predictions
    )
    nonpositive_prediction_count = sum(
        math.isfinite(float(prediction)) and float(prediction) <= 0.0
        for prediction in predictions
    )
    valid_pairs = [
        (float(actual), float(prediction))
        for actual, prediction in zip(actuals, predictions)
        if math.isfinite(float(actual)) and math.isfinite(float(prediction))
    ]
    if not valid_pairs:
        return {
            "r2": None,
            "mae": None,
            "rmse": None,
            "relative_mae": None,
            "mean_absolute_percentage_error": None,
            "maximum_absolute_percentage_error": None,
            "smape": None,
            "p95_absolute_error": None,
            "mean_actual": None,
            "prediction_count": prediction_count,
            "nonfinite_prediction_count": nonfinite_prediction_count,
            "nonpositive_prediction_count": nonpositive_prediction_count,
            "min_prediction": None,
            "max_prediction": None,
        }

    valid_actuals = [pair[0] for pair in valid_pairs]
    valid_predictions = [pair[1] for pair in valid_pairs]
    errors = [
        actual - predicted
        for actual, predicted in zip(valid_actuals, valid_predictions)
    ]
    absolute_errors = [abs(error) for error in errors]
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_actual = sum(valid_actuals) / len(valid_actuals)
    ss_tot = sum((actual - mean_actual) ** 2 for actual in valid_actuals)
    ss_res = sum(error * error for error in errors)
    r2 = None if ss_tot <= 1e-24 else 1.0 - (ss_res / ss_tot)
    relative_mae = None if mean_actual <= 0.0 else mae / mean_actual
    absolute_percentage_errors = [
        abs(actual - predicted) / abs(actual)
        for actual, predicted in zip(valid_actuals, valid_predictions)
        if abs(actual) > 0.0
    ]
    mean_absolute_percentage_error = (
        None
        if not absolute_percentage_errors
        else sum(absolute_percentage_errors) / len(absolute_percentage_errors)
    )
    maximum_absolute_percentage_error = (
        None
        if not absolute_percentage_errors
        else max(absolute_percentage_errors)
    )
    smape_terms = [
        2.0 * abs(actual - predicted) / (abs(actual) + abs(predicted))
        for actual, predicted in zip(valid_actuals, valid_predictions)
        if abs(actual) + abs(predicted) > 0.0
    ]
    smape = None if not smape_terms else sum(smape_terms) / len(smape_terms)
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "relative_mae": relative_mae,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "maximum_absolute_percentage_error": maximum_absolute_percentage_error,
        "smape": smape,
        "p95_absolute_error": float(np.percentile(absolute_errors, 95)),
        "mean_actual": mean_actual,
        "prediction_count": prediction_count,
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "nonpositive_prediction_count": nonpositive_prediction_count,
        "min_prediction": min(valid_predictions),
        "max_prediction": max(valid_predictions),
    }


def _latency_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "latency_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required modeling columns: {sorted(missing)}")

    model_df = df[list(required)].copy()
    for col in ("cpu_cores", "mem_cap_gb", "input_scale", "latency_s"):
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df[
        model_df["cpu_cores"].notna()
        & model_df["mem_cap_gb"].notna()
        & model_df["input_scale"].notna()
        & model_df["latency_s"].notna()
        & (model_df["cpu_cores"] > 0)
        & (model_df["mem_cap_gb"] > 0)
        & (model_df["input_scale"] > 0)
        & (model_df["latency_s"] > 0)
    ].copy()
    model_df["source_row"] = model_df.index
    model_df["hardware_model"] = model_df["gpu_mode"].map(_hardware_model_name)
    return model_df.sort_values(
        ["hardware_model", "cpu_cores", "mem_cap_gb", "input_scale", "source_row"]
    ).reset_index(drop=True)


def _aggregate_latency_model_points(model_df: pd.DataFrame) -> list:
    """Collapse repeated windows before any fit or validation split."""
    point_df = model_df.groupby(
        ["hardware_model", "cpu_cores", "mem_cap_gb", "input_scale"],
        as_index=False,
        sort=True,
    ).agg(
        latency_s=("latency_s", "median"),
        latency_mean_s=("latency_s", "mean"),
        latency_std_s=("latency_s", lambda values: values.std(ddof=0)),
        repeat_count=("latency_s", "size"),
    )
    point_df["latency_std_s"] = point_df["latency_std_s"].fillna(0.0)
    return list(point_df.itertuples(index=False))


def _point_key(row) -> tuple[str, float, float, float]:
    return (
        str(row.hardware_model),
        float(row.cpu_cores),
        float(row.mem_cap_gb),
        float(row.input_scale),
    )


def _resource_configuration_validation(
    rows: list,
    hardware_model: str,
) -> tuple[dict, dict[tuple[str, float, float, float], float]]:
    """Leave one complete (CPU, memory) configuration out per fold."""
    configurations = sorted({
        (float(row.cpu_cores), float(row.mem_cap_gb)) for row in rows
    })
    if len(configurations) < 3:
        return {
            "available": False,
            "reason": (
                "need at least 3 resource configurations so each held-out fold "
                "retains at least 2 training configurations"
            ),
            "folds": 0,
            "completed_folds": 0,
            "failed_folds": [],
            "r2_quality_gate_applicable": True,
            "train_test_group_overlap_count": 0,
            "metrics": _regression_metrics([], []),
        }, {}

    predictions: dict[tuple[str, float, float, float], float] = {}
    failed_folds = []
    fold_details = []
    train_test_group_overlap_count = 0
    completed_folds = 0
    for cpu_cores, mem_cap_gb in configurations:
        test_rows = [
            row
            for row in rows
            if (
                float(row.cpu_cores) == cpu_cores
                and float(row.mem_cap_gb) == mem_cap_gb
            )
        ]
        train_rows = [
            row
            for row in rows
            if not (
                float(row.cpu_cores) == cpu_cores
                and float(row.mem_cap_gb) == mem_cap_gb
            )
        ]
        train_groups = {
            (float(row.cpu_cores), float(row.mem_cap_gb)) for row in train_rows
        }
        test_groups = {
            (float(row.cpu_cores), float(row.mem_cap_gb)) for row in test_rows
        }
        fold_overlap_count = len(train_groups & test_groups)
        train_test_group_overlap_count += fold_overlap_count
        fold_detail = {
            "held_out_cpu_cores": cpu_cores,
            "held_out_mem_cap_gb": mem_cap_gb,
            "train_case_rows": len(train_rows),
            "test_case_rows": len(test_rows),
            "train_test_group_overlap_count": fold_overlap_count,
        }
        try:
            fold_model = _fit_log_latency_model(train_rows, hardware_model)
            if not (
                LATENCY_MODEL_RESOURCE_FEATURES[hardware_model]
                & set(fold_model["selected_feature_names"])
            ):
                raise ValueError(
                    "held-out fold has no identifiable CPU or memory predictor"
                )
        except ValueError as exc:
            fold_detail["status"] = "failed"
            fold_detail["reason"] = str(exc)
            fold_details.append(fold_detail)
            failed_folds.append({
                "cpu_cores": cpu_cores,
                "mem_cap_gb": mem_cap_gb,
                "reason": str(exc),
            })
            continue

        fold_predictions = [
            _predict_latency(row, fold_model) for row in test_rows
        ]
        for row, prediction in zip(test_rows, fold_predictions):
            predictions[_point_key(row)] = prediction
        fold_detail["status"] = "ok"
        fold_detail["metrics"] = _regression_metrics(
            [float(row.latency_s) for row in test_rows],
            fold_predictions,
        )
        fold_details.append(fold_detail)
        completed_folds += 1

    predicted_rows = [row for row in rows if _point_key(row) in predictions]
    actuals = [float(row.latency_s) for row in predicted_rows]
    predicted = [predictions[_point_key(row)] for row in predicted_rows]
    case_relative_errors = [
        abs(actual - prediction) / actual
        for actual, prediction in zip(actuals, predicted)
    ]
    worst_case_idx = (
        None
        if not predicted_rows
        else max(
            range(len(predicted_rows)),
            key=lambda idx: case_relative_errors[idx],
        )
    )
    worst_case = (
        None if worst_case_idx is None else predicted_rows[worst_case_idx]
    )
    completed_fold_details = [
        fold_detail
        for fold_detail in fold_details
        if fold_detail.get("status") == "ok"
    ]
    worst_relative_mae_fold = (
        None
        if not completed_fold_details
        else max(
            completed_fold_details,
            key=lambda fold_detail: (
                math.inf
                if fold_detail["metrics"].get("relative_mae") is None
                else float(fold_detail["metrics"]["relative_mae"])
            ),
        )
    )
    return {
        "available": not failed_folds and len(predicted_rows) == len(rows),
        "method": "leave-one-(cpu_cores,mem_cap_gb)-configuration-out",
        "r2_quality_gate_applicable": True,
        "group_columns": ["cpu_cores", "mem_cap_gb"],
        "all_input_scales_move_with_held_out_configuration": True,
        "folds": len(configurations),
        "completed_folds": completed_folds,
        "failed_folds": failed_folds,
        "fold_details": fold_details,
        "train_test_group_overlap_count": train_test_group_overlap_count,
        "test_case_rows": len(predicted_rows),
        "worst_fold_relative_mae": (
            None
            if worst_relative_mae_fold is None
            else worst_relative_mae_fold["metrics"]["relative_mae"]
        ),
        "worst_fold_configuration": (
            None
            if worst_relative_mae_fold is None
            else {
                "cpu_cores": worst_relative_mae_fold["held_out_cpu_cores"],
                "mem_cap_gb": worst_relative_mae_fold["held_out_mem_cap_gb"],
            }
        ),
        "worst_case_relative_error": (
            None
            if worst_case_idx is None
            else case_relative_errors[worst_case_idx]
        ),
        "worst_case_configuration": (
            None
            if worst_case is None
            else {
                "cpu_cores": float(worst_case.cpu_cores),
                "mem_cap_gb": float(worst_case.mem_cap_gb),
                "input_scale": float(worst_case.input_scale),
            }
        ),
        "metrics": _regression_metrics(actuals, predicted),
    }, predictions


def _input_scale_validation(
    rows: list,
    hardware_model: str,
) -> tuple[dict, dict[tuple[str, float, float, float], float]]:
    """Train on smaller scales and hold out the largest scale for extrapolation."""
    input_scales = sorted({float(row.input_scale) for row in rows})
    if len(input_scales) < 3:
        return {
            "available": False,
            "reason": (
                "need at least 3 input scales so the maximum can be held out "
                "while at least 2 distinct scales remain for training"
            ),
            "held_out_input_scale": None,
            "group_columns": ["input_scale"],
            "r2_quality_gate_applicable": False,
            "r2_quality_gate_note": (
                "A single held-out scale tests level extrapolation; R-squared "
                "within that scale only measures resource-configuration "
                "variation and is not used as an extrapolation gate."
            ),
            "train_test_group_overlap_count": 0,
            "metrics": _regression_metrics([], []),
        }, {}

    held_out_scale = input_scales[-1]
    train_rows = [row for row in rows if float(row.input_scale) < held_out_scale]
    test_rows = [row for row in rows if float(row.input_scale) == held_out_scale]
    train_scale_groups = {float(row.input_scale) for row in train_rows}
    test_scale_groups = {float(row.input_scale) for row in test_rows}
    train_test_group_overlap_count = len(train_scale_groups & test_scale_groups)
    try:
        scale_model = _fit_log_latency_model(train_rows, hardware_model)
        if "log_input_scale" not in scale_model["selected_feature_names"]:
            raise ValueError(
                "maximum-scale training fold has no identifiable input-scale predictor"
            )
    except ValueError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "held_out_input_scale": held_out_scale,
            "group_columns": ["input_scale"],
            "r2_quality_gate_applicable": False,
            "r2_quality_gate_note": (
                "A single held-out scale tests level extrapolation; R-squared "
                "within that scale only measures resource-configuration "
                "variation and is not used as an extrapolation gate."
            ),
            "train_input_scale_max": max(
                (float(row.input_scale) for row in train_rows),
                default=None,
            ),
            "train_test_group_overlap_count": train_test_group_overlap_count,
            "metrics": _regression_metrics([], []),
        }, {}

    predictions = {
        _point_key(row): _predict_latency(row, scale_model) for row in test_rows
    }
    actuals = [float(row.latency_s) for row in test_rows]
    predicted = [predictions[_point_key(row)] for row in test_rows]
    case_relative_errors = [
        abs(actual - prediction) / actual
        for actual, prediction in zip(actuals, predicted)
    ]
    worst_case_idx = max(
        range(len(test_rows)),
        key=lambda idx: case_relative_errors[idx],
    )
    worst_case = test_rows[worst_case_idx]
    return {
        "available": True,
        "method": "forward holdout of maximum input_scale",
        "group_columns": ["input_scale"],
        "r2_quality_gate_applicable": False,
        "r2_quality_gate_note": (
            "A single held-out scale tests level extrapolation; R-squared "
            "within that scale only measures resource-configuration variation "
            "and is reported for context, not used as an extrapolation gate."
        ),
        "held_out_input_scale": held_out_scale,
        "train_input_scale_max": max(float(row.input_scale) for row in train_rows),
        "test_input_scale_min": min(float(row.input_scale) for row in test_rows),
        "strict_extrapolation": (
            max(float(row.input_scale) for row in train_rows)
            < min(float(row.input_scale) for row in test_rows)
        ),
        "train_test_group_overlap_count": train_test_group_overlap_count,
        "train_case_rows": len(train_rows),
        "test_case_rows": len(test_rows),
        "worst_case_relative_error": case_relative_errors[worst_case_idx],
        "worst_case_configuration": {
            "cpu_cores": float(worst_case.cpu_cores),
            "mem_cap_gb": float(worst_case.mem_cap_gb),
            "input_scale": float(worst_case.input_scale),
        },
        "metrics": _regression_metrics(actuals, predicted),
    }, predictions


def _validation_quality_failures(validation_name: str, validation: dict) -> list[str]:
    if not validation.get("available"):
        return [
            f"{validation_name} unavailable: "
            f"{validation.get('reason', 'incomplete validation folds')}"
        ]

    metrics = validation["metrics"]
    failures = []
    if int(metrics.get("prediction_count") or 0) < LATENCY_MODEL_MIN_VALIDATION_POINTS:
        failures.append(
            f"{validation_name} has fewer than "
            f"{LATENCY_MODEL_MIN_VALIDATION_POINTS} predictions"
        )
    if int(validation.get("train_test_group_overlap_count") or 0) > 0:
        failures.append(f"{validation_name} has train/test group leakage")
    failed_quality_folds = []
    for fold_detail in validation.get("fold_details", []):
        if fold_detail.get("status") != "ok":
            continue
        fold_metrics = fold_detail["metrics"]
        fold_relative_mae = fold_metrics.get("relative_mae")
        fold_mape = fold_metrics.get("mean_absolute_percentage_error")
        if (
            fold_relative_mae is None
            or not math.isfinite(float(fold_relative_mae))
            or float(fold_relative_mae)
            > LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE
            or fold_mape is None
            or not math.isfinite(float(fold_mape))
            or float(fold_mape) > LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE
            or int(fold_metrics.get("nonfinite_prediction_count") or 0) > 0
            or int(fold_metrics.get("nonpositive_prediction_count") or 0) > 0
        ):
            failed_quality_folds.append(fold_detail)
    if failed_quality_folds:
        worst_fold = max(
            failed_quality_folds,
            key=lambda fold_detail: max(
                math.inf
                if (
                    fold_detail["metrics"].get("relative_mae") is None
                    or not math.isfinite(
                        float(fold_detail["metrics"]["relative_mae"])
                    )
                )
                else (
                    float(fold_detail["metrics"]["relative_mae"])
                    / LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE
                ),
                math.inf
                if (
                    fold_detail["metrics"].get(
                        "mean_absolute_percentage_error"
                    )
                    is None
                    or not math.isfinite(
                        float(
                            fold_detail["metrics"][
                                "mean_absolute_percentage_error"
                            ]
                        )
                    )
                )
                else (
                    float(
                        fold_detail["metrics"][
                            "mean_absolute_percentage_error"
                        ]
                    )
                    / LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE
                ),
            ),
        )
        failures.append(
            f"{validation_name} has {len(failed_quality_folds)} held-out "
            "configuration fold(s) above the relative-MAE/MAPE/validity "
            "threshold; "
            f"worst=(cpu_cores={worst_fold['held_out_cpu_cores']},"
            f"mem_cap_gb={worst_fold['held_out_mem_cap_gb']}), "
            f"relative_mae={worst_fold['metrics'].get('relative_mae')!r}, "
            "mean_absolute_percentage_error="
            f"{worst_fold['metrics'].get('mean_absolute_percentage_error')!r}"
        )
    worst_case_relative_error = validation.get("worst_case_relative_error")
    if (
        worst_case_relative_error is not None
        and (
            not math.isfinite(float(worst_case_relative_error))
            or float(worst_case_relative_error)
            > LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR
        )
    ):
        failures.append(
            f"{validation_name} worst case relative_error="
            f"{worst_case_relative_error!r} exceeds "
            f"{LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR}; "
            f"configuration={validation.get('worst_case_configuration')!r}"
        )
    if validation.get("r2_quality_gate_applicable", True):
        r2 = metrics.get("r2")
        if (
            r2 is None
            or not math.isfinite(float(r2))
            or float(r2) < LATENCY_MODEL_MIN_VALIDATION_R2
        ):
            failures.append(
                f"{validation_name} r2={r2!r} is below "
                f"{LATENCY_MODEL_MIN_VALIDATION_R2}"
            )
    relative_mae = metrics.get("relative_mae")
    if (
        relative_mae is None
        or not math.isfinite(float(relative_mae))
        or float(relative_mae) > LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE
    ):
        failures.append(
            f"{validation_name} relative_mae={relative_mae!r} exceeds "
            f"{LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE}"
        )
    mean_absolute_percentage_error = metrics.get(
        "mean_absolute_percentage_error"
    )
    if (
        mean_absolute_percentage_error is None
        or not math.isfinite(float(mean_absolute_percentage_error))
        or float(mean_absolute_percentage_error) > LATENCY_MODEL_MAX_VALIDATION_MAPE
    ):
        failures.append(
            f"{validation_name} mean_absolute_percentage_error="
            f"{mean_absolute_percentage_error!r} exceeds "
            f"{LATENCY_MODEL_MAX_VALIDATION_MAPE}"
        )
    if int(metrics.get("nonfinite_prediction_count") or 0) > 0:
        failures.append(f"{validation_name} contains non-finite predictions")
    if int(metrics.get("nonpositive_prediction_count") or 0) > 0:
        failures.append(f"{validation_name} contains non-positive predictions")
    return failures


def _validation_is_evaluable(validation: dict) -> bool:
    if not validation.get("available"):
        return False
    metrics = validation.get("metrics", {})
    metric_names = ["relative_mae", "mean_absolute_percentage_error"]
    if validation.get("r2_quality_gate_applicable", True):
        metric_names.append("r2")
    return bool(
        int(metrics.get("prediction_count") or 0)
        >= LATENCY_MODEL_MIN_VALIDATION_POINTS
        and all(
            metrics.get(metric_name) is not None
            and math.isfinite(float(metrics[metric_name]))
            for metric_name in metric_names
        )
    )


def _training_range(rows: list) -> dict[str, dict[str, float]]:
    return {
        "cpu_cores": {
            "min": min(float(row.cpu_cores) for row in rows),
            "max": max(float(row.cpu_cores) for row in rows),
        },
        "mem_cap_gb": {
            "min": min(float(row.mem_cap_gb) for row in rows),
            "max": max(float(row.mem_cap_gb) for row in rows),
        },
        "input_scale": {
            "min": min(float(row.input_scale) for row in rows),
            "max": max(float(row.input_scale) for row in rows),
        },
    }

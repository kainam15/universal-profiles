"""结果 CSV 整理、分组和基础聚合。"""

import math
import os

import pandas as pd

from acprof.metric_registry import NUMERIC_FIELDS

from acprof.pixel_metrics import (
    PIXEL_COUNT_FIELDS,
    PIXEL_RATE_SOURCES,
    pixel_rate_metrics,
)

from acprof.analysis.latency_model import (
    GPU_MODE_OFF_VALUES,
    GPU_MODE_ON_VALUES,
)

from acprof.plotting.config import BYTES_PER_GIB, COMPUTE_NUMERIC_COLUMNS, PLOT_OUTPUT_DIRS


def make_config_label(row) -> str:
    cpu = int(row["cpu_cores"])
    mem = int(row["mem_cap_gb"])
    if is_gpu_on(row["gpu_mode"]):
        return f"GPU+CPU{cpu}+Mem{mem}"
    else:
        return f"CPU+CPU{cpu}+Mem{mem}"


def is_gpu_on(gpu_mode: object) -> bool:
    return str(gpu_mode).strip().lower() in GPU_MODE_ON_VALUES


def is_gpu_off(gpu_mode: object) -> bool:
    return str(gpu_mode).strip().lower() in GPU_MODE_OFF_VALUES


def build_plot_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split plot inputs into CPU-only, GPU-only, and comparison groups."""
    gpu_on = df["gpu_mode"].map(is_gpu_on)
    gpu_off = df["gpu_mode"].map(is_gpu_off)
    cpu_df = df[gpu_off].copy()
    gpu_df = df[gpu_on].copy()
    combined_df = (
        df[gpu_off | gpu_on].copy()
        if not cpu_df.empty and not gpu_df.empty
        else df.iloc[0:0].copy()
    )
    return list(zip(PLOT_OUTPUT_DIRS, (cpu_df, gpu_df, combined_df)))


def _with_pixel_metrics(df: pd.DataFrame, static_meta: dict) -> pd.DataFrame:
    """只使用 CSV 的显式像素数；不从历史计划补写缺失的测量字段。"""
    derived_rows = []
    for row in df.to_dict("records"):
        counts = {}
        for field in PIXEL_COUNT_FIELDS:
            try:
                number = float(row.get(field, "nan"))
            except (TypeError, ValueError, OverflowError):
                number = float("nan")
            counts[field] = number if math.isfinite(number) and number > 0 and number.is_integer() else float("nan")
        derived_rows.append({**counts, **pixel_rate_metrics({**row, **counts})})

    derived = pd.DataFrame(derived_rows, index=df.index,
                           columns=[*PIXEL_COUNT_FIELDS, *PIXEL_RATE_SOURCES])
    result = df.drop(columns=derived.columns, errors="ignore").join(derived)
    family = static_meta.get("task_family", "")
    result.attrs["pixel_scale_kind"] = {"diffusion": "output", "cv": "input", "multimodal": "input"}.get(family, "")
    result.attrs["input_scale_type"] = str(static_meta.get("input_scale_type") or "")
    return result


def normalized_metric_spec(
    df: pd.DataFrame, metric: str, title: str, ylabel: str, xlabel: str,
) -> tuple[str, str, str]:
    """分辨率图使用像素指标；缺少几何信息时显示 No data，不退回边长分母。"""
    sources = {
        "container_attributed_j_per_input_unit": ("container_attributed_j", "Estimated energy (J/Mpixel)"),
        "latency_s_per_input_unit": ("latency_s", "Latency (s/Mpixel)"),
        "latency_app_s_per_input_unit": ("latency_app_s", "Latency (s/Mpixel)"),
    }
    scale_type = df.attrs.get("input_scale_type") or xlabel
    if metric not in sources or scale_type not in {"resolution_px", "resolution_scale"}:
        return metric, title, ylabel
    direction = df.attrs.get("pixel_scale_kind")
    if direction not in {"input", "output"}:
        available = [
            kind for kind in ("input", "output")
            if f"{kind}_pixels_per_request" in df
            and pd.to_numeric(df[f"{kind}_pixels_per_request"], errors="coerce").gt(0).any()
        ]
        direction = available[0] if len(available) == 1 else "unknown"
    prefix, pixel_ylabel = sources[metric]
    unit_title = f"{direction.title()} Megapixel" if direction != "unknown" else "Megapixel (geometry unavailable)"
    return (
        f"{prefix}_per_{direction}_megapixel",
        title.replace("Input Unit", unit_title),
        pixel_ylabel if ylabel else "",
    )


def prepare_df(
    csv_path: str,
    *,
    only_ok: bool = True,
    exclude_warmup: bool = True,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [str(col).strip() for col in df.columns]
    from acprof.result_csv import require_current_fields
    require_current_fields(df.columns)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    static_meta = read_static_meta(csv_path)
    if "gpu_mem_total_bytes" not in df.columns and static_meta.get("gpu_mem_total_bytes"):
        df["gpu_mem_total_bytes"] = pd.to_numeric(
            static_meta["gpu_mem_total_bytes"],
            errors="coerce",
        )

    num_cols = [*NUMERIC_FIELDS, "gpu_mem_total_bytes", *COMPUTE_NUMERIC_COLUMNS]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bytes_to_gib = {
        "container_mem_usage_avg_bytes": "container_mem_usage_avg_gib",
        "container_mem_usage_peak_bytes": "container_mem_usage_peak_gib",
        "container_mem_peak_cgroup_bytes": "container_mem_peak_cgroup_gib",
        "container_mem_anon_bytes_end": "container_mem_anon_end_gib",
        "container_mem_file_bytes_end": "container_mem_file_end_gib",
        "container_mem_slab_bytes_end": "container_mem_slab_end_gib",
        "container_swap_limit_bytes": "container_swap_limit_gib",
        "container_swap_usage_avg_bytes": "container_swap_usage_avg_gib",
        "container_swap_usage_peak_bytes": "container_swap_usage_peak_gib",
        "gpu_mem_used_avg_bytes": "gpu_mem_used_avg_gib",
        "gpu_mem_used_peak_bytes": "gpu_mem_used_peak_gib",
        "gpu_mem_total_bytes": "gpu_mem_total_gib",
        "cpu_heap_peak_total_bytes_massif": "cpu_heap_peak_total_gib_massif",
    }
    for source_col, target_col in bytes_to_gib.items():
        if source_col in df.columns:
            df[target_col] = df[source_col] / float(BYTES_PER_GIB)

    df = _with_pixel_metrics(df, static_meta)

    if only_ok and "status" in df.columns:
        normalized_status = df["status"].astype(str).str.strip().str.lower()
        df = df[normalized_status == "ok"].copy()

    if exclude_warmup and "warmup" in df.columns:
        df = df[df["warmup"].fillna(0).astype(int) == 0].copy()

    need = {"input_scale", "cpu_cores", "mem_cap_gb", "gpu_mode"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if df.empty:
        df["config"] = pd.Series(index=df.index, dtype="object")
    else:
        df["config"] = df.apply(make_config_label, axis=1)
    df = df[df["input_scale"].notna() & (df["input_scale"] > 0)].copy()

    return df


def read_static_meta(csv_path: str) -> dict[str, object]:
    from acprof.artifacts import read_static_metadata
    return read_static_metadata(os.path.dirname(csv_path) or ".")


def aggregate_metric(df: pd.DataFrame, metric: str, *, agg_func: str = "mean") -> pd.DataFrame:
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]
    metric_df = df[group_cols + [metric]].copy()
    metric_df = metric_df[metric_df[metric].notna()].copy()
    if metric_df.empty:
        return metric_df

    if agg_func == "median":
        agg_df = metric_df.groupby(group_cols, as_index=False)[metric].median()
    else:
        agg_df = metric_df.groupby(group_cols, as_index=False)[metric].mean()

    agg_df["config"] = agg_df.apply(make_config_label, axis=1)
    agg_df["gpu_on"] = agg_df["gpu_mode"].map(is_gpu_on)
    return agg_df


def aggregate_cold_start(df: pd.DataFrame, *, agg_func: str = "mean") -> pd.DataFrame:
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode"]
    metric_df = df[group_cols + ["cold_start_s"]].copy()
    metric_df = metric_df[metric_df["cold_start_s"].notna()].copy()
    if metric_df.empty:
        return metric_df

    if agg_func == "median":
        agg_df = metric_df.groupby(group_cols, as_index=False)["cold_start_s"].median()
    else:
        agg_df = metric_df.groupby(group_cols, as_index=False)["cold_start_s"].mean()

    agg_df["config"] = agg_df.apply(make_config_label, axis=1)
    agg_df["gpu_on"] = agg_df["gpu_mode"].map(is_gpu_on)
    return agg_df

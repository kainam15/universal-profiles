"""像素计数与每百万像素指标的纯计算；不采样、不加载模型或素材。"""
from __future__ import annotations

import math
from typing import Any, Mapping


PIXEL_COUNT_FIELDS = ("input_pixels_per_request", "output_pixels_per_request")
PIXEL_RATE_SOURCES = {
    f"{prefix}_per_{direction}_megapixel": (source, f"{direction}_pixels_per_request")
    for direction in ("input", "output")
    for prefix, source in (
        ("container_attributed_j", "container_attributed_energy_eff_j"),
        ("latency_s", "latency_s"),
        ("latency_app_s", "latency_app_s"),
    )
}


def _number(value: Any) -> float:
    try:
        return float(value) if not isinstance(value, bool) else float("nan")
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _positive_count(value: Any) -> float:
    number = _number(value)
    return number if math.isfinite(number) and number > 0 and number.is_integer() else float("nan")


def pixel_counts_from_metadata(
    metadata: Mapping[str, Any] | None,
    batch_size: Any,
    *,
    task_family: str,
    pipeline_tag: str = "",
    workload: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """按计划中的真实宽高/像素数计数；视频包括全部帧，batch 仅乘一次。

    CV/多模态统计 processor 处理前的输入像素，Diffusion 统计输出像素。
    没有尺寸的 token、秒数或去噪步数不能作为像素面积使用。
    """
    counts = dict.fromkeys(PIXEL_COUNT_FIELDS, float("nan"))
    batch = _positive_count(batch_size)
    if not math.isfinite(batch) or not isinstance(metadata, Mapping):
        return counts

    pixels = float("nan")
    if task_family == "diffusion":
        direction = "output"
        if "output_pixel_count_per_video" in metadata:
            pixels = _positive_count(metadata["output_pixel_count_per_video"])
        elif "output_pixel_count_per_frame" in metadata or "output_num_frames" in metadata:
            pixels = _positive_count(metadata.get("output_pixel_count_per_frame")) * _positive_count(
                metadata.get("output_num_frames")
            )
        elif "output_pixel_count_per_image" in metadata:
            pixels = _positive_count(metadata["output_pixel_count_per_image"])
    elif task_family == "cv":
        direction = "input"
        frames = metadata.get("num_frames", float("nan") if pipeline_tag == "video-classification" else 1)
        pixels = (
            _positive_count(metadata.get("image_width"))
            * _positive_count(metadata.get("image_height"))
            * _positive_count(frames)
        )
    elif task_family == "multimodal":
        direction = "input"
        pixels = 0.0
        if "image_width" in metadata or "image_height" in metadata:
            pixels += _positive_count(metadata.get("image_width")) * _positive_count(metadata.get("image_height"))
        if "video_num_frames" in metadata:
            fixed_media = workload.get("fixed_media", {}) if isinstance(workload, Mapping) else {}
            fixed_side = fixed_media.get("image_resolution") if isinstance(fixed_media, Mapping) else None
            pixels += (
                _positive_count(metadata.get("video_frame_width", fixed_side))
                * _positive_count(metadata.get("video_frame_height", fixed_side))
                * _positive_count(metadata["video_num_frames"])
            )
    else:
        return counts

    counts[f"{direction}_pixels_per_request"] = _positive_count(pixels * batch)
    return counts


def per_megapixel(value: Any, pixels: Any) -> float:
    numerator = _number(value)
    denominator = _positive_count(pixels)
    if not math.isfinite(numerator) or numerator < 0 or not math.isfinite(denominator):
        return float("nan")
    rate = numerator / denominator * 1_000_000.0
    return rate if math.isfinite(rate) else float("nan")


def pixel_rate_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    """原始值均已按单请求平均，此处不再除以窗口内请求次数。"""
    return {
        field: per_megapixel(row.get(source), row.get(denominator))
        for field, (source, denominator) in PIXEL_RATE_SOURCES.items()
    }

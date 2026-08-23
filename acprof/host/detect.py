"""AC-Prof Universal Profiler - HuggingFace Model Task Auto-Detection.

Three-level fallback:
  Level 1: HF Hub API (pipeline_tag + library_name)
  Level 2: config.json / AutoConfig architecture inference
  Level 3: CLI manual override (always takes precedence)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from acprof.config import (
    ARCHITECTURE_TO_TASK,
    DEFAULT_BACKEND,
    LIBRARY_TO_BACKEND,
    PIPELINE_TAG_TO_FAMILY,
)


@dataclass
class TaskInfo:
    model_id: str
    pipeline_tag: str
    task_family: str
    runtime_backend: str
    library_name: str
    model_revision: str
    detection_method: str  # "hub_api" / "config_infer" / "manual"
    parameter_count: Optional[int] = None
    parameter_bytes: Optional[int] = None
    precision_dtype: Optional[str] = None
    parameter_dtype_counts: dict[str, int] = field(default_factory=dict)
    quantized: Optional[bool] = None
    quantization_method: Optional[str] = None
    quantization_config: dict[str, Any] = field(default_factory=dict)
    model_license: Optional[str] = None
    model_metadata_source: Optional[str] = None


_HUB_DTYPE_NAMES = {
    "BOOL": "BOOL",
    "F64": "FP64",
    "F32": "FP32",
    "F16": "FP16",
    "BF16": "BF16",
    "F8_E5M2": "FP8_E5M2",
    "F8_E4M3": "FP8_E4M3",
    "I64": "INT64",
    "I32": "INT32",
    "I16": "INT16",
    "I8": "INT8",
    "U64": "UINT64",
    "U32": "UINT32",
    "U16": "UINT16",
    "U8": "UINT8",
}
_QUANTIZED_PRECISIONS = {
    "FP8_E5M2",
    "FP8_E4M3",
    "INT32",
    "UINT32",
    "INT16",
    "UINT16",
    "INT8",
    "UINT8",
    "INT4",
    "UINT4",
}
_QUANTIZATION_TAGS = {
    "4-bit",
    "8-bit",
    "aqlm",
    "awq",
    "bitsandbytes",
    "bnb-4bit",
    "compressed-tensors",
    "eetq",
    "fp8",
    "gguf",
    "gptq",
    "hqq",
    "int4",
    "int8",
    "quanto",
    "spqr",
}
_DTYPE_BYTE_WIDTHS = {
    "BOOL": 1,
    "FP64": 8,
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "FP8_E5M2": 1,
    "FP8_E4M3": 1,
    "INT64": 8,
    "INT32": 4,
    "INT16": 2,
    "INT8": 1,
    "UINT64": 8,
    "UINT32": 4,
    "UINT16": 2,
    "UINT8": 1,
}


def _normalize_dtype_name(raw_dtype: object) -> str:
    value = str(raw_dtype or "").strip().upper().replace("TORCH.", "")
    aliases = {
        "FLOAT": "FP32",
        "FLOAT32": "FP32",
        "FLOAT16": "FP16",
        "HALF": "FP16",
        "BFLOAT16": "BF16",
        "FLOAT64": "FP64",
        "DOUBLE": "FP64",
        "INT64": "INT64",
        "LONG": "INT64",
        "INT32": "INT32",
        "INT16": "INT16",
        "INT8": "INT8",
        "UINT8": "UINT8",
    }
    return _HUB_DTYPE_NAMES.get(value, aliases.get(value, value))


def _mapping_value(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _json_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return dict(converted) if isinstance(converted, dict) else {}
    return {}


def _parameter_bytes_from_dtype_counts(
    dtype_counts: dict[str, int],
) -> Optional[int]:
    """Return logical tensor payload bytes, or ``None`` if not exact."""
    if not dtype_counts:
        return None

    total = 0
    for dtype, count in dtype_counts.items():
        byte_width = _DTYPE_BYTE_WIDTHS.get(dtype)
        if byte_width is None or count < 0:
            return None
        total += count * byte_width
    return total


def _hub_model_metadata(info: object) -> dict[str, Any]:
    """Extract reproducible model-card metadata from a Hub ModelInfo object."""
    safetensors = getattr(info, "safetensors", None)
    raw_dtype_counts = _mapping_value(safetensors, "parameters")
    dtype_counts: dict[str, int] = {}
    if isinstance(raw_dtype_counts, dict):
        for raw_dtype, raw_count in raw_dtype_counts.items():
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            dtype = _normalize_dtype_name(raw_dtype)
            if dtype:
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count

    parameter_count: Optional[int] = None
    raw_parameter_count = _mapping_value(safetensors, "total")
    try:
        if raw_parameter_count is not None:
            parameter_count = int(raw_parameter_count)
    except (TypeError, ValueError):
        pass

    config = _json_mapping(getattr(info, "config", None))
    quantization_config = _json_mapping(config.get("quantization_config"))
    tags = {
        str(tag).strip().lower()
        for tag in (getattr(info, "tags", None) or [])
        if str(tag).strip()
    }

    precision_dtype: Optional[str] = None
    if dtype_counts:
        # Small INT64/BOOL tensors are commonly non-weight buffers. The
        # dominant dtype is the most useful single precision label.
        precision_dtype = max(dtype_counts.items(), key=lambda item: item[1])[0]
    else:
        config_dtype = config.get("torch_dtype") or config.get("dtype")
        if config_dtype:
            precision_dtype = _normalize_dtype_name(config_dtype)

    quantization_method = (
        quantization_config.get("quant_method")
        or quantization_config.get("quantization_method")
        or quantization_config.get("quant_type")
    )
    matching_quantization_tags = sorted(tags & _QUANTIZATION_TAGS)
    if not quantization_method and matching_quantization_tags:
        quantization_method = matching_quantization_tags[0]

    if quantization_config or matching_quantization_tags:
        quantized: Optional[bool] = True
    elif precision_dtype in _QUANTIZED_PRECISIONS:
        quantized = True
    elif precision_dtype:
        quantized = False
    else:
        quantized = None

    card_data = getattr(info, "card_data", None)
    model_license = _mapping_value(card_data, "license")
    if not model_license:
        license_tags = sorted(
            tag.split(":", 1)[1]
            for tag in tags
            if tag.startswith("license:") and ":" in tag
        )
        model_license = license_tags[0] if license_tags else None

    return {
        "parameter_count": parameter_count,
        "parameter_bytes": _parameter_bytes_from_dtype_counts(dtype_counts),
        "precision_dtype": precision_dtype,
        "parameter_dtype_counts": dtype_counts,
        "quantized": quantized,
        "quantization_method": (
            str(quantization_method) if quantization_method else None
        ),
        "quantization_config": quantization_config,
        "model_license": str(model_license) if model_license else None,
        "model_metadata_source": "huggingface_hub",
    }


def _format_failure(exc: Exception) -> str:
    detail = str(exc).strip().replace("\n", " ")
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _record_failure(diagnostics: Optional[list[str]], stage: str, reason: str) -> None:
    if diagnostics is not None:
        diagnostics.append(f"{stage}: {reason}")


def _detect_from_hub(
    model_id: str,
    diagnostics: Optional[list[str]] = None,
) -> Optional[TaskInfo]:
    """Level 1: Query HuggingFace Hub API."""
    try:
        from huggingface_hub import model_info as hf_model_info

        info = hf_model_info(model_id)
    except Exception as exc:
        _record_failure(diagnostics, "hub_api", _format_failure(exc))
        return None

    pipeline_tag = getattr(info, "pipeline_tag", None)
    library_name = getattr(info, "library_name", None) or ""
    sha = getattr(info, "sha", None) or "main"

    # Determine task_family from pipeline_tag
    task_family = None
    if pipeline_tag:
        task_family = PIPELINE_TAG_TO_FAMILY.get(pipeline_tag)

    # Determine runtime_backend from library_name
    runtime_backend = LIBRARY_TO_BACKEND.get(library_name, DEFAULT_BACKEND)

    # Special handling: chronos models
    if library_name == "chronos" or (
        not pipeline_tag and "chronos" in model_id.lower()
    ):
        pipeline_tag = pipeline_tag or "time-series-forecasting"
        task_family = "timeseries"
        runtime_backend = "chronos"

    if not pipeline_tag:
        _record_failure(
            diagnostics,
            "hub_api",
            f"metadata returned no pipeline_tag (library_name={library_name or 'unknown'})",
        )
        return None

    if not task_family:
        _record_failure(
            diagnostics,
            "hub_api",
            f"unsupported pipeline_tag '{pipeline_tag}' (library_name={library_name or 'unknown'})",
        )
        return None

    return TaskInfo(
        model_id=model_id,
        pipeline_tag=pipeline_tag,
        task_family=task_family,
        runtime_backend=runtime_backend,
        library_name=library_name,
        model_revision=sha,
        detection_method="hub_api",
        **_hub_model_metadata(info),
    )


def _detect_from_config(
    model_id: str,
    diagnostics: Optional[list[str]] = None,
) -> Optional[TaskInfo]:
    """Level 2: Infer task from model architecture name."""
    architectures = []
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        architectures = config_data.get("architectures") or []
        if not architectures:
            _record_failure(diagnostics, "config_json", "config.json has no architectures field")
    except Exception as exc:
        _record_failure(diagnostics, "config_json", _format_failure(exc))

    if not architectures:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            architectures = getattr(config, "architectures", None) or []
        except Exception as exc:
            _record_failure(diagnostics, "AutoConfig", _format_failure(exc))
            return None

        if not architectures:
            _record_failure(diagnostics, "AutoConfig", "config has no architectures")
            return None

    pipeline_tag = None
    for arch in architectures:
        for suffix, task in ARCHITECTURE_TO_TASK.items():
            if arch.endswith(suffix):
                pipeline_tag = task
                break
        if pipeline_tag:
            break

    if not pipeline_tag:
        _record_failure(
            diagnostics,
            "architecture_infer",
            f"unsupported architectures: {', '.join(str(arch) for arch in architectures)}",
        )
        return None

    task_family = PIPELINE_TAG_TO_FAMILY.get(pipeline_tag)
    if not task_family:
        _record_failure(
            diagnostics,
            "architecture_infer",
            f"inferred unsupported pipeline_tag '{pipeline_tag}'",
        )
        return None

    return TaskInfo(
        model_id=model_id,
        pipeline_tag=pipeline_tag,
        task_family=task_family,
        runtime_backend=DEFAULT_BACKEND,
        library_name="transformers",
        model_revision="main",
        detection_method="config_infer",
    )


def detect_task(
    model_id: str,
    override_tag: Optional[str] = None,
    override_family: Optional[str] = None,
    override_backend: Optional[str] = None,
) -> TaskInfo:
    """Detect model task with three-level fallback.

    CLI overrides (Level 3) always take precedence over auto-detection.
    """
    # Start with auto-detection
    diagnostics: list[str] = []
    info = _detect_from_hub(model_id, diagnostics)
    if info is None:
        info = _detect_from_config(model_id, diagnostics)

    # If auto-detection failed entirely, require manual override
    if info is None:
        if not override_tag and not override_family:
            details = "\n".join(f"  - {reason}" for reason in diagnostics)
            if not details:
                details = "  - no diagnostic details were captured"
            print(
                f"[ERROR] Cannot auto-detect task for '{model_id}'.\n"
                f"  Auto-detection attempts:\n"
                f"{details}\n"
                f"  Please specify --task and/or --task-family manually.\n"
                f"  Example: --task text-generation --task-family nlp",
                file=sys.stderr,
            )
            sys.exit(1)
        info = TaskInfo(
            model_id=model_id,
            pipeline_tag=override_tag or "unknown",
            task_family=override_family or "nlp",
            runtime_backend=override_backend or DEFAULT_BACKEND,
            library_name="unknown",
            model_revision="main",
            detection_method="manual",
        )

    # Apply CLI overrides (Level 3 - highest priority)
    if override_tag:
        info.pipeline_tag = override_tag
        if override_tag in PIPELINE_TAG_TO_FAMILY:
            info.task_family = PIPELINE_TAG_TO_FAMILY[override_tag]
        info.detection_method = "manual"
    if override_family:
        info.task_family = override_family
        info.detection_method = "manual"
    if override_backend:
        info.runtime_backend = override_backend
        info.detection_method = "manual"

    return info

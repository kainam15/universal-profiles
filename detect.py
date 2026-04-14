"""AC-Prof Universal Profiler - HuggingFace Model Task Auto-Detection.

Three-level fallback:
  Level 1: HF Hub API (pipeline_tag + library_name)
  Level 2: AutoConfig architecture inference
  Level 3: CLI manual override (always takes precedence)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Optional

from config import (
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


def _detect_from_hub(model_id: str) -> Optional[TaskInfo]:
    """Level 1: Query HuggingFace Hub API."""
    try:
        from huggingface_hub import model_info as hf_model_info

        info = hf_model_info(model_id)
    except Exception:
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

    if not pipeline_tag or not task_family:
        return None

    return TaskInfo(
        model_id=model_id,
        pipeline_tag=pipeline_tag,
        task_family=task_family,
        runtime_backend=runtime_backend,
        library_name=library_name,
        model_revision=sha,
        detection_method="hub_api",
    )


def _detect_from_config(model_id: str) -> Optional[TaskInfo]:
    """Level 2: Infer task from model architecture name."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None

    architectures = getattr(config, "architectures", None) or []

    pipeline_tag = None
    for arch in architectures:
        for suffix, task in ARCHITECTURE_TO_TASK.items():
            if arch.endswith(suffix):
                pipeline_tag = task
                break
        if pipeline_tag:
            break

    if not pipeline_tag:
        return None

    task_family = PIPELINE_TAG_TO_FAMILY.get(pipeline_tag)
    if not task_family:
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
    info = _detect_from_hub(model_id)
    if info is None:
        info = _detect_from_config(model_id)

    # If auto-detection failed entirely, require manual override
    if info is None:
        if not override_tag and not override_family:
            print(
                f"[ERROR] Cannot auto-detect task for '{model_id}'.\n"
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

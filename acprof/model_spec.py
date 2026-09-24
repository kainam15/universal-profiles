"""Declarative model interfaces shared by discovery, image identity and loaders."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


SPEC_ENV = "ACPROF_MODEL_SPEC_B64"
MAX_SPEC_BYTES = 64 * 1024
FORMAT_BACKENDS = {"onnxruntime": "onnxruntime", "transformers-pipeline": "transformers_pipeline",
                   "torchscript": "torchscript", "skops": "skops"}


def validate_model_spec(spec: Any) -> dict:
    if not isinstance(spec, dict) or type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("model spec must declare schema_version=1")
    if (not isinstance(spec.get("format"), str) or spec["format"] not in FORMAT_BACKENDS
            or not isinstance(spec.get("task"), str) or not spec["task"]):
        raise ValueError("model spec requires a supported format and an explicit task")
    if len(json.dumps(spec, allow_nan=False).encode()) > MAX_SPEC_BYTES:
        raise ValueError("model spec exceeds 64 KiB")
    if spec["format"] == "transformers-pipeline":
        if set(spec) - {"schema_version", "format", "task", "pipeline_task"}:
            raise ValueError("transformers-pipeline model spec has unknown fields")
        if not isinstance(spec.get("pipeline_task"), str) or not spec["pipeline_task"]:
            raise ValueError("transformers-pipeline model spec requires pipeline_task")
    else:
        name = spec.get("model_file")
        path = PurePosixPath(name) if isinstance(name, str) else None
        if (not name or path is None or path.is_absolute() or ".." in path.parts or "\\" in name
                or str(path) != name):
            raise ValueError("model spec model_file must be a relative snapshot path")
        if spec["format"] == "onnxruntime" and path.suffix.lower() != ".onnx":
            raise ValueError("ONNX model_file must end in .onnx")
        if "feature_dim" in spec and (type(spec["feature_dim"]) is not int or not 0 < spec["feature_dim"] <= 65536):
            raise ValueError("model spec feature_dim must be a positive integer <= 65536")
    return spec


def read_model_spec(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_SPEC_BYTES + 1)
    if len(data) > MAX_SPEC_BYTES:
        raise ValueError("model spec exceeds 64 KiB")
    return validate_model_spec(json.loads(data))


def task_model_spec(task_info: Any) -> dict:
    override = getattr(task_info, "model_spec", {})
    metadata = getattr(task_info, "repository_metadata", {}) or {}
    return override or metadata.get("acprof_model.json", {})


def encode_model_spec(spec: dict) -> str:
    if not spec:
        return ""
    validate_model_spec(spec)
    return base64.b64encode(json.dumps(spec, sort_keys=True, allow_nan=False,
                                       separators=(",", ":")).encode()).decode("ascii")


def load_model_spec(model_source: str, task: str, *, expected_format: str) -> dict:
    """Load the image-baked declaration, falling back to the local snapshot only."""
    encoded = os.getenv(SPEC_ENV, "")
    path = Path(model_source) / "acprof_model.json"
    if encoded:
        if len(encoded) > MAX_SPEC_BYTES * 2:
            raise ValueError("baked model spec exceeds 64 KiB")
        spec = validate_model_spec(json.loads(base64.b64decode(encoded, validate=True)))
    elif path.is_file():
        spec = read_model_spec(path)
    else:
        return {}
    if spec["task"] != task or spec["format"] != expected_format:
        raise ValueError("model spec task/format differs from selected interface")
    return spec


def custom_code_files(config: dict) -> list[str]:
    """Resolve declared Python filenames without importing repository modules."""
    auto_map = config.get("auto_map") or {}
    pipelines = config.get("custom_pipelines") or {}
    if not isinstance(auto_map, dict) or not isinstance(pipelines, dict):
        raise ValueError("auto_map/custom_pipelines must be JSON objects")
    references = list(auto_map.values())
    for entry in pipelines.values():
        if not isinstance(entry, dict) or not entry.get("impl"):
            raise ValueError("custom pipeline must declare its code implementation (impl)")
        references.append(entry["impl"])
    files = set()
    for value in references:
        for reference in value if isinstance(value, (list, tuple)) else [value]:
            if reference is None:
                continue
            if not isinstance(reference, str) or "--" in reference:
                raise ValueError("custom code must belong to the pinned model snapshot; cross-repository code requires an adapter")
            module, separator, name = reference.rpartition(".")
            if not separator or not name.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
                raise ValueError(f"invalid custom code reference: {reference!r}")
            files.add(module.replace(".", "/") + ".py")
    return sorted(files)


def pipeline_task(model_source: str, task: str) -> str:
    """Bridge an explicitly declared custom pipeline to an existing task protocol."""
    spec = load_model_spec(model_source, task, expected_format="transformers-pipeline")
    if not spec:
        return task
    root = Path(model_source)
    if not root.is_dir():
        raise ValueError("custom pipeline requires a baked local model snapshot")
    config = json.loads((root / "config.json").read_text())
    selected = spec["pipeline_task"]
    if selected not in (config.get("custom_pipelines") or {}):
        raise ValueError(f"pipeline_task {selected!r} is not declared in config.custom_pipelines")
    for name in custom_code_files(config):
        # Hub snapshot entries can link to the pinned content-addressed blob store.
        if not (root / name).is_file():
            raise ValueError(f"custom pipeline code is missing from snapshot: {name}")
    return selected

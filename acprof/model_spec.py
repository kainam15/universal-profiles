"""Declarative model interfaces shared by discovery, image identity and loaders."""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any


SPEC_ENV = "ACPROF_MODEL_SPEC_B64"
DEPENDENCIES_ENV = "ACPROF_MODEL_DEPENDENCIES_B64"
MAX_SPEC_BYTES = 64 * 1024
FORMAT_BACKENDS = {"onnxruntime": "onnxruntime", "transformers-pipeline": "transformers_pipeline",
                   "torchscript": "torchscript", "skops": "skops"}
MULTIMODAL_PIPELINE_INPUTS = {
    "audio-text-to-text": {"text", "audio", "sampling_rate"},
    "image-text-to-text": {"text", "image"},
    "video-text-to-text": {"text", "video", "fps"},
}


def validate_dependencies(dependencies: Any) -> list[dict]:
    """Offline repositories are explicit, bounded, and pinned before a build."""
    if not isinstance(dependencies, list) or len(dependencies) > 16:
        raise ValueError("model dependencies must be a list of at most 16 repositories")
    seen = set()
    for item in dependencies:
        if not isinstance(item, dict) or set(item) - {"repo_id", "revision", "allow_patterns"}:
            raise ValueError("invalid model dependency fields")
        repo = item.get("repo_id")
        if (not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", repo, re.ASCII)
                or any(part in {".", ".."} for part in repo.split("/")) or repo in seen):
            raise ValueError("model dependency repo_id must be a unique Hub repository")
        if not isinstance(item.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", item["revision"]):
            raise ValueError("model dependency revision must be a fixed commit SHA")
        patterns = item.get("allow_patterns")
        if patterns is not None and (not isinstance(patterns, list) or not patterns or len(patterns) > 128 or any(
            not isinstance(pattern, str) or not pattern or pattern.startswith("/") or "\\" in pattern
            or ".." in pattern.split("/") for pattern in patterns
        )):
            raise ValueError("model dependency allow_patterns must contain relative file patterns")
        seen.add(repo)
    return dependencies


def validate_multimodal_pipeline(task: str, protocol: Any) -> dict:
    required = MULTIMODAL_PIPELINE_INPUTS.get(task)
    if required is None or not isinstance(protocol, dict) or set(protocol) - {"inputs", "forward_kwargs"}:
        raise ValueError("multimodal pipeline requires a supported text-output task and protocol")
    inputs = protocol.get("inputs")
    if (not isinstance(inputs, dict) or not inputs or any(not isinstance(key, str) or not key.isidentifier()
            or not isinstance(value, str) or value not in required for key, value in inputs.items())
            or set(inputs.values()) != required):
        raise ValueError(f"multimodal inputs must map all of {sorted(required)}")
    kwargs = protocol.get("forward_kwargs", {"max_new_tokens": "$max_new_tokens", "do_sample": "$do_sample"})
    if (not isinstance(kwargs, dict) or not kwargs or any(not isinstance(key, str) or not key.isidentifier()
            or not isinstance(value, (str, bool, int, float, type(None))) for key, value in kwargs.items())
            or any(isinstance(value, str) and value.startswith("$") and value not in {"$max_new_tokens", "$do_sample"}
                   for value in kwargs.values()) or "$max_new_tokens" not in kwargs.values()):
        raise ValueError("multimodal forward_kwargs require bounded generation and known parameter references")
    if (kwargs.get("do_sample", False) not in (False, "$do_sample")
            or kwargs.get("temperature", 0) not in (0, None)
            or not ("$do_sample" in kwargs.values() or kwargs.get("do_sample") is False
                    or ("temperature" in kwargs and kwargs["temperature"] == 0))):
        raise ValueError("multimodal generation parameters must declare deterministic sampling")
    return protocol


def validate_model_spec(spec: Any) -> dict:
    if not isinstance(spec, dict) or type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("model spec must declare schema_version=1")
    if (not isinstance(spec.get("format"), str) or spec["format"] not in FORMAT_BACKENDS
            or not isinstance(spec.get("task"), str) or not spec["task"]):
        raise ValueError("model spec requires a supported format and an explicit task")
    if len(json.dumps(spec, allow_nan=False).encode()) > MAX_SPEC_BYTES:
        raise ValueError("model spec exceeds 64 KiB")
    validate_dependencies(spec.get("dependencies", []))
    if spec["format"] == "transformers-pipeline":
        if set(spec) - {"schema_version", "format", "task", "pipeline_task", "multimodal", "dependencies"}:
            raise ValueError("transformers-pipeline model spec has unknown fields")
        if not isinstance(spec.get("pipeline_task"), str) or not spec["pipeline_task"]:
            raise ValueError("transformers-pipeline model spec requires pipeline_task")
        if "multimodal" in spec:
            validate_multimodal_pipeline(spec["task"], spec["multimodal"])
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


def declared_multimodal_pipeline(task_info: Any) -> bool:
    spec = task_model_spec(task_info)
    if not spec or "multimodal" not in spec:
        return False
    validate_model_spec(spec)
    return (spec["format"] == "transformers-pipeline" and spec["task"] == task_info.pipeline_tag
            and task_info.runtime_backend == "transformers_pipeline")


def encode_model_dependencies(spec: dict) -> str:
    dependencies = validate_dependencies(spec.get("dependencies", []))
    if not dependencies:
        return ""
    return base64.b64encode(json.dumps(dependencies, sort_keys=True, separators=(",", ":")).encode()).decode("ascii")


def load_model_dependencies() -> list[dict]:
    encoded = os.getenv(DEPENDENCIES_ENV, "")
    if not encoded:
        return []
    if len(encoded) > MAX_SPEC_BYTES * 2:
        raise ValueError("baked model dependencies exceed 64 KiB")
    return validate_dependencies(json.loads(base64.b64decode(encoded, validate=True)))


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

"""Resolve metadata and native interfaces without importing inference libraries."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from acprof.model_spec import FORMAT_BACKENDS, custom_code_files, task_model_spec, validate_model_spec


# Tasks select upstream operations; checkpoint names never take part in this map.
_TASK_MAPPINGS = {
    "text-generation": ("CAUSAL_LM",), "conversational": ("CAUSAL_LM",),
    "text2text-generation": ("SEQ_TO_SEQ_CAUSAL_LM",), "translation": ("SEQ_TO_SEQ_CAUSAL_LM",),
    "summarization": ("SEQ_TO_SEQ_CAUSAL_LM",), "fill-mask": ("MASKED_LM",),
    "text-classification": ("SEQUENCE_CLASSIFICATION",), "text-ranking": ("SEQUENCE_CLASSIFICATION",),
    "zero-shot-classification": ("SEQUENCE_CLASSIFICATION",), "token-classification": ("TOKEN_CLASSIFICATION",),
    "question-answering": ("QUESTION_ANSWERING",), "feature-extraction": ("",),
    "sentence-similarity": ("",), "image-classification": ("IMAGE_CLASSIFICATION",),
    "image-feature-extraction": ("IMAGE", ""), "image-to-text": ("VISION_2_SEQ", "IMAGE_TEXT_TO_TEXT"),
    "image-text-to-text": ("IMAGE_TEXT_TO_TEXT",), "video-text-to-text": ("IMAGE_TEXT_TO_TEXT",),
    "visual-question-answering": ("VISUAL_QUESTION_ANSWERING",),
    "document-question-answering": ("DOCUMENT_QUESTION_ANSWERING",),
    "table-question-answering": ("TABLE_QUESTION_ANSWERING", "SEQ_TO_SEQ_CAUSAL_LM"),
    "object-detection": ("OBJECT_DETECTION",), "zero-shot-object-detection": ("ZERO_SHOT_OBJECT_DETECTION",),
    "image-segmentation": ("IMAGE_SEGMENTATION", "SEMANTIC_SEGMENTATION", "INSTANCE_SEGMENTATION", "UNIVERSAL_SEGMENTATION"),
    "depth-estimation": ("DEPTH_ESTIMATION",), "video-classification": ("VIDEO_CLASSIFICATION",),
    "zero-shot-image-classification": ("ZERO_SHOT_IMAGE_CLASSIFICATION",), "mask-generation": ("MASK_GENERATION",),
    "keypoint-detection": ("KEYPOINT_DETECTION", "KEYPOINT_MATCHING"),
    "automatic-speech-recognition": ("CTC", "SPEECH_SEQ_2_SEQ"), "audio-classification": ("AUDIO_CLASSIFICATION",),
    "text-to-speech": ("TEXT_TO_WAVEFORM", "TEXT_TO_SPECTROGRAM"),
}


@lru_cache(maxsize=None)
def transformers_support(version: str) -> dict:
    path = Path(__file__).parent / "extensions" / "transformers" / f"{version}.json"
    if not path.is_file():
        raise ValueError(f"No static Auto registry for locked transformers=={version}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("version") != version:
        raise ValueError(f"Invalid Transformers support catalog: {path}")
    return data["mappings"]


def audio_text_loader(version: str, config: dict) -> tuple[str, str | None] | None:
    """Select a native text-output Auto interface, shared by preflight and load.

    Composite models may expose a separately registered multimodal text head.
    Select that head without instantiating speech synthesis components. Never
    select a plain language submodel, which would discard the audio encoder.
    """
    mappings = transformers_support(version)
    model_type = config.get("model_type")
    for operation, auto_class in (
        ("SEQ_TO_SEQ_CAUSAL_LM", "AutoModelForSeq2SeqLM"),
        ("IMAGE_TEXT_TO_TEXT", "AutoModelForImageTextToText"),
    ):
        if model_type in mappings.get(f"MODEL_FOR_{operation}_MAPPING_NAMES", {}):
            return auto_class, None
    if model_type not in mappings.get("MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", {}):
        return None
    multimodal = mappings.get("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", {})
    text_heads = [key for key, value in config.items()
                  if isinstance(value, dict) and value.get("model_type") in multimodal]
    if len(text_heads) == 1:
        return "AutoModelForImageTextToText", text_heads[0]
    return None


def supports_transformers_task(
    version: str, task: str, model_type: str, model_config: dict | None = None,
) -> bool | None:
    if task == "audio-text-to-text" and model_type:
        return audio_text_loader(version, model_config or {"model_type": model_type}) is not None
    if not model_type or task not in _TASK_MAPPINGS:
        return None
    mappings = transformers_support(version)
    return any(model_type in mappings.get(f"MODEL_FOR_{name}_MAPPING_NAMES" if name else "MODEL_MAPPING_NAMES", {})
               for name in _TASK_MAPPINGS[task])


_AUTO_TASKS = {
    "AutoModelForCausalLM": "text-generation", "AutoModelForSeq2SeqLM": "text2text-generation",
    "AutoModelForMaskedLM": "fill-mask", "AutoModelForSequenceClassification": "text-classification",
    "AutoModelForTokenClassification": "token-classification", "AutoModelForQuestionAnswering": "question-answering",
    "AutoModelForImageClassification": "image-classification", "AutoModelForAudioClassification": "audio-classification",
    "AutoModelForCTC": "automatic-speech-recognition", "AutoModelForSpeechSeq2Seq": "automatic-speech-recognition",
    "AutoModelForImageTextToText": "image-text-to-text", "AutoModelForVision2Seq": "image-to-text",
}


def discover_model_candidates(task_info: Any, *, override_tag: str | None = None,
                              override_backend: str | None = None) -> dict:
    """Collect static evidence before selecting a route. Never import model code."""
    from acprof.extensions import CATALOG

    config = task_info.model_config or {}
    metadata = task_info.repository_metadata or {}
    files: set[str] = set(task_info.repository_files or ())
    spec = task_model_spec(task_info)
    errors: list[str] = []
    conflicts: list[str] = []
    candidates: list[dict[str, Any]] = []
    if spec:
        try:
            validate_model_spec(spec)
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
            spec = {}
    hub_task = (task_info.pipeline_tag if task_info.pipeline_tag != "unknown"
                and task_info.detection_method != "config_infer" else None)
    if spec.get("pipeline_task") and hub_task == spec["pipeline_task"]:
        hub_task = spec["task"]
    declared_task = spec.get("task")
    if declared_task and hub_task and hub_task != declared_task and not override_tag:
        conflicts.append(f"task conflict: Hub={hub_task}, model spec={declared_task}; select --task explicitly")
    if override_tag and declared_task and override_tag != declared_task:
        conflicts.append("--task conflicts with model spec task; update --model-spec")
    declared_backend = FORMAT_BACKENDS.get(spec.get("format"))
    if override_backend and declared_backend and override_backend != declared_backend:
        conflicts.append("--backend conflicts with model spec format; update --model-spec")

    def backend_for(task: str) -> str:
        if override_backend or declared_backend:
            return override_backend or declared_backend
        family = CATALOG.task_families.get(task, "unknown")
        # An export-only repository cannot be loaded by a native weight loader.
        has_native_weights = any(name.endswith((".safetensors", ".bin")) for name in files)
        if any(name.lower().endswith(".onnx") for name in files) and not has_native_weights:
            return "onnxruntime"
        if task in {"feature-extraction", "sentence-similarity"} and "modules.json" in metadata:
            return "sentence_transformers"
        return CATALOG.default_backend(task, family, task_info.library_name, task_info.runtime_backend)

    def add(task: str | None, source: str) -> None:
        if not task:
            return
        backend = backend_for(task)
        previous = next((item for item in candidates if (item["task"], item["backend"]) == (task, backend)), None)
        if previous is not None:
            if source not in previous["evidence"]:
                previous["evidence"].append(source)
            return
        family = CATALOG.task_families.get(task, "unknown")
        candidates.append({"task": task, "family": family, "backend": backend, "evidence": [source]})

    add(override_tag, "explicit_task")
    add(declared_task, "local_model_spec" if getattr(task_info, "model_spec", {}) else "repository_model_spec")
    add(hub_task, "hub_metadata" if task_info.detection_method == "hub_api" else task_info.detection_method)
    for architecture in config.get("architectures") or []:
        if isinstance(architecture, str):
            add(CATALOG.infer_task([architecture], str(config.get("model_type") or "")), "config.architectures")
    if config.get("model_type"):
        add(CATALOG.infer_task([], str(config["model_type"])), "config.model_type")
    auto_map = config.get("auto_map") or {}
    if isinstance(auto_map, dict):
        for auto_class in auto_map:
            add(_AUTO_TASKS.get(auto_class), "config.auto_map")
    custom_tasks = config.get("custom_pipelines") or {}
    if isinstance(custom_tasks, dict):
        for custom_task in custom_tasks:
            if custom_task in CATALOG.task_families:
                add(custom_task, "config.custom_pipelines")
    if not candidates and "modules.json" in metadata:
        add("feature-extraction", "modules.json")

    selected = override_tag or declared_task or hub_task
    status = "candidate"
    if not selected:
        if len(candidates) == 1:
            selected = candidates[0]["task"]
        elif len(candidates) > 1:
            status = "ambiguous"
        else:
            status = "needs_configuration"
            if any(name.lower().endswith(".onnx") for name in files):
                for extension in CATALOG.extensions.values():
                    if "onnxruntime" in extension.backends:
                        for candidate_task in extension.tasks:
                            add(candidate_task, "artifact.onnx; task semantics required")
            errors.append("task semantics are unknown; provide --task or --model-spec")
    if errors:
        status = "needs_configuration"
    if conflicts:
        status = "ambiguous"
    if selected:
        task_info.pipeline_tag = selected
        task_info.task_family = CATALOG.task_families.get(selected, "unknown")
        task_info.runtime_backend = backend_for(selected)
    return {"schema_version": 1, "status": status, "model_revision": task_info.model_revision,
            "candidates": candidates, "conflicts": conflicts, "missing": errors,
            "selection": {"task": selected, "backend": task_info.runtime_backend if selected else None,
                          "source": "explicit" if override_tag or override_backend or
                                    getattr(task_info, "model_spec", {}) else "metadata"}}


def require_resolved_candidate(task_info: Any) -> None:
    resolution = getattr(task_info, "model_resolution", {}) or {}
    if resolution.get("status") in {"ambiguous", "needs_configuration"}:
        choices = ", ".join(f"{item['task']}/{item['backend']}" for item in resolution.get("candidates", []))
        details = "; ".join([*resolution.get("conflicts", []), *resolution.get("missing", [])])
        raise ValueError(f"model resolution {resolution['status']}: {details}; candidates: {choices or 'none'}; "
                         "use --task / --backend / --model-spec to complete the interface")


def resolve_model_interface(task_info: Any) -> dict:
    """Reject known incompatible layouts; a successful resolution is only a candidate."""
    from acprof.extensions import select_extension

    require_resolved_candidate(task_info)
    extension = select_extension(task_info)
    backend, library = task_info.runtime_backend, task_info.library_name
    config = getattr(task_info, "model_config", {}) or {}
    files: set[str] = set(getattr(task_info, "repository_files", ()) or ())
    metadata = getattr(task_info, "repository_metadata", {}) or {}
    errors = getattr(task_info, "metadata_errors", ())
    if errors:
        raise ValueError("Model metadata could not be resolved: " + "; ".join(errors))
    spec = task_model_spec(task_info)
    if spec:
        validate_model_spec(spec)
        if spec["task"] != task_info.pipeline_tag or FORMAT_BACKENDS[spec["format"]] != backend:
            raise ValueError("model spec task/format conflicts with selected interface")
        if files and spec.get("model_file") and spec["model_file"] not in files:
            raise ValueError(f"model spec model_file is absent from snapshot: {spec['model_file']}")
    code_files = set()
    for item in [config, *(value for value in metadata.values() if isinstance(value, dict))]:
        code_files.update(custom_code_files(item))
    if files and code_files - files:
        raise ValueError(f"custom code is absent from snapshot: {sorted(code_files - files)}")
    pipeline_name = spec.get("pipeline_task", task_info.pipeline_tag)
    if spec.get("format") == "transformers-pipeline":
        from acprof.model_spec import declared_multimodal_pipeline
        if (task_info.task_family not in {"nlp", "cv", "audio"} and not declared_multimodal_pipeline(task_info)) or task_info.pipeline_tag in {
            "video-classification", "keypoint-detection", "sentence-similarity", "text-ranking",
        }:
            raise ValueError("custom pipeline needs an adapter for this task protocol")
        if pipeline_name not in (config.get("custom_pipelines") or {}):
            raise ValueError(f"pipeline_task {pipeline_name!r} is not declared in config.custom_pipelines")
    if any(item["repo_id"] == task_info.model_id for item in spec.get("dependencies", [])):
        raise ValueError("model dependencies must not override the primary snapshot")
    format_name, loader, operation = "unknown", backend, "predict"
    if backend in {"transformers_model", "transformers_pipeline", "sentence_transformers", "cross_encoder"}:
        # Unknown ecosystem tags may still wrap a registered native checkpoint;
        # accept actual native metadata, never an unrelated task tag alone.
        native = any(supports_transformers_task(version, task_info.pipeline_tag, str(config.get("model_type", "")), config)
                     for version in ("4.57.6", "5.6.0"))
        if library not in {"", "unknown", "transformers", "sentence-transformers", "timm"} and not native and extension.adapter == "family-default":
            raise ValueError(f"library {library!r} has no shared {backend} interface; select a registered backend")
        if files and "config.json" not in files and "modules.json" not in files:
            suffix = "GGUF" if any(name.lower().endswith(".gguf") for name in files) else "missing config.json/modules.json"
            raise ValueError(f"{backend} cannot load this artifact layout ({suffix})")
        if "adapter_config.json" in files:
            base = (metadata.get("adapter_config.json") or {}).get("base_model_name_or_path")
            raise ValueError(f"adapter requires an offline base model dependency ({base or 'unknown'}); a standalone snapshot is incomplete")
        format_name = "sentence_transformers" if backend == "sentence_transformers" else "transformers"
        if backend == "sentence_transformers":
            loader, operation = "SentenceTransformer", "encode"
        elif backend == "cross_encoder":
            loader = "CrossEncoder"
        else:
            loader = "Transformers Auto/pipeline"
    elif backend == "chronos":
        format_name, loader = "chronos", "BaseChronosPipeline"
    elif backend == "diffusers":
        if files and "model_index.json" not in files:
            format_name = "GGUF" if any(name.lower().endswith(".gguf") for name in files) else "single-file/components"
            raise ValueError(f"DiffusionPipeline requires model_index.json; {format_name} needs a compatible artifact loader")
        format_name, loader, operation = "diffusers_pipeline", "DiffusionPipeline", "__call__"
    elif backend == "onnxruntime":
        format_name, loader, operation = "onnx", "InferenceSession", "run"
        artifacts = [name for name in files if name.lower().endswith(".onnx")]
        if files and not spec and len(artifacts) != 1:
            raise ValueError("ONNX requires one artifact or --model-spec selecting model_file")
        if files and task_info.task_family in {"cv", "nlp"} and not spec:
            raise ValueError("ONNX image/text preprocessing requires --model-spec (acprof_model.json)")
    elif backend in {"torchscript", "skops"}:
        format_name = backend
    return {**getattr(task_info, "model_resolution", {}),
            "schema_version": 1, "status": "candidate", "task": task_info.pipeline_tag,
            "backend": backend, "library": library, "artifact_format": format_name,
            "loader": loader, "operation": operation, "model_revision": task_info.model_revision,
            "metadata_files": sorted(metadata), "model_type": config.get("model_type"),
            "interface_kind": "custom_pipeline" if pipeline_name in (config.get("custom_pipelines") or {}) else
                              "custom_auto" if config.get("auto_map") else "standard",
            "pipeline_task": pipeline_name if backend == "transformers_pipeline" else None,
            "code_revision": task_info.model_revision if code_files else None,
            "code_files": sorted(code_files), "model_spec": spec}

"""Resolve metadata and native interfaces without importing inference libraries."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


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


def resolve_model_interface(task_info: Any) -> dict:
    """Reject known incompatible layouts; a successful resolution is only a candidate."""
    from acprof.extensions import select_extension

    extension = select_extension(task_info)
    backend, library = task_info.runtime_backend, task_info.library_name
    config = getattr(task_info, "model_config", {}) or {}
    files = set(getattr(task_info, "repository_files", ()) or ())
    metadata = getattr(task_info, "repository_metadata", {}) or {}
    errors = getattr(task_info, "metadata_errors", ())
    if errors:
        raise ValueError("Model metadata could not be resolved: " + "; ".join(errors))
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
    elif backend in {"torchscript", "skops"}:
        format_name = backend
    return {"schema_version": 1, "status": "candidate", "task": task_info.pipeline_tag,
            "backend": backend, "library": library, "artifact_format": format_name,
            "loader": loader, "operation": operation, "model_revision": task_info.model_revision,
            "metadata_files": sorted(metadata), "model_type": config.get("model_type")}

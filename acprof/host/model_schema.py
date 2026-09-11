"""任务输入输出描述及设备推理精度。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from acprof.host.detect import TaskInfo


def _json_object_schema(
    properties: Dict[str, Any],
    required: List[str],
) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
    }


def _model_io_formats(task_info: TaskInfo) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Describe the actual /predict JSON contract used by the selected handler."""
    string_schema = {"type": "string"}
    params_schema = {"type": "object"}
    input_properties: Dict[str, Any]
    input_required: List[str]
    output_properties: Dict[str, Any]
    output_required = ["task"]

    if task_info.task_family == "nlp":
        if task_info.pipeline_tag == "question-answering":
            input_properties = {
                "question": string_schema,
                "context": string_schema,
                "params": params_schema,
            }
            input_required = ["question", "context"]
        else:
            input_properties = {
                "text": string_schema,
                "params": params_schema,
            }
            input_required = ["text"]
        output_properties = {
            "task": string_schema,
            "output_type": {
                "type": "string",
                "enum": ["text", "label"],
            },
            "n_results": {"type": "integer"},
            "effective_input_scale": {"type": "number"},
        }
        output_required.extend(["output_type", "n_results"])
        input_properties["batch_size"] = {"type": "integer", "minimum": 1}
        if task_info.pipeline_tag == "table-question-answering":
            input_properties = {
                "table": {"type": "object", "additionalProperties": {"type": "array", "items": string_schema}},
                "query": string_schema, "params": params_schema,
                "batch_size": {"type": "integer", "minimum": 1},
            }
            input_required = ["table", "query"]
            output_properties["output_type"]["enum"] = ["table_answer"]
        elif task_info.pipeline_tag in {"sentence-similarity", "text-ranking"}:
            input_properties = {
                "query": string_schema,
                "documents": {"type": "array", "items": string_schema, "minItems": 1},
                "params": params_schema, "batch_size": {"type": "integer", "minimum": 1},
            }
            input_required = ["query", "documents"]
            output_properties["output_type"]["enum"] = [
                "similarity" if task_info.pipeline_tag == "sentence-similarity" else "ranking"
            ]
        elif task_info.pipeline_tag == "zero-shot-classification":
            input_properties["candidate_labels"] = {"type": "array", "items": string_schema, "minItems": 1}
            input_properties["hypothesis_template"] = string_schema
            input_required.append("candidate_labels")
        elif task_info.pipeline_tag == "feature-extraction":
            output_properties["output_type"]["enum"] = ["embedding"]
    elif task_info.task_family == "cv":
        input_properties = {
            "image_base64": {
                "type": "string",
                "contentEncoding": "base64",
                "contentMediaType": "image/png",
            },
            "params": params_schema,
            "input_scale": {"type": "number", "exclusiveMinimum": 0},
        }
        input_required = ["image_base64"]
        output_properties = {
            "task": string_schema,
            "output_type": {
                "type": "string",
                "enum": ["classification", "detection"],
            },
            "n_results": {"type": "integer"},
            "effective_input_scale": {"type": "number"},
        }
        output_required.extend(["output_type", "n_results"])
        if task_info.pipeline_tag == "image-to-text":
            output_properties["output_type"]["enum"] = ["caption"]
            output_properties.update({
                "captions": {"type": "array", "items": {"type": "string"}},
                "output_length": {"type": "integer"},
                "output_token_count": {"type": ["integer", "null"]},
            })
            output_required.extend(["captions", "output_length", "output_token_count"])
        else:
            output_types = {
                "depth-estimation": "depth", "image-segmentation": "segmentation",
                "mask-generation": "masks", "image-feature-extraction": "features",
                "keypoint-detection": "keypoints",
            }
            output_properties["output_type"]["enum"] = [output_types.get(
                task_info.pipeline_tag,
                "classification" if "classification" in task_info.pipeline_tag else "detection",
            )]
        if task_info.pipeline_tag == "video-classification":
            image_schema = input_properties.pop("image_base64")
            input_properties["frames_base64"] = {
                "type": "array", "items": image_schema, "minItems": 1,
            }
            input_required = ["frames_base64"]
        if task_info.pipeline_tag.startswith("zero-shot-"):
            input_properties["candidate_labels"] = {
                "type": "array", "items": string_schema, "minItems": 1,
            }
            input_required.append("candidate_labels")
        if task_info.pipeline_tag == "keypoint-detection":
            input_properties["boxes"] = {
                "type": "array", "description": "COCO xywh boxes in input-image pixels for pose estimation",
            }
            output_properties["keypoint_count"] = {"type": "integer"}
        if task_info.pipeline_tag in {"depth-estimation", "image-feature-extraction"}:
            name = "depth_shape" if task_info.pipeline_tag == "depth-estimation" else "feature_shape"
            output_properties[name] = {"type": "array", "items": {"type": "integer"}}
        if task_info.pipeline_tag == "video-classification":
            output_properties["classifications"] = {"type": "array", "items": {"type": "object"}}
    elif task_info.task_family == "audio":
        input_properties = {
            "audio_base64": {
                "type": "string",
                "contentEncoding": "base64",
                "contentMediaType": "audio/wav",
            },
            "audio_format": {"type": "string", "enum": ["wav"]},
            "sample_rate": {"type": "integer", "unit": "Hz"},
            "params": params_schema,
        }
        input_required = ["audio_base64", "audio_format", "sample_rate"]
        output_properties = {
            "task": string_schema,
            "output_type": {
                "type": "string",
                "enum": ["transcription", "classification", "unknown"],
            },
            "text": string_schema,
            "output_length": {"type": "integer"},
            "output_token_count": {"type": ["integer", "null"]},
            "n_results": {"type": "integer"},
            "effective_input_scale": {"type": "number"},
        }
        output_required.append("output_type")
        if task_info.pipeline_tag in {"text-to-speech", "text-to-audio"}:
            input_properties = {"text": string_schema, "params": params_schema}
            input_required = ["text"]
        if task_info.pipeline_tag in {"text-to-speech", "text-to-audio", "audio-to-audio"}:
            output_properties["output_type"]["enum"] = ["audio"]
            output_properties.update({
                "audio_shape": {"type": "array", "items": {"type": "integer"}},
                "audio_num_samples": {"type": "integer"},
                "audio_sample_rate": {"type": "integer", "unit": "Hz"},
                "audio_duration_s": {"type": "number", "unit": "s"},
            })
        elif task_info.pipeline_tag == "voice-activity-detection":
            output_properties["output_type"]["enum"] = ["voice_activity"]
            output_properties.update({
                "segments": {"type": "array", "items": _json_object_schema({
                    "start": {"type": "number", "unit": "s"}, "end": {"type": "number", "unit": "s"},
                }, ["start", "end"])},
                "timestamp_unit": {"type": "string", "enum": ["seconds"]},
                "speech_duration_s": {"type": "number", "unit": "s"},
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "frame_duration_s": {"type": "number", "unit": "s"},
                "segmentation_policy": {"type": "string", "enum": ["adjacent_frames_above_threshold"]},
            })
    elif task_info.task_family == "structured":
        is_graph = task_info.pipeline_tag == "graph-ml"
        is_table = task_info.pipeline_tag in {"tabular-classification", "tabular-regression"}
        field_name = "graphs" if is_graph else "features" if is_table else "observations"
        matrix_schema = {"type": "array", "minItems": 1, "items": {
            "type": "array", "minItems": 1, "items": {"type": "number", "format": "float32"},
        }}
        values_schema = matrix_schema
        if is_graph:
            values_schema = {"type": "array", "minItems": 1, "items": _json_object_schema({
                "node_features": matrix_schema,
                "edge_index": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
                    "type": "array", "items": {"type": "integer", "minimum": 0},
                }},
            }, ["node_features", "edge_index"])}
        input_properties = {
            field_name: values_schema,
            "batch_size": {"type": "integer", "minimum": 1},
            "input_scale": {"type": "integer", "minimum": 1},
        }
        input_required = [field_name, "batch_size", "input_scale"]
        output_type = {
            "tabular-classification": "classification", "tabular-regression": "regression",
            "reinforcement-learning": "actions", "robotics": "actions", "graph-ml": "graph",
        }[task_info.pipeline_tag]
        output_properties = {
            "task": string_schema, "output_type": {"type": "string", "enum": [output_type]},
            "output_shape": {"type": "array", "items": {"type": "integer"}},
            "n_results": {"type": "integer"}, "effective_input_scale": {"type": "number"},
        }
        output_required.extend(["output_type", "output_shape", "n_results"])
    elif task_info.task_family == "timeseries":
        input_properties = {
            "context": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number", "format": "float32"},
                },
            },
            "prediction_length": {"type": "integer"},
        }
        input_required = ["context", "prediction_length"]
        output_properties = {
            "task": string_schema,
            "forecast_shape": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "output_type": {
                "type": "string",
                "enum": ["forecast"],
            },
        }
    elif task_info.task_family == "multimodal":
        input_properties = {
            "samples": {
                "type": "array", "minItems": 1, "maxItems": 1,
                "items": _json_object_schema({
                    "text": string_schema,
                    "image_base64": {"type": "string", "contentEncoding": "base64", "contentMediaType": "image/png"},
                    "audio_base64": {"type": "string", "contentEncoding": "base64", "contentMediaType": "audio/wav"},
                    "sampling_rate": {"type": "integer", "unit": "Hz"},
                    "video_frames_base64": {"type": "array", "items": string_schema},
                    "fps": {"type": "number", "unit": "frames/s"},
                    "words": {"type": "array", "items": string_schema},
                    "boxes": {"type": "array", "description": "OCR boxes normalized to 0..1000"},
                }, ["text"]),
            },
            "params": params_schema,
            "input_scale": {"type": "number", "exclusiveMinimum": 0},
            "input_scale_type": {"type": "string", "enum": ["resolution_px", "duration_s", "frame_count"]},
        }
        input_required = ["samples", "input_scale", "input_scale_type"]
        output_properties = {
            "task": string_schema,
            "output_type": string_schema,
            "n_results": {"type": "integer"},
            "output_length": {"type": ["integer", "null"]},
            "output_token_count": {"type": ["integer", "null"]},
            "effective_input_scale": {"type": "number"},
        }
        if task_info.pipeline_tag == "visual-document-retrieval":
            output_properties["scores"] = {"type": "array", "description": "query-by-page MaxSim scores, including both encoders"}
            output_properties.update({"n_queries": {"type": "integer"}, "n_documents": {"type": "integer"}, "retrieval_scope": string_schema})
        elif task_info.pipeline_tag in {"visual-question-answering", "document-question-answering"}:
            output_properties["answers"] = {"type": "array", "items": {"type": "object"}}
        else:
            output_properties["texts"] = {"type": "array", "items": string_schema}
        if task_info.pipeline_tag == "any-to-any":
            output_properties.update({
                "audio_num_samples": {"type": "integer"},
                "audio_sample_rate": {"type": "integer", "unit": "Hz"},
                "audio_duration_s": {"type": "number", "unit": "s"},
            })
        output_required.extend(["output_type", "n_results"])
    elif task_info.task_family == "diffusion":
        input_properties = {
            "prompt": {
                "oneOf": [
                    string_schema,
                    {
                        "type": "array",
                        "items": string_schema,
                        "minItems": 1,
                    },
                ],
            },
            "resolution": {
                "type": "integer",
                "minimum": 64,
                "multipleOf": 8,
                "unit": "px",
            },
            "params": params_schema,
        }
        input_required = ["prompt", "resolution"]
        output_properties = {
            "task": string_schema,
            "output_type": {
                "type": "string",
                "enum": ["image"],
            },
            "n_results": {"type": "integer"},
            "output_length": {"type": "integer"},
            "image_width": {"type": ["integer", "null"], "unit": "px"},
            "image_height": {"type": ["integer", "null"], "unit": "px"},
            "effective_input_scale": {"type": "number"},
        }
        output_required.extend(["output_type", "n_results"])
        if task_info.pipeline_tag in {
            "image-text-to-image", "image-text-to-video", "image-to-image",
            "image-to-video", "image-to-3d",
        }:
            input_properties["image_base64"] = {
                "type": "string", "contentEncoding": "base64", "contentMediaType": "image/png",
            }
            input_required.append("image_base64")
        if task_info.pipeline_tag in {"unconditional-image-generation", "image-to-3d", "image-to-video", "image-to-image"}:
            input_required.remove("prompt")
        if task_info.pipeline_tag in {"image-to-video", "image-to-image"}:
            input_properties["prompt_optional"] = {
                "type": "boolean",
                "description": "Only synthetic default text may be omitted by an image-only native pipeline",
            }
        if task_info.pipeline_tag == "video-to-video":
            input_properties["frames_base64"] = {
                "type": "array", "minItems": 1,
                "items": {"type": "string", "contentEncoding": "base64", "contentMediaType": "image/png"},
            }
            input_required.append("frames_base64")
        if task_info.pipeline_tag in {"image-text-to-video", "image-to-video", "text-to-video", "video-to-video"}:
            output_properties["output_type"]["enum"] = ["video"]
            output_properties.pop("image_width")
            output_properties.pop("image_height")
            output_properties.update({
                "video_frame_count": {"type": "integer"},
                "video_width": {"type": "integer", "unit": "px"},
                "video_height": {"type": "integer", "unit": "px"},
                "output_shape": {"type": "array", "items": {"type": "integer"}},
            })
        if task_info.pipeline_tag in {"unconditional-image-generation", "text-to-3d", "image-to-3d"}:
            input_properties.pop("resolution")
            input_required.remove("resolution")
            input_properties["input_scale"] = {
                "type": "integer", "minimum": 1, "unit": "denoising steps",
            }
            input_required.append("input_scale")
        if task_info.pipeline_tag in {"text-to-3d", "image-to-3d"}:
            output_properties["output_type"]["enum"] = ["mesh"]
            output_properties.pop("image_width")
            output_properties.pop("image_height")
            output_properties.update({
                "mesh_vertex_counts": {"type": "array", "items": {"type": "integer"}},
                "mesh_face_counts": {"type": "array", "items": {"type": "integer"}},
            })
    else:
        input_properties = {}
        input_required = []
        output_properties = {}
        output_required = []

    common = {
        "transport": "HTTP",
        "media_type": "application/json",
    }
    input_format = {
        **common,
        "method": "POST",
        "endpoint": "/predict",
        "json_schema": _json_object_schema(input_properties, input_required),
    }
    output_format = {
        **common,
        "status": 200,
        "json_schema": _json_object_schema(
            output_properties,
            output_required,
        ),
    }
    return input_format, output_format


def _inference_precision_by_device(task_info: TaskInfo) -> Dict[str, str]:
    if task_info.pipeline_tag == "voice-activity-detection":
        return {"cpu": "artifact-defined; FP32 inputs"}
    if task_info.pipeline_tag == "audio-to-audio":
        return {"cpu": "FP32", "gpu": "FP32"}
    if task_info.runtime_backend == "torchscript":
        return {"cpu": "artifact-defined; FP32 inputs", "gpu": "artifact-defined; FP32 inputs"}
    if task_info.runtime_backend == "skops":
        return {"cpu": "estimator-defined; FP32 inputs"}
    if task_info.pipeline_tag == "any-to-any" and task_info.task_family == "multimodal":
        return {"cpu": "FP32", "gpu": "mixed FP16 (thinker/talker), FP32 (token2wav)"}
    if (
        task_info.runtime_backend == "diffusers"
        and task_info.task_family == "diffusion"
    ):
        return {"cpu": "FP32", "gpu": "FP16"}
    if (
        task_info.runtime_backend in {"transformers_pipeline", "transformers_model", "sentence_transformers", "cross_encoder"}
        and task_info.task_family in {"nlp", "cv", "audio", "multimodal"}
    ):
        return {"cpu": "FP32", "gpu": "FP16"}
    if task_info.precision_dtype:
        return {
            "cpu": task_info.precision_dtype,
            "gpu": task_info.precision_dtype,
        }
    return {}

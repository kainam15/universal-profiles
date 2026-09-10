"""CV task handler - image-classification, object-detection, segmentation, etc."""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, Optional

import numpy as np

from acprof.container.handlers import (
    BaseHandler,
    HandlerRegistry,
    model_revision_kwargs,
    transformers_pipeline_load_kwargs,
)


class CVHandler(BaseHandler):

    def load(
        self,
        model_source: str,
        task_type: str,
        backend: str,
        device: str,
        model_revision: str = "main",
        load_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import torch
        from transformers import pipeline as hf_pipeline

        device_map = device if device == "cpu" else "auto"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        if task_type in {"video-classification", "keypoint-detection"}:
            return self._load_direct(model_source, task_type, device, model_revision, load_options)

        try:
            pipe = hf_pipeline(
                task=task_type,
                model=model_source,
                **model_revision_kwargs(model_source, model_revision),
                **transformers_pipeline_load_kwargs(load_options),
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except KeyError as exc:
            if task_type == "image-to-text" and "Unknown task image-to-text" in str(exc):
                raise RuntimeError(
                    "image-to-text requires the CV image with transformers==4.57.6; "
                    "rebuild it without --skip-build / 取消“复用现有镜像”后重新构建"
                ) from exc
            raise
        return {
            "pipeline": pipe,
            "task_type": task_type,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }

    def _load_direct(self, model_source: str, task_type: str, device: str,
                     model_revision: str, load_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVideoClassification

        revision = model_revision_kwargs(model_source, model_revision)
        options = transformers_pipeline_load_kwargs(load_options).get("model_kwargs", {})
        config = AutoConfig.from_pretrained(model_source, **revision, trust_remote_code=True)
        keypoint_kind = None
        if task_type == "video-classification":
            model_cls = AutoModelForVideoClassification
        elif config.model_type == "vitpose":
            from transformers import VitPoseForPoseEstimation

            model_cls = VitPoseForPoseEstimation
            keypoint_kind = "vitpose"
        else:
            from transformers import AutoModelForKeypointDetection

            model_cls = AutoModelForKeypointDetection
            keypoint_kind = "superpoint"
            if config.model_type != "superpoint":
                raise ValueError("keypoint-detection supports SuperPoint and VitPose with transformers 4.57.6; "
                                 f"unsupported model_type={config.model_type!r}")
        processor = AutoImageProcessor.from_pretrained(model_source, **revision, trust_remote_code=True, use_fast=False)
        model = model_cls.from_pretrained(
            model_source, **revision, config=config, trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            **options,
        ).to(device).eval()
        return {"model": model, "processor": processor, "task_type": task_type, "device": device,
                "keypoint_kind": keypoint_kind, "model_revision": model_revision or "main",
                "load_options": dict(load_options or {})}

    @staticmethod
    def _decode_image(value: Any):
        from PIL import Image

        if not isinstance(value, str) or not value:
            raise ValueError("image_base64 must contain a base64-encoded image")
        try:
            with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as source:
                return source.convert("RGB")
        except (ValueError, OSError) as exc:
            raise ValueError("invalid base64 image payload") from exc

    @staticmethod
    def _params(raw_input: Dict[str, Any]) -> Dict[str, Any]:
        params = raw_input.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        reserved = set(params) & {"batch_size", "num_workers", "candidate_labels", "num_frames", "boxes"}
        if reserved:
            raise ValueError("unsupported CV params: " + ", ".join(sorted(reserved)))
        return dict(params)

    @staticmethod
    def _scale_metadata(raw_input: Dict[str, Any], image: Any) -> Dict[str, Any]:
        scale = raw_input.get("input_scale", image.width / 224.0)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("input_scale must be a finite positive resolution multiplier")
        if "input_scale" in raw_input:
            side = max(1, int(224 * scale))
            if image.size != (side, side):
                raise ValueError("input_scale does not match the actual image resolution")
        return {"_effective_input_scale": float(scale), "_truncated_by_limit": False,
                "_probe_reason": "raw image resolution verified; processor may resize or pad internally"}

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        task = model_ctx["task_type"]
        params = self._params(raw_input)
        if task == "video-classification":
            frames = raw_input.get("frames_base64")
            if not isinstance(frames, list) or not frames:
                raise ValueError("video-classification requires non-empty frames_base64")
            expected = getattr(model_ctx["model"].config, "num_frames", None)
            if isinstance(expected, int) and expected > 0 and len(frames) != expected:
                raise ValueError(f"video-classification model requires num_frames={expected}; received {len(frames)}; "
                                 "set workload num_frames or provide matching video_frames")
            unknown = set(params) - {"top_k", "function_to_apply"}
            if unknown:
                raise ValueError("unsupported video-classification params: " + ", ".join(sorted(unknown)))
            top_k = params.get("top_k", 5)
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer")
            if params.get("function_to_apply", "softmax") not in {"softmax", "sigmoid", "none"}:
                raise ValueError("function_to_apply must be softmax, sigmoid, or none")
            images = [self._decode_image(frame) for frame in frames]
            if len({image.size for image in images}) != 1:
                raise ValueError("all video frames must have the same resolution")
            inputs = model_ctx["processor"](images, return_tensors="pt")
            return {"inputs": inputs, "params": params, **self._scale_metadata(raw_input, images[0])}
        image = self._decode_image(raw_input.get("image_base64"))
        processed = {"image": image, "params": params, **self._scale_metadata(raw_input, image)}
        if task in {"zero-shot-image-classification", "zero-shot-object-detection"}:
            processed["candidate_labels"] = self._candidate_labels(raw_input)
        if task == "keypoint-detection":
            allowed = {"dataset_index"} if model_ctx["keypoint_kind"] == "vitpose" else set()
            if set(params) - allowed:
                raise ValueError("unsupported keypoint-detection params: " + ", ".join(sorted(set(params) - allowed)))
            if model_ctx["keypoint_kind"] == "vitpose":
                boxes = raw_input.get("boxes", [[0, 0, image.width, image.height]])
                if not isinstance(boxes, list) or not boxes:
                    raise ValueError("boxes must contain pixel xywh boxes")
                for box in boxes:
                    if not isinstance(box, list) or len(box) != 4 or any(
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box
                    ):
                        raise ValueError("boxes must contain finite pixel xywh boxes")
                    x, y, width, height = box
                    if min(x, y) < 0 or min(width, height) <= 0 or x + width > image.width or y + height > image.height:
                        raise ValueError("boxes must fit within the image")
                dataset_index = params.get("dataset_index", 0)
                if isinstance(dataset_index, bool) or not isinstance(dataset_index, int) or dataset_index < 0:
                    raise ValueError("dataset_index must be a non-negative integer")
                processed["boxes"] = [boxes]
                processed["inputs"] = model_ctx["processor"](image, boxes=[boxes], return_tensors="pt")
            else:
                processed["inputs"] = model_ctx["processor"](image, return_tensors="pt")
            processed["target_sizes"] = [(image.height, image.width)]
        return processed

    @staticmethod
    def _candidate_labels(raw_input: Dict[str, Any]) -> list[str]:
        labels = raw_input.get("candidate_labels")
        if not isinstance(labels, list) or not labels or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ):
            raise ValueError("candidate_labels must be a non-empty list of non-empty strings")
        return list(labels)

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        task = model_ctx["task_type"]
        params = self._params(processed_input)
        if task in {"video-classification", "keypoint-detection"}:
            import torch

            model = model_ctx["model"]
            inputs = processed_input["inputs"].to(model.device)
            # Preserve integer tensors while matching floating inputs to model dtype.
            inputs = {name: value.to(dtype=model.dtype) if value.is_floating_point() else value
                      for name, value in inputs.items()}
            if model_ctx.get("keypoint_kind") == "vitpose":
                inputs["dataset_index"] = torch.full(
                    (len(processed_input["boxes"][0]),), params.get("dataset_index", 0),
                    dtype=torch.long, device=model.device,
                )
            with torch.inference_mode():
                outputs = model(**inputs)
            return {"outputs": outputs, "params": params,
                    "boxes": processed_input.get("boxes"), "target_sizes": processed_input.get("target_sizes")}
        pipe = model_ctx["pipeline"]
        if task in {"zero-shot-image-classification", "zero-shot-object-detection"}:
            params["candidate_labels"] = self._candidate_labels(processed_input)
        return pipe(processed_input["image"], **params)

    @staticmethod
    def _caption_token_count(pipe: Any, captions: list[str]) -> Optional[int]:
        """Count decoded text tokens, not generation steps or special tokens."""
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            return None
        total = 0
        try:
            for text in captions:
                if callable(getattr(tokenizer, "encode", None)):
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                elif callable(tokenizer):
                    token_ids = tokenizer(text, add_special_tokens=False).get("input_ids")
                else:
                    return None
                if token_ids is None:
                    return None
                total += int(np.asarray(token_ids).size)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None
        return total

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        task_type = model_ctx["task_type"]

        if task_type == "video-classification":
            logits = raw_output["outputs"].logits[0]
            params = raw_output.get("params", {})
            function = params.get("function_to_apply", "softmax")
            if function == "softmax":
                scores = logits.softmax(dim=-1)
            elif function == "sigmoid":
                scores = logits.sigmoid()
            else:
                scores = logits
            values, indices = scores.topk(min(params.get("top_k", 5), scores.shape[-1]))
            labels = getattr(model_ctx["model"].config, "id2label", {})
            classifications = [
                {"label": labels.get(index, str(index)), "score": float(score)}
                for score, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist())
            ]
            return {"task": task_type, "output_type": "classification",
                    "n_results": len(classifications), "classifications": classifications}

        if task_type == "keypoint-detection":
            processor = model_ctx["processor"]
            if model_ctx["keypoint_kind"] == "vitpose":
                grouped = processor.post_process_pose_estimation(raw_output["outputs"], boxes=raw_output["boxes"])
                results = [person for people in grouped for person in people]
            else:
                results = processor.post_process_keypoint_detection(raw_output["outputs"], raw_output["target_sizes"])
            return {"task": task_type, "output_type": "keypoints", "n_results": len(results),
                    "keypoint_count": sum(len(record["keypoints"]) for record in results)}

        if task_type == "image-to-text":
            records = [raw_output] if isinstance(raw_output, dict) else raw_output
            if not isinstance(records, list) or not records or any(
                not isinstance(record, dict)
                or not isinstance(record.get("generated_text"), str)
                for record in records
            ):
                raise ValueError("invalid caption output: expected generated_text strings")
            captions = [record["generated_text"] for record in records]
            return {
                "task": task_type,
                "output_type": "caption",
                "captions": captions,
                "n_results": len(captions),
                "output_length": sum(len(text) for text in captions),
                "output_token_count": self._caption_token_count(
                    model_ctx.get("pipeline"), captions
                ),
            }

        if task_type == "depth-estimation":
            if not isinstance(raw_output, dict) or "predicted_depth" not in raw_output:
                raise ValueError("invalid depth output: expected predicted_depth")
            depth = raw_output["predicted_depth"]
            shape = list(depth.shape) if hasattr(depth, "shape") else list(np.asarray(depth).shape)
            return {"task": task_type, "output_type": "depth", "n_results": 1, "depth_shape": shape}

        if task_type == "mask-generation":
            if not isinstance(raw_output, dict) or "masks" not in raw_output:
                raise ValueError("invalid mask output: expected masks")
            return {"task": task_type, "output_type": "masks", "n_results": len(raw_output["masks"])}

        if task_type == "image-feature-extraction":
            shape = list(raw_output.shape) if hasattr(raw_output, "shape") else list(np.asarray(raw_output).shape)
            return {"task": task_type, "output_type": "features", "n_results": 1, "feature_shape": shape}

        if isinstance(raw_output, list):
            n_results = len(raw_output)
        elif isinstance(raw_output, dict):
            n_results = 1
        else:
            n_results = 1

        output_type = {
            "image-classification": "classification", "zero-shot-image-classification": "classification",
            "image-segmentation": "segmentation", "object-detection": "detection",
            "zero-shot-object-detection": "detection",
        }.get(task_type)
        if output_type is None:
            raise ValueError(f"unsupported CV task_type={task_type!r}")
        return {
            "task": task_type,
            "output_type": output_type,
            "n_results": n_results,
        }


HandlerRegistry.register("cv", "transformers_pipeline", CVHandler)
HandlerRegistry.register("cv", "transformers_model", CVHandler)

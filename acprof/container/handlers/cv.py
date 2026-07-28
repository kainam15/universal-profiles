"""CV task handler - image-classification, object-detection, segmentation, etc."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

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

        pipe = hf_pipeline(
            task=task_type,
            model=model_source,
            **model_revision_kwargs(model_source, model_revision),
            **transformers_pipeline_load_kwargs(load_options),
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        return {
            "pipeline": pipe,
            "task_type": task_type,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        from PIL import Image

        image_b64 = raw_input.get("image_base64", "")
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            # Fallback: create a dummy image
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return {"image": image, "params": raw_input.get("params", {})}

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        pipe = model_ctx["pipeline"]
        image = processed_input["image"]
        return pipe(image)

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        task_type = model_ctx["task_type"]

        if isinstance(raw_output, list):
            n_results = len(raw_output)
        elif isinstance(raw_output, dict):
            n_results = 1
        else:
            n_results = 1

        return {
            "task": task_type,
            "output_type": "classification" if "classification" in task_type else "detection",
            "n_results": n_results,
        }


HandlerRegistry.register("cv", "transformers_pipeline", CVHandler)
HandlerRegistry.register("cv", "transformers_model", CVHandler)

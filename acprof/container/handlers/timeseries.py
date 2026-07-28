"""Time-series handler - chronos forecasting models."""

from __future__ import annotations

from typing import Any, Dict, Optional

from acprof.container.handlers import (
    BaseHandler,
    HandlerRegistry,
    model_revision_kwargs,
    transformers_pipeline_load_kwargs,
)


class ChronosHandler(BaseHandler):
    """Handler for Chronos / ChronosBolt time-series forecasting models."""

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

        chronos_load_options: Dict[str, Any] = {}
        if load_options:
            attention_implementation = load_options.get(
                "attention_implementation"
            )
            if attention_implementation != "eager":
                raise ValueError(
                    "attention_implementation must be 'eager' for compute profiling"
                )
            chronos_load_options["attn_implementation"] = "eager"

        # Try ChronosBolt first (faster), fall back to base Chronos
        try:
            from chronos import ChronosBoltPipeline
            pipeline = ChronosBoltPipeline.from_pretrained(
                model_source,
                **model_revision_kwargs(model_source, model_revision),
                **chronos_load_options,
                device_map=device,
                local_files_only=True,
            )
            pipeline_type = "bolt"
        except Exception:
            from chronos import ChronosPipeline
            pipeline = ChronosPipeline.from_pretrained(
                model_source,
                **model_revision_kwargs(model_source, model_revision),
                **chronos_load_options,
                device_map=device,
                local_files_only=True,
            )
            pipeline_type = "base"

        return {
            "pipeline": pipeline,
            "pipeline_type": pipeline_type,
            "task_type": task_type,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        import torch

        raw = raw_input["context"]
        pred_len = int(raw_input.get("prediction_length", 64))

        context = torch.tensor(raw, dtype=torch.float32)
        if context.ndim == 1:
            context = context.unsqueeze(0)

        return {"context": context, "prediction_length": pred_len}

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        pipeline = model_ctx["pipeline"]
        context = processed_input["context"]
        pred_len = processed_input["prediction_length"]

        device = model_ctx["device"]
        if device != "cpu":
            context = context.to(device)

        forecast = pipeline.predict(context, prediction_length=pred_len)
        return forecast

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        return {
            "task": model_ctx["task_type"],
            "forecast_shape": list(raw_output.shape),
        }


class TimeseriesTransformersHandler(BaseHandler):
    """Fallback handler for time-series models using transformers backend."""

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
        pipe = hf_pipeline(
            task="time-series-forecasting",
            model=model_source,
            **model_revision_kwargs(model_source, model_revision),
            **transformers_pipeline_load_kwargs(load_options),
            device_map=device_map,
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
        return raw_input

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        return model_ctx["pipeline"](processed_input)

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        return {"task": model_ctx["task_type"], "output_type": "forecast"}


HandlerRegistry.register("timeseries", "chronos", ChronosHandler)
HandlerRegistry.register("timeseries", "transformers_pipeline", TimeseriesTransformersHandler)

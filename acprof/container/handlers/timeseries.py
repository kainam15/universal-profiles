"""Time-series handler - chronos forecasting models."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from acprof.container.handlers import (
    BaseHandler,
    model_revision_kwargs,
)


class ChronosHandler(BaseHandler):
    """Use the upstream registry for native Chronos forecasting checkpoints."""

    def load(
        self,
        model_source: str,
        task_type: str,
        backend: str,
        device: str,
        model_revision: str = "main",
        load_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import torch  # noqa: F401 -- 保留所选运行时在加载前的 Torch 依赖检查。

        if task_type != "time-series-forecasting" or backend != "chronos":
            raise ValueError("time-series-forecasting supports Chronos/ChronosBolt with backend='chronos'")
        chronos_load_options: Dict[str, Any] = {}
        if load_options:
            unknown = set(load_options) - {"attention_implementation"}
            if unknown:
                raise ValueError("unsupported Chronos load options: " + ", ".join(sorted(unknown)))
            attention_implementation = load_options.get(
                "attention_implementation"
            )
            if attention_implementation != "eager":
                raise ValueError(
                    "attention_implementation must be 'eager' for compute profiling"
                )
            chronos_load_options["attn_implementation"] = "eager"

        from chronos import BaseChronosPipeline

        pipeline = BaseChronosPipeline.from_pretrained(
            model_source,
            **model_revision_kwargs(model_source, model_revision),
            **chronos_load_options,
            device_map=device,
            local_files_only=True,
        )

        return {
            "pipeline": pipeline,
            "pipeline_type": type(pipeline).__name__,
            "task_type": task_type,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        import torch

        raw = raw_input.get("context")
        if not isinstance(raw, list) or not raw:
            raise ValueError("context must be a non-empty finite sequence or batch of sequences")
        rows = raw if isinstance(raw[0], list) else [raw]
        if any(not isinstance(row, list) or not row for row in rows) or len({len(row) for row in rows}) != 1:
            raise ValueError("context batch must contain non-empty sequences of equal length")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               or abs(value) > 3.4028234663852886e38 for row in rows for value in row):
            raise ValueError("context must contain finite float32 numbers")
        context_limit = self._context_limit(model_ctx)
        if context_limit is not None and len(rows[0]) > context_limit:
            raise ValueError(f"context length exceeds model context limit {context_limit}; reduce input scales to avoid silent truncation")
        pred_len = raw_input.get("prediction_length", 64)
        if (isinstance(pred_len, bool) or not isinstance(pred_len, (int, float))
                or not math.isfinite(pred_len) or pred_len <= 0 or int(pred_len) != pred_len):
            raise ValueError("prediction_length must be a positive integer")
        pred_len = int(pred_len)

        context = torch.tensor(raw, dtype=torch.float32)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        # Native pipelines own batching and device transfer. Chronos 2 pins
        # the incoming CPU tensors in its DataLoader before moving them.

        return {"context": context, "series": [context[index] for index in range(context.shape[0])],
                "prediction_length": pred_len,
                "_effective_input_scale": float(len(rows[0])), "_truncated_by_limit": False,
                "_probe_reason": "validated context length against available Chronos model limit"}

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        pipeline = model_ctx["pipeline"]
        context = processed_input["context"]
        series = processed_input.get("series")
        if series is None:
            series = [context[index] for index in range(context.shape[0])]
        pred_len = processed_input["prediction_length"]

        import torch

        with torch.inference_mode():
            # A list of 1-D series is supported by all native Chronos
            # generations; their batched tensor ranks differ (2-D vs 3-D).
            forecast = pipeline.predict(series, prediction_length=pred_len)
        # Chronos 2 returns one (variates, quantiles, horizon) tensor per
        # series; this workload is univariate. Keep the shared B x Q x H
        # contract without losing a series or flattening its horizon.
        if isinstance(forecast, (list, tuple)):
            if (len(forecast) != context.shape[0] or not forecast or
                    any(len(item.shape) != 3 or item.shape[0] != 1 or
                        item.shape[-1] != pred_len for item in forecast)):
                raise ValueError("Chronos forecast must contain one univariate prediction per input series")
            forecast = torch.cat(forecast, dim=0)
        return forecast

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        return {
            "task": model_ctx["task_type"],
            "forecast_shape": list(raw_output.shape),
        }

    @staticmethod
    def _context_limit(model_ctx: Dict[str, Any]) -> Optional[int]:
        pipeline = model_ctx["pipeline"]
        model = getattr(pipeline, "model", None)
        config = getattr(model, "config", None)
        chronos_config = getattr(model, "chronos_config", None)
        serialized_config = getattr(config, "chronos_config", None)
        candidates = [getattr(pipeline, "model_context_length", None),
                      getattr(config, "context_length", None),
                      getattr(chronos_config, "context_length", None)]
        if isinstance(serialized_config, dict):
            candidates.append(serialized_config.get("context_length"))
        for value in candidates:
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    def get_scale_metadata(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
        limit = self._context_limit(model_ctx)
        return {"context_length": limit, "max_effective_input_scale": limit,
                "input_scale_type": "context_length",
                "reason": "Chronos model context limit; larger raw contexts would be silently truncated"}


class TimeseriesTransformersHandler(BaseHandler):
    """Reject the old placeholder: Transformers has no forecasting pipeline."""

    def load(
        self,
        model_source: str,
        task_type: str,
        backend: str,
        device: str,
        model_revision: str = "main",
        load_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise ValueError(
            "time-series-forecasting currently supports Chronos/ChronosBolt with backend='chronos'; "
            "Transformers has no generic time-series-forecasting pipeline"
        )

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        raise ValueError("use the Chronos handler for time-series-forecasting")

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        raise ValueError("use the Chronos handler for time-series-forecasting")

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        raise ValueError("use the Chronos handler for time-series-forecasting")

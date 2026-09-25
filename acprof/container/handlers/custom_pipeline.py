"""Bridge declared text-output pipelines without executing their whole __call__.

Reference: Transformers Pipeline subclass contract (Apache-2.0). Decode/transfer,
model execution, and output decoding retain the existing measurement phases.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path

from acprof.model_spec import pipeline_task
from acprof.model_transforms import transform_inputs


def load_custom_pipeline(model_source, task_type, device, dtype, spec, attention_options):
    import transformers

    if not Path(model_source).is_dir():
        raise ValueError("custom multimodal pipeline requires a baked local snapshot")
    name = pipeline_task(model_source, task_type)
    pipe = transformers.pipeline(
        task=name, model=model_source, trust_remote_code=True,
        device_map="cpu" if device == "cpu" else "auto", torch_dtype=dtype,
        model_kwargs={"local_files_only": True, **attention_options.get("model_kwargs", {})},
    )
    for method in ("preprocess", "_forward", "postprocess"):
        if not callable(getattr(pipe, method, None)):
            raise ValueError(f"custom multimodal pipeline has no {method} phase")
    pipe.model.eval()
    return {"pipeline": pipe, "model": pipe.model, "processor": getattr(pipe, "processor", pipe),
            "mode": "custom_pipeline", "pipeline_protocol": spec["multimodal"], "model_spec": spec}


def pipeline_inputs(model_ctx, content, media_kwargs):
    values = {"text": next(item["text"] for item in content if item["type"] == "text")}
    for name, plural in (("audio", "audio"), ("image", "images"), ("video", "videos")):
        if plural in media_kwargs:
            values[name] = media_kwargs[plural][0]
    values.update({key: media_kwargs[key] for key in ("sampling_rate", "fps") if key in media_kwargs})
    payload = transform_inputs(model_ctx["pipeline_protocol"]["inputs"], values)
    inputs = model_ctx["pipeline"].preprocess(payload)
    if not isinstance(inputs, Mapping):
        raise ValueError("custom multimodal preprocess must return a tensor mapping")
    return dict(inputs)


def pipeline_forward_kwargs(model_ctx, inputs, params):
    declaration = model_ctx["pipeline_protocol"].get(
        "forward_kwargs", {"max_new_tokens": "$max_new_tokens", "do_sample": "$do_sample"},
    )
    kwargs = {key: params[value[1:]] if isinstance(value, str) and value.startswith("$") else value
              for key, value in declaration.items()}
    # Invalid declarations fail during preprocessing, before a measurement window.
    inspect.signature(model_ctx["pipeline"]._forward).bind(inputs, **kwargs)
    return kwargs


def pipeline_texts(model_ctx, raw_output):
    import torch

    pipe = model_ctx["pipeline"]
    output = pipe._ensure_tensor_on_device(raw_output, device=torch.device("cpu"))
    text = pipe.postprocess(output)
    texts = [text] if isinstance(text, str) else text
    if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str):
        raise ValueError("custom multimodal postprocess must return one text string")
    return texts

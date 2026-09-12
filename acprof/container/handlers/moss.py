"""MOSS 自定义模型适配，复用固定 snapshot 的官方模型及 processor。

Reference: OpenMOSS/MOSS-Transcribe-Diarize (Apache-2.0), inference_utils.py.
音频解码、特征计算和张量传输在 preprocess，generate 在 predict，文本解码在 postprocess。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from acprof.container.handlers import HandlerRegistry, transformers_pipeline_load_kwargs
from acprof.runtime_profiles import MOSS_ADAPTER
from acprof.container.handlers.multimodal import MultimodalHandler


class MossTranscribeDiarizeHandler(MultimodalHandler):
    def load(
        self, model_source: str, task_type: str, backend: str, device: str,
        model_revision: str = "main", load_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if task_type != "audio-text-to-text" or backend not in {"transformers_model", "transformers_pipeline"}:
            raise ValueError("MOSS requires audio-text-to-text with a Transformers backend")
        source = Path(model_source)
        if not source.is_dir():
            raise ValueError("MOSS adapter requires an offline, baked model snapshot")
        config = json.loads((source / "config.json").read_text())
        if config.get("model_type") != "moss_transcribe_diarize":
            raise ValueError("MOSS adapter received an incompatible model_type")
        attention = transformers_pipeline_load_kwargs(load_options).get("model_kwargs", {})
        attention.setdefault("attn_implementation", "sdpa")
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
            dtype=dtype, **attention,
        ).to(device).eval()
        return {
            "task_type": task_type, "device": device, "model": model, "processor": processor,
            "model_type": "moss_transcribe_diarize", "mode": "generate", "audio_chunking": True,
            "model_revision": model_revision, "load_options": dict(load_options or {}),
            "attention_implementation": attention["attn_implementation"],
        }

    def _generation_inputs(self, model_ctx, content, media_kwargs):
        processor = model_ctx["processor"]
        # Official MOSS message order: instruction, then audio placeholder.
        content = sorted(content, key=lambda item: item["type"] != "text")
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        return processor(text=prompt, audio=media_kwargs["audio"], return_tensors="pt")

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        import torch

        if str(model_ctx["device"]).startswith("cuda"):
            with torch.autocast("cuda", dtype=model_ctx["model"].dtype):
                return super().predict(model_ctx, processed_input)
        return super().predict(model_ctx, processed_input)


for _backend in ("transformers_model", "transformers_pipeline"):
    HandlerRegistry.register_adapter(MOSS_ADAPTER, "multimodal", _backend, MossTranscribeDiarizeHandler)

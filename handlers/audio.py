"""Audio task handler - speech recognition, audio classification, etc."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from handlers import BaseHandler, HandlerRegistry


class AudioHandler(BaseHandler):

    def load(self, model_id: str, task_type: str, backend: str, device: str) -> Dict[str, Any]:
        import torch
        from transformers import pipeline as hf_pipeline

        device_map = device if device == "cpu" else "auto"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32

        pipe = hf_pipeline(
            task=task_type,
            model=model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        return {
            "pipeline": pipe,
            "task_type": task_type,
            "device": device,
        }

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        audio_samples = raw_input.get("audio_samples", [])
        sample_rate = raw_input.get("sample_rate", 16000)

        if audio_samples:
            audio_array = np.array(audio_samples, dtype=np.float32)
        else:
            # Fallback: 1 second of silence
            audio_array = np.zeros(sample_rate, dtype=np.float32)

        return {
            "audio": audio_array,
            "sample_rate": sample_rate,
            "params": raw_input.get("params", {}),
        }

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        pipe = model_ctx["pipeline"]
        audio = processed_input["audio"]
        sample_rate = processed_input["sample_rate"]

        # transformers pipeline accepts dict with raw audio
        audio_input = {"raw": audio, "sampling_rate": sample_rate}
        return pipe(audio_input)

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        task_type = model_ctx["task_type"]

        if isinstance(raw_output, dict):
            text = raw_output.get("text", "")
            return {
                "task": task_type,
                "output_type": "transcription",
                "output_length": len(text),
            }
        elif isinstance(raw_output, list):
            return {
                "task": task_type,
                "output_type": "classification",
                "n_results": len(raw_output),
            }
        return {
            "task": task_type,
            "output_type": "unknown",
        }


HandlerRegistry.register("audio", "transformers_pipeline", AudioHandler)
HandlerRegistry.register("audio", "transformers_model", AudioHandler)

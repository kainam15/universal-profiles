"""NLP task handler - text-generation, classification, fill-mask, QA, etc."""

from __future__ import annotations

from typing import Any, Dict

from handlers import BaseHandler, HandlerRegistry

# Tasks that generate text output
_GENERATIVE_TASKS = {
    "text-generation", "text2text-generation", "summarization",
    "translation", "conversational",
}


class NLPHandler(BaseHandler):

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
        task_type = model_ctx["task_type"]
        text = raw_input.get("text", "")
        params = raw_input.get("params", {})

        if task_type == "question-answering":
            return {
                "question": raw_input.get("question", text),
                "context": raw_input.get("context", text),
                "params": params,
            }
        return {"text": text, "params": params}

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        pipe = model_ctx["pipeline"]
        task_type = model_ctx["task_type"]
        params = processed_input.get("params", {})

        if task_type == "question-answering":
            return pipe(
                question=processed_input["question"],
                context=processed_input["context"],
            )
        elif task_type in _GENERATIVE_TASKS:
            max_new_tokens = params.get("max_new_tokens", 64)
            return pipe(
                processed_input["text"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            return pipe(processed_input["text"])

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
            "output_type": "text" if task_type in _GENERATIVE_TASKS else "label",
            "n_results": n_results,
        }


HandlerRegistry.register("nlp", "transformers_pipeline", NLPHandler)
HandlerRegistry.register("nlp", "transformers_model", NLPHandler)

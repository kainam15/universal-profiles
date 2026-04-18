"""NLP task handler - text-generation, classification, fill-mask, QA, etc."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from handlers import BaseHandler, HandlerRegistry

# Tasks that generate text output
_GENERATIVE_TASKS = {
    "text-generation", "text2text-generation", "summarization",
    "translation", "conversational",
}


class NLPHandler(BaseHandler):

    @staticmethod
    def _get_tokenizer_max_length(tokenizer: Any) -> Optional[int]:
        max_len = getattr(tokenizer, "model_max_length", None)
        if not max_len or max_len >= 1_000_000:
            return None
        try:
            return int(max_len)
        except (TypeError, ValueError):
            return None

    def _truncate_single_text(
        self,
        pipe: Any,
        text: str,
        task_type: str,
    ) -> Tuple[str, Optional[int]]:
        tokenizer = getattr(pipe, "tokenizer", None)
        if not tokenizer:
            return text, None

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        max_len = self._get_tokenizer_max_length(tokenizer)
        if max_len is not None:
            special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
            available = max(1, max_len - int(special_tokens))
            if len(token_ids) > available:
                if task_type == "fill-mask":
                    mask_token = tokenizer.mask_token or "[MASK]"
                    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
                    try:
                        mask_pos = token_ids.index(mask_token_id)
                    except ValueError:
                        mask_pos = len(token_ids) // 2
                    if mask_pos < available:
                        token_ids = token_ids[:available]
                    else:
                        token_ids = token_ids[:available - 1] + [mask_token_id]
                    text = tokenizer.decode(token_ids, skip_special_tokens=False)
                    if mask_token not in text:
                        text = text + " " + mask_token
                else:
                    token_ids = token_ids[:available]
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
                token_ids = tokenizer.encode(text, add_special_tokens=False)

        return text, len(token_ids)

    def _truncate_qa_context(
        self,
        pipe: Any,
        question: str,
        context: str,
    ) -> Tuple[str, Optional[int]]:
        tokenizer = getattr(pipe, "tokenizer", None)
        if not tokenizer:
            return context, None

        context_ids = tokenizer.encode(context, add_special_tokens=False)
        max_len = self._get_tokenizer_max_length(tokenizer)
        if max_len is not None:
            question_ids = tokenizer.encode(question, add_special_tokens=False)
            special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
            available = max(0, max_len - len(question_ids) - int(special_tokens))
            if len(context_ids) > available:
                context_ids = context_ids[:available]
                context = tokenizer.decode(context_ids, skip_special_tokens=True)
                context_ids = tokenizer.encode(context, add_special_tokens=False)

        return context, len(context_ids)

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
        pipe = model_ctx["pipeline"]

        if task_type == "question-answering":
            question = raw_input.get("question", text)
            context = raw_input.get("context", text)
            context, effective_input_scale = self._truncate_qa_context(pipe, question, context)
            return {
                "question": question,
                "context": context,
                "params": params,
                "_effective_input_scale": effective_input_scale,
            }

        text, effective_input_scale = self._truncate_single_text(pipe, text, task_type)
        return {
            "text": text,
            "params": params,
            "_effective_input_scale": effective_input_scale,
        }

    # Pipelines whose _sanitize_parameters does not accept truncation directly.
    # For these, pass truncation via tokenizer_kwargs instead.
    _TRUNCATION_VIA_KWARGS_TASKS = {"fill-mask"}

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        pipe = model_ctx["pipeline"]
        task_type = model_ctx["task_type"]
        params = processed_input.get("params", {})

        if task_type == "question-answering":
            return pipe(
                question=processed_input["question"],
                context=processed_input["context"],
                truncation="only_second",
            )
        elif task_type in _GENERATIVE_TASKS:
            max_new_tokens = params.get("max_new_tokens", 64)
            return pipe(
                processed_input["text"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                truncation=True,
            )
        elif task_type in self._TRUNCATION_VIA_KWARGS_TASKS:
            return pipe(processed_input["text"], tokenizer_kwargs={"truncation": True})
        else:
            return pipe(processed_input["text"], truncation=True)

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

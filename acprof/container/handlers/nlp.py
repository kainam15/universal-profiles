"""NLP task handler - text-generation, classification, fill-mask, QA, etc."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from acprof.container.handlers import (
    BaseHandler,
    HandlerRegistry,
    model_revision_kwargs,
    transformers_pipeline_load_kwargs,
)

# Tasks that generate text output
_GENERATIVE_TASKS = {
    "text-generation", "text2text-generation", "summarization",
    "translation", "conversational",
}


class NLPHandler(BaseHandler):

    # Host workloads use one model-independent placeholder.  The container has
    # the loaded tokenizer, so it is the only layer that can safely translate
    # that placeholder to the model's native mask token.
    _FILL_MASK_PLACEHOLDER = "[MASK]"

    @staticmethod
    def _get_tokenizer_max_length(tokenizer: Any) -> Optional[int]:
        max_len = getattr(tokenizer, "model_max_length", None)
        try:
            value = int(max_len)
            return value if 0 < value < 1_000_000 else None
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _get_model_max_length(cls, pipe: Any) -> Optional[int]:
        limits = [cls._get_tokenizer_max_length(getattr(pipe, "tokenizer", None))]
        # Sentence Transformers may configure a shorter sequence limit than
        # the underlying tokenizer; CrossEncoder exposes max_length instead.
        for name in ("max_seq_length", "max_length"):
            value = getattr(pipe, name, None)
            if isinstance(value, (int, float)) and 0 < value < 1_000_000:
                limits.append(int(value))
        config = getattr(getattr(pipe, "model", None), "config", None)
        for name in ("max_position_embeddings", "n_positions"):
            value = getattr(config, name, None)
            if isinstance(value, (int, float)) and 0 < value < 1_000_000:
                limits.append(int(value))
        known = [value for value in limits if value is not None]
        return min(known) if known else None

    @staticmethod
    def _generation_token_reserve(pipe: Any, task_type: str, params: Dict[str, Any]) -> int:
        config = getattr(getattr(pipe, "model", None), "config", None)
        if task_type != "text-generation" or getattr(config, "is_encoder_decoder", False) is True:
            return 0
        value = params.get("max_new_tokens", 64)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        return value

    @staticmethod
    def _single_text_budget(max_len: int, special_tokens: int, reserved_tokens: int) -> int:
        available = max_len - special_tokens - reserved_tokens
        if reserved_tokens and available < 1:
            raise ValueError("max_new_tokens leaves no input token budget within the model context limit")
        return max(1, available)

    @classmethod
    def _prepare_fill_mask_input(
        cls,
        tokenizer: Any,
        text: str,
    ) -> Tuple[str, list, Any]:
        raw_mask_token = getattr(tokenizer, "mask_token", None)
        if raw_mask_token is None or not str(raw_mask_token):
            raise ValueError(
                "fill-mask task requires a tokenizer with a configured mask_token"
            )
        mask_token = str(raw_mask_token)

        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert_tokens_to_ids):
                mask_token_id = convert_tokens_to_ids(mask_token)
        if mask_token_id is None:
            raise ValueError(
                "fill-mask task requires a tokenizer with a configured mask_token_id"
            )

        if mask_token != cls._FILL_MASK_PLACEHOLDER:
            text = text.replace(cls._FILL_MASK_PLACEHOLDER, mask_token)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if mask_token_id not in token_ids:
            separator = "" if not text or text[-1].isspace() else " "
            text = f"{text}{separator}{mask_token}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)

        if mask_token_id not in token_ids:
            raise ValueError(
                "tokenizer could not encode its configured mask_token "
                f"{mask_token!r} as mask_token_id={mask_token_id!r}"
            )

        return text, token_ids, mask_token_id

    def _truncate_single_text(
        self,
        pipe: Any,
        text: str,
        task_type: str,
        reserved_tokens: int = 0,
    ) -> Tuple[str, Optional[int], Optional[bool], str]:
        tokenizer = getattr(pipe, "tokenizer", None)
        if not tokenizer:
            return text, None, None, "tokenizer_missing"

        mask_token_id = None
        if task_type == "fill-mask":
            text, token_ids, mask_token_id = self._prepare_fill_mask_input(
                tokenizer,
                text,
            )
        else:
            token_ids = tokenizer.encode(text, add_special_tokens=False)

        max_len = self._get_model_max_length(pipe)
        if max_len is None:
            return text, len(token_ids), None, "max_length_unknown"

        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        available = self._single_text_budget(max_len, int(special_tokens), reserved_tokens)
        truncated_by_limit = len(token_ids) > available
        if truncated_by_limit:
            if task_type == "fill-mask":
                mask_pos = token_ids.index(mask_token_id)
                if mask_pos < available:
                    token_ids = token_ids[:available]
                else:
                    token_ids = token_ids[:available - 1] + [mask_token_id]
                text = tokenizer.decode(token_ids, skip_special_tokens=False)
            else:
                token_ids = token_ids[:available]
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if task_type == "fill-mask" and mask_token_id not in token_ids:
                raise ValueError(
                    "tokenizer decode/encode round trip removed the configured "
                    "mask token while truncating fill-mask input"
                )

        reason = (
            f"truncated_to_model_limit(max_length={max_len},available={available})"
            if truncated_by_limit
            else f"within_model_limit(max_length={max_len},available={available})"
        )
        if reserved_tokens:
            reason += f"; reserved_output_tokens={reserved_tokens}"
        return text, len(token_ids), truncated_by_limit, reason

    def _truncate_qa_context(
        self,
        pipe: Any,
        question: str,
        context: str,
    ) -> Tuple[str, Optional[int], Optional[bool], str]:
        tokenizer = getattr(pipe, "tokenizer", None)
        if not tokenizer:
            return context, None, None, "tokenizer_missing"

        context_ids = tokenizer.encode(context, add_special_tokens=False)
        max_len = self._get_model_max_length(pipe)
        if max_len is None:
            return context, len(context_ids), None, "max_length_unknown"

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
        available = max(0, max_len - len(question_ids) - int(special_tokens))
        truncated_by_limit = len(context_ids) > available
        if truncated_by_limit:
            context_ids = context_ids[:available]
            context = tokenizer.decode(context_ids, skip_special_tokens=True)
            context_ids = tokenizer.encode(context, add_special_tokens=False)

        reason = (
            f"truncated_context_to_model_limit(max_length={max_len},available={available})"
            if truncated_by_limit
            else f"within_model_limit(max_length={max_len},available={available})"
        )
        return context, len(context_ids), truncated_by_limit, reason

    def _max_effective_single_text_length(
        self, pipe: Any, reserved_tokens: int = 0,
    ) -> Tuple[Optional[int], str]:
        tokenizer = getattr(pipe, "tokenizer", None)
        if not tokenizer:
            return None, "tokenizer_missing"

        max_len = self._get_model_max_length(pipe)
        if max_len is None:
            return None, "max_length_unknown"

        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        available = self._single_text_budget(max_len, int(special_tokens), reserved_tokens)
        reason = f"max_effective_single_text_length(max_length={max_len},available={available})"
        if reserved_tokens:
            reason += f"; reserved_output_tokens={reserved_tokens}"
        return available, reason

    def _max_effective_qa_context_length(
        self,
        pipe: Any,
        question: str,
    ) -> Tuple[Optional[int], str]:
        tokenizer = getattr(pipe, "tokenizer", None)
        if not tokenizer:
            return None, "tokenizer_missing"

        max_len = self._get_model_max_length(pipe)
        if max_len is None:
            return None, "max_length_unknown"

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
        available = max(0, max_len - len(question_ids) - int(special_tokens))
        return available, (
            "max_effective_qa_context_length("
            f"max_length={max_len},question_tokens={len(question_ids)},available={available})"
        )

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
        device_map = device if device == "cpu" else "auto"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        pipeline_options = transformers_pipeline_load_kwargs(load_options)
        if task_type in {"sentence-similarity", "text-ranking"}:
            if task_type == "sentence-similarity":
                from sentence_transformers import SentenceTransformer as Encoder
            else:
                from sentence_transformers import CrossEncoder as Encoder
            pipe = Encoder(
                model_source,
                **model_revision_kwargs(model_source, model_revision),
                device=device,
                local_files_only=os.path.isdir(model_source),
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch_dtype,
                              **pipeline_options.get("model_kwargs", {})},
            )
            if task_type == "text-ranking" and pipe.num_labels != 1:
                raise ValueError("text-ranking requires a single relevance score (num_labels=1)")
        else:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                task=task_type,
                model=model_source,
                **model_revision_kwargs(model_source, model_revision),
                **pipeline_options,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            if task_type == "text-generation":
                tokenizer = getattr(pipe, "tokenizer", None)
                if tokenizer is not None:
                    tokenizer.padding_side = "left"
                    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                        pipe.generation_config.pad_token_id = tokenizer.eos_token_id
        return {
            "pipeline": pipe,
            "task_type": task_type,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }

    @staticmethod
    def _batch_size(raw_input: Dict[str, Any]) -> int:
        value = raw_input.get("batch_size", 1)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < 1:
            raise ValueError("batch_size must be a positive integer")
        return int(value)

    @staticmethod
    def _string_list(raw_input: Dict[str, Any], field: str) -> list[str]:
        values = raw_input.get(field)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(f"{field} must be a nonempty list of nonempty strings")
        return list(values)

    def _zero_shot_hypotheses(self, pipe: Any, raw_input: Dict[str, Any]) -> Tuple[list[str], str, str]:
        labels = self._string_list(raw_input, "candidate_labels")
        template = raw_input.get("hypothesis_template", "This example is {}.")
        if not isinstance(template, str) or "{}" not in template:
            raise ValueError("hypothesis_template must contain a {} placeholder")
        hypotheses = [template.format(label) for label in labels]
        tokenizer = getattr(pipe, "tokenizer", None)
        longest = max(hypotheses, key=lambda text: len(
            tokenizer.encode(text, add_special_tokens=False)
        ) if tokenizer else len(text))
        return labels, template, longest

    def _preprocess_table(self, pipe: Any, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        import pandas as pd

        table = raw_input.get("table")
        if not isinstance(table, dict) or not table or not all(
            isinstance(name, str) and isinstance(column, list)
            for name, column in table.items()
        ):
            raise ValueError("table must be a nonempty mapping of column names to lists")
        row_counts = {len(column) for column in table.values()}
        if len(row_counts) != 1 or 0 in row_counts:
            raise ValueError("table columns must contain the same number of rows, greater than zero")
        if not all(isinstance(value, str) for column in table.values() for value in column):
            raise ValueError("table cells must be strings for the table QA tokenizer")
        query = raw_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("table-question-answering requires a nonempty query")
        frame = pd.DataFrame(table)
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("table-question-answering requires a table tokenizer")
        encoded = tokenizer(frame, query, truncation=False)
        token_ids = encoded["input_ids"]
        if len(token_ids) and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        max_len = self._get_model_max_length(pipe)
        if max_len is not None and len(token_ids) > max_len:
            raise ValueError(f"table exceeds model token limit ({len(token_ids)} > {max_len}); reduce table_rows")
        return {"table": frame, "query": query, "_effective_input_scale": next(iter(row_counts)),
                "_truncated_by_limit": False, "_probe_reason": "complete_table_rows_no_truncation"}

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        task_type = model_ctx["task_type"]
        text = raw_input.get("text", "")
        params = raw_input.get("params", {})
        pipe = model_ctx["pipeline"]
        common = {"params": params, "batch_size": self._batch_size(raw_input)}

        if task_type == "table-question-answering":
            return {**common, **self._preprocess_table(pipe, raw_input)}

        if task_type in {"sentence-similarity", "text-ranking"}:
            documents = self._string_list(raw_input, "documents")
            query = raw_input.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{task_type} requires a nonempty query")
            checked_query, _, query_truncated, _ = self._truncate_single_text(pipe, query, task_type)
            if query_truncated:
                raise ValueError("query exceeds model token limit")
            results = [self._truncate_qa_context(pipe, checked_query, doc)
                       if task_type == "text-ranking" else
                       self._truncate_single_text(pipe, doc, task_type) for doc in documents]
            if any(item[1] == 0 for item in results):
                raise ValueError("query leaves no model token budget for documents")
            scales = [item[1] for item in results if item[1] is not None]
            return {**common, "query": checked_query, "documents": [item[0] for item in results],
                    "_effective_input_scale": max(scales) if scales else None,
                    "_truncated_by_limit": any(item[2] for item in results),
                    "_probe_reason": "per_candidate_tokens; " + results[0][3]}

        if task_type == "zero-shot-classification":
            labels, template, longest = self._zero_shot_hypotheses(pipe, raw_input)
            text, scale, truncated, reason = self._truncate_qa_context(pipe, longest, text)
            if scale == 0:
                raise ValueError("candidate hypotheses leave no model token budget for text")
            return {**common, "text": text, "candidate_labels": labels,
                    "hypothesis_template": template, "_effective_input_scale": scale,
                    "_truncated_by_limit": truncated, "_probe_reason": reason}

        if task_type == "question-answering":
            question = raw_input.get("question", text)
            context = raw_input.get("context", text)
            context, effective_input_scale, truncated_by_limit, reason = self._truncate_qa_context(
                pipe,
                question,
                context,
            )
            return {
                **common,
                "question": question,
                "context": context,
                "params": params,
                "_effective_input_scale": effective_input_scale,
                "_truncated_by_limit": truncated_by_limit,
                "_probe_reason": reason,
            }

        text, effective_input_scale, truncated_by_limit, reason = self._truncate_single_text(
            pipe,
            text,
            task_type,
            reserved_tokens=self._generation_token_reserve(pipe, task_type, params),
        )
        return {
            **common,
            "text": text,
            "params": params,
            "_effective_input_scale": effective_input_scale,
            "_truncated_by_limit": truncated_by_limit,
            "_probe_reason": reason,
        }

    def get_scale_metadata(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
        task_type = model_ctx["task_type"]
        pipe = model_ctx["pipeline"]

        if task_type == "table-question-answering":
            return {"input_scale_type": "table_rows", "max_effective_input_scale": None,
                    "reason": "table capacity depends on tokenized cells and query; no row truncation"}

        if task_type in {"text-ranking", "zero-shot-classification"}:
            fixed_text = (raw_input.get("query", "") if task_type == "text-ranking"
                          else self._zero_shot_hypotheses(pipe, raw_input)[2])
            available, reason = self._max_effective_qa_context_length(pipe, fixed_text)
            return {"input_scale_type": "seq_length", "max_effective_input_scale": available,
                    "reason": reason}

        if task_type == "question-answering":
            question = raw_input.get("question", raw_input.get("text", ""))
            available, reason = self._max_effective_qa_context_length(pipe, question)
            return {
                "input_scale_type": "seq_length",
                "max_effective_input_scale": available,
                "reason": reason,
            }

        available, reason = self._max_effective_single_text_length(
            pipe, reserved_tokens=self._generation_token_reserve(pipe, task_type, raw_input.get("params", {})),
        )
        return {
            "input_scale_type": "seq_length",
            "max_effective_input_scale": available,
            "reason": reason,
        }

    # Pipelines whose _sanitize_parameters does not accept truncation directly.
    # For these, pass truncation via tokenizer_kwargs instead.
    _TRUNCATION_VIA_KWARGS_TASKS = {"fill-mask"}

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        pipe = model_ctx["pipeline"]
        task_type = model_ctx["task_type"]
        params = processed_input.get("params", {})
        batch_size = processed_input.get("batch_size", 1)

        if task_type == "table-question-answering":
            # TAPAS batches queries natively for one table. The generic
            # Pipeline batch iterator cannot unpack its nested model_inputs
            # output, so keep the outer iterator at one table per item.
            query = processed_input["query"]
            return pipe(table=processed_input["table"],
                        query=[query] * batch_size if batch_size > 1 else query,
                        batch_size=1, truncation=False, sequential=False)

        if task_type == "sentence-similarity":
            documents = processed_input["documents"]
            sample = [processed_input["query"], *documents]
            embeddings = pipe.encode(sample * batch_size, batch_size=len(sample) * batch_size,
                                     show_progress_bar=False, convert_to_tensor=True, prompt="")
            return [pipe.similarity(embeddings[offset:offset + 1],
                                    embeddings[offset + 1:offset + len(sample)])[0]
                    for offset in range(0, len(sample) * batch_size, len(sample))]

        if task_type == "text-ranking":
            documents = processed_input["documents"]
            pairs = [[processed_input["query"], document] for document in documents] * batch_size
            scores = pipe.predict(pairs, batch_size=len(pairs), show_progress_bar=False,
                                  convert_to_numpy=True)
            count = len(documents)
            return [sorted([{"corpus_id": index, "score": float(scores[offset + index])}
                            for index in range(count)], key=lambda item: item["score"], reverse=True)
                    for offset in range(0, len(pairs), count)]

        if task_type == "question-answering":
            if batch_size > 1:
                sample = {"question": processed_input["question"], "context": processed_input["context"]}
                return pipe([dict(sample) for _ in range(batch_size)], batch_size=batch_size)
            return pipe(
                question=processed_input["question"],
                context=processed_input["context"],
            )
        text = processed_input["text"]
        inputs = [text] * batch_size if batch_size > 1 else text
        batch_kwargs = {"batch_size": batch_size} if batch_size > 1 else {}
        if task_type == "zero-shot-classification":
            return pipe(inputs, candidate_labels=processed_input["candidate_labels"],
                        hypothesis_template=processed_input["hypothesis_template"],
                        multi_label=bool(params.get("multi_label", False)), **batch_kwargs)
        if task_type in _GENERATIVE_TASKS:
            max_new_tokens = params.get("max_new_tokens", 64)
            return pipe(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                truncation=True,
                **batch_kwargs,
            )
        elif task_type in self._TRUNCATION_VIA_KWARGS_TASKS:
            return pipe(inputs, tokenizer_kwargs={"truncation": True}, **batch_kwargs)
        elif task_type == "token-classification":
            # Transformers 4.x token classification truncates internally and
            # its sanitizer rejects a caller-supplied truncation keyword.
            return pipe(inputs, **batch_kwargs)
        else:
            return pipe(inputs, truncation=True, **batch_kwargs)

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
            "output_type": ("text" if task_type in _GENERATIVE_TASKS
                            else {"table-question-answering": "table_answer",
                                  "sentence-similarity": "similarity", "text-ranking": "ranking",
                                  "feature-extraction": "embedding"}.get(task_type, "label")),
            "n_results": n_results,
        }


HandlerRegistry.register("nlp", "transformers_pipeline", NLPHandler)
HandlerRegistry.register("nlp", "transformers_model", NLPHandler)
HandlerRegistry.register("nlp", "sentence_transformers", NLPHandler)
HandlerRegistry.register("nlp", "cross_encoder", NLPHandler)

"""NLP workload generator - synthetic text at target token lengths."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from acprof.workloads import WorkloadGenerator, register_generator

# Fixed corpus for deterministic text generation
_CORPUS_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a journey of a thousand miles begins with a single step "
    "to be or not to be that is the question "
    "all that glitters is not gold "
    "the early bird catches the worm "
    "knowledge is power and power is knowledge "
    "actions speak louder than words in every situation "
    "time and tide wait for no man in this world "
).split()

DEFAULT_QA_QUESTION = "What is the main topic?"
DEFAULT_PAIR_QUERY = "What is the main topic?"
DEFAULT_CANDIDATE_LABELS = ["science", "sports", "politics"]
DEFAULT_HYPOTHESIS_TEMPLATE = "This example is {}."


class NLPWorkloadGenerator(WorkloadGenerator):
    """Generate synthetic text of target sequence length (in approximate tokens)."""

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        super().__init__(model_id, task_type, batch_size)
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

    def _generate_text_from_word_count(self, word_count: int) -> str:
        n_words = max(1, int(word_count))
        words = []
        for i in range(n_words):
            words.append(_CORPUS_WORDS[i % len(_CORPUS_WORDS)])
        return " ".join(words)

    def generate_for_word_count(self, word_count: int) -> Dict[str, Any]:
        text = self._generate_text_from_word_count(word_count)

        payload: Dict[str, Any] = {
            "text": text, "params": {}, "batch_size": self.batch_size,
        }

        if self.task_type == "fill-mask":
            # Portable placeholder: the container replaces it with the loaded
            # tokenizer's native mask token (for example, RoBERTa uses <mask>).
            words = text.split()
            if len(words) > 1:
                mask_pos = len(words) // 2
                words[mask_pos] = "[MASK]"
            else:
                words.append("[MASK]")
            payload["text"] = " ".join(words)
        elif self.task_type in ("text-generation", "text2text-generation",
                              "summarization", "translation", "conversational"):
            payload["params"]["max_new_tokens"] = 64
        elif self.task_type == "question-answering":
            payload["question"] = DEFAULT_QA_QUESTION
            payload["context"] = text
        elif self.task_type == "zero-shot-classification":
            payload["candidate_labels"] = list(DEFAULT_CANDIDATE_LABELS)
            payload["hypothesis_template"] = DEFAULT_HYPOTHESIS_TEMPLATE
        elif self.task_type in {"sentence-similarity", "text-ranking"}:
            payload["query"] = DEFAULT_PAIR_QUERY
            # Two candidates are part of one request sample. Length is varied,
            # while candidate count and the query stay fixed across scales.
            payload["documents"] = [text, " ".join(reversed(text.split()))]

        return payload

    def generate(self, scale_value: float) -> Dict[str, Any]:
        if self.task_type == "table-question-answering":
            if not math.isfinite(scale_value) or scale_value < 1 or int(scale_value) != scale_value:
                raise ValueError("table_rows must be a positive integer")
            rows = int(scale_value)
            return {
                "table": {
                    "name": [f"item {index + 1}" for index in range(rows)],
                    "value": [str(index + 1) for index in range(rows)],
                },
                "query": "What is the value of item 1?",
                "params": {},
                "batch_size": self.batch_size,
                "input_scale_type": "table_rows",
            }
        return self.generate_for_word_count(int(scale_value))

    def scale_label(self, scale_value: float) -> str:
        if self.task_type == "table-question-answering":
            return f"rows{int(scale_value)}"
        return f"seq{int(scale_value)}"

    def default_input_scales(self) -> Optional[List[float]]:
        if self.task_type == "table-question-answering":
            return [1, 2, 4, 8, 16, 32]
        return None

    def plan_metadata(self) -> Dict[str, Any]:
        if self.task_type == "table-question-answering":
            return {"input_scale_type": "table_rows", "columns": ["name", "value"]}
        if self.task_type in {"sentence-similarity", "text-ranking"}:
            return {"input_scale_type": "seq_length", "scale_scope": "per_candidate",
                    "candidate_count": 2, "query": DEFAULT_PAIR_QUERY}
        if self.task_type == "zero-shot-classification":
            return {"candidate_labels": list(DEFAULT_CANDIDATE_LABELS),
                    "hypothesis_template": DEFAULT_HYPOTHESIS_TEMPLATE}
        return {}


register_generator("nlp", NLPWorkloadGenerator)

"""NLP workload generator - synthetic text at target token lengths."""

from __future__ import annotations

import random
from typing import Any, Dict

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

BASE_SEED = 12345
DEFAULT_QA_QUESTION = "What is the main topic?"


class NLPWorkloadGenerator(WorkloadGenerator):
    """Generate synthetic text of target sequence length (in approximate tokens)."""

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        super().__init__(model_id, task_type, batch_size)
        self._rng = random.Random(BASE_SEED)

    def _generate_text_from_word_count(self, word_count: int) -> str:
        n_words = max(1, int(word_count))
        words = []
        for i in range(n_words):
            words.append(_CORPUS_WORDS[i % len(_CORPUS_WORDS)])
        return " ".join(words)

    def generate_for_word_count(self, word_count: int) -> Dict[str, Any]:
        text = self._generate_text_from_word_count(word_count)

        payload: Dict[str, Any] = {"text": text, "params": {}}

        if self.task_type == "fill-mask":
            # fill-mask requires a [MASK] token in the input
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

        return payload

    def generate(self, scale_value: float) -> Dict[str, Any]:
        return self.generate_for_word_count(int(scale_value))

    def scale_label(self, scale_value: float) -> str:
        return f"seq{int(scale_value)}"


register_generator("nlp", NLPWorkloadGenerator)

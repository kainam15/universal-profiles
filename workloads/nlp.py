"""NLP workload generator - synthetic text at target token lengths."""

from __future__ import annotations

import random
from typing import Any, Dict

from workloads import WorkloadGenerator, register_generator

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


class NLPWorkloadGenerator(WorkloadGenerator):
    """Generate synthetic text of target sequence length (in approximate tokens)."""

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        super().__init__(model_id, task_type, batch_size)
        self._rng = random.Random(BASE_SEED)

    def _generate_text(self, target_tokens: int) -> str:
        # Approximate: 1 word ~ 1.3 tokens, so generate more words than needed
        n_words = int(target_tokens * 1.0)
        words = []
        for i in range(n_words):
            words.append(_CORPUS_WORDS[i % len(_CORPUS_WORDS)])
        return " ".join(words)

    def generate(self, scale_value: float) -> Dict[str, Any]:
        seq_len = int(scale_value)
        text = self._generate_text(seq_len)

        payload: Dict[str, Any] = {"text": text, "params": {}}

        if self.task_type in ("text-generation", "text2text-generation",
                              "summarization", "translation", "conversational"):
            payload["params"]["max_new_tokens"] = 64
        elif self.task_type == "question-answering":
            payload["question"] = "What is the main topic?"
            payload["context"] = text

        return payload

    def scale_label(self, scale_value: float) -> str:
        return f"seq{int(scale_value)}"


register_generator("nlp", NLPWorkloadGenerator)

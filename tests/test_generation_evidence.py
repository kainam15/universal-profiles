"""Actual generated IDs remain distinct from budgets and decoded token counts."""

from types import SimpleNamespace
from collections import UserDict
import unittest
from unittest.mock import Mock

import numpy as np

from acprof.container.handlers.nlp import NLPHandler
from acprof.container.handlers.multimodal import MultimodalHandler


class FakeGenerationPipeline:
    def __init__(self, sequences, *, seq2seq=False):
        self.sequences = sequences
        self.seq2seq = seq2seq
        self.generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0, decoder_start_token_id=0)
        self.model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=seq2seq))
        self.calls = []

    def preprocess(self, _text, **_kwargs):
        return UserDict({"input_ids": np.array([[10, 11, 12]]), "attention_mask": np.array([[1, 1, 1]])})

    def postprocess(self, _output, **_kwargs):
        return [{"generated_text": "decoded words"}]

    def __call__(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        records = []
        for index, text in enumerate(inputs if isinstance(inputs, list) else [inputs]):
            prepared = self.preprocess(text)
            sequence = self.sequences[index]
            outputs = {"output_ids" if self.seq2seq else "generated_sequence": np.array([[sequence]])}
            if not self.seq2seq:
                outputs["input_ids"] = prepared["input_ids"]
            records.append(self.postprocess(outputs))
        return records if isinstance(inputs, list) else records[0]


class GenerationEvidenceTests(unittest.TestCase):
    def _nlp(self, pipe, *, budget=8, batch=1):
        handler = NLPHandler()
        ctx = {"pipeline": pipe, "task_type": "text2text-generation" if pipe.seq2seq else "text-generation"}
        handler._observe_generation(ctx)
        raw = handler.predict(ctx, {"text": "input", "batch_size": batch, "params": {"max_new_tokens": budget}})
        return raw, handler.postprocess(ctx, raw)

    def test_causal_eos_and_padding_counts_ids_without_changing_pipeline_output(self):
        pipe = FakeGenerationPipeline([[10, 11, 12, 71, 2, 0, 0]])
        raw, response = self._nlp(pipe)
        self.assertEqual(raw, [{"generated_text": "decoded words"}])
        self.assertEqual(response["actual_output_tokens"], 2)
        self.assertEqual(response["actual_output_tokens_per_sequence"], [2])
        self.assertEqual(response["actual_input_tokens"], 3)
        self.assertEqual(response["stop_reason"], "eos")
        self.assertEqual(pipe.calls, [("input", {"max_new_tokens": 8, "do_sample": False, "truncation": True})])

    def test_batch_counts_each_sequence_and_distinguishes_mixed_stop_reasons(self):
        pipe = FakeGenerationPipeline([[10, 11, 12, 2, 0], [10, 11, 12, 71, 72]])
        _, response = self._nlp(pipe, budget=2, batch=2)
        self.assertEqual(response["actual_output_tokens_per_sequence"], [1, 2])
        self.assertEqual(response["actual_output_tokens"], 3)
        self.assertEqual(response["actual_input_tokens"], 6)
        self.assertIsNone(response["stop_reason"])
        self.assertEqual(response["stop_reason_per_sequence"], ["eos", "length"])

    def test_partial_batch_token_evidence_is_not_reported_as_complete_work(self):
        pipe = FakeGenerationPipeline([[10, 11, 12, 71, 2], None])
        _, response = self._nlp(pipe, budget=8, batch=2)
        self.assertEqual(response["actual_input_tokens"], 6)
        self.assertIsNone(response["actual_output_tokens"])
        self.assertIsNone(response["actual_output_tokens_per_sequence"])
        self.assertIsNone(response["stop_reason"])

    def test_seq2seq_excludes_decoder_start_and_does_not_use_budget_as_actual(self):
        pipe = FakeGenerationPipeline([[0, 71, 2, 0]], seq2seq=True)
        _, response = self._nlp(pipe, budget=99)
        self.assertEqual(response["actual_output_tokens"], 2)
        self.assertEqual(response["actual_input_tokens"], 3)
        self.assertEqual(response["stop_reason"], "eos")

    def test_decoder_start_equal_eos_is_not_an_immediate_generated_stop(self):
        pipe = FakeGenerationPipeline([[2, 71, 72]], seq2seq=True)
        pipe.generation_config.decoder_start_token_id = 2
        _, response = self._nlp(pipe, budget=2)
        self.assertEqual(response["actual_output_tokens"], 2)
        self.assertEqual(response["stop_reason"], "length")

    def test_unknown_seq2seq_prefix_keeps_actual_output_unavailable(self):
        pipe = FakeGenerationPipeline([[0, 71, 2]], seq2seq=True)
        pipe.generation_config.decoder_start_token_id = None
        _, response = self._nlp(pipe, budget=8)
        self.assertIsNone(response["actual_output_tokens"])
        self.assertIsNone(response["stop_reason"])

    def test_observer_absence_is_unavailable_and_no_stale_request_evidence(self):
        handler = NLPHandler()
        ctx = {"pipeline": lambda *_a, **_kw: [{"generated_text": "answer"}], "task_type": "text-generation"}
        handler._observe_generation(ctx)
        response = handler.postprocess(ctx, handler.predict(ctx, {"text": "input", "params": {}}))
        self.assertIsNone(response["actual_output_tokens"])
        self.assertIsNone(response["actual_input_tokens"])
        pipe = FakeGenerationPipeline([[10, 11, 12, 71, 2]])
        ctx = {"pipeline": pipe, "task_type": "text-generation"}
        handler._observe_generation(ctx)
        handler.predict(ctx, {"text": "input", "params": {}})
        pipe.sequences = [[10, 11, 12, 72]]
        response = handler.postprocess(ctx, handler.predict(ctx, {"text": "input", "params": {"max_new_tokens": 8}}))
        self.assertEqual(response["actual_output_tokens"], 1)
        self.assertIsNone(response["stop_reason"])

    def test_multimodal_preserves_decoded_token_metric_and_counts_raw_generated_ids(self):
        processor = Mock()
        processor.batch_decode.return_value = ["one_word"]
        processor.tokenizer.encode.return_value = [99]
        model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False),
                                generation_config=SimpleNamespace(eos_token_id=2, decoder_start_token_id=0))
        ctx = {"task_type": "image-text-to-text", "mode": "generate", "model": model, "processor": processor}
        raw = {"generated": np.array([[10, 11, 12, 71, 72, 2, 0]]), "prompt_length": 3,
               "actual_input_tokens": 3, "max_new_tokens": 8}
        response = MultimodalHandler().postprocess(ctx, raw)
        self.assertEqual(response["output_token_count"], 1)
        self.assertEqual(response["actual_output_tokens"], 3)
        self.assertEqual(response["actual_input_tokens"], 3)
        self.assertEqual(response["stop_reason"], "eos")

    def test_multimodal_seq2seq_decoder_start_is_not_generated_work(self):
        processor = Mock()
        processor.batch_decode.return_value = ["answer"]
        processor.tokenizer.encode.return_value = [99]
        model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True),
                                generation_config=SimpleNamespace(eos_token_id=2, decoder_start_token_id=0))
        ctx = {"task_type": "image-text-to-text", "mode": "generate", "model": model, "processor": processor}
        response = MultimodalHandler().postprocess(ctx, {"generated": np.array([[0, 71, 72]]),
                    "prompt_length": 20, "actual_input_tokens": 20, "max_new_tokens": 2})
        self.assertEqual(response["actual_output_tokens"], 2)
        self.assertEqual(response["stop_reason"], "length")


if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from acprof.container.handlers.nlp import NLPHandler


class FakeTokenizer:
    def __init__(
        self,
        mask_token,
        *,
        mask_token_id=99,
        model_max_length=16,
        special_tokens=2,
    ):
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self._encoded_mask_token_id = (
            99 if mask_token_id is None else mask_token_id
        )
        self.model_max_length = model_max_length
        self._special_tokens = special_tokens
        self._next_token_id = 100
        self._token_to_id = {}
        self._id_to_token = {}

    def convert_tokens_to_ids(self, token):
        if token == self.mask_token:
            return self._encoded_mask_token_id
        return self._id_for_token(token)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [
            self._encoded_mask_token_id
            if token == self.mask_token
            else self._id_for_token(token)
            for token in text.split()
        ]

    def decode(self, token_ids, skip_special_tokens):
        tokens = []
        for token_id in token_ids:
            if token_id == self._encoded_mask_token_id:
                if not skip_special_tokens:
                    tokens.append(self.mask_token)
            else:
                tokens.append(self._id_to_token[token_id])
        return " ".join(tokens)

    def num_special_tokens_to_add(self, pair):
        del pair
        return self._special_tokens

    def _id_for_token(self, token):
        if token not in self._token_to_id:
            token_id = self._next_token_id
            self._next_token_id += 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


class NLPHandlerFillMaskTests(unittest.TestCase):
    def setUp(self):
        self.handler = NLPHandler()

    def _preprocess(self, tokenizer, text, task_type="fill-mask"):
        model_ctx = {
            "task_type": task_type,
            "pipeline": SimpleNamespace(tokenizer=tokenizer),
        }
        return self.handler.preprocess(
            model_ctx,
            {"text": text, "params": {}},
        )

    def test_keeps_bert_native_mask_token(self):
        processed = self._preprocess(
            FakeTokenizer("[MASK]"),
            "the [MASK] token",
        )

        self.assertEqual(processed["text"], "the [MASK] token")
        self.assertEqual(processed["_effective_input_scale"], 3)
        self.assertFalse(processed["_truncated_by_limit"])

    def test_translates_portable_placeholder_to_roberta_mask_token(self):
        processed = self._preprocess(
            FakeTokenizer("<mask>"),
            "the [MASK] token",
        )

        self.assertEqual(processed["text"], "the <mask> token")
        self.assertEqual(processed["_effective_input_scale"], 3)
        self.assertFalse(processed["_truncated_by_limit"])

    def test_uses_arbitrary_tokenizer_mask_token_without_model_name_rules(self):
        tokenizer = FakeTokenizer("<custom-mask>", mask_token_id=None)

        processed = self._preprocess(
            tokenizer,
            "the [MASK] token",
        )

        self.assertEqual(processed["text"], "the <custom-mask> token")
        self.assertIn(99, tokenizer.encode(processed["text"]))

    def test_appends_native_mask_when_input_has_no_placeholder(self):
        processed = self._preprocess(
            FakeTokenizer("<mask>"),
            "the token",
        )

        self.assertEqual(processed["text"], "the token <mask>")
        self.assertEqual(processed["_effective_input_scale"], 3)

    def test_preserves_native_mask_when_truncating_past_its_position(self):
        tokenizer = FakeTokenizer("<mask>", model_max_length=6)

        processed = self._preprocess(
            tokenizer,
            "one two three four five [MASK]",
        )

        self.assertEqual(processed["text"], "one two three <mask>")
        self.assertEqual(processed["_effective_input_scale"], 4)
        self.assertTrue(processed["_truncated_by_limit"])
        self.assertIn(
            tokenizer.mask_token_id,
            tokenizer.encode(processed["text"]),
        )

    def test_does_not_normalize_mask_placeholder_for_other_tasks(self):
        processed = self._preprocess(
            FakeTokenizer("<mask>"),
            "the [MASK] token",
            task_type="text-classification",
        )

        self.assertEqual(processed["text"], "the [MASK] token")

    def test_rejects_fill_mask_tokenizer_without_mask_token(self):
        tokenizer = FakeTokenizer(None, mask_token_id=None)

        with self.assertRaisesRegex(
            ValueError,
            "tokenizer with a configured mask_token",
        ):
            self._preprocess(tokenizer, "the [MASK] token")


class NLPTaskCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.handler = NLPHandler()

    def context(self, task, *, limit=16, **kwargs):
        pipe = Mock()
        pipe.tokenizer = FakeTokenizer("[MASK]", model_max_length=limit)
        pipe.max_seq_length = None
        pipe.max_length = None
        for name, value in kwargs.items():
            setattr(pipe, name, value)
        return {"task_type": task, "pipeline": pipe}

    def test_zero_shot_reserves_tokens_for_longest_candidate_hypothesis(self):
        ctx = self.context("zero-shot-classification", limit=12)
        processed = self.handler.preprocess(ctx, {
            "text": "one two three four five six seven eight nine",
            "candidate_labels": ["news", "world politics"],
            "hypothesis_template": "This is {}.",
        })
        self.assertEqual(processed["_effective_input_scale"], 6)
        self.assertTrue(processed["_truncated_by_limit"])
        self.handler.predict(ctx, processed)
        self.assertEqual(ctx["pipeline"].call_args.kwargs["candidate_labels"],
                         ["news", "world politics"])
        self.assertNotIn("truncation", ctx["pipeline"].call_args.kwargs)

    def test_zero_shot_requires_candidate_labels(self):
        with self.assertRaisesRegex(ValueError, "candidate_labels"):
            self.handler.preprocess(self.context("zero-shot-classification"),
                                    {"text": "one two"})

    def test_sentence_similarity_uses_encoder_limit_and_real_candidates(self):
        ctx = self.context("sentence-similarity", max_seq_length=6)
        ctx["pipeline"].encode.return_value = np.eye(3)
        ctx["pipeline"].similarity.return_value = np.array([[0.2, 0.9]])
        processed = self.handler.preprocess(ctx, {
            "query": "one two", "documents": ["one two three four five"] * 2,
        })
        self.assertEqual(processed["documents"], ["one two three four"] * 2)
        self.assertEqual(processed["_effective_input_scale"], 4)
        self.assertTrue(processed["_truncated_by_limit"])
        output = self.handler.predict(ctx, processed)
        np.testing.assert_equal(output[0], [0.2, 0.9])
        args, kwargs = ctx["pipeline"].encode.call_args
        self.assertEqual(args[0], ["one two", "one two three four", "one two three four"])
        self.assertFalse(kwargs["show_progress_bar"])

    def test_ranking_preserves_query_budget_and_orders_document_scores(self):
        ctx = self.context("text-ranking", limit=10, max_length=8)
        ctx["pipeline"].predict.return_value = np.array([0.1, 0.8, 0.6, 0.2])
        processed = self.handler.preprocess(ctx, {
            "query": "one two", "documents": ["one two three four five"] * 2,
            "batch_size": 2,
        })
        self.assertEqual(processed["_effective_input_scale"], 4)
        output = self.handler.predict(ctx, processed)
        self.assertEqual([item[0]["corpus_id"] for item in output], [1, 0])
        self.assertEqual(len(ctx["pipeline"].predict.call_args.args[0]), 4)
        self.assertEqual(ctx["pipeline"].predict.call_args.kwargs["batch_size"], 4)

    def test_pair_task_metadata_reserves_query_and_honors_encoder_limit(self):
        ctx = self.context("text-ranking", limit=20, max_length=10)
        metadata = self.handler.get_scale_metadata(ctx, {"query": "one two three"})
        self.assertEqual(metadata["max_effective_input_scale"], 5)

    def test_pair_tasks_reject_absent_candidates(self):
        for task in ("sentence-similarity", "text-ranking"):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "documents"):
                self.handler.preprocess(self.context(task), {"query": "a"})

    def test_token_classification_calls_supported_pipeline_signature_and_batches(self):
        ctx = self.context("token-classification")
        processed = self.handler.preprocess(ctx, {"text": "one two", "batch_size": 3})
        self.handler.predict(ctx, processed)
        args, kwargs = ctx["pipeline"].call_args
        self.assertEqual(args[0], ["one two"] * 3)
        self.assertEqual(kwargs["batch_size"], 3)
        self.assertNotIn("truncation", kwargs)

    def test_question_answering_batches_complete_question_context_pairs(self):
        ctx = self.context("question-answering")
        processed = self.handler.preprocess(ctx, {
            "question": "who", "context": "one two", "batch_size": 2,
        })
        self.handler.predict(ctx, processed)
        self.assertEqual(ctx["pipeline"].call_args.args[0],
                         [{"question": "who", "context": "one two"}] * 2)

    def test_table_qa_passes_rectangular_table_and_preserves_row_scale(self):
        ctx = self.context("table-question-answering", limit=128)
        ctx["pipeline"].tokenizer = Mock(model_max_length=128)
        ctx["pipeline"].tokenizer.return_value = {"input_ids": [1] * 14}
        table = {"name": ["A", "B"], "value": ["1", "2"]}
        processed = self.handler.preprocess(ctx, {"table": table, "query": "which name", "batch_size": 2})
        self.assertEqual(processed["_effective_input_scale"], 2)
        self.assertFalse(processed["_truncated_by_limit"])
        self.handler.predict(ctx, processed)
        _, kwargs = ctx["pipeline"].call_args
        self.assertEqual(kwargs["table"].to_dict(orient="list"), table)
        self.assertEqual(kwargs["query"], ["which name", "which name"])
        self.assertEqual(kwargs["batch_size"], 1)
        self.assertIs(kwargs["truncation"], False)

    def test_table_qa_rejects_ragged_columns_and_never_silently_drops_rows(self):
        ctx = self.context("table-question-answering", limit=10)
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            self.handler.preprocess(ctx, {"table": {"a": ["1"], "b": []}, "query": "q"})
        ctx["pipeline"].tokenizer = Mock(model_max_length=10)
        ctx["pipeline"].tokenizer.return_value = {"input_ids": [1] * 11}
        with self.assertRaisesRegex(ValueError, "table.*model.*limit"):
            self.handler.preprocess(ctx, {"table": {"a": ["1"]}, "query": "q"})

    def test_sentence_transformer_load_preserves_revision_and_profiler_options(self):
        constructor = Mock(return_value=SimpleNamespace())
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(float32="fp32", float16="fp16"),
            "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
            "transformers": SimpleNamespace(pipeline=Mock()),
        }):
            self.handler.load("org/model", "sentence-similarity", "sentence_transformers",
                              "cpu", model_revision="fixed-sha",
                              load_options={"attention_implementation": "eager"})
        self.assertEqual(constructor.call_args.args, ("org/model",))
        self.assertEqual(constructor.call_args.kwargs["revision"], "fixed-sha")
        self.assertEqual(constructor.call_args.kwargs["model_kwargs"]["attn_implementation"], "eager")

    def test_cross_encoder_local_load_is_offline_and_rejects_multiclass_ranker(self):
        constructor = Mock(return_value=SimpleNamespace(num_labels=3))
        with tempfile.TemporaryDirectory() as local, patch.dict("sys.modules", {
            "torch": SimpleNamespace(float32="fp32", float16="fp16"),
            "sentence_transformers": SimpleNamespace(CrossEncoder=constructor),
            "transformers": SimpleNamespace(pipeline=Mock()),
        }):
            with self.assertRaisesRegex(ValueError, "single.*score|num_labels"):
                self.handler.load(local, "text-ranking", "cross_encoder", "cpu", "fixed-sha")
        self.assertTrue(constructor.call_args.kwargs["local_files_only"])
        self.assertNotIn("revision", constructor.call_args.kwargs)

    def test_causal_generation_load_configures_padding_for_real_batches(self):
        tokenizer = SimpleNamespace(pad_token_id=None, eos_token="</s>", eos_token_id=2,
                                    pad_token=None, padding_side="right")
        pipe = SimpleNamespace(tokenizer=tokenizer, generation_config=SimpleNamespace(pad_token_id=None))
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(float32="fp32", float16="fp16"),
            "transformers": SimpleNamespace(pipeline=Mock(return_value=pipe)),
        }):
            self.handler.load("org/model", "text-generation", "transformers_pipeline", "cpu")
        self.assertEqual(tokenizer.pad_token, "</s>")
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(pipe.generation_config.pad_token_id, 2)

    def test_decoder_generation_reserves_output_with_architecture_context_limit(self):
        ctx = self.context("text-generation", limit=100)
        ctx["pipeline"].model = SimpleNamespace(config=SimpleNamespace(
            is_encoder_decoder=False, max_position_embeddings=12,
        ))
        raw_input = {"text": "one two three four five six seven eight nine",
                     "params": {"max_new_tokens": 4}}
        metadata = self.handler.get_scale_metadata(ctx, raw_input)
        processed = self.handler.preprocess(ctx, raw_input)
        self.assertEqual(metadata["max_effective_input_scale"], 6)
        self.assertEqual(processed["_effective_input_scale"], 6)
        self.assertTrue(processed["_truncated_by_limit"])
        self.assertIn("reserved_output_tokens=4", processed["_probe_reason"])

    def test_encoder_decoder_input_limit_is_independent_of_output_tokens(self):
        ctx = self.context("summarization", limit=12)
        ctx["pipeline"].model = SimpleNamespace(config=SimpleNamespace(
            is_encoder_decoder=True, max_position_embeddings=12,
        ))
        raw_input = {"text": "one two three four five six seven eight nine",
                     "params": {"max_new_tokens": 4}}
        self.assertEqual(self.handler.get_scale_metadata(ctx, raw_input)["max_effective_input_scale"], 10)
        processed = self.handler.preprocess(ctx, raw_input)
        self.assertEqual(processed["_effective_input_scale"], 9)
        self.assertFalse(processed["_truncated_by_limit"])

    def test_generation_rejects_output_budget_that_fills_entire_context(self):
        ctx = self.context("text-generation", limit=8)
        with self.assertRaisesRegex(ValueError, "max_new_tokens.*input token budget"):
            self.handler.get_scale_metadata(ctx, {"params": {"max_new_tokens": 8}})


if __name__ == "__main__":
    unittest.main()

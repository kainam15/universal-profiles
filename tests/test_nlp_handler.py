import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()

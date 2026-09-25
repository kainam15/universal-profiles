"""Bounded declarative input adapters preserve media and reject executable DSLs."""
import copy
import unittest

from acprof.container.handlers.custom_pipeline import pipeline_inputs
from acprof.model_spec import validate_model_spec


def spec(inputs):
    return {"schema_version": 1, "format": "transformers-pipeline", "task": "audio-text-to-text",
            "pipeline_task": "chat-audio", "multimodal": {"inputs": inputs,
            "forward_kwargs": {"max_new_tokens": "$max_new_tokens", "do_sample": False}}}


class ModelTransformTests(unittest.TestCase):
    def test_chat_template_reaches_preprocess_with_original_media(self):
        declaration = spec({"turns": {"template": [{"role": "user", "content": {"from": "text"}}]},
                            "audio": {"from": "audio"}, "rate": "sampling_rate",
                            "mode": {"literal": "transcribe"}})
        validate_model_spec(declaration)
        class Pipeline:
            def preprocess(self, payload):
                return payload
        audio = object()
        result = pipeline_inputs({"pipeline": Pipeline(), "pipeline_protocol": declaration["multimodal"]},
                                 [{"type": "text", "text": "你好"}], {"audio": [audio], "sampling_rate": 16000})
        self.assertEqual(result["turns"], [{"role": "user", "content": "你好"}])
        self.assertIs(result["audio"], audio)
        self.assertEqual(result["rate"], 16000)
        self.assertEqual(result["mode"], "transcribe")
        result["turns"][0]["role"] = "changed"
        self.assertEqual(declaration["multimodal"]["inputs"]["turns"]["template"][0]["role"], "user")

    def test_invalid_or_unbounded_transforms_are_rejected(self):
        nested = {"from": "text"}
        for _ in range(12):
            nested = [nested]
        bad = [{"python": "import os"}, {"from": "os.environ"}, {"from": "text", "extra": 1},
               {"literal": float("nan")}, {"template": nested}, {"template": ["x"] * 300},
               {"template": {"content": {"from": "missing"}}}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                validate_model_spec(spec({"prompt": value, "audio": "audio", "rate": "sampling_rate"}))

    def test_literal_cannot_substitute_for_required_text(self):
        with self.assertRaisesRegex(ValueError, "text"):
            validate_model_spec(spec({"turns": {"literal": [{"content": "text"}]},
                                      "audio": "audio", "rate": "sampling_rate"}))

    def test_old_rename_mapping_remains_valid(self):
        original = spec({"prompt": "text", "audio": "audio", "rate": "sampling_rate"})
        self.assertEqual(validate_model_spec(copy.deepcopy(original)), original)

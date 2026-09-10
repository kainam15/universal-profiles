import sys
import types
import unittest
from unittest.mock import Mock, patch

from acprof.container.handlers.cv import CVHandler
from acprof.host.detect import TaskInfo
from acprof.host.orchestrator import _model_io_formats
from acprof.host.task_support import require_task_support
from acprof.workloads.cv import CVWorkloadGenerator


class CVCaptionTests(unittest.TestCase):
    def setUp(self):
        self.handler = CVHandler()
        self.tokenizer = Mock()
        self.tokenizer.encode.side_effect = lambda text, **kw: {
            "a cat": [10, 20], "蓝色的猫": [30, 31, 32], "": [],
        }[text]
        self.pipe = Mock(tokenizer=self.tokenizer)
        self.ctx = {"pipeline": self.pipe, "task_type": "image-to-text"}

    def test_caption_text_and_counts_describe_all_returned_sequences(self):
        result = self.handler.postprocess(self.ctx, [
            {"generated_text": "a cat"}, {"generated_text": "蓝色的猫"},
        ])
        self.assertEqual(result, {
            "task": "image-to-text", "output_type": "caption",
            "captions": ["a cat", "蓝色的猫"], "n_results": 2,
            "output_length": 9, "output_token_count": 5,
        })
        self.tokenizer.encode.assert_any_call("a cat", add_special_tokens=False)

    def test_empty_generated_text_is_a_valid_zero_length_caption(self):
        result = self.handler.postprocess(self.ctx, {"generated_text": ""})
        self.assertEqual(result["captions"], [""])
        self.assertEqual(result["output_length"], 0)
        self.assertEqual(result["output_token_count"], 0)

    def test_unavailable_token_count_is_null_without_losing_caption(self):
        for tokenizer in (None, Mock(encode=Mock(side_effect=ValueError("no tokenizer")))):
            with self.subTest(tokenizer=tokenizer):
                self.pipe.tokenizer = tokenizer
                result = self.handler.postprocess(self.ctx, [{"generated_text": "a cat"}])
                self.assertEqual(result["captions"], ["a cat"])
                self.assertEqual(result["output_length"], 5)
                self.assertIsNone(result["output_token_count"])

    def test_callable_tokenizer_and_partial_count_failure(self):
        self.pipe.tokenizer = lambda text, **kw: {"input_ids": [10, 20]}
        result = self.handler.postprocess(self.ctx, [{"generated_text": "a cat"}])
        self.assertEqual(result["output_token_count"], 2)
        self.pipe.tokenizer = self.tokenizer
        self.tokenizer.encode.side_effect = [[10, 20], ValueError("second caption failed")]
        result = self.handler.postprocess(self.ctx, [
            {"generated_text": "a cat"}, {"generated_text": "蓝色的猫"},
        ])
        self.assertIsNone(result["output_token_count"])
        self.assertEqual(result["output_length"], 9)

    def test_malformed_caption_output_fails_instead_of_reporting_detection(self):
        for output in (None, [], "a cat", [{"label": "cat"}], [{"generated_text": None}]):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "caption"):
                self.handler.postprocess(self.ctx, output)

    def test_generation_parameters_reach_pipeline(self):
        params = {"max_new_tokens": 12, "generate_kwargs": {"do_sample": False}}
        processed = {"image": "decoded-image", "params": params}
        self.handler.predict(self.ctx, processed)
        self.pipe.assert_called_once_with("decoded-image", **params)
        self.assertEqual(processed["params"], params)

    def test_invalid_generation_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "params must be an object"):
            self.handler.predict(self.ctx, {"image": "image", "params": []})

    def test_other_cv_tasks_keep_their_output_and_call_contract(self):
        for task, output_type in (("image-classification", "classification"), ("object-detection", "detection")):
            ctx = {"pipeline": self.pipe, "task_type": task}
            result = self.handler.postprocess(ctx, [{"label": "cat", "score": 0.9}])
            self.assertEqual(result, {"task": task, "output_type": output_type, "n_results": 1})
            self.handler.predict(ctx, {"image": "image", "params": {}})
            self.pipe.assert_called_with("image")

    def test_obsolete_image_error_explains_rebuild_without_hiding_other_errors(self):
        torch = types.ModuleType("torch")
        torch.float16, torch.float32 = "fp16", "fp32"
        transformers = types.ModuleType("transformers")
        transformers.pipeline = Mock(side_effect=KeyError("Unknown task image-to-text, available tasks are []"))
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            with self.assertRaisesRegex(RuntimeError, "4.57.6.*--skip-build"):
                self.handler.load("example/model", "image-to-text", "transformers_pipeline", "cpu")
            transformers.pipeline.side_effect = KeyError("broken_model_config")
            with self.assertRaisesRegex(KeyError, "broken_model_config"):
                self.handler.load("example/model", "image-to-text", "transformers_pipeline", "cpu")

    def test_caption_task_and_output_schema_are_available(self):
        info = TaskInfo(
            model_id="example/caption-model", pipeline_tag="image-to-text",
            task_family="cv", runtime_backend="transformers_pipeline",
            library_name="transformers", model_revision="revision", detection_method="unit",
        )
        require_task_support(info)
        _, output = _model_io_formats(info)
        properties = output["json_schema"]["properties"]
        self.assertEqual(properties["output_type"]["enum"], ["caption"])
        self.assertEqual(properties["captions"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(properties["output_token_count"], {"type": ["integer", "null"]})
        self.assertIn("output_length", output["json_schema"]["required"])

    def test_caption_workload_does_not_silently_ignore_batch_size(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            CVWorkloadGenerator("example/model", "image-to-text", 2)
        generator = CVWorkloadGenerator("example/model", "image-to-text", 1)
        payload = generator.generate(0.5)
        self.assertEqual(payload, generator.generate(0.5))
        self.assertIn("image_base64", payload)


if __name__ == "__main__":
    unittest.main()

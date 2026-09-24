"""Declared multimodal protocols must work independently of checkpoint names."""
import copy
import contextlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from acprof.host.detect import TaskInfo
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.host.runtime_images import model_fingerprint, request_fingerprint
from acprof.model_spec import validate_model_spec
from acprof.container.handlers.multimodal import MultimodalHandler
from test_multimodal_handler import audio_payload


def pipeline_spec():
    return {
        "schema_version": 1,
        "format": "transformers-pipeline",
        "task": "audio-text-to-text",
        "pipeline_task": "listen-and-answer",
        "multimodal": {
            "inputs": {"waveform": "audio", "rate": "sampling_rate", "question": "text"},
            "forward_kwargs": {"limit": "$max_new_tokens", "temperature": 0.0},
        },
        "dependencies": [{
            "repo_id": "example/base", "revision": "b" * 40,
            "allow_patterns": ["*.json", "*.safetensors"],
        }],
    }


def custom_task(spec=None, *, model_id="unseen/speech-model", model_type="unseen_audio"):
    config = {"model_type": model_type, "auto_map": {
        "AutoConfig": "custom_model.AudioConfig", "AutoModel": "custom_model.AudioModel",
    }, "custom_pipelines": {
        "listen-and-answer": {"impl": "custom_pipeline.AudioPipeline", "pt": ["AutoModel"],
                              "type": "multimodal"},
    }}
    return TaskInfo(
        model_id=model_id, model_revision="a" * 40, library_name="transformers",
        pipeline_tag="audio-text-to-text", task_family="multimodal",
        runtime_backend="transformers_pipeline", detection_method="manual",
        model_config=config, repository_files=("config.json", "custom_model.py", "custom_pipeline.py"),
        model_spec=pipeline_spec() if spec is None else spec,
    )


class CustomMultimodalDeclarationTests(unittest.TestCase):
    def test_distinct_unknown_architectures_use_the_declared_shared_protocol(self):
        for name in ("unseen_audio", "another_speech_architecture"):
            with self.subTest(architecture=name):
                task = custom_task(model_id=f"arbitrary/{name}", model_type=name)
                require_task_support(task)
                self.assertEqual(task.model_adapter, "family-default")
                self.assertEqual(task.runtime_profile_id, "custom-multimodal-cu128")
                self.assertEqual(task.model_resolution["interface_kind"], "custom_pipeline")
                self.assertEqual(task.model_resolution["pipeline_task"], "listen-and-answer")
                self.assertEqual(task.model_resolution["status"], "candidate")

    def test_undeclared_custom_architecture_still_fails_before_loading(self):
        with self.assertRaisesRegex(TaskSupportError, "custom|Custom|multimodal"):
            require_task_support(custom_task({}))

    def test_declared_pipeline_rejects_a_native_environment_override(self):
        task = custom_task()
        task.runtime_profile_id = "multimodal-transformers4576"
        with self.assertRaisesRegex(TaskSupportError, "custom-multimodal"):
            require_task_support(task)

    def test_audio_protocol_requires_text_waveform_and_sampling_rate(self):
        for missing in ("question", "waveform", "rate"):
            spec = pipeline_spec()
            del spec["multimodal"]["inputs"][missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "inputs|audio|text|sampling_rate"):
                validate_model_spec(spec)

    def test_unknown_parameter_references_and_unbounded_generation_are_rejected(self):
        for kwargs in ({"limit": "$unknown", "temperature": 0.0}, {"temperature": 0.0},
                       {"limit": "$max_new_tokens", "do_sample": True}):
            spec = pipeline_spec()
            spec["multimodal"]["forward_kwargs"] = kwargs
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "generation|forward_kwargs|parameter"):
                validate_model_spec(spec)

    def test_dependency_revisions_must_be_immutable_and_unique(self):
        for dependencies in (
            [{"repo_id": "example/base", "revision": "main"}],
            [{"repo_id": "../base", "revision": "b" * 40}],
            [{"repo_id": "example/base", "revision": "b" * 40}] * 2,
            [{"repo_id": "example/base", "revision": "b" * 40, "allow_patterns": ["../*"]}],
        ):
            spec = pipeline_spec()
            spec["dependencies"] = dependencies
            with self.subTest(dependencies=dependencies), self.assertRaisesRegex(ValueError, "dependenc"):
                validate_model_spec(spec)

    def test_dependencies_change_weights_identity_but_input_mapping_changes_only_service_identity(self):
        task = custom_task()
        before_weights = model_fingerprint(task, "sha256:" + "c" * 64)
        before_service = request_fingerprint(task)
        changed_inputs = copy.deepcopy(task)
        changed_inputs.model_spec["multimodal"]["inputs"]["prompt"] = changed_inputs.model_spec["multimodal"]["inputs"].pop("question")
        self.assertEqual(model_fingerprint(changed_inputs, "sha256:" + "c" * 64), before_weights)
        self.assertNotEqual(request_fingerprint(changed_inputs), before_service)
        task.model_spec["dependencies"][0]["revision"] = "d" * 40
        self.assertNotEqual(model_fingerprint(task, "sha256:" + "c" * 64), before_weights)


class CustomMultimodalPhaseTests(unittest.TestCase):
    def setUp(self):
        self.handler = MultimodalHandler()
        self.pipeline = Mock()
        self.pipeline.preprocess.return_value = {
            "input_ids": np.array([[1, 2]]), "audio_values": np.ones((1, 80, 2)),
        }
        self.pipeline._forward.side_effect = lambda inputs, **kwargs: inputs.pop("input_ids")
        self.pipeline.postprocess.return_value = "heard speech"
        self.pipeline._ensure_tensor_on_device.side_effect = lambda value, **kwargs: value
        self.pipeline.tokenizer.encode.return_value = [4, 5]
        self.context = {
            "task_type": "audio-text-to-text", "mode": "custom_pipeline", "device": "cpu",
            "pipeline": self.pipeline, "pipeline_protocol": pipeline_spec()["multimodal"],
            "model": SimpleNamespace(device="cpu", config=SimpleNamespace(is_encoder_decoder=False)),
            "processor": SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000)),
        }
        self.payload = {"samples": [{"text": "What is said?", "audio_base64": audio_payload(),
                                     "sampling_rate": 16000}], "params": {"max_new_tokens": 7}}
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext, device=lambda value: value)
        mocked = patch.dict(sys.modules, {"torch": fake_torch})
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_pipeline_phases_map_inputs_preserve_repeated_requests_and_validate_text(self):
        processed = self.handler.preprocess(self.context, self.payload)
        payload = self.pipeline.preprocess.call_args.args[0]
        self.assertEqual(payload["question"], "What is said?")
        self.assertEqual(payload["rate"], 16000)
        self.assertEqual(payload["waveform"].shape, (160,))
        self.assertEqual(processed["_effective_input_scale"], 0.01)
        self.pipeline._forward.assert_not_called()
        self.pipeline.preprocess.reset_mock()
        for _ in range(2):
            raw = self.handler.predict(self.context, processed)
            self.assertIn("input_ids", processed["inputs"])
            self.pipeline.postprocess.assert_not_called()
            result = self.handler.postprocess(self.context, raw)
            self.assertEqual(result["texts"], ["heard speech"])
            self.assertEqual(result["output_token_count"], 2)
            self.assertNotIn("actual_generated_tokens", result)
            self.pipeline.postprocess.reset_mock()
        self.pipeline.preprocess.assert_not_called()
        self.assertEqual(self.pipeline._forward.call_args.kwargs, {"limit": 7, "temperature": 0.0})

    def test_preprocessor_cannot_silently_drop_audio(self):
        self.pipeline.preprocess.return_value.pop("audio_values")
        with self.assertRaisesRegex(ValueError, "discarded.*audio"):
            self.handler.preprocess(self.context, self.payload)
        self.pipeline._forward.assert_not_called()

    def test_custom_output_must_be_a_single_text_result(self):
        for value in ({"unrelated": True}, ["one", "two"], 123):
            self.pipeline.postprocess.return_value = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "text"):
                self.handler.postprocess(self.context, {"pipeline_output": [1, 2]})


if __name__ == "__main__":
    unittest.main()

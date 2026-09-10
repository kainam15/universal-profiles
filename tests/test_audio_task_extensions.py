import contextlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from acprof.container.handlers.audio import AudioHandler
from acprof.workloads.audio import AudioWorkloadGenerator


class TextTokenizer:
    model_max_length = 6

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class AudioTaskExtensionTests(unittest.TestCase):
    def setUp(self):
        self.handler = AudioHandler()

    def test_text_audio_workloads_are_deterministic_text_and_keep_scale_kind(self):
        for task in ("text-to-speech", "text-to-audio"):
            with self.subTest(task=task):
                generator = AudioWorkloadGenerator("model", task, 1)
                payload = generator.generate(4)
                self.assertEqual(payload, generator.generate_for_word_count(4))
                self.assertEqual(len(payload["text"].split()), 4)
                self.assertNotIn("audio_base64", payload)
                self.assertEqual(generator.plan_metadata()["input_scale"]["type"], "seq_length")
                self.assertEqual(generator.scale_label(4), "seq4")

    def test_waveform_tasks_reuse_verified_speech_without_asr_inference_options(self):
        for task in ("audio-classification", "audio-to-audio", "voice-activity-detection"):
            with self.subTest(task=task):
                generator = AudioWorkloadGenerator("model", task, 1)
                payload = generator.generate(1)
                self.assertEqual(payload["sample_rate"], 16000)
                self.assertNotIn("asr_task", payload["params"])
                metadata = generator.plan_metadata()
                self.assertEqual(metadata["pipeline_tag"], task)
                self.assertEqual(metadata["provenance"]["license"], "CC-BY-4.0")
                self.assertEqual(generator.input_metadata(1, payload)["input_num_samples"], 16000)

    def test_text_audio_preprocessing_uses_text_token_count_and_limit(self):
        context = {"task_type": "text-to-speech", "pipeline": SimpleNamespace(tokenizer=TextTokenizer())}
        output = self.handler.preprocess(context, {"text": "one two three four five", "params": {}})
        self.assertEqual(output["text"], "one two three four")
        self.assertEqual(output["_effective_input_scale"], 4)
        self.assertTrue(output["_truncated_by_limit"])
        metadata = self.handler.get_scale_metadata(context, {})
        self.assertEqual(metadata["input_scale_type"], "seq_length")
        self.assertEqual(metadata["max_effective_input_scale"], 4)

    def test_bark_input_limit_uses_semantic_generation_window_without_special_tokens(self):
        tokenizer = TextTokenizer()
        tokenizer.model_max_length = 1024
        context = {
            "task_type": "text-to-speech",
            "pipeline": SimpleNamespace(
                tokenizer=tokenizer,
                model=SimpleNamespace(config=SimpleNamespace(model_type="bark")),
                generation_config=SimpleNamespace(semantic_config={"max_input_semantic_length": 256}),
            ),
        }
        metadata = self.handler.get_scale_metadata(context, {})
        self.assertEqual(metadata["max_effective_input_scale"], 256)
        output = self.handler.preprocess(context, {"text": "word " * 300})
        self.assertEqual(output["_effective_input_scale"], 256)
        self.assertTrue(output["_truncated_by_limit"])

    def test_text_audio_rejects_tokenizer_overrides_that_invalidate_measured_scale(self):
        pipe = Mock()
        context = {"task_type": "text-to-speech", "pipeline": pipe}
        with self.assertRaisesRegex(ValueError, "preprocess_params.*input scale"):
            self.handler.predict(context, {"text": "test", "params": {"pipeline_kwargs": {"preprocess_params": {"max_length": 1}}}})
        pipe.assert_not_called()

    def test_text_audio_predict_passes_generation_controls_and_returns_waveform_metadata(self):
        pipe = Mock(return_value={"audio": np.zeros((1, 24000)), "sampling_rate": 24000})
        context = {"task_type": "text-to-audio", "pipeline": pipe}
        processed = {"text": "calm music", "params": {"pipeline_kwargs": {"generate_kwargs": {"max_new_tokens": 64}}}}
        result = self.handler.postprocess(context, self.handler.predict(context, processed))
        pipe.assert_called_once_with("calm music", generate_kwargs={"max_new_tokens": 64})
        self.assertEqual(result["output_type"], "audio")
        self.assertEqual(result["audio_duration_s"], 1.0)
        self.assertEqual(result["audio_num_samples"], 24000)
        self.assertEqual(result["audio_shape"], [1, 24000])
        self.assertNotIn("audio", result)

    def test_audio_output_rejects_non_finite_or_missing_waveform(self):
        context = {"task_type": "text-to-speech"}
        for result in ({"audio": np.array([np.nan]), "sampling_rate": 24000}, {"sampling_rate": 24000}):
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, "audio"):
                self.handler.postprocess(context, result)

    def test_audio_classification_resamples_in_preprocess_and_preserves_source_duration(self):
        context = {"task_type": "audio-classification", "audio_metadata": {"sampling_rate": 32000}}
        signal = types.ModuleType("scipy.signal")
        signal.resample_poly = Mock(side_effect=lambda value, up, down: np.repeat(value, up))
        with patch.dict(sys.modules, {"scipy": types.ModuleType("scipy"), "scipy.signal": signal}):
            output = self.handler.preprocess(context, {"audio_samples": [0.25] * 160, "sample_rate": 16000})
        self.assertEqual(output["sample_rate"], 32000)
        self.assertEqual(len(output["audio"]), 320)
        self.assertEqual(output["_effective_input_scale"], 0.01)
        self.assertEqual(output["_input_num_samples"], 160)
        self.assertEqual(output["_model_input_num_samples"], 320)
        self.assertEqual(output["_source_sample_rate"], 16000)

    def test_codec_load_uses_explicit_supported_architecture_and_pinned_revision(self):
        transformer = types.ModuleType("transformers")
        transformer.AutoConfig = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(model_type="encodec", audio_channels=1)))
        model = Mock(config=SimpleNamespace(model_type="encodec", sampling_rate=24000, audio_channels=1))
        model.to.return_value = model
        model.eval.return_value = model
        transformer.EncodecModel = SimpleNamespace(from_pretrained=Mock(return_value=model))
        processor = SimpleNamespace(sampling_rate=24000)
        transformer.AutoProcessor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        transformer.pipeline = Mock(side_effect=AssertionError("audio-to-audio is not a Transformers pipeline"))
        torch = types.ModuleType("torch")
        torch.float32 = "float32"
        with patch.dict(sys.modules, {"transformers": transformer, "torch": torch}):
            context = self.handler.load("model", "audio-to-audio", "transformers_model", "cpu", "pinned")
        self.assertIs(context["model"], model)
        self.assertIs(context["processor"], processor)
        self.assertEqual(context["audio_metadata"]["sampling_rate"], 24000)
        self.assertEqual(transformer.EncodecModel.from_pretrained.call_args.kwargs["revision"], "pinned")

    def test_codec_predict_returns_model_waveform(self):
        model = Mock(return_value=SimpleNamespace(audio_values=np.zeros((1, 1, 24))))
        context = {"task_type": "audio-to-audio", "model": model, "audio_metadata": {"sampling_rate": 24000}}
        processed = {"model_inputs": {"input_values": np.ones((1, 1, 24))}, "params": {"pipeline_kwargs": {"bandwidth": 6.0}}}
        torch = types.ModuleType("torch")
        torch.inference_mode = contextlib.nullcontext
        with patch.dict(sys.modules, {"torch": torch}):
            output = self.handler.predict(context, processed)
        result = self.handler.postprocess(context, output)
        self.assertEqual(result["audio_num_samples"], 24)
        self.assertEqual(result["audio_duration_s"], 0.001)
        self.assertEqual(model.call_args.kwargs["bandwidth"], 6.0)

    def test_vad_load_uses_local_jit_and_rejects_cuda(self):
        torch = types.ModuleType("torch")
        model = Mock()
        model.eval.return_value = model
        torch.jit = SimpleNamespace(load=Mock(return_value=model))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silero_vad.jit"
            path.write_bytes(b"fake-model")
            with patch.dict(sys.modules, {"torch": torch}):
                context = self.handler.load(directory, "voice-activity-detection", "torchscript", "cpu")
                self.assertIs(context["model"], model)
                self.assertIs(self.handler.get_scale_metadata(context, {"sample_rate": 16000})["short_form_fixed_padding"], False)
                torch.jit.load.assert_called_once_with(str(path), map_location="cpu")
                with self.assertRaisesRegex(ValueError, "CPU"):
                    self.handler.load(directory, "voice-activity-detection", "torchscript", "cuda")

    def test_vad_postprocessing_joins_adjacent_positive_frames_and_clips_padding(self):
        result = self.handler.postprocess(
            {"task_type": "voice-activity-detection"},
            {"probabilities": [0.1, 0.8, 0.9, 0.2, 0.8], "frame_samples": 512, "sample_rate": 16000, "input_num_samples": 2200, "threshold": 0.5},
        )
        self.assertEqual(result["output_type"], "voice_activity")
        self.assertEqual(result["segments"], [{"start": 0.032, "end": 0.096}, {"start": 0.128, "end": 0.1375}])
        self.assertEqual(result["n_results"], 2)
        self.assertAlmostEqual(result["speech_duration_s"], 0.0735)

    def test_vad_predict_resets_recurrent_state_for_each_request(self):
        model = Mock(side_effect=[0.8, 0.2, 0.8, 0.2])
        context = {"task_type": "voice-activity-detection", "model": model}
        frames = [np.zeros(512), np.zeros(512)]
        processed = {"frames": frames, "audio": np.zeros(900), "params": {}}
        torch = types.ModuleType("torch")
        torch.inference_mode = contextlib.nullcontext
        with patch.dict(sys.modules, {"torch": torch}):
            first = self.handler.predict(context, processed)
            second = self.handler.predict(context, processed)
        self.assertEqual(first["probabilities"], second["probabilities"])
        self.assertEqual(model.reset_states.call_count, 2)
        self.assertEqual(first["input_num_samples"], 900)

    def test_vad_load_rejects_ambiguous_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for subdir in ("v5", "v6"):
                folder = Path(directory) / subdir
                folder.mkdir()
                (folder / "silero_vad.jit").write_bytes(b"model")
            with self.assertRaisesRegex(ValueError, "exactly one.*found 2"):
                self.handler.load(directory, "voice-activity-detection", "torchscript", "cpu")


if __name__ == "__main__":
    unittest.main()

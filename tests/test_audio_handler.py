import base64
import io
import sys
import types
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from acprof.container.handlers.audio import AudioHandler


def wav_base64(
    samples,
    *,
    sample_rate=16000,
    channels=1,
    sample_width=2,
):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        if sample_width == 2:
            payload = np.asarray(samples, dtype="<i2").tobytes()
        elif sample_width == 1:
            payload = np.asarray(samples, dtype=np.uint8).tobytes()
        else:
            raise AssertionError("test helper only supports 8-bit and 16-bit WAV")
        wav_file.writeframes(payload)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RecordingPipeline:
    def __init__(self, *, model_type="whisper", tokenizer=None):
        self.calls = []
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                model_type=model_type,
                num_mel_bins=128,
                max_source_positions=1500,
                max_target_positions=448,
            )
        )
        self.feature_extractor = SimpleNamespace(
            sampling_rate=16000,
            chunk_length=30,
            n_samples=480000,
            nb_max_frames=3000,
            hop_length=160,
        )
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"text": "hello world"}


class AudioHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = AudioHandler()

    def context(self, pipe=None, task_type="automatic-speech-recognition"):
        pipe = pipe or RecordingPipeline()
        return {
            "pipeline": pipe,
            "task_type": task_type,
            "audio_metadata": self.handler._extract_audio_metadata(pipe),
        }

    def valid_request(self, sample_count=16000, **overrides):
        request = {
            "audio_base64": wav_base64(np.arange(sample_count) % 100),
            "audio_format": "wav",
            "sample_rate": 16000,
            "params": {},
        }
        request.update(overrides)
        return request

    def test_load_extracts_whisper_audio_limits(self):
        pipe = RecordingPipeline()
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.pipeline = lambda **kwargs: pipe

        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            context = self.handler.load(
                "openai/whisper-large-v3",
                "automatic-speech-recognition",
                "transformers_pipeline",
                "cpu",
            )

        self.assertEqual(
            context["audio_metadata"],
            {
                "sampling_rate": 16000,
                "max_short_form_duration_s": 30.0,
                "model_input_num_samples": 480000,
                "model_input_frames": 3000,
                "short_form_fixed_padding": True,
                "fixed_frontend_num_samples": 480000,
                "fixed_frontend_num_frames": 3000,
                "frontend_feature_bins": 128,
                "encoder_positions": 1500,
                "decoder_output_token_limit": 448,
                "model_type": "whisper",
            },
        )

    def test_scale_metadata_distinguishes_audio_and_decoder_limits(self):
        metadata = self.handler.get_scale_metadata(self.context(), {})

        self.assertEqual(metadata["input_scale_type"], "duration_s")
        self.assertEqual(metadata["required_sampling_rate"], 16000)
        self.assertEqual(metadata["max_short_form_duration_s"], 30.0)
        self.assertEqual(metadata["max_effective_input_scale"], 30.0)
        self.assertEqual(metadata["model_input_num_samples"], 480000)
        self.assertEqual(metadata["model_input_frames"], 3000)
        self.assertTrue(metadata["short_form_fixed_padding"])
        self.assertEqual(metadata["fixed_frontend_num_samples"], 480000)
        self.assertEqual(metadata["fixed_frontend_num_frames"], 3000)
        self.assertEqual(metadata["frontend_feature_bins"], 128)
        self.assertEqual(metadata["encoder_positions"], 1500)
        self.assertEqual(metadata["decoder_output_token_limit"], 448)
        self.assertEqual(metadata["model_type"], "whisper")
        self.assertIn("output-token limit", metadata["reason"])
        self.assertIn("not an audio input-length limit", metadata["reason"])
        self.assertIn("pads every accepted short-form", metadata["reason"])

    def test_preprocess_decodes_pcm16_mono_wav_and_reports_scale(self):
        samples = np.array([-32768, -1, 0, 16384, 32767], dtype=np.int16)
        processed = self.handler.preprocess(
            self.context(),
            {
                "audio_base64": wav_base64(samples),
                "audio_format": "wav",
                "sample_rate": 16000,
                "params": {"mode": "short_form"},
            },
        )

        np.testing.assert_allclose(
            processed["audio"],
            samples.astype(np.float32) / 32768.0,
        )
        self.assertEqual(processed["sample_rate"], 16000)
        self.assertEqual(processed["_input_num_samples"], 5)
        self.assertEqual(processed["_duration_s"], 5 / 16000)
        self.assertEqual(processed["_effective_input_scale"], 5 / 16000)
        self.assertFalse(processed["_truncated_by_limit"])
        self.assertIn("within", processed["_probe_reason"])

    def test_preprocess_keeps_legacy_float_samples_compatible(self):
        processed = self.handler.preprocess(
            self.context(),
            {
                "audio_samples": [0.0, 0.25, -0.5],
                "sample_rate": 16000,
                "params": {},
            },
        )

        np.testing.assert_array_equal(
            processed["audio"], np.array([0.0, 0.25, -0.5], dtype=np.float32)
        )
        self.assertEqual(processed["_input_num_samples"], 3)

    def test_preprocess_rejects_missing_empty_and_ambiguous_audio(self):
        with self.assertRaisesRegex(ValueError, "missing audio input"):
            self.handler.preprocess(self.context(), {"params": {}})
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            self.handler.preprocess(
                self.context(), {"audio_samples": [], "sample_rate": 16000}
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.handler.preprocess(
                self.context(),
                {
                    "audio_base64": self.valid_request()["audio_base64"],
                    "audio_samples": [0.0],
                    "audio_format": "wav",
                    "sample_rate": 16000,
                },
            )

    def test_preprocess_strictly_validates_base64_wav_contract(self):
        cases = [
            (
                {"audio_base64": "%%%", "audio_format": "wav", "sample_rate": 16000},
                "not valid Base64",
            ),
            (
                {
                    "audio_base64": self.valid_request()["audio_base64"],
                    "audio_format": "flac",
                    "sample_rate": 16000,
                },
                "audio_format must be 'wav'",
            ),
            (
                {
                    "audio_base64": self.valid_request()["audio_base64"],
                    "audio_format": "WAV",
                    "sample_rate": 16000,
                },
                "audio_format must be 'wav'",
            ),
            (
                {
                    "audio_base64": self.valid_request()["audio_base64"],
                    "audio_format": "wav",
                },
                "sample_rate is required",
            ),
            (
                {
                    "audio_base64": wav_base64([0, 1], sample_rate=8000),
                    "audio_format": "wav",
                    "sample_rate": 16000,
                },
                "does not match the WAV header",
            ),
            (
                {
                    "audio_base64": wav_base64([0, 1, 2, 3], channels=2),
                    "audio_format": "wav",
                    "sample_rate": 16000,
                },
                "must be mono",
            ),
            (
                {
                    "audio_base64": wav_base64([0, 1], sample_width=1),
                    "audio_format": "wav",
                    "sample_rate": 16000,
                },
                "signed 16-bit PCM",
            ),
        ]
        for request, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.handler.preprocess(self.context(), request)

    def test_preprocess_rejects_model_sample_rate_mismatch(self):
        with self.assertRaisesRegex(ValueError, "model feature extractor"):
            self.handler.preprocess(
                self.context(),
                {
                    "audio_base64": wav_base64([0, 1], sample_rate=8000),
                    "audio_format": "wav",
                    "sample_rate": 8000,
                },
            )

    def test_probe_diagnoses_over_limit_without_truncating(self):
        processed = self.handler.preprocess(
            self.context(),
            {
                "audio_samples": np.zeros(480001, dtype=np.float32),
                "sample_rate": 16000,
            },
        )

        self.assertEqual(processed["_input_num_samples"], 480001)
        self.assertTrue(processed["_truncated_by_limit"])
        self.assertIn("exceeds", processed["_probe_reason"])
        self.assertEqual(processed["audio"].size, 480001)

    def test_predict_maps_whisper_semantics_and_pipeline_kwargs(self):
        pipe = RecordingPipeline()
        context = self.context(pipe)
        processed = self.handler.preprocess(
            context,
            self.valid_request(
                16000,
                params={
                    "mode": "short_form",
                    "asr_task": "transcribe",
                    "language": "en",
                    "return_timestamps": False,
                    "pipeline_kwargs": {
                        "batch_size": 1,
                        "generate_kwargs": {"max_new_tokens": 64},
                    },
                },
            ),
        )

        self.handler.predict(context, processed)

        args, kwargs = pipe.calls[0]
        self.assertEqual(args[0]["sampling_rate"], 16000)
        np.testing.assert_array_equal(args[0]["raw"], processed["audio"])
        self.assertEqual(kwargs["batch_size"], 1)
        self.assertFalse(kwargs["return_timestamps"])
        self.assertEqual(
            kwargs["generate_kwargs"],
            {"max_new_tokens": 64, "task": "transcribe", "language": "en"},
        )

    def test_predict_rejects_translation_and_timestamp_modes(self):
        context = self.context()
        processed = {
            "audio": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
            "_duration_s": 1.0,
            "params": {"asr_task": "translate"},
        }
        with self.assertRaisesRegex(ValueError, "translation must be profiled"):
            self.handler.predict(context, processed)

        processed["params"] = {"return_timestamps": True}
        with self.assertRaisesRegex(ValueError, "requires return_timestamps=false"):
            self.handler.predict(context, processed)

        processed["params"] = {
            "pipeline_kwargs": {"stride_length_s": 2},
        }
        with self.assertRaisesRegex(ValueError, "chunked long-form setting"):
            self.handler.predict(context, processed)

    def test_predict_rejects_whisper_over_30_seconds(self):
        context = self.context()
        processed = {
            "audio": np.zeros(1, dtype=np.float32),
            "sample_rate": 16000,
            "_duration_s": 30.0001,
            "params": {},
        }

        with self.assertRaisesRegex(ValueError, "limited to 30s"):
            self.handler.predict(context, processed)

    def test_non_whisper_pipeline_does_not_receive_language_or_task(self):
        pipe = RecordingPipeline(model_type="wav2vec2")
        context = self.context(pipe)
        processed = {
            "audio": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
            "_duration_s": 1.0,
            "params": {
                "asr_task": "transcribe",
                "language": "en",
                "pipeline_kwargs": {"top_k": 3},
            },
        }

        self.handler.predict(context, processed)

        _, kwargs = pipe.calls[0]
        self.assertEqual(kwargs, {"top_k": 3})

    def test_postprocess_returns_asr_text_character_and_token_counts(self):
        tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens: [10, 11, 12]
        )
        context = self.context(RecordingPipeline(tokenizer=tokenizer))

        result = self.handler.postprocess(context, {"text": "hello world"})

        self.assertEqual(result["output_type"], "transcription")
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["output_length"], 11)
        self.assertEqual(result["output_token_count"], 3)

    def test_postprocess_keeps_audio_classification_compatible(self):
        context = self.context(
            RecordingPipeline(model_type="wav2vec2"),
            task_type="audio-classification",
        )

        list_result = self.handler.postprocess(context, [{"label": "speech"}])
        dict_result = self.handler.postprocess(context, {"label": "speech"})

        self.assertEqual(list_result["output_type"], "classification")
        self.assertEqual(list_result["n_results"], 1)
        self.assertEqual(dict_result["output_type"], "classification")
        self.assertEqual(dict_result["n_results"], 1)


if __name__ == "__main__":
    unittest.main()

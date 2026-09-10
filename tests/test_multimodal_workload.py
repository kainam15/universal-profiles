import base64
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

from PIL import Image

from acprof.workloads import get_generator


class MultimodalWorkloadTests(unittest.TestCase):
    def generator(self, task, spec=None, batch=1):
        return get_generator("multimodal", "example/model", task, batch, workload_spec_path=spec)

    def test_image_tasks_share_reproducible_scene_and_document_annotations(self):
        for task in ("image-text-to-text", "visual-question-answering", "document-question-answering", "visual-document-retrieval"):
            with self.subTest(task=task):
                gen = self.generator(task)
                payload = gen.generate(224)
                self.assertEqual(payload, self.generator(task).generate(224))
                sample = payload["samples"][0]
                image = Image.open(io.BytesIO(base64.b64decode(sample["image_base64"])))
                self.assertEqual(image.size, (224, 224))
                self.assertTrue(sample["text"])
                self.assertEqual(payload["input_scale_type"], "resolution_px")
                self.assertEqual(gen.input_metadata(224, payload)["input_num_samples"], 1)
                if task == "document-question-answering":
                    self.assertEqual(len(sample["words"]), len(sample["boxes"]))
                    self.assertIn("42", sample["words"])

    def test_audio_is_prefix_of_same_real_waveform_with_provenance(self):
        gen = self.generator("audio-text-to-text")
        def read(payload):
            with wave.open(io.BytesIO(base64.b64decode(payload["samples"][0]["audio_base64"]))) as audio:
                self.assertEqual(audio.getframerate(), 16000)
                return audio.readframes(audio.getnframes())
        short, long = gen.generate(1), gen.generate(2)
        self.assertEqual(len(read(short)), 32000)
        self.assertEqual(read(long)[:32000], read(short))
        self.assertEqual(short["input_scale_type"], "duration_s")
        self.assertEqual(gen.plan_metadata()["assets"][0]["sha256"], "c67f163f3b1aa88157123dfa0264d80a34d7a5b51e4d3f2b6718b60bcca34157")
        with self.assertRaisesRegex(ValueError, "duration|时长"):
            gen.generate(31)

    def test_video_scales_frame_prefix_and_records_fps(self):
        gen = self.generator("video-text-to-text")
        small, large = gen.generate(2), gen.generate(4)
        self.assertEqual(small["samples"][0]["video_frames_base64"], large["samples"][0]["video_frames_base64"][:2])
        self.assertEqual(len(large["samples"][0]["video_frames_base64"]), 4)
        self.assertEqual(large["input_scale_type"], "frame_count")
        self.assertEqual(gen.input_metadata(4, large)["video_num_frames"], 4)

    def test_custom_manifest_resolves_relative_assets_and_freezes_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 30), "red").save(root / "page.png")
            spec = root / "workload.json"
            spec.write_text(json.dumps({"schema_version": 1, "task": "image-text-to-text", "image_path": "page.png", "text": "What color?", "input_scales": [64, 128], "params": {"max_new_tokens": 8}}))
            gen = self.generator("image-text-to-text", str(spec))
            first = gen.generate(64)
            Image.new("RGB", (20, 30), "blue").save(root / "page.png")
            self.assertEqual(first, gen.generate(64))
            self.assertEqual(first["samples"][0]["text"], "What color?")
            self.assertEqual(first["params"]["max_new_tokens"], 8)
            self.assertEqual(gen.default_input_scales(), [64.0, 128.0])
            self.assertEqual(len(gen.plan_metadata()["assets"][0]["sha256"]), 64)

    def test_invalid_batch_scale_and_unknown_manifest_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch"):
            self.generator("image-text-to-text", batch=2)
        gen = self.generator("image-text-to-text")
        for scale in (0, -1, float("nan"), 1.5, True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                gen.generate(scale)
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "invalid.json"
            spec.write_text('{"schema_version":1,"promtp":"typo"}')
            with self.assertRaisesRegex(ValueError, "promtp"):
                self.generator("image-text-to-text", str(spec))

    def test_any_to_any_can_mix_image_and_audio_with_one_scaling_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "any.json"
            spec.write_text(json.dumps({"schema_version": 1, "modalities": ["image", "audio"], "scale_modality": "audio", "params": {"return_audio": True}}))
            gen = self.generator("any-to-any", str(spec))
            small, large = gen.generate(1), gen.generate(2)
            self.assertEqual(small["samples"][0]["image_base64"], large["samples"][0]["image_base64"])
            self.assertNotEqual(small["samples"][0]["audio_base64"], large["samples"][0]["audio_base64"])
            self.assertTrue(small["params"]["return_audio"])
            self.assertEqual(gen.plan_metadata()["scale_modality"], "audio")

    def test_manifest_rejects_unused_assets_and_active_axis_fixed_values(self):
        for extra in ({"audio_path": "unused.wav"}, {"video_frames": ["unused.png"]}, {"image_resolution": 128}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                spec = Path(tmp) / "invalid.json"
                spec.write_text(json.dumps({"schema_version": 1, **extra}))
                with self.assertRaises(ValueError):
                    self.generator("image-text-to-text", str(spec))
        gen = self.generator("image-text-to-text")
        self.assertNotIn("image_resolution", gen.plan_metadata()["fixed_media"])

    def test_custom_video_frames_respect_fixed_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 30), "red").save(root / "frame.png")
            spec = root / "video.json"
            spec.write_text(json.dumps({"schema_version": 1, "video_frames": ["frame.png"], "image_resolution": 64, "input_scales": [1]}))
            gen = self.generator("video-text-to-text", str(spec))
            frame = gen.generate(1)["samples"][0]["video_frames_base64"][0]
            self.assertEqual(Image.open(io.BytesIO(base64.b64decode(frame))).size, (64, 64))

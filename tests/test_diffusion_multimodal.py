import base64
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from acprof.container.handlers.diffusion import DiffusionHandler
from acprof.workloads.diffusion import DiffusionWorkloadGenerator


class FakeGenerator:
    def __init__(self, *, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class ImagePipeline:
    vae_scale_factor = 8

    def __call__(self, prompt, image, num_inference_steps=20, guidance_scale=7.5,
                 generator=None, output_type="pil", return_dict=True, strength=0.8):
        self.kwargs = locals()
        return types.SimpleNamespace(images=[image.copy()])


class VideoPipeline:
    vae_scale_factor_spatial = 8
    vae_scale_factor_temporal = 4
    transformer = types.SimpleNamespace(config=types.SimpleNamespace(patch_size=2))

    def __call__(self, prompt, image, height, width, num_frames=17,
                 num_inference_steps=20, guidance_scale=7.5, generator=None,
                 output_type="pil", return_dict=True):
        self.kwargs = locals()
        return types.SimpleNamespace(frames=[[
            Image.new("RGB", (width, height)) for _ in range(num_frames)
        ]])


class DiffusionMultimodalWorkloadTests(unittest.TestCase):
    def test_image_condition_and_its_hash_are_reproducible_across_generators(self):
        first = DiffusionWorkloadGenerator("example/edit", "image-text-to-image", 1)
        second = DiffusionWorkloadGenerator("example/edit", "image-text-to-image", 1)
        payload = first.generate(128)
        self.assertEqual(payload, second.generate(128))
        image_bytes = base64.b64decode(payload["image_base64"], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertEqual(image.size, (128, 128))
            self.assertGreater(len(image.getcolors(maxcolors=128 * 128)), 1)
        metadata = first.input_metadata(128, payload)
        self.assertEqual(metadata["condition_image_sha256"], hashlib.sha256(image_bytes).hexdigest())
        self.assertEqual(metadata["input_num_samples"], 1)
        self.assertEqual(metadata["input_scale_type"], "resolution_px")
        self.assertEqual(first.plan_metadata()["input_scale_type"], "resolution_px")

    def test_manifest_materializes_relative_image_prompt_steps_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (200, 120), (200, 30, 10)).save(root / "condition.png")
            spec = {
                "schema_version": 1,
                "workload_id": "custom-video-v1",
                "image_path": "condition.png",
                "prompt": "pan slowly to the left",
                "input_scales": [128, 256],
                "params": {"num_inference_steps": 5, "num_frames": 9, "seed": 99},
            }
            manifest = root / "workload.json"
            manifest.write_text(json.dumps(spec))
            generator = DiffusionWorkloadGenerator(
                "example/video", "image-text-to-video", 1, workload_spec_path=str(manifest)
            )
            payload = generator.generate(128)
            self.assertEqual(payload["prompt"], ["pan slowly to the left"])
            self.assertEqual(payload["params"]["num_frames"], 9)
            self.assertEqual(payload["params"]["seed"], 99)
            self.assertEqual(generator.default_input_scales(), [128.0, 256.0])
            self.assertEqual(generator.input_metadata(128, payload)["output_num_frames"], 9)
            self.assertEqual(generator.plan_metadata()["workload_id"], "custom-video-v1")
            self.assertEqual(generator.plan_metadata()["workload_spec_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_manifest_rejects_unknown_params_and_image_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 64)).save(root / "condition.png")
            manifest = root / "workload.json"
            for spec, message in [
                ({"params": {"strenth": 0.7}}, "strenth"),
                ({"image_path": "condition.png", "image_sha256": "0" * 64}, "SHA256"),
            ]:
                with self.subTest(spec=spec):
                    manifest.write_text(json.dumps(spec))
                    with self.assertRaisesRegex(ValueError, message):
                        DiffusionWorkloadGenerator("example/edit", "image-text-to-image", 1, workload_spec_path=str(manifest))

    def test_hub_aliases_generate_and_execute_conditioned_workloads(self):
        for alias, canonical in [("image-to-image", "image-text-to-image"), ("image-to-video", "image-text-to-video")]:
            with self.subTest(alias=alias):
                generator = DiffusionWorkloadGenerator("example/model", alias, 1)
                reference = DiffusionWorkloadGenerator("example/model", canonical, 1)
                self.assertEqual(generator.generate(128), reference.generate(128))

    def test_conditioned_tasks_reject_batch_greater_than_one(self):
        for task in ["image-text-to-image", "image-text-to-video"]:
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "batch_size=1"):
                DiffusionWorkloadGenerator("example/model", task, 2)

    def test_manifest_rejects_invalid_generation_values_before_materializing(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "workload.json"
            for params in [
                {"num_frames": "9"}, {"num_inference_steps": 0},
                {"guidance_scale": True}, {"strength": 1.1},
                {"seed": 1.5}, {"negative_prompt": ["first", "second"]},
            ]:
                with self.subTest(params=params):
                    manifest.write_text(json.dumps({"params": params}))
                    with self.assertRaises(ValueError):
                        DiffusionWorkloadGenerator(
                            "example/video", "image-text-to-video", 1,
                            workload_spec_path=str(manifest),
                        )


class DiffusionMultimodalHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = DiffusionHandler()
        self.fake_torch = types.ModuleType("torch")
        self.fake_torch.Generator = FakeGenerator

    @staticmethod
    def payload(task="image-text-to-image", resolution=128):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (100, 20, 30)).save(buf, format="PNG")
        params = {"seed": 27, "num_inference_steps": 10}
        if task == "image-text-to-video":
            params["num_frames"] = 9
        return {"prompt": ["make the scene brighter"], "resolution": resolution,
                "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"), "params": params}

    def run_pipeline(self, pipeline, task="image-text-to-image", payload=None):
        context = {"task_type": task, "pipeline": pipeline, "device": "cpu"}
        processed = self.handler.preprocess(context, payload or self.payload(task))
        with patch.dict(sys.modules, {"torch": self.fake_torch}):
            output = self.handler.predict(context, processed)
        return self.handler.postprocess(context, output)

    def test_image_prompt_seed_strength_and_resolution_reach_native_pipeline(self):
        pipe = ImagePipeline()
        payload = self.payload()
        payload["params"]["strength"] = 0.6
        response = self.run_pipeline(pipe, payload=payload)
        self.assertEqual(pipe.kwargs["prompt"], ["make the scene brighter"])
        self.assertEqual(pipe.kwargs["image"].size, (128, 128))
        self.assertEqual(pipe.kwargs["generator"].seed, 27)
        self.assertEqual(pipe.kwargs["strength"], 0.6)
        self.assertEqual(response["output_shape"], [1, 128, 128, 3])
        self.assertNotIn("images", response)

    def test_video_metadata_counts_decoded_frames_without_encoding(self):
        pipe = VideoPipeline()
        payload = self.payload("image-text-to-video")
        with patch.object(Image.Image, "save", side_effect=AssertionError("must not encode output")):
            response = self.run_pipeline(pipe, "image-text-to-video", payload)
        self.assertEqual(pipe.kwargs["num_frames"], 9)
        self.assertEqual(response["output_type"], "video")
        self.assertEqual(response["n_results"], 1)
        self.assertEqual(response["output_length"], 9)
        self.assertEqual(response["video_frame_count"], 9)
        self.assertEqual(response["output_shape"], [1, 9, 128, 128, 3])
        self.assertNotIn("frames", response)

    def test_missing_or_invalid_condition_is_rejected_instead_of_generated(self):
        context = {"task_type": "image-text-to-image", "pipeline": ImagePipeline()}
        for value in [None, "", "not base64!"]:
            payload = self.payload()
            payload["image_base64"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "image_base64"):
                self.handler.preprocess(context, payload)

    def test_pipeline_cannot_swallow_missing_image_support_through_kwargs(self):
        def text_only(prompt, num_inference_steps=20, guidance_scale=7.5,
                      generator=None, output_type="pil", return_dict=True, **kwargs):
            raise AssertionError("unsupported pipeline must not execute")
        with self.assertRaisesRegex(ValueError, "image"):
            self.run_pipeline(text_only)

    def test_unsupported_explicit_parameter_is_rejected_even_with_kwargs(self):
        def no_strength(prompt, image, num_inference_steps=20, guidance_scale=7.5,
                        generator=None, output_type="pil", return_dict=True, **kwargs):
            raise AssertionError("unsupported strength must not be ignored")
        payload = self.payload()
        payload["params"]["strength"] = 0.5
        with self.assertRaisesRegex(ValueError, "strength"):
            self.run_pipeline(no_strength, payload=payload)

    def test_video_enforces_model_spatial_and_temporal_alignment(self):
        context = {"task_type": "image-text-to-video", "pipeline": VideoPipeline()}
        for resolution, frames, message in [(72, 9, "16"), (128, 8, "num_frames")]:
            payload = self.payload("image-text-to-video", resolution)
            payload["params"]["num_frames"] = frames
            with self.subTest(resolution=resolution, frames=frames), self.assertRaisesRegex(ValueError, message):
                self.handler.preprocess(context, payload)

    def test_pipeline_output_cannot_silently_change_requested_frames_or_size(self):
        class WrongOutput(VideoPipeline):
            def __call__(self, prompt, image, height, width, num_frames=17,
                         num_inference_steps=20, guidance_scale=7.5, generator=None,
                         output_type="pil", return_dict=True):
                return types.SimpleNamespace(frames=[[Image.new("RGB", (64, 64))]])
        with self.assertRaisesRegex(ValueError, "requested|expected"):
            self.run_pipeline(WrongOutput(), "image-text-to-video")

    def test_prediction_does_not_run_postprocess_inside_profiler_window(self):
        context = {"task_type": "image-text-to-video", "pipeline": VideoPipeline(), "device": "cpu"}
        processed = self.handler.preprocess(context, self.payload("image-text-to-video"))
        with patch.dict(sys.modules, {"torch": self.fake_torch}), patch.object(
            self.handler, "postprocess", side_effect=AssertionError("metadata belongs in postprocess")
        ):
            output = self.handler.predict(context, processed)
        self.assertEqual(self.handler.postprocess(context, output)["video_frame_count"], 9)

    def test_strength_that_produces_no_denoising_steps_is_rejected(self):
        payload = self.payload()
        payload["params"]["strength"] = 0.01
        with self.assertRaisesRegex(ValueError, "strength.*num_inference_steps|denoising"):
            self.run_pipeline(ImagePipeline(), payload=payload)

    def test_handler_rejects_batched_conditioned_request(self):
        payload = self.payload()
        payload["prompt"] = ["first", "second"]
        with self.assertRaisesRegex(ValueError, "one prompt|batch_size=1"):
            self.run_pipeline(ImagePipeline(), payload=payload)

    def test_load_rejects_image_only_video_pipeline_and_preserves_revision(self):
        calls = []
        class ImageOnly:
            def __call__(self, image, num_frames=17):
                pass
        class Loader:
            @classmethod
            def from_pretrained(cls, model_source, **kwargs):
                calls.append(kwargs)
                return ImageOnly()
        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.DiffusionPipeline = Loader
        self.fake_torch.float16 = "float16"
        self.fake_torch.float32 = "float32"
        with patch.dict(sys.modules, {"torch": self.fake_torch, "diffusers": fake_diffusers}):
            with self.assertRaisesRegex(ValueError, "prompt"):
                self.handler.load("example/video", "image-text-to-video", "diffusers", "cpu", "revision123")
        self.assertEqual(calls[0]["revision"], "revision123")
        self.assertTrue(calls[0]["local_files_only"])

    def test_native_video_load_keeps_transformer_and_refuses_unverified_eager_flops(self):
        class LoadableVideo(VideoPipeline):
            def to(self, device):
                self.device = device
                return self

            def set_progress_bar_config(self, *, disable):
                self.progress_disabled = disable

        pipe = LoadableVideo()
        calls = []

        class Loader:
            @classmethod
            def from_pretrained(cls, model_source, **kwargs):
                calls.append(kwargs)
                return pipe

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.DiffusionPipeline = Loader
        self.fake_torch.float16 = "float16"
        self.fake_torch.float32 = "float32"
        with tempfile.TemporaryDirectory() as snapshot, patch.dict(
            sys.modules, {"torch": self.fake_torch, "diffusers": fake_diffusers}
        ):
            context = self.handler.load(snapshot, "image-to-video", "diffusers", "cpu", "revision123")
            with self.assertRaisesRegex(ValueError, "eager UNet"):
                self.handler.load(
                    snapshot, "image-to-video", "diffusers", "cpu", "revision123",
                    load_options={"attention_implementation": "eager"},
                )
        self.assertIs(context["model"], pipe.transformer)
        self.assertEqual(context["task_type"], "image-to-video")
        self.assertEqual(pipe.device, "cpu")
        self.assertTrue(pipe.progress_disabled)
        self.assertNotIn("revision", calls[0])
        self.assertTrue(calls[0]["local_files_only"])


if __name__ == "__main__":
    unittest.main()

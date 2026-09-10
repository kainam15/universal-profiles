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
from acprof.container.handlers import diffusion as diffusion_handler
from acprof.workloads.diffusion import DiffusionWorkloadGenerator


class Generator:
    def __init__(self, *, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class DDPMPipeline:
    unet = types.SimpleNamespace(config=types.SimpleNamespace(sample_size=32))
    scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))

    def __call__(self, batch_size=1, generator=None, num_inference_steps=1000,
                 output_type="pil", return_dict=True):
        self.kwargs = locals()
        return types.SimpleNamespace(images=[Image.new("RGB", (32, 32))])


class TextVideoPipeline:
    vae_scale_factor_temporal = 4
    vae_scale_factor = 8

    def __call__(self, prompt, height, width, num_frames=17,
                 num_inference_steps=20, guidance_scale=7.5, generator=None,
                 output_type="pil", return_dict=True):
        self.kwargs = locals()
        return types.SimpleNamespace(frames=[[
            Image.new("RGB", (width, height)) for _ in range(num_frames)
        ]])


class VideoVideoPipeline:
    vae_scale_factor_temporal = 4
    vae_scale_factor = 8

    def __call__(self, video, prompt, height=None, width=None,
                 num_inference_steps=20, strength=0.8, guidance_scale=7.5,
                 generator=None, output_type="pil", return_dict=True):
        self.kwargs = locals()
        return types.SimpleNamespace(frames=[video])


class ImageVideoPipeline:
    vae_scale_factor = 8

    def __call__(self, image, height, width, num_frames=17,
                 num_inference_steps=20, min_guidance_scale=1.0,
                 max_guidance_scale=3.0, generator=None,
                 output_type="pil", return_dict=True):
        self.kwargs = locals()
        return types.SimpleNamespace(frames=[[image.copy() for _ in range(num_frames)]])


class ShapEPipeline:
    shap_e_renderer = types.SimpleNamespace(decode_to_mesh=lambda *args: None)

    def __call__(self, prompt, num_inference_steps=25, generator=None,
                 guidance_scale=4.0, frame_size=64, output_type="pil", return_dict=True):
        self.kwargs = locals()
        mesh = types.SimpleNamespace(
            verts=types.SimpleNamespace(shape=(12, 3)),
            faces=types.SimpleNamespace(shape=(20, 3)),
        )
        return types.SimpleNamespace(images=[mesh])


class ShapEImagePipeline(ShapEPipeline):
    def __call__(self, image, num_inference_steps=25, generator=None,
                 guidance_scale=3.0, frame_size=64, output_type="pil", return_dict=True):
        result = super().__call__("unused", num_inference_steps, generator,
                                  guidance_scale, frame_size, output_type, return_dict)
        self.kwargs = locals()
        return result


class GenerativeVisionTests(unittest.TestCase):
    def run_pipeline(self, task, pipe, scale=128, payload=None):
        handler = DiffusionHandler()
        ctx = {"task_type": task, "pipeline": pipe, "device": "cpu"}
        if payload is None:
            payload = DiffusionWorkloadGenerator("example/model", task, 1).generate(scale)
        processed = handler.preprocess(ctx, payload)
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(Generator=Generator)}):
            result = handler.predict(ctx, processed)
        return handler.postprocess(ctx, result), processed

    def test_text_to_video_passes_prompt_and_reports_decoded_frames(self):
        pipe = TextVideoPipeline()
        output, _ = self.run_pipeline("text-to-video", pipe)
        self.assertEqual(output["output_shape"], [1, 17, 128, 128, 3])
        self.assertEqual(pipe.kwargs["generator"].seed, 12345)
        self.assertIsInstance(pipe.kwargs["prompt"][0], str)

    def test_video_to_video_supplies_ordered_real_frames_without_num_frames_kwarg(self):
        pipe = VideoVideoPipeline()
        gen = DiffusionWorkloadGenerator("example/model", "video-to-video", 1)
        payload = gen.generate(128)
        output, _ = self.run_pipeline("video-to-video", pipe, payload=payload)
        self.assertEqual(output["video_frame_count"], 17)
        self.assertNotEqual(pipe.kwargs["video"][0].tobytes(), pipe.kwargs["video"][1].tobytes())
        metadata = gen.input_metadata(128, payload)
        self.assertEqual(metadata["condition_frame_count"], 17)
        self.assertEqual(len(metadata["condition_frame_sha256"]), 17)

    def test_video_manifest_preserves_order_hashes_and_declared_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i in range(5):
                name = f"frame-{i}.png"
                Image.new("RGB", (64, 64), (i * 40, 0, 0)).save(root / name)
                paths.append(name)
            spec = root / "workload.json"
            spec.write_text(json.dumps({"video_frames": paths, "params": {"num_frames": 5}}))
            gen = DiffusionWorkloadGenerator("example/model", "video-to-video", 1, str(spec))
            payload = gen.generate(64)
            frame = base64.b64decode(payload["frames_base64"][1])
            self.assertEqual(Image.open(io.BytesIO(frame)).getpixel((0, 0)), (40, 0, 0))
            self.assertEqual(gen.input_metadata(64, payload)["condition_frame_sha256"][1], hashlib.sha256(frame).hexdigest())
            spec.write_text(json.dumps({"video_frames": paths, "params": {"num_frames": 9}}))
            with self.assertRaisesRegex(ValueError, "num_frames"):
                DiffusionWorkloadGenerator("example/model", "video-to-video", 1, str(spec))

    def test_image_to_video_uses_native_image_only_pipeline(self):
        pipe = ImageVideoPipeline()
        gen = DiffusionWorkloadGenerator("example/model", "image-to-video", 1)
        payload = gen.generate(128)
        self.assertTrue(payload["prompt_optional"])
        output, _ = self.run_pipeline("image-to-video", pipe, payload=payload)
        self.assertEqual(output["video_frame_count"], 17)
        self.assertEqual(pipe.kwargs["max_guidance_scale"], 3.0)

    def test_unconditional_scale_is_steps_and_keeps_unet_native_32px_output(self):
        gen = DiffusionWorkloadGenerator("example/ddpm", "unconditional-image-generation", 1)
        payload = gen.generate(4)
        self.assertNotIn("prompt", payload)
        self.assertNotIn("resolution", payload)
        self.assertEqual(payload["input_scale"], 4)
        self.assertEqual(payload["params"]["num_inference_steps"], 4)
        self.assertEqual(gen.plan_metadata()["input_scale_type"], "denoising_steps")
        pipe = DDPMPipeline()
        output, processed = self.run_pipeline("unconditional-image-generation", pipe, payload=payload)
        self.assertEqual(pipe.kwargs["num_inference_steps"], 4)
        self.assertEqual(output["output_shape"], [1, 32, 32, 3])
        self.assertEqual(processed["_effective_input_scale"], 4.0)

    def test_shap_e_returns_mesh_and_steps_do_not_pretend_to_scale_frame_size(self):
        for task, pipe in [("text-to-3d", ShapEPipeline()), ("image-to-3d", ShapEImagePipeline())]:
            with self.subTest(task=task):
                gen = DiffusionWorkloadGenerator("example/shape", task, 1)
                payload = gen.generate(4)
                output, _ = self.run_pipeline(task, pipe, payload=payload)
                self.assertEqual(pipe.kwargs["output_type"], "mesh")
                self.assertEqual(pipe.kwargs["frame_size"], 64)
                self.assertEqual(pipe.kwargs["num_inference_steps"], 4)
                self.assertEqual(output["output_type"], "mesh")
                self.assertEqual(output["mesh_vertex_counts"], [12])
                self.assertEqual(output["mesh_face_counts"], [20])
                self.assertNotIn("output_pixel_count_per_image", gen.input_metadata(4, payload))

    def test_native_generation_rejects_ignored_or_conflicting_inputs(self):
        cases = [
            ("unconditional-image-generation", DDPMPipeline(), 4, {"resolution": 128}),
            ("image-to-video", ImageVideoPipeline(), 128, {"prompt": ["ignored"], "prompt_optional": False}),
            ("text-to-3d", ShapEPipeline(), 4, {"params": {"seed": 3, "num_inference_steps": 4, "frame_size": 128}}),
        ]
        for task, pipe, scale, update in cases:
            with self.subTest(task=task):
                payload = DiffusionWorkloadGenerator("example/model", task, 1).generate(scale)
                payload.update(update)
                with self.assertRaises(ValueError):
                    self.run_pipeline(task, pipe, payload=payload)

    def test_video_rejects_temporal_mismatch_before_predict(self):
        gen = DiffusionWorkloadGenerator("example/model", "video-to-video", 1)
        payload = gen.generate(128)
        payload["frames_base64"] = payload["frames_base64"][:8]
        payload["params"]["num_frames"] = 8
        with self.assertRaisesRegex(ValueError, "temporal"):
            self.run_pipeline("video-to-video", VideoVideoPipeline(), payload=payload)

    def test_three_dimensional_task_rejects_unverified_non_shap_e_pipeline(self):
        pipe = ShapEPipeline()
        pipe.shap_e_renderer = None
        with self.assertRaisesRegex(ValueError, "ShapE|mesh"):
            self.run_pipeline("text-to-3d", pipe, 4)

    def test_step_scale_cannot_disagree_with_actual_denoising_steps(self):
        payload = DiffusionWorkloadGenerator("example/model", "unconditional-image-generation", 1).generate(4)
        payload["params"]["num_inference_steps"] = 8
        with self.assertRaisesRegex(ValueError, "input_scale"):
            self.run_pipeline("unconditional-image-generation", DDPMPipeline(), payload=payload)

    def test_checkpoint_pipeline_is_adapted_to_native_video_task_without_reloading_weights(self):
        pipe = type("CogVideoXPipeline", (TextVideoPipeline,), {})()
        converted = VideoVideoPipeline()
        converter = types.SimpleNamespace(from_pipe=lambda source: converted if source is pipe else None)
        with patch.dict(sys.modules, {"diffusers": types.SimpleNamespace(CogVideoXVideoToVideoPipeline=converter)}):
            actual = diffusion_handler._pipeline_for_task(pipe, "video-to-video")
        self.assertIs(actual, converted)

    def test_wan_dual_transformer_is_not_silently_dropped_during_task_adaptation(self):
        pipe = type("WanPipeline", (TextVideoPipeline,), {})()
        pipe.transformer_2 = object()
        with self.assertRaisesRegex(ValueError, "transformer_2|dual"):
            diffusion_handler._pipeline_for_task(pipe, "video-to-video")

    def test_image_conditioned_cogvideo_weights_are_not_reused_for_video_to_video(self):
        pipe = type("CogVideoXImageToVideoPipeline", (ImageVideoPipeline,), {})()
        converter = types.SimpleNamespace(from_pipe=lambda source: VideoVideoPipeline())
        with patch.dict(sys.modules, {"diffusers": types.SimpleNamespace(CogVideoXVideoToVideoPipeline=converter)}):
            with self.assertRaisesRegex(ValueError, "verified"):
                diffusion_handler._pipeline_for_task(pipe, "video-to-video")

    def test_image_alias_keeps_explicit_prompt_for_native_conditioned_pipelines(self):
        class ImagePromptVideo(TextVideoPipeline):
            def __call__(self, image, prompt, height, width, num_frames=17,
                         num_inference_steps=20, guidance_scale=7.5, generator=None,
                         output_type="pil", return_dict=True):
                return super().__call__(prompt, height, width, num_frames,
                                        num_inference_steps, guidance_scale, generator,
                                        output_type, return_dict)
        pipe = ImagePromptVideo()
        self.run_pipeline("image-to-video", pipe)
        self.assertEqual(len(pipe.kwargs["prompt"]), 1)

    def test_native_image_variation_consumes_image_and_rejects_fixed_upscale(self):
        class ImageVariation:
            def __call__(self, image, height=None, width=None, num_inference_steps=20,
                         guidance_scale=7.5, generator=None, output_type="pil", return_dict=True):
                return types.SimpleNamespace(images=[image])
        result, _ = self.run_pipeline("image-to-image", ImageVariation())
        self.assertEqual(result["output_shape"], [1, 128, 128, 3])

        class Upscale:
            def __call__(self, image, num_inference_steps=20, guidance_scale=7.5,
                         generator=None, output_type="pil", return_dict=True):
                return types.SimpleNamespace(images=[image.resize((512, 512))])
        with self.assertRaisesRegex(ValueError, "requested"):
            self.run_pipeline("image-to-image", Upscale())

    def test_known_fixed_upscalers_are_rejected_before_model_execution(self):
        def call(self, prompt, image, num_inference_steps=20, guidance_scale=7.5,
                 generator=None, output_type="pil", return_dict=True):
            raise AssertionError("unsupported fixed upscaler must not execute")

        for name in ("StableDiffusionUpscalePipeline", "StableDiffusionLatentUpscalePipeline"):
            for task in ("image-to-image", "image-text-to-image"):
                with self.subTest(pipeline=name, task=task):
                    pipe = type(name, (), {"__call__": call})()
                    payload = DiffusionWorkloadGenerator("example/upscale", task, 1).generate(128)
                    with self.assertRaisesRegex(ValueError, "fixed-factor upscaler.*resolution"):
                        DiffusionHandler().preprocess({"task_type": task, "pipeline": pipe}, payload)


if __name__ == "__main__":
    unittest.main()

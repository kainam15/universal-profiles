import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.container.handlers.diffusion import DiffusionHandler
from acprof.host import orchestrator
from acprof.host.detect import TaskInfo
from acprof.host.orchestrator import (
    ImageInfo,
    _inference_precision_by_device,
    _model_io_formats,
    plan_input_scales,
)
from acprof.workloads.diffusion import (
    BASE_SEED,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_RESOLUTIONS,
    DiffusionWorkloadGenerator,
)


class DiffusionWorkloadTests(unittest.TestCase):
    def test_default_workload_is_deterministic_and_scales_resolution(self) -> None:
        generator = DiffusionWorkloadGenerator(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "text-to-image",
            2,
        )

        payload = generator.generate(256)

        self.assertEqual(payload["resolution"], 256)
        self.assertEqual(len(payload["prompt"]), 2)
        self.assertEqual(
            payload["params"],
            {
                "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
                "guidance_scale": DEFAULT_GUIDANCE_SCALE,
                "seed": BASE_SEED,
            },
        )
        self.assertEqual(
            generator.default_input_scales(),
            [float(value) for value in DEFAULT_RESOLUTIONS],
        )
        self.assertEqual(generator.effective_input_scale(256, payload), 256.0)
        self.assertEqual(generator.scale_label(256), "res256px")
        self.assertEqual(
            generator.input_metadata(256, payload)["output_pixel_count_per_image"],
            256 * 256,
        )

    def test_workload_rejects_non_aligned_resolution(self) -> None:
        generator = DiffusionWorkloadGenerator("example/model", "text-to-image", 1)

        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            generator.generate(255)


class DiffusionHandlerTests(unittest.TestCase):
    def test_load_uses_local_snapshot_and_eager_attention_processor(self) -> None:
        calls = []

        class FakeAttnProcessor:
            pass

        class FakeUNet:
            def __init__(self):
                self.config = types.SimpleNamespace()
                self.processor = None

            def set_attn_processor(self, processor):
                self.processor = processor

        class FakePipeline:
            def __init__(self):
                self.unet = FakeUNet()
                self.device = None
                self.progress_disabled = False

            def to(self, device):
                self.device = device
                return self

            def set_progress_bar_config(self, *, disable):
                self.progress_disabled = disable

        pipe = FakePipeline()

        class FakeDiffusionPipeline:
            @classmethod
            def from_pretrained(cls, model_source, **kwargs):
                calls.append((model_source, kwargs))
                return pipe

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.DiffusionPipeline = FakeDiffusionPipeline
        fake_models = types.ModuleType("diffusers.models")
        fake_attention = types.ModuleType("diffusers.models.attention_processor")
        fake_attention.AttnProcessor = FakeAttnProcessor

        with tempfile.TemporaryDirectory() as model_source, patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "diffusers": fake_diffusers,
                "diffusers.models": fake_models,
                "diffusers.models.attention_processor": fake_attention,
            },
        ):
            context = DiffusionHandler().load(
                model_source,
                "text-to-image",
                "diffusers",
                "cpu",
                "0123456789abcdef",
                load_options={"attention_implementation": "eager"},
            )

        self.assertEqual(calls[0][0], model_source)
        self.assertNotIn("revision", calls[0][1])
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertEqual(calls[0][1]["torch_dtype"], "float32")
        self.assertEqual(pipe.device, "cpu")
        self.assertTrue(pipe.progress_disabled)
        self.assertIsInstance(pipe.unet.processor, FakeAttnProcessor)
        self.assertEqual(pipe.unet.config._attn_implementation, "eager")
        self.assertIs(context["model"], pipe.unet)

    def test_predict_is_seeded_and_postprocess_returns_metadata_only(self) -> None:
        generator_calls = []
        pipeline_calls = []

        class FakeGenerator:
            def __init__(self, *, device):
                self.device = device
                self.seed = None
                generator_calls.append(self)

            def manual_seed(self, seed):
                self.seed = seed
                return self

        def fake_pipeline(**kwargs):
            pipeline_calls.append(kwargs)
            return types.SimpleNamespace(
                images=[
                    types.SimpleNamespace(size=(256, 256)),
                    types.SimpleNamespace(size=(256, 256)),
                ]
            )

        fake_torch = types.ModuleType("torch")
        fake_torch.Generator = FakeGenerator
        handler = DiffusionHandler()
        processed = handler.preprocess(
            {},
            {
                "prompt": ["first prompt", "second prompt"],
                "resolution": 256,
                "params": {
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "seed": 42,
                },
            },
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            raw_output = handler.predict(
                {
                    "pipeline": fake_pipeline,
                    "device": "cuda",
                    "task_type": "text-to-image",
                },
                processed,
            )
        response = handler.postprocess(
            {"task_type": "text-to-image"},
            raw_output,
        )

        self.assertEqual([item.seed for item in generator_calls], [42, 43])
        self.assertEqual(pipeline_calls[0]["height"], 256)
        self.assertEqual(pipeline_calls[0]["width"], 256)
        self.assertEqual(pipeline_calls[0]["prompt"], ["first prompt", "second prompt"])
        self.assertEqual(response["output_type"], "image")
        self.assertEqual(response["n_results"], 2)
        self.assertEqual(response["image_width"], 256)
        self.assertNotIn("images", response)

    def test_preprocess_rejects_invalid_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            DiffusionHandler().preprocess(
                {},
                {"prompt": "test", "resolution": 250, "params": {}},
            )


class DiffusionMetadataTests(unittest.TestCase):
    def _task_info(self) -> TaskInfo:
        return TaskInfo(
            model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
            pipeline_tag="text-to-image",
            task_family="diffusion",
            runtime_backend="diffusers",
            library_name="diffusers",
            model_revision="main",
            detection_method="hub_api",
        )

    def test_static_io_contract_and_precision_are_diffusion_specific(self) -> None:
        input_format, output_format = _model_io_formats(self._task_info())

        self.assertEqual(
            input_format["json_schema"]["properties"]["resolution"]["multipleOf"],
            8,
        )
        self.assertEqual(
            output_format["json_schema"]["properties"]["output_type"]["enum"],
            ["image"],
        )
        self.assertEqual(
            _inference_precision_by_device(self._task_info()),
            {"cpu": "FP32", "gpu": "FP16"},
        )

    def test_default_scale_plan_uses_supported_resolutions(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            planned = plan_input_scales(
                task_info=self._task_info(),
                image_info=ImageInfo(tag="acprof-diffusion-test:latest"),
                cpu_list=[4],
                mem_list=[16],
                gpu_list=["on"],
                batch_size=1,
                output_dir=output_dir,
            )
            self.assertEqual(
                planned.scales,
                [float(value) for value in DEFAULT_RESOLUTIONS],
            )
            self.assertEqual(planned.source, "workload_default")
            self.assertTrue(os.path.isfile(planned.plan_file or ""))
            with open(planned.plan_file or "", "r", encoding="utf-8") as handle:
                plan = json.load(handle)

        self.assertEqual(plan["task_family"], "diffusion")
        self.assertEqual(
            [entry["input_scale"] for entry in plan["entries"]],
            [float(value) for value in DEFAULT_RESOLUTIONS],
        )
        self.assertEqual(
            plan["entries"][-1]["payload"]["resolution"],
            max(DEFAULT_RESOLUTIONS),
        )

    def test_diffusion_image_build_receives_selected_torch_wheel(self) -> None:
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        project_dir = str(Path(__file__).resolve().parents[1])
        with patch.object(orchestrator, "_run", side_effect=fake_run), patch.object(
            orchestrator,
            "_select_nlp_torch_index_url",
            return_value="https://download.pytorch.org/whl/cu124",
        ), patch.object(
            orchestrator,
            "_select_nlp_torch_spec",
            return_value="torch>=2.6,<2.7",
        ):
            image = orchestrator.build_image(self._task_info(), project_dir)

        self.assertEqual(
            image.tag,
            "acprof-diffusion-stable-diffusion-v1-5--stable-diffusion-v1-5:latest",
        )
        self.assertEqual(len(commands), 2)
        family_command = commands[1]
        self.assertIn(
            str(Path(project_dir, "dockerfiles", "diffusion.Dockerfile")),
            family_command,
        )
        self.assertIn("TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124", family_command)
        self.assertIn("TORCH_PACKAGE_SPEC=torch>=2.6,<2.7", family_command)


if __name__ == "__main__":
    unittest.main()

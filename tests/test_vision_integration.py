"""Vision task routing and materialized protocol contracts without model downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import detect
from acprof.host.orchestrator import ImageInfo, _model_io_formats, plan_input_scales
from acprof.host.task_support import TaskSupportError, require_task_support


CV_TASKS = (
    "depth-estimation", "image-classification", "object-detection",
    "image-segmentation", "image-to-text", "video-classification",
    "zero-shot-image-classification", "mask-generation",
    "zero-shot-object-detection", "image-feature-extraction", "keypoint-detection",
)
GENERATION_TASKS = (
    "text-to-image", "image-to-image", "image-to-video",
    "unconditional-image-generation", "text-to-video", "text-to-3d",
    "image-to-3d", "video-to-video",
)


def task_info(task):
    family = "cv" if task in CV_TASKS else "diffusion"
    backend = "transformers_pipeline" if family == "cv" else "diffusers"
    return detect.TaskInfo("example/model", task, family, backend,
                           "transformers" if family == "cv" else "diffusers",
                           "pinned-revision", "hub_api")


class VisionIntegrationTests(unittest.TestCase):
    def test_all_nineteen_hub_tasks_route_to_collection(self):
        for task in CV_TASKS + GENERATION_TASKS:
            expected = task_info(task)
            with self.subTest(task=task), patch("huggingface_hub.model_info", return_value=SimpleNamespace(
                pipeline_tag=task, library_name=expected.library_name, sha="pinned-revision",
            )):
                actual = detect.detect_task("example/model")
                self.assertEqual(actual.task_family, expected.task_family)
                self.assertEqual(actual.runtime_backend, expected.runtime_backend)
                self.assertEqual(actual.model_revision, "pinned-revision")
                require_task_support(actual)

    def test_cv_rejects_unimplemented_batch_and_backend_before_measurement(self):
        for task in CV_TASKS:
            with self.subTest(task=task):
                with self.assertRaisesRegex(TaskSupportError, "batch-size 1"):
                    require_task_support(task_info(task), batch_size=2)
        info = task_info("video-classification")
        info.runtime_backend = "diffusers"
        with self.assertRaisesRegex(TaskSupportError, "Transformers"):
            require_task_support(info)

    def test_specific_vision_architectures_do_not_use_generic_detection(self):
        for arch, task in (
            ("VideoMAEForVideoClassification", "video-classification"),
            ("OwlViTForObjectDetection", "zero-shot-object-detection"),
            ("Owlv2ForObjectDetection", "zero-shot-object-detection"),
            ("SamModel", "mask-generation"), ("Sam2Model", "mask-generation"),
            ("SuperPointForKeypointDetection", "keypoint-detection"),
            ("VitPoseForPoseEstimation", "keypoint-detection"),
            ("Mask2FormerForUniversalSegmentation", "image-segmentation"),
        ):
            with self.subTest(architecture=arch), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps({"architectures": [arch]}))
                with patch("huggingface_hub.hf_hub_download", return_value=str(path)):
                    result = detect._detect_from_config("example/model")
                self.assertIsNotNone(result)
                self.assertEqual((result.pipeline_tag, result.task_family), (task, "cv"))

    def test_missing_diffusers_tag_uses_pinned_native_pipeline_class(self):
        for pipeline, task in (
            ("DDPMPipeline", "unconditional-image-generation"),
            ("ShapEPipeline", "text-to-3d"),
            ("ShapEImg2ImgPipeline", "image-to-3d"),
            ("StableVideoDiffusionPipeline", "image-to-video"),
            ("CogVideoXPipeline", "text-to-video"),
            ("CogVideoXVideoToVideoPipeline", "video-to-video"),
        ):
            with self.subTest(pipeline=pipeline), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model_index.json"
                path.write_text(json.dumps({"_class_name": pipeline}))
                with patch("huggingface_hub.model_info", return_value=SimpleNamespace(
                    pipeline_tag=None, library_name="diffusers", sha="pinned-revision",
                )), patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download:
                    result = detect.detect_task("example/model")
                self.assertEqual(result.pipeline_tag, task)
                self.assertEqual(result.model_revision, "pinned-revision")
                download.assert_called_once_with(repo_id="example/model", filename="model_index.json", revision="pinned-revision")

    def test_cv_manifest_reaches_materialized_plan_and_preserves_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "workload.json"
            manifest.write_text(json.dumps({
                "schema_version": 1, "input_scales": [0.5, 1.0],
                "candidate_labels": ["cat", "person"], "params": {"threshold": 0.2},
            }))
            planned = plan_input_scales(
                task_info("zero-shot-object-detection"), ImageInfo("unused"),
                [1], [4], ["off"], 1, tmp, workload_spec_path=str(manifest),
            )
            self.assertEqual(planned.scales, [0.5, 1.0])
            data = json.loads(Path(planned.plan_file).read_text())
            self.assertEqual(data["entries"][0]["payload"]["params"]["threshold"], 0.2)
            self.assertEqual(data["entries"][0]["input_metadata"]["input_num_samples"], 1)
            self.assertEqual(data["workload"]["input_scale_type"], "resolution_scale")
            self.assertEqual(len(planned.plan_sha256), 64)

    def test_static_contract_uses_correct_media_and_output_types(self):
        for task, expected in (
            ("depth-estimation", "depth"), ("image-segmentation", "segmentation"),
            ("mask-generation", "masks"), ("image-feature-extraction", "features"),
            ("keypoint-detection", "keypoints"), ("text-to-video", "video"),
            ("text-to-3d", "mesh"), ("image-to-3d", "mesh"),
        ):
            with self.subTest(task=task):
                _, outputs = _model_io_formats(task_info(task))
                self.assertEqual(outputs["json_schema"]["properties"]["output_type"]["enum"], [expected])
        for task in ("video-classification", "video-to-video"):
            inputs, _ = _model_io_formats(task_info(task))
            self.assertIn("frames_base64", inputs["json_schema"]["required"])
            self.assertNotIn("image_base64", inputs["json_schema"]["required"])
        for task in ("unconditional-image-generation", "image-to-3d", "image-to-video"):
            inputs, _ = _model_io_formats(task_info(task))
            self.assertNotIn("prompt", inputs["json_schema"]["required"])

    def test_denoising_scale_plan_preserves_steps_and_omits_fake_resolution(self):
        for task in ("unconditional-image-generation", "text-to-3d", "image-to-3d"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                planned = plan_input_scales(
                    task_info(task), ImageInfo("unused"), [1], [4], ["off"], 1,
                    tmp, input_scales="2,4",
                )
                self.assertEqual(planned.scales, [2.0, 4.0])
                self.assertEqual(planned.workload["input_scale_type"], "denoising_steps")
                data = json.loads(Path(planned.plan_file).read_text())
                for entry, steps in zip(data["entries"], (2, 4)):
                    self.assertEqual(entry["payload"]["params"]["num_inference_steps"], steps)
                    self.assertNotIn("resolution", entry["payload"])
                    self.assertEqual(entry["input_metadata"]["input_num_samples"], 1)


if __name__ == "__main__":
    unittest.main()

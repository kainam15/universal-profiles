import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host.detect import TaskInfo, detect_task
from acprof.host.orchestrator import ImageInfo
from acprof.host.static_metadata import StaticMeta, enrich_static_meta_from_input_plan
from acprof.host.model_schema import _model_io_formats
from acprof.host.input_plan import plan_input_scales
from acprof.host.task_support import TaskSupportError, require_task_support


MULTIMODAL_TASKS = ("audio-text-to-text", "image-text-to-text", "visual-question-answering", "document-question-answering", "video-text-to-text", "visual-document-retrieval", "any-to-any")


def info(task, family="multimodal", backend="transformers_model"):
    return TaskInfo("example/model", task, family, backend, "transformers", "rev", "hub_api")


class MultimodalIntegrationTests(unittest.TestCase):
    def test_nine_hub_tags_route_to_available_task_families(self):
        for task in MULTIMODAL_TASKS + ("image-text-to-image", "image-text-to-video", "image-to-image", "image-to-video"):
            diffusion = task in {"image-text-to-image", "image-text-to-video", "image-to-image", "image-to-video"}
            family = "diffusion" if diffusion else "multimodal"
            with self.subTest(task=task), patch("huggingface_hub.model_info", return_value=SimpleNamespace(pipeline_tag=task, library_name="diffusers" if diffusion else "transformers", sha="rev")):
                detected = detect_task("example/model")
                self.assertEqual(detected.task_family, family)
                require_task_support(detected)
                with self.assertRaisesRegex(TaskSupportError, "batch-size 1"):
                    require_task_support(detected, batch_size=2)

    def test_manifest_plan_persists_exact_payload_parameters_and_scale_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "workload.json"
            manifest.write_text(json.dumps({"schema_version": 1, "input_scales": [64, 128], "params": {"max_new_tokens": 7}}))
            planned = plan_input_scales(info("image-text-to-text"), ImageInfo("test-image"), [1], [4], ["off"], 1, tmp, workload_spec_path=str(manifest))
            self.assertEqual(planned.scales, [64.0, 128.0])
            self.assertEqual(planned.workload["input_scale_type"], "resolution_px")
            data = json.loads(Path(planned.plan_file).read_text())
            self.assertEqual(data["entries"][0]["payload"]["params"]["max_new_tokens"], 7)
            self.assertEqual(data["entries"][0]["input_metadata"]["input_num_samples"], 1)
            self.assertEqual(len(planned.plan_sha256), 64)
            static = StaticMeta(
                model_name="example/model", model_revision="rev", task_family="multimodal",
                pipeline_tag="image-text-to-text", runtime_backend="transformers_model",
                image_tag="test-image", batch_size=1, input_scale_type="media_scale",
                run_command="original command", model_download_url="", gpu="unknown",
                gpu_mem_total_bytes=None, model_cache_bytes=0, docker_image_bytes=0,
                environment="test", cpu_power_source="unavailable", vcpu_power_method="unavailable",
                cpu_governor="unknown", cpu_boost="unknown",
            )
            enriched = enrich_static_meta_from_input_plan(static, planned)
            self.assertEqual(enriched.input_scale_type, "resolution_px")
            self.assertEqual(enriched.input_scale_plan_sha256, planned.plan_sha256)
            self.assertEqual(enriched.run_command, "original command")

    def test_io_contract_describes_multimodal_request_and_retrieval_summary(self):
        inputs, outputs = _model_io_formats(info("visual-document-retrieval"))
        self.assertIn("samples", inputs["json_schema"]["required"])
        self.assertIn("scores", outputs["json_schema"]["properties"])

    def test_specific_architectures_do_not_fall_through_to_text_generation(self):
        for architecture, expected_task in (("Qwen2AudioForConditionalGeneration", "audio-text-to-text"), ("Qwen2_5OmniForConditionalGeneration", "any-to-any"), ("Qwen2_5_VLForConditionalGeneration", "image-text-to-text"), ("ColPaliForRetrieval", "visual-document-retrieval"), ("ViltForQuestionAnswering", "visual-question-answering"), ("LayoutLMv3ForQuestionAnswering", "document-question-answering")):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps({"architectures": [architecture]}))
                with patch("huggingface_hub.model_info", side_effect=RuntimeError("offline")), patch("huggingface_hub.hf_hub_download", return_value=str(path)):
                    detected = detect_task("example/model")
                self.assertEqual((detected.pipeline_tag, detected.task_family), (expected_task, "multimodal"))

"""Task detection, preflight and scale semantics for non-vision Hub tasks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import detect, orchestrator
from acprof.host.task_support import TaskSupportError, require_task_support


def info(task, family="nlp", backend="transformers_pipeline"):
    return detect.TaskInfo("example/model", task, family, backend,
                           "transformers", "fixed-revision", "hub_api")


class TaskCatalogIntegrationTests(unittest.TestCase):
    def test_chronos_default_scales_respect_loaded_model_context(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(orchestrator, "_start_probe_session", return_value=SimpleNamespace(name="probe")), patch.object(
            orchestrator, "_stop_container_session"
        ), patch.object(orchestrator, "_request_scale_meta", return_value={
            "input_scale_type": "context_length", "max_effective_input_scale": 512,
            "reason": "model context length",
        }) as metadata:
            planned = orchestrator.plan_input_scales(
                info("time-series-forecasting", "timeseries", "chronos"), orchestrator.ImageInfo("unused"),
                [1], [2], ["off"], 1, tmp,
            )
            metadata.assert_called_once()
            self.assertEqual(max(planned.scales), 512)
            self.assertEqual(len(planned.scales), 6)
            self.assertEqual(planned.workload["model_constraints"]["max_effective_input_scale"], 512)

    def test_new_hub_tasks_select_the_appropriate_runtime(self):
        for task, library, family, backend in (
            ("table-question-answering", "transformers", "nlp", "transformers_pipeline"),
            ("text-ranking", "sentence-transformers", "nlp", "cross_encoder"),
            ("sentence-similarity", "sentence-transformers", "nlp", "sentence_transformers"),
            ("text-to-audio", "transformers", "audio", "transformers_pipeline"),
            ("audio-to-audio", "transformers", "audio", "transformers_model"),
            ("voice-activity-detection", "unknown", "audio", "torchscript"),
            ("tabular-classification", "unknown", "structured", "torchscript"),
            ("tabular-regression", "sklearn", "structured", "skops"),
            ("reinforcement-learning", "unknown", "structured", "torchscript"),
            ("robotics", "unknown", "structured", "torchscript"),
            ("graph-ml", "unknown", "structured", "torchscript"),
        ):
            with self.subTest(task=task), patch("huggingface_hub.model_info", return_value=SimpleNamespace(
                pipeline_tag=task, library_name=library, sha="fixed-revision",
            )):
                actual = detect.detect_task("example/model")
                self.assertEqual((actual.task_family, actual.runtime_backend), (family, backend))
                self.assertEqual(actual.model_revision, "fixed-revision")
                require_task_support(actual)

    def test_explicit_backend_override_is_preserved_and_incompatibility_rejected(self):
        with patch.object(detect, "_detect_from_hub", return_value=info("robotics", "structured")):
            actual = detect.detect_task("example/model", override_backend="diffusers")
        self.assertEqual(actual.runtime_backend, "diffusers")
        with self.assertRaisesRegex(TaskSupportError, "后端"):
            require_task_support(actual)

    def test_table_planning_uses_rows_instead_of_token_binary_search(self):
        generator = SimpleNamespace(
            default_input_scales=lambda: [1, 4],
            generate=lambda scale: {"table": {"name": ["a"] * int(scale)}, "query": "How many?"},
            effective_input_scale=lambda scale, payload: float(len(payload["table"]["name"])),
            scale_label=lambda scale: f"rows{int(scale)}",
            plan_metadata=lambda: {"input_scale_type": "table_rows"},
            input_metadata=lambda scale, payload: {"table_rows": len(payload["table"]["name"])},
        )
        with tempfile.TemporaryDirectory() as tmp, patch("acprof.workloads.get_generator", return_value=generator), patch.object(
            orchestrator, "_start_probe_session", side_effect=AssertionError("table rows entered token planner")
        ):
            planned = orchestrator.plan_input_scales(
                info("table-question-answering"), orchestrator.ImageInfo("unused"),
                [1], [2], ["off"], 1, tmp,
            )
            self.assertEqual(planned.scales, [1.0, 4.0])
            self.assertEqual(planned.workload["input_scale_type"], "table_rows")
            saved = json.loads(Path(planned.plan_file).read_text())
            self.assertEqual(saved["entries"][1]["input_metadata"]["table_rows"], 4)

    def test_text_audio_uses_token_planning_and_forwards_manifest(self):
        for task in ("text-to-speech", "text-to-audio"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp, patch.object(
                orchestrator, "_plan_manual_nlp_scales", return_value="planned tokens"
            ) as plan, patch.object(orchestrator, "_plan_audio_scales", side_effect=AssertionError("text treated as waveform")):
                result = orchestrator.plan_input_scales(
                    info(task, "audio"), orchestrator.ImageInfo("unused"),
                    [1], [2], ["off"], 1, tmp, input_scales="8,16", workload_spec_path="text.json",
                )
                self.assertEqual(result, "planned tokens")
                self.assertEqual(plan.call_args.kwargs["workload_spec_path"], "text.json")

    def test_static_input_contracts_describe_task_inputs(self):
        for task, family, required, output_type in (
            ("table-question-answering", "nlp", {"table", "query"}, "table_answer"),
            ("sentence-similarity", "nlp", {"query", "documents"}, "similarity"),
            ("text-ranking", "nlp", {"query", "documents"}, "ranking"),
            ("text-to-speech", "audio", {"text"}, "audio"),
            ("text-to-audio", "audio", {"text"}, "audio"),
        ):
            with self.subTest(task=task):
                inputs, outputs = orchestrator._model_io_formats(info(task, family))
                self.assertTrue(required.issubset(inputs["json_schema"]["required"]))
                self.assertIn(output_type, outputs["json_schema"]["properties"]["output_type"]["enum"])


if __name__ == "__main__":
    unittest.main()

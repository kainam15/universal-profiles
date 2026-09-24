"""Model discovery retains evidence and requires explicit task semantics."""
import base64
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.host.detect import TaskInfo, detect_task
from acprof.host.input_plan import _get_task_generator
from acprof.host.runtime_images import request_fingerprint
from acprof.host.task_support import TaskSupportError, require_task_support


REVISION = "a" * 40


class ModelDiscoveryTests(unittest.TestCase):
    def discover(self, metadata, *, tag=None, library=None, files=(), **options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in metadata.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            hub = SimpleNamespace(pipeline_tag=tag, library_name=library, sha=REVISION,
                                  config={}, siblings=[SimpleNamespace(rfilename=name)
                                                       for name in [*metadata, *files]])

            def download(**kwargs):
                self.assertEqual(kwargs.get("revision"), REVISION)
                return str(root / kwargs["filename"])

            with patch("huggingface_hub.model_info", return_value=hub), patch(
                "huggingface_hub.hf_hub_download", side_effect=download,
            ), patch("sys.stderr", io.StringIO()):
                return detect_task("example/model", **options)

    def test_onnx_without_hub_task_uses_declared_interface(self):
        spec = {"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                "model_file": "iris.onnx", "feature_dim": 4}
        task = self.discover({"acprof_model.json": spec}, files=("iris.onnx",))
        require_task_support(task)
        self.assertEqual((task.pipeline_tag, task.runtime_backend), ("tabular-classification", "onnxruntime"))
        self.assertEqual(task.model_revision, REVISION)
        self.assertEqual(task.model_resolution["status"], "candidate")
        self.assertEqual(task.model_resolution["candidates"][0]["evidence"], ["repository_model_spec"])

    def test_bare_onnx_retains_candidates_but_never_guesses_task(self):
        task = self.discover({}, files=("iris.onnx",))
        self.assertEqual(task.model_resolution["status"], "needs_configuration")
        self.assertIn("tabular-classification", {item["task"] for item in task.model_resolution["candidates"]})
        self.assertIn("tabular-regression", {item["task"] for item in task.model_resolution["candidates"]})
        with self.assertRaisesRegex(TaskSupportError, "model-spec|--task"):
            require_task_support(task)

    def test_explicit_task_selects_onnx_backend_from_artifact(self):
        task = self.discover({}, files=("iris.onnx",), override_tag="tabular-classification")
        require_task_support(task)
        self.assertEqual(task.runtime_backend, "onnxruntime")
        self.assertEqual(task.model_revision, REVISION)

    def test_multiple_onnx_artifacts_require_a_selection(self):
        task = self.discover({}, files=("encoder.onnx", "decoder.onnx"), override_tag="text-classification")
        with self.assertRaisesRegex(TaskSupportError, "model_file"):
            require_task_support(task)

    def test_config_inference_keeps_hub_snapshot_and_file_evidence(self):
        task = self.discover({"config.json": {"architectures": ["BertForMaskedLM"], "model_type": "bert"}},
                             files=("model.safetensors",), library="transformers")
        self.assertEqual(task.pipeline_tag, "fill-mask")
        self.assertEqual(task.model_revision, REVISION)
        self.assertIn("model.safetensors", task.repository_files)
        self.assertIn("config.json", task.repository_metadata)

    def test_two_architectures_are_ambiguous_instead_of_first_match(self):
        task = self.discover({"config.json": {"architectures": ["BertForMaskedLM", "BertForSequenceClassification"]}},
                             library="transformers")
        self.assertEqual(task.model_resolution["status"], "ambiguous")
        with self.assertRaisesRegex(TaskSupportError, "fill-mask.*text-classification"):
            require_task_support(task)

    def test_hub_and_declared_tasks_conflict_until_user_selects_one(self):
        spec = {"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                "model_file": "iris.onnx", "feature_dim": 4}
        task = self.discover({"acprof_model.json": spec}, tag="tabular-regression", files=("iris.onnx",))
        with self.assertRaisesRegex(TaskSupportError, "conflict|冲突"):
            require_task_support(task)
        selected = self.discover({"acprof_model.json": spec}, tag="tabular-regression", files=("iris.onnx",),
                                  override_tag="tabular-classification")
        require_task_support(selected)

    def test_custom_pipeline_alias_uses_explicit_standard_protocol(self):
        config = {"model_type": "bert", "custom_pipelines": {
            "acme-classify": {"impl": "custom_pipeline.Classifier", "pt": ["AutoModelForSequenceClassification"]}}}
        spec = {"schema_version": 1, "format": "transformers-pipeline", "task": "text-classification",
                "pipeline_task": "acme-classify"}
        task = self.discover({"config.json": config, "acprof_model.json": spec},
                             tag="acme-classify", library="transformers", files=("custom_pipeline.py",))
        require_task_support(task)
        self.assertEqual(task.pipeline_tag, "text-classification")
        self.assertEqual(task.model_resolution["pipeline_task"], "acme-classify")
        self.assertEqual(task.model_resolution["interface_kind"], "custom_pipeline")

    def test_custom_auto_class_is_a_candidate_without_host_import(self):
        config = {"auto_map": {"AutoModelForSequenceClassification": "custom_model.Classifier"}}
        task = self.discover({"config.json": config}, library="transformers", files=("custom_model.py",))
        require_task_support(task)
        self.assertEqual(task.pipeline_tag, "text-classification")
        self.assertEqual(task.model_resolution["interface_kind"], "custom_auto")

    def test_cross_repository_code_is_not_treated_as_offline_complete(self):
        config = {"auto_map": {"AutoModelForSequenceClassification": "other/repo--custom_model.Classifier"}}
        task = self.discover({"config.json": config}, tag="text-classification", library="transformers")
        with self.assertRaisesRegex(TaskSupportError, "code|代码"):
            require_task_support(task)

    def test_malformed_custom_metadata_reports_configuration_error(self):
        task = self.discover({"config.json": {"auto_map": ["not-a-map"]}},
                             tag="image-text-to-text", library="transformers")
        with self.assertRaisesRegex(TaskSupportError, "auto_map"):
            require_task_support(task)


class ModelSpecificationTests(unittest.TestCase):
    def test_invalid_format_reports_a_declaration_error(self):
        from acprof.model_spec import validate_model_spec
        with self.assertRaisesRegex(ValueError, "format"):
            validate_model_spec({"schema_version": 1, "format": [], "task": "text-classification"})

    def test_custom_classification_requires_actual_labels_and_scores(self):
        from acprof.container.validation import OutputValidationError, validate_output
        context = {"task_type": "text-classification", "model_spec": {"format": "transformers-pipeline"}}
        for output in ([], [{"unrelated": True}], [{"label": "", "score": 0.5}]):
            with self.subTest(output=output), self.assertRaisesRegex(OutputValidationError, "label|classification"):
                validate_output(context, {"input_scale": 2}, {"_effective_input_scale": 2}, output,
                                {"task": "text-classification", "output_type": "label", "n_results": len(output)})

    def test_model_spec_content_changes_resume_identity(self):
        from acprof.host.run_state import run_options
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            args = SimpleNamespace(cpus="1", mems="2", gpus="off", model_spec=str(path))
            path.write_text('{"model_file":"first.onnx"}')
            before = run_options(args)
            path.write_text('{"model_file":"second.onnx"}')
            after = run_options(args)
        self.assertNotEqual(before["model_spec_sha256"], after["model_spec_sha256"])

    def task(self, spec):
        info = TaskInfo("example/model", "tabular-classification", "structured", "onnxruntime",
                        "onnx", REVISION, "manual")
        info.repository_metadata["acprof_model.json"] = spec
        return info

    def test_model_declaration_drives_workload_feature_width(self):
        task = self.task({"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                          "model_file": "iris.onnx", "feature_dim": 4})
        generator = _get_task_generator(task, 1)
        self.assertEqual(len(generator.generate(2)["features"][0]), 4)

    def test_workload_cannot_silently_override_model_feature_width(self):
        task = self.task({"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                          "model_file": "iris.onnx", "feature_dim": 4})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps({"schema_version": 1, "feature_dim": 8}))
            with self.assertRaisesRegex(ValueError, "feature_dim"):
                _get_task_generator(task, 1, workload_spec_path=str(path))

    def test_declaration_changes_service_image_identity(self):
        task = self.task({"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                          "model_file": "first.onnx", "feature_dim": 4})
        before = request_fingerprint(task)
        task.repository_metadata["acprof_model.json"]["model_file"] = "second.onnx"
        self.assertNotEqual(before, request_fingerprint(task))

    def test_baked_declaration_takes_precedence_over_repository_file(self):
        from acprof.model_spec import load_model_spec
        spec = {"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                "model_file": "selected.onnx", "feature_dim": 4}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "acprof_model.json").write_text(json.dumps({**spec, "model_file": "old.onnx"}))
            encoded = base64.b64encode(json.dumps(spec).encode()).decode()
            with patch.dict(os.environ, {"ACPROF_MODEL_SPEC_B64": encoded}):
                self.assertEqual(load_model_spec(directory, "tabular-classification", expected_format="onnxruntime"), spec)

    def test_local_spec_can_resolve_a_repository_without_task_metadata(self):
        spec = {"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                "model_file": "selected.onnx", "feature_dim": 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(spec))
            task = ModelDiscoveryTests().discover({}, files=("selected.onnx",), model_spec_path=str(path))
        require_task_support(task)
        self.assertEqual(task.model_spec, spec)
        self.assertEqual(task.runtime_backend, "onnxruntime")
        self.assertEqual(task.model_resolution["selection"]["source"], "explicit")
        self.assertEqual(task.model_resolution["candidates"][0]["evidence"], ["local_model_spec"])

    def test_pipeline_bridge_keeps_semantic_task_and_uses_custom_entry(self):
        from acprof.container.handlers.nlp import NLPHandler
        spec = {"schema_version": 1, "format": "transformers-pipeline", "task": "text-classification",
                "pipeline_task": "acme-classify"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "acprof_model.json").write_text(json.dumps(spec))
            (root / "config.json").write_text(json.dumps({"custom_pipelines": {
                "acme-classify": {"impl": "custom_pipeline.Classifier"}}}))
            (root / "custom_pipeline.py").write_text("raise AssertionError('host must not import this')\n")
            with patch.dict("sys.modules", {"torch": SimpleNamespace(float32="float32", float16="float16"),
                                           "transformers": SimpleNamespace()}):
                with patch("transformers.pipeline", create=True) as pipeline:
                    pipeline.return_value = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
                    context = NLPHandler().load(str(root), "text-classification", "transformers_pipeline", "cpu")
            self.assertEqual(pipeline.call_args.kwargs["task"], "acme-classify")
            self.assertEqual(context["task_type"], "text-classification")


if __name__ == "__main__":
    unittest.main()

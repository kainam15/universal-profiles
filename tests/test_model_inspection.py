"""Inspection exposes provenance without starting measurement or loading models."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_model_contract as fixture
from acprof.cli.main import main


class ModelInspectionTests(unittest.TestCase):
    def test_native_probe_uses_family_defaults_when_generator_has_no_declared_scales(self):
        from types import SimpleNamespace
        from acprof.host.detect import TaskInfo
        from acprof.host.model_inspection import probe_model_contract
        for family, tag, expected_scale, field in (
                ("nlp", "fill-mask", 64, "text"), ("cv", "image-classification", 0.1, "image_base64")):
            task = TaskInfo("example/native", tag, family, "transformers_pipeline", "transformers", "a" * 40, "hub")
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory, patch(
                    "acprof.host.docker_runtime.prepare_image", return_value=SimpleNamespace(tag="sha256:" + "b" * 64)), patch(
                    "acprof.host.preflight.require_native_linux_host"), patch(
                    "acprof.host.preflight.require_native_docker"), patch(
                    "acprof.host.run_state.MeasurementLock"), patch(
                    "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}) as validate:
                probe_model_contract(task, directory, mode="full")
                entry = json.loads(Path(validate.call_args.kwargs["planned"].plan_file).read_text())["entries"][0]
                self.assertEqual(entry["input_scale"], expected_scale)
                self.assertTrue(entry["payload"][field])
                if family == "nlp":
                    self.assertIn("[MASK]", entry["payload"][field])

    def test_full_probe_accepts_native_models_and_preserves_feature_width(self):
        from types import SimpleNamespace
        from acprof.host.detect import TaskInfo
        from acprof.host.model_inspection import probe_model_contract
        task = TaskInfo("example/iris", "tabular-classification", "structured", "onnxruntime",
                        "onnx", "a" * 40, "manual")
        task.model_spec = {"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                           "model_file": "iris.onnx", "feature_dim": 4}
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.docker_runtime.prepare_image",
                return_value=SimpleNamespace(tag="sha256:" + "b" * 64)), patch(
                "acprof.host.preflight.require_native_linux_host"), patch(
                "acprof.host.preflight.require_native_docker"), patch(
                "acprof.host.run_state.MeasurementLock"), patch(
                "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}) as validate:
            report = probe_model_contract(task, directory, mode="full")
            self.assertEqual(report["status"], "ok")
            plan = json.loads(Path(validate.call_args.kwargs["planned"].plan_file).read_text())
            self.assertEqual(len(plan["entries"][0]["payload"]["features"][0]), 4)
            self.assertNotIn("params", plan["entries"][0]["payload"])

    def test_review_provenance_survives_export_and_subprocess_inspection(self):
        from acprof.model_contract import write_model_resolution
        from acprof.model_spec import task_model_spec
        task = fixture.ModelContractTests().discover()
        original = task.model_resolution["contract"]
        declaration = task_model_spec(task)
        author_task = fixture.ModelContractTests().discover(spec=declaration)
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=author_task), patch(
            "acprof.host.env_utils.bootstrap_project_env",
        ), contextlib.redirect_stdout(io.StringIO()):
            write_model_resolution(task, directory)
            path = Path(directory, "acprof_model.json")
            path.write_text(json.dumps(declaration))
            self.assertEqual(main(["inspect", task.model_id, "--model-spec", str(path),
                                   "--expected-revision", task.model_revision, "--output-dir", directory]), 0)
            saved = json.loads(Path(directory, "model_resolution.json").read_text())["contract"]
            self.assertEqual(saved["cache_key"], original["cache_key"])
            self.assertEqual(saved["fields"]["multimodal.inputs.prompt"]["sources"],
                             original["fields"]["multimodal.inputs.prompt"]["sources"])

    def test_changed_revision_refuses_probe_and_preserves_previous_report(self):
        task = fixture.ModelContractTests().discover()
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
            "acprof.host.env_utils.bootstrap_project_env",
        ), patch("acprof.host.docker_runtime.prepare_image") as build:
            path = Path(directory, "model_resolution.json")
            path.write_text('{"preserve":true}')
            self.assertEqual(main(["inspect", task.model_id, "--expected-revision", "b" * 40,
                                   "--probe", "full", "--output-dir", directory]), 2)
            self.assertEqual(path.read_text(), '{"preserve":true}')
            build.assert_not_called()

    def test_inspect_explains_and_exports_the_static_contract(self):
        task = fixture.ModelContractTests().discover()
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
            "acprof.host.env_utils.bootstrap_project_env",
        ), patch("acprof.host.docker_runtime.prepare_image") as build, contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["inspect", task.model_id, "--explain", "--output-dir", directory])
            self.assertEqual(code, 0)
            report = json.loads(Path(directory, "model_resolution.json").read_text())
            self.assertEqual(report["contract"]["runtime_validation"], "not_run")
            self.assertIn("multimodal.inputs.prompt", output.getvalue())
            self.assertIn("pipeline.py", output.getvalue())
            build.assert_not_called()

    def test_unresolved_inspection_exports_draft_and_returns_two(self):
        task = fixture.ModelContractTests().discover(fixture.SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["turns"]'))
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
            "acprof.host.env_utils.bootstrap_project_env",
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["inspect", task.model_id, "--output-dir", directory]), 2)
            self.assertIn("multimodal.inputs.turns", output.getvalue())
            self.assertTrue(Path(directory, "model_resolution.json").is_file())

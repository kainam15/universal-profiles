"""Coverage keeps a frozen denominator and does not count resource failures as semantic errors."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.cli.main import main
from test_resolution_decisions import candidate


class ModelCoverageTests(unittest.TestCase):
    def test_runtime_failures_keep_semantic_review_and_frozen_denominator(self):
        from acprof.host.model_coverage import run_sample
        from acprof.host.model_inspection import ProbePreparationError
        task = candidate(tag="text-generation")
        manifest = {"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                    "models": [{"model_id": f"example/model-{index}", "revision": "a" * 40, "weight": 1,
                                "semantic_reference": {"task": "text-generation", "source": "manual_fixture"}}
                               for index in range(3)]}
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
                "acprof.host.automation.check_repository_access"), patch(
                "acprof.host.model_inspection.probe_model_contract", side_effect=[
                    {"status": "ok"}, {"status": "resource_limited"}, ProbePreparationError("build failed")]):
            report = run_sample(manifest, Path(directory, "report"), probe="full")
            summary = report["summary"]
            self.assertEqual(summary["weighted_runtime_success_rate"], 1 / 3)
            self.assertEqual(summary["weighted_resource_limited_rate"], 1 / 3)
            self.assertEqual(summary["semantic_accuracy_on_reviewed_selections"], 1)
            self.assertEqual(summary["semantic_wrong_selections"], 0)
            self.assertEqual(summary["failure_stages"], {"resource": 1, "environment": 1})

    def test_static_coverage_keeps_unmeasured_separate_and_requires_reference(self):
        first = candidate(tag="text-generation")
        second = candidate(tag="fill-mask", hub={"transformers_info": {"pipeline_tag": "text-generation"}})
        manifest = {"schema_version": 1, "sampling": "fixed_regression", "weight_basis": "fixed_test_weights",
                    "models": [{"model_id": "example/first", "revision": "a" * 40, "weight": 3,
                                "semantic_reference": {"task": "text-generation", "source": "manual_fixture"}},
                               {"model_id": "example/second", "revision": "a" * 40, "weight": 1}]}
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", side_effect=[first, second]), patch(
                "acprof.host.env_utils.bootstrap_project_env"), contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / "models.json"
            source.write_text(json.dumps(manifest))
            self.assertEqual(main(["coverage", "run", str(source), "--output-dir", str(root / "report")]), 0)
            report = json.loads((root / "report/coverage.json").read_text())
            self.assertEqual(report["summary"]["sample_weight"], 4)
            self.assertEqual(report["summary"]["weighted_resolution_rate"], 0.75)
            self.assertEqual(report["summary"]["weighted_abstain_rate"], 0.25)
            self.assertIsNone(report["summary"]["weighted_runtime_success_rate"])
            self.assertIsNone(report["summary"]["weighted_access_denied_rate"])
            self.assertEqual(report["summary"]["semantic_accuracy_on_reviewed_selections"], 1.0)
            self.assertEqual(report["rows"][1]["runtime_status"], "not_requested")

    def test_snapshot_freezes_sha_and_weights_without_creating_semantic_oracles(self):
        model = SimpleNamespace(id="example/model", sha="a" * 40, downloads=7, pipeline_tag="text-generation",
                                library_name="transformers")
        wrapper = SimpleNamespace(id="timm/wrapper", sha="b" * 40, downloads=9, pipeline_tag="text-generation",
                                  library_name="timm")
        with tempfile.TemporaryDirectory() as directory, patch("huggingface_hub.HfApi.list_models", return_value=[wrapper, model]), patch(
                "acprof.host.env_utils.bootstrap_project_env"), contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory, "sample.json")
            self.assertEqual(main(["coverage", "snapshot", "--stratum", "text-generation:transformers",
                                   "--output", str(path), "--limit", "1"]), 0)
            data = json.loads(path.read_text())
            self.assertEqual(data["models"][0]["revision"], "a" * 40)
            self.assertEqual(data["models"][0]["weight"], 7)
            self.assertEqual(len(data["models"]), 1)
            self.assertNotIn("semantic_reference", data["models"][0])

    def test_unpinned_manifest_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task") as detect, patch(
                "acprof.host.env_utils.bootstrap_project_env"), contextlib.redirect_stderr(io.StringIO()):
            path = Path(directory, "models.json")
            path.write_text(json.dumps({"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                                        "models": [{"model_id": "example/model", "revision": "main", "weight": 1}]}))
            self.assertEqual(main(["coverage", "run", str(path), "--output-dir", str(Path(directory, "report"))]), 2)
            detect.assert_not_called()

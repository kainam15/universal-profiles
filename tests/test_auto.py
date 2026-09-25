"""Automatic preparation preserves requested scope and never guesses model identity."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acprof.cli.main import main
from acprof.host.doctor import DoctorCheck
from test_resolution_decisions import candidate


class AutoTests(unittest.TestCase):
    def test_preflight_artifacts_can_be_handed_to_run_state_without_allowing_changes(self):
        from acprof.cli.run_args import build_parser
        from acprof.host.automation import AutomaticRun
        from acprof.host.run_state import RunState, RunStateError
        with tempfile.TemporaryDirectory() as directory, patch(
                "acprof.host.doctor.collect_checks", return_value=[DoctorCheck("docker", "available", "ok")]), patch(
                "acprof.host.automation.check_repository_access", return_value={"status": "accessible"}), patch(
                "acprof.host.detect.detect_task", return_value=candidate(tag="text-generation")), patch(
                "acprof.host.run_state.host_identity", return_value={}):
            args = build_parser(automatic=True).parse_args(["example/model", "--profiling-mode", "basic",
                                                           "--output-dir", directory])
            automatic = AutomaticRun(args)
            automatic.prepare()
            fingerprints = automatic.preparation_artifacts
            original = automatic.path.read_bytes()
            automatic.path.write_text('{"changed":true}')
            with self.assertRaises(RunStateError):
                RunState(automatic.root, {}, resume=False, project_dir=directory, preparation_artifacts=fingerprints)
            automatic.path.write_bytes(original)
            state = RunState(automatic.root, {}, resume=False, project_dir=directory, preparation_artifacts=fingerprints)
            state.close()
            self.assertTrue((automatic.root / "run_state.json").is_file())

    def test_partial_collection_cannot_be_reported_as_automatic_success(self):
        from argparse import Namespace
        from acprof.host.automation import AutomaticRun
        with tempfile.TemporaryDirectory() as directory:
            run = AutomaticRun(Namespace(model="example/model", output_dir=directory, profiling_mode="full"))
            run.started = True
            run.root.mkdir()
            (run.root / "run_state.json").write_text(json.dumps({"status": "complete", "outcome": "ok"}))
            (run.root / "capability_report.json").write_text(json.dumps({"collection_succeeded": True,
                                                                        "requested_measurements_complete": False}))
            self.assertEqual(run.finish(), 2)
            self.assertEqual(json.loads(run.path.read_text())["status"], "failed")

    def invoke(self, directory, *, mode="auto", task=None, checks=None, access_error=None):
        task = task or candidate(tag="text-generation")
        checks = checks or [DoctorCheck("docker", "available", "ok"),
                            DoctorCheck("rapl", "unavailable", "no RAPL")]

        def run(**kwargs):
            args = kwargs["args"]
            self.assertEqual(args.profiling_mode, "basic")
            self.assertEqual(kwargs["prepared_task"].model_revision, "a" * 40)
            output = Path(directory, "example--model")
            (output / "run_state.json").write_text(json.dumps({"status": "complete", "outcome": "ok"}))
            (output / "capability_report.json").write_text(json.dumps({"requested_measurements_complete": True,
                                                                       "collection_succeeded": True}))
            return 0

        with patch("acprof.host.env_utils.bootstrap_project_env"), patch(
                "acprof.host.doctor.collect_checks", return_value=checks), patch(
                "huggingface_hub.HfApi.auth_check", side_effect=access_error), patch(
                "acprof.host.detect.detect_task", return_value=task), patch(
                "acprof.host.run_state.MeasurementLock"), patch(
                "acprof.cli.run.main", side_effect=run) as collect, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["auto", "example/model", "--profiling-mode", mode,
                         "--gpus", "off", "--output-dir", directory, "--notify", "none"])
            return code, collect.call_count

    def test_auto_mode_selects_basic_and_records_requested_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.invoke(directory), (0, 1))
            report = json.loads(Path(directory, "example--model/auto_report.json").read_text())
            self.assertEqual(report["requested_profiling_mode"], "auto")
            self.assertEqual(report["profiling_mode"], "basic")
            self.assertEqual(report["status"], "succeeded")

    def test_explicit_full_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls = self.invoke(directory, mode="full")
            self.assertNotEqual(code, 0)
            self.assertEqual(calls, 0)

    def test_semantic_conflict_exports_explanation_without_collection(self):
        task = candidate(tag="fill-mask", hub={"transformers_info": {"pipeline_tag": "text-generation"}})
        with tempfile.TemporaryDirectory() as directory:
            code, calls = self.invoke(directory, task=task)
            self.assertEqual((code, calls), (2, 0))
            self.assertTrue(Path(directory, "example--model/model_resolution.json").is_file())

    def test_host_failure_never_becomes_basic_success(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls = self.invoke(directory, checks=[DoctorCheck("docker", "unavailable", "no Docker")])
            self.assertNotEqual(code, 0)
            self.assertEqual(calls, 0)

    def test_access_failure_stops_and_does_not_export_provider_details(self):
        import httpx
        from huggingface_hub.errors import GatedRepoError
        with tempfile.TemporaryDirectory() as directory:
            response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/api/models/example/model"))
            code, calls = self.invoke(directory, access_error=GatedRepoError("sensitive-provider-detail", response=response))
            self.assertEqual((code, calls), (2, 0))
            data = Path(directory, "example--model/auto_report.json").read_text()
            self.assertIn("GatedRepoError", data)
            self.assertNotIn("sensitive-provider-detail", data)

    def test_resume_reuses_saved_revision_without_hub_queries(self):
        from argparse import Namespace
        from dataclasses import asdict
        from acprof.host.automation import AutomaticRun
        task = candidate(tag="text-generation")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "example--model")
            root.mkdir()
            (root / "auto_report.json").write_text(json.dumps({"schema_version": 1, "requested_profiling_mode": "basic", "status": "failed"}))
            (root / "run_state.json").write_text(json.dumps({"schema_version": 1, "runtime": {"task": asdict(task)},
                                                            "options": {"profiling_mode": "basic"}}))
            args = Namespace(model=task.model_id, output_dir=directory, resume=True, revision=None,
                             profiling_mode="basic", batch_size=1, gpus="off", sniff_iface="docker0")
            with patch("acprof.host.detect.detect_task") as detect, patch("huggingface_hub.HfApi.auth_check") as access, patch(
                    "acprof.host.doctor.collect_checks", return_value=[DoctorCheck("docker", "available", "ok")]):
                restored = AutomaticRun(args).prepare()
            self.assertEqual(restored.model_revision, "a" * 40)
            self.assertEqual(args.revision, "a" * 40)
            detect.assert_not_called()
            access.assert_not_called()

    def test_existing_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "example--model")
            target.mkdir()
            (target / "auto_report.json").write_text('{"preserve":true}')
            code, calls = self.invoke(directory)
            self.assertNotEqual(code, 0)
            self.assertEqual(calls, 0)
            self.assertEqual((target / "auto_report.json").read_text(), '{"preserve":true}')

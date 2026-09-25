"""Contract probes run outside measurements and publish only observed evidence."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_runtime_validation as runtime_fixture
import test_model_contract as contract_fixture
from acprof.host.runtime_validation import validate_runtime


class ModelProbeTests(unittest.TestCase):
    def test_empty_device_set_and_mutable_image_cannot_claim_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            options = runtime_fixture.RuntimeValidationTests().fixture(Path(directory))
            with self.assertRaises(ValueError):
                validate_runtime(**{**options, "gpu_list": []})
            options["image_info"].tag = "example:latest"
            with self.assertRaises(ValueError):
                validate_runtime(**options)

    def test_basic_mode_does_not_claim_inference_and_only_uses_cpu(self):
        response = {"status": "ok", "mode": "basic", "stages": [
            {"stage": stage, "status": "verified"} for stage in ("import", "signature")], "inference": "not_run"}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.runtime_validation.subprocess.run", return_value=subprocess.CompletedProcess([], 0,
                stdout="ACPROF_RUNTIME_VALIDATION=" + json.dumps(response), stderr=""),
        ) as run, patch("acprof.host.docker_runtime._inspect_container_state", return_value={}):
            task = contract_fixture.ModelContractTests().discover()
            options = runtime_fixture.RuntimeValidationTests().fixture(Path(directory))
            options.update(task_info=task, gpu_list=["off"], mode="basic")
            report = validate_runtime(**options)
            self.assertEqual(report["mode"], "basic")
            self.assertEqual(task.model_resolution["contract"]["runtime_validation"]["status"], "basic_verified")
            command = run.call_args_list[0].args[0]
            self.assertIn("ACPROF_CONTRACT_PROBE_MODE=basic", command)
            self.assertNotIn("--gpus", command)
            with self.assertRaises(ValueError):
                validate_runtime(**{**options, "gpu_list": ["on"]})

    def test_probe_is_readonly_and_updates_the_contract_report(self):
        commands = []
        stages = [{"stage": stage, "status": "verified"} for stage in
                  ("load", "preprocess", "predict", "postprocess", "validate_output")]
        response = {"status": "ok", "stages": stages, "validation": {
            "protocol": {"status": "verified"}, "task": {"status": "verified"}}}
        def run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="ACPROF_RUNTIME_VALIDATION=" + json.dumps(response), stderr="")
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.runtime_validation.subprocess.run", side_effect=run,
        ), patch("acprof.host.docker_runtime._inspect_container_state", return_value={}):
            task = contract_fixture.ModelContractTests().discover()
            options = runtime_fixture.RuntimeValidationTests().fixture(Path(directory))
            options.update(task_info=task, gpu_list=["off"])
            validate_runtime(**options)
            command = commands[0]
            self.assertIn("--read-only", command)
            self.assertIn("no-new-privileges", command)
            self.assertNotIn("--gpus", command)
            saved = json.loads(Path(directory, "model_resolution.json").read_text())["contract"]
            self.assertEqual(saved["runtime_validation"]["status"], "verified")
            self.assertEqual(saved["runtime_validation"]["image_id"], "sha256:" + "b" * 64)
            self.assertEqual(saved["runtime_validation"]["mode"], "full")
            self.assertEqual(list(Path(directory).glob("*.csv")), [])

    def test_incomplete_runtime_record_never_becomes_verified(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.runtime_validation.subprocess.run", return_value=subprocess.CompletedProcess([], 0,
                    stdout='ACPROF_RUNTIME_VALIDATION={"status":"ok"}', stderr=""),
        ), patch("acprof.host.docker_runtime._inspect_container_state", return_value={}):
            task = contract_fixture.ModelContractTests().discover()
            options = runtime_fixture.RuntimeValidationTests().fixture(Path(directory))
            options.update(task_info=task, gpu_list=["off"])
            with self.assertRaisesRegex(RuntimeError, "contract|validation"):
                validate_runtime(**options)
            self.assertEqual(task.model_resolution["contract"]["runtime_validation"]["status"], "error")

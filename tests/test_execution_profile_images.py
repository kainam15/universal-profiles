import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.host.profilers import execution_environment as environment


IMAGE_ID = "sha256:" + "a" * 64
MODEL_TAG = "acprof-test:latest"


def image_result(labels=None):
    return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
        "Id": IMAGE_ID, "Config": {"Labels": labels}, "RepoTags": [MODEL_TAG],
    }))


class ExecutionProfileImageTests(unittest.TestCase):
    def test_both_tools_use_the_same_immutable_model_image(self):
        labels = {environment.EXECUTION_RUNTIME_LABEL_PREFIX + tool: "1"
                  for tool in ("massif", "nsys")}
        with patch.object(environment, "_run", return_value=image_result(labels)) as run:
            for tool in ("massif", "nsys"):
                self.assertEqual(environment.require_execution_image(MODEL_TAG, tool), IMAGE_ID)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][:4], ["docker", "image", "inspect", MODEL_TAG])

    def test_missing_or_outdated_capability_fails_without_build_or_tag(self):
        for tool in ("massif", "nsys"):
            for labels in (None, {}, {environment.EXECUTION_RUNTIME_LABEL_PREFIX + tool: "0"}):
                with self.subTest(tool=tool, labels=labels), patch.object(
                    environment, "_run", return_value=image_result(labels),
                ) as run, self.assertRaisesRegex(RuntimeError, tool + "_runtime_unavailable.*rebuild"):
                    environment.require_execution_image(MODEL_TAG, tool)
                self.assertEqual(run.call_count, 1)

    def test_capability_is_checked_for_the_requested_tool(self):
        labels = {environment.EXECUTION_RUNTIME_LABEL_PREFIX + "nsys": "1"}
        with patch.object(environment, "_run", return_value=image_result(labels)) as run:
            with self.assertRaisesRegex(RuntimeError, "massif_runtime_unavailable"):
                environment.require_execution_image(MODEL_TAG, "massif")
        self.assertEqual(run.call_count, 1)

    def test_missing_image_does_not_trigger_a_pull_or_build(self):
        missing = SimpleNamespace(returncode=1, stdout="", stderr="Error: No such image: test")
        with patch.object(environment, "_run", return_value=missing) as run:
            with self.assertRaisesRegex(RuntimeError, "nsys_runtime_unavailable.*image not found"):
                environment.require_execution_image(MODEL_TAG, "nsys")
        self.assertEqual(run.call_count, 1)

    def test_daemon_failure_is_reported_without_rebuilding(self):
        failed = SimpleNamespace(returncode=1, stdout="", stderr="Cannot connect to Docker daemon")
        with patch.object(environment, "_run", return_value=failed) as run:
            with self.assertRaisesRegex(RuntimeError, "execution_image_inspect_failed.*Docker daemon"):
                environment.require_execution_image(MODEL_TAG, "nsys")
        self.assertEqual(run.call_count, 1)

    def test_invalid_image_metadata_is_rejected(self):
        for payload in ("not json", "[]", '{"Id":"latest"}', json.dumps({
            "Id": IMAGE_ID, "Config": {"Labels": ["invalid"]},
        })):
            result = SimpleNamespace(returncode=0, stdout=payload, stderr="")
            with self.subTest(payload=payload), patch.object(environment, "_run", return_value=result) as run:
                with self.assertRaisesRegex(RuntimeError, "invalid_metadata"):
                    environment.require_execution_image(MODEL_TAG, "nsys")
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()

"""Independent validation identifies failure stages without producing measurements."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from acprof.container import runtime_validate


class ValidationStageTests(unittest.TestCase):
    def fixtures(self, stack, *, failed=None):
        handler = Mock()
        handler.load.return_value = {}
        handler.preprocess.return_value = {"_effective_input_scale": 1}
        handler.predict.return_value = [1.0]
        handler.postprocess.return_value = {"task": "text-classification"}
        handler.validate_output.return_value = {"protocol": {"status": "verified"},
                                                "task": {"status": "verified"}, "workload_contract": {}}
        if failed:
            getattr(handler, failed).side_effect = ValueError("bad input contract")
        execution = SimpleNamespace(inference_context=nullcontext, metadata=lambda: {})
        stack.enter_context(patch.dict(os.environ, {"TASK_FAMILY": "nlp", "TASK_TYPE": "text-classification",
                                                   "RUNTIME_BACKEND": "transformers_pipeline", "MODEL_ID": "fixture",
                                                   "MODEL_REVISION": "a" * 40, "USE_GPU": "0"}))
        stack.enter_context(patch("acprof.container.handlers.HandlerRegistry.get", return_value=handler))
        stack.enter_context(patch("acprof.container.handlers.load_handler", side_effect=lambda *args: handler.load()))
        stack.enter_context(patch("acprof.container.execution.configured_execution", return_value=(execution, "cpu")))
        stack.enter_context(patch("acprof.container.execution.complete_prediction", side_effect=lambda e, c, output: output))
        return handler

    def test_success_records_each_phase_separately(self):
        with ExitStack() as stack:
            self.fixtures(stack)
            result = runtime_validate.validate({"text": "hello"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual([item["stage"] for item in result["stages"]],
                         ["execution", "load", "preprocess", "predict", "completion", "postprocess", "validate_output", "metadata"])
        self.assertTrue(all(item["status"] == "verified" for item in result["stages"]))

    def test_preprocess_failure_is_reported_without_running_prediction(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            payload = Path(directory) / "input.json"
            payload.write_text('{"text":"hello"}')
            stack.enter_context(patch("sys.argv", ["runtime_validate", str(payload)]))
            handler = self.fixtures(stack, failed="preprocess")
            output = io.StringIO()
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(patch("acprof.container.runtime_validate.traceback.print_exc"))
            self.assertEqual(runtime_validate.main(), 1)
            record = json.loads(output.getvalue().split(runtime_validate.RESULT_PREFIX)[1])
            handler.predict.assert_not_called()
        self.assertEqual(record["failed_stage"], "preprocess")
        self.assertEqual(record["stages"][-1]["status"], "error")
        self.assertIn("ValueError: bad input contract", record["error"])


if __name__ == "__main__":
    unittest.main()

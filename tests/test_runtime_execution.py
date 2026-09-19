import os
import subprocess
import sys
import unittest


class RuntimeExecutionTests(unittest.TestCase):
    def test_generic_runner_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, '-c', '''
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("generic runner attempted a Torch import")
sys.meta_path.insert(0, NoTorch())
import acprof.container.compute_profile_runner
assert "torch" not in sys.modules
'''], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selected_cpu_runtime_validation_needs_no_torch(self):
        result = subprocess.run([sys.executable, '-c', '''
import sys, types
from contextlib import nullcontext
from unittest.mock import patch
from acprof.container.runtime_validate import validate
from acprof.container.handlers import BaseHandler
class Handler(BaseHandler):
    def load(self, *args): return {"task_type": "tabular-regression"}
    def preprocess(self, ctx, payload): return {"_effective_input_scale": 1}
    def predict(self, *args): return [[3.]]
    def postprocess(self, *args):
        return {"task": "tabular-regression", "output_type": "regression", "output_shape": [1,1], "n_results": 1}
with patch("acprof.container.handlers.HandlerRegistry.get", return_value=Handler()):
    result = validate({"input_scale": 1, "features": [[1., 2.]]})
assert result["status"] == "ok"
assert result["validation"]["protocol"]["status"] == "verified"
assert "torch" not in sys.modules
'''], env={**os.environ, 'TASK_FAMILY': 'structured', 'TASK_TYPE': 'tabular-regression',
          'RUNTIME_BACKEND': 'onnxruntime', 'MODEL_ID': 'local/fixture', 'MODEL_REVISION': 'fixture',
          'USE_GPU': '0', 'ACPROF_MODEL_ADAPTER': 'family-default'},
          text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

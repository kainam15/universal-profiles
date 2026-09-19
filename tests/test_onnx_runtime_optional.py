"""真实 ORT 容器接口回归；宿主机不安装推理框架。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('onnx', 'onnxruntime', 'numpy')),
                     'requires the no-Torch ONNX Runtime CPU container')
class ONNXRuntimeIntegrationTests(unittest.TestCase):
    def test_dynamic_linear_model_runs_full_handler_and_validation_contract(self):
        from examples.onnxruntime.smoke import create_linear_fixture, exercise
        with tempfile.TemporaryDirectory() as temporary:
            result = exercise(create_linear_fixture(Path(temporary)))
        self.assertTrue(result['known_reference_passed'])
        self.assertFalse(result['torch_installed'])
        self.assertEqual(result['response']['output_shape'], [4, 1])

    def test_fixed_batch_cannot_silently_change_workload(self):
        from examples.onnxruntime.smoke import create_linear_fixture
        from acprof.container.handlers import HandlerRegistry
        with tempfile.TemporaryDirectory() as temporary:
            root = create_linear_fixture(Path(temporary), fixed_rows=1)
            handler = HandlerRegistry.get('structured', 'onnxruntime')
            context = handler.load(str(root), 'tabular-regression', 'onnxruntime', 'cpu')
            with self.assertRaisesRegex(ValueError, 'fixed input shape'):
                handler.preprocess(context, {'input_scale': 2, 'batch_size': 1,
                                             'features': [[0.0] * 8, [1.0] * 8]})


if __name__ == '__main__':
    unittest.main()

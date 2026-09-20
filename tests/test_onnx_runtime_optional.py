"""真实 ORT 容器接口回归；宿主机不安装推理框架。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('onnx', 'onnxruntime', 'numpy')),
                     'requires the no-Torch ONNX Runtime CPU container')
class ONNXRuntimeIntegrationTests(unittest.TestCase):
    def test_environment_has_no_torch_or_transformers(self):
        self.assertIsNone(importlib.util.find_spec('torch'))
        self.assertIsNone(importlib.util.find_spec('transformers'))

    def test_dynamic_linear_model_runs_full_handler_and_validation_contract(self):
        from examples.onnxruntime.smoke import create_linear_fixture, exercise
        with tempfile.TemporaryDirectory() as temporary:
            result = exercise(create_linear_fixture(Path(temporary)))
        self.assertTrue(result['known_reference_passed'])
        self.assertFalse(result['torch_installed'])
        self.assertEqual(result['response']['output_shape'], [4, 1])

    def test_dynamic_rows_have_known_results_at_distinct_shapes(self):
        import numpy as np
        from examples.onnxruntime.smoke import create_linear_fixture
        from acprof.container.handlers import HandlerRegistry
        with tempfile.TemporaryDirectory() as temporary:
            handler = HandlerRegistry.get('structured', 'onnxruntime')
            root = create_linear_fixture(Path(temporary))
            context = handler.load(str(root), 'tabular-regression', 'onnxruntime', 'cpu')
            for count in (1, 3, 7):
                with self.subTest(rows=count):
                    payload = {'input_scale': count, 'batch_size': 1,
                               'features': [[float(row)] * 8 for row in range(count)]}
                    prepared = handler.preprocess(context, payload)
                    raw = handler.predict(context, prepared)
                    np.testing.assert_allclose(raw, [[8.0 * row + 0.25] for row in range(count)],
                                               rtol=0, atol=1e-6)
                    response = handler.postprocess(context, raw)
                    self.assertEqual(response['output_shape'], [count, 1])
                    self.assertEqual(handler.validate_output(context, payload, prepared, raw,
                                                             response)['protocol']['status'], 'verified')

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

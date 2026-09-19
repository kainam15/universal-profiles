import importlib
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class ONNXHandlerTests(unittest.TestCase):
    def handler(self, root, *, input_shape=('N', 4)):
        self.assertIsNotNone(importlib.util.find_spec('acprof.container.handlers.onnxruntime'),
                             'ONNX Runtime handler is missing')
        fake = types.SimpleNamespace(
            __version__='test', SessionOptions=lambda: types.SimpleNamespace(),
            InferenceSession=lambda *args, **kwargs: types.SimpleNamespace(
                get_inputs=lambda: [types.SimpleNamespace(name='x', shape=list(input_shape), type='tensor(float)')],
                get_outputs=lambda: [types.SimpleNamespace(name='y', shape=['N', 1], type='tensor(float)')],
                get_providers=lambda: ['CPUExecutionProvider'],
                run=lambda names, inputs: [inputs['x'].sum(axis=1, keepdims=True)],
            ),
        )
        with patch.dict('sys.modules', {'onnxruntime': fake}):
            module = importlib.import_module('acprof.container.handlers.onnxruntime')
            handler = module.ONNXRuntimeHandler()
            with patch.object(module, 'ort', fake):
                context = handler.load(str(root), 'tabular-regression', 'onnxruntime', 'cpu')
        return handler, context

    def test_full_protocol_and_known_numeric_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.onnx').write_bytes(b'fixture')
            handler, ctx = self.handler(root)
            payload = {'input_scale': 2, 'batch_size': 1,
                       'features': [[1., 2., 3., 4.], [5., 6., 7., 8.]]}
            prepared = handler.preprocess(ctx, payload)
            output = handler.predict(ctx, prepared)
            self.assertEqual(output.tolist(), [[10.], [26.]])
            response = handler.postprocess(ctx, output)
            self.assertEqual(response['output_shape'], [2, 1])
            self.assertEqual(handler.validate_output(ctx, payload, prepared, output, response)['protocol']['status'], 'verified')

    def test_fixed_model_batch_is_enforced_without_splitting_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.onnx').write_bytes(b'fixture')
            handler, ctx = self.handler(root, input_shape=(1, 4))
            with self.assertRaisesRegex(ValueError, 'fixed|shape'):
                handler.preprocess(ctx, {'input_scale': 2, 'features': [[1.] * 4] * 2})

    def test_missing_or_ambiguous_artifact_is_explicit(self):
        for filenames in ([], ['a.onnx', 'b.onnx']):
            with self.subTest(files=filenames), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in filenames:
                    (root / name).write_bytes(b'fixture')
                with self.assertRaisesRegex(ValueError, 'one.*onnx|model_file'):
                    self.handler(root)

    def test_nonfinite_input_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.onnx').write_bytes(b'fixture')
            handler, ctx = self.handler(root)
            with self.assertRaisesRegex(ValueError, 'finite'):
                handler.preprocess(ctx, {'input_scale': 1, 'features': [[float('nan')] * 4]})


if __name__ == '__main__':
    unittest.main()

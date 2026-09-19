import importlib
import importlib.util
import json
import unittest

import numpy as np


class ValidationContractTests(unittest.TestCase):
    def validator(self):
        self.assertIsNotNone(importlib.util.find_spec('acprof.container.validation'),
                             'the shared output validation contract is missing')
        return importlib.import_module('acprof.container.validation').validate_output

    def fixture(self):
        return ({'task_type': 'tabular-regression', 'feature_dim': 2},
                {'input_scale': 2, 'batch_size': 1, 'features': [[1., 2.], [3., 4.]]},
                {'_effective_input_scale': 2., 'args': (np.ones((2, 2)),)},
                np.array([[3.], [7.]]),
                {'task': 'tabular-regression', 'output_type': 'regression',
                 'output_shape': [2, 1], 'n_results': 2})

    def test_valid_shape_finite_workload_and_serialization(self):
        result = self.validator()(*self.fixture())
        self.assertEqual(result['protocol']['status'], 'verified')
        self.assertEqual(result['task']['status'], 'verified')
        contract = result['workload_contract']
        self.assertEqual(contract['input']['rows'], 2)
        self.assertEqual(contract['output']['shape'], [2, 1])
        self.assertEqual(contract['scenario'], {'type': 'serial'})
        json.dumps(result, allow_nan=False)

    def test_wrong_output_shape_rejected(self):
        args = list(self.fixture())
        args[3] = np.ones((1, 1))
        args[4] = {**args[4], 'output_shape': [1, 1], 'n_results': 1}
        with self.assertRaisesRegex(ValueError, 'shape|rows'):
            self.validator()(*args)

    def test_nan_and_inf_raw_output_rejected_even_when_summary_is_finite(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=bad):
                args = list(self.fixture())
                args[3][0, 0] = bad
                with self.assertRaisesRegex(ValueError, 'finite'):
                    self.validator()(*args)

    def test_effective_workload_mismatch_rejected(self):
        args = list(self.fixture())
        args[2]['_effective_input_scale'] = 1.
        with self.assertRaisesRegex(ValueError, 'scale'):
            self.validator()(*args)

    def test_missing_output_field_rejected(self):
        args = list(self.fixture())
        del args[4]['output_shape']
        with self.assertRaisesRegex(ValueError, 'output_shape'):
            self.validator()(*args)

    def test_requested_generation_limit_is_never_used_as_actual_count(self):
        result = self.validator()(
            {'task_type': 'text-generation'},
            {'params': {'max_new_tokens': 256}, 'batch_size': 1},
            {'_effective_input_scale': 5, 'text': 'hello'},
            [{'generated_text': 'hello world'}],
            {'task': 'text-generation', 'output_type': 'text', 'n_results': 1},
        )
        generation = result['workload_contract']['generation']
        self.assertEqual(generation['max_output_tokens'], 256)
        self.assertIsNone(generation['actual_output_tokens'])
        self.assertEqual(generation['actual_output_tokens_status'], 'unavailable')
        self.assertIsNone(generation['stop_reason'])

    def test_json_nonfinite_and_missing_task_rejected(self):
        for response in ({'task': 'tabular-regression', 'score': float('inf')}, {}):
            args = list(self.fixture())
            args[4] = response
            with self.assertRaises(ValueError):
                self.validator()(*args)

    def test_missing_raw_output_or_effective_scale_cannot_verify(self):
        for position, replacement, expected in ((3, None, 'raw output'), (2, {}, 'actual.*scale')):
            args = list(self.fixture())
            args[position] = replacement
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, expected):
                self.validator()(*args)

    def test_nlp_and_cv_require_actual_output_summary_fields(self):
        for task in ('text-classification', 'image-classification'):
            base = {'task': task, 'output_type': 'classification', 'n_results': 1}
            for missing in ('output_type', 'n_results'):
                response = {key: value for key, value in base.items() if key != missing}
                with self.subTest(task=task, missing=missing), self.assertRaisesRegex(ValueError, missing):
                    self.validator()({'task_type': task}, {'input_scale': 1},
                                     {'_effective_input_scale': 1}, [{'label': 'a', 'score': 0.9}], response)

    def test_generation_requires_text_or_observed_tokens(self):
        for task, output_type in (('text-generation', 'text'), ('automatic-speech-recognition', 'transcription')):
            response = {'task': task, 'output_type': output_type, 'n_results': 1, 'text': ''}
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'empty|evidence'):
                self.validator()({'task_type': task}, {'input_scale': 1},
                                 {'_effective_input_scale': 1}, {'text': ''}, response)

    def test_missing_shape_is_explicit_and_is_not_claimed_as_checked(self):
        report = self.validator()({'task_type': 'text-generation'}, {'input_scale': 1},
                                 {'_effective_input_scale': 1}, [{'generated_text': 'hello'}],
                                 {'task': 'text-generation', 'output_type': 'text', 'n_results': 1})
        self.assertNotIn('shape', report['protocol']['checks'])
        self.assertEqual(report['protocol']['aspects']['shape']['status'], 'unavailable')

    def test_runtime_probe_refuses_custom_validator_unverified_status(self):
        import os
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch
        from acprof.container.runtime_validate import validate

        for status in ('unsupported', 'unavailable', 'error'):
            handler = SimpleNamespace(
                preprocess=lambda *args: {'_effective_input_scale': 1},
                predict=lambda *args: [[1.]], postprocess=lambda *args: {'task': 'tabular-regression'},
                validate_output=lambda *args: {'protocol': {'status': 'verified'},
                                              'task': {'status': status}, 'workload_contract': {}},
            )
            runtime = SimpleNamespace(inference_context=nullcontext, metadata=lambda: {})
            with self.subTest(status=status), patch.dict(os.environ, {
                'TASK_FAMILY': 'structured', 'RUNTIME_BACKEND': 'onnxruntime',
                'TASK_TYPE': 'tabular-regression', 'MODEL_ID': 'test/model', 'MODEL_REVISION': 'fixed',
            }), patch('acprof.container.handlers.HandlerRegistry.get', return_value=handler), patch(
                'acprof.container.handlers.load_handler', return_value={},
            ), patch('acprof.container.handlers.resolve_model_source', return_value='test/model'), patch(
                'acprof.container.execution.configured_execution', return_value=(runtime, 'cpu'),
            ), self.assertRaisesRegex(ValueError, 'validation.*verified'):
                validate({'input_scale': 1})


if __name__ == '__main__':
    unittest.main()

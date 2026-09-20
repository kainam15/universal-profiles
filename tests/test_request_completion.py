"""Controlled async backend: protocol evidence, not a GPU benchmark."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from importlib import import_module
import os
from threading import Event
import types
import unittest
from unittest.mock import patch

from acprof.container.handlers import BaseHandler
from acprof.container.runtime_validate import validate


class RequestCompletionTests(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)

    def setup_backend(self, future, events, entered):
        class AsyncHandler(BaseHandler):
            def load(self, *args):
                return {'task_type': 'tabular-regression', 'device': 'cpu'}

            def preprocess(self, context, payload):
                return {'_effective_input_scale': 1}

            def predict(self, *args):
                events.append('submitted')
                return future

            def postprocess(self, context, output):
                if isinstance(output, Future):
                    raise AssertionError('postprocess received an unfinished request')
                events.append('postprocessed')
                return {'task': 'tabular-regression', 'output_type': 'regression',
                        'output_shape': [1, 1], 'n_results': 1}

            def validate_output(self, *args):
                events.append('validated')
                return super().validate_output(*args)

        def wait_for_completion(context, output, *, timeout_s):
            entered.set()
            value = output.result(timeout=timeout_s)
            events.append('completed')
            return value

        runtime = types.SimpleNamespace(inference_context=nullcontext, metadata=lambda: {},
                                        wait_for_completion=wait_for_completion)
        self.contexts.enter_context(patch('acprof.container.handlers.HandlerRegistry.get', return_value=AsyncHandler()))
        # Import-isolation fixtures may leave a stale attribute on the parent
        # package. Patch the module that validate() actually imports, including
        # on Python 3.10 where dotted patch targets can use that stale attribute.
        execution = import_module('acprof.container.execution')
        self.contexts.enter_context(patch.object(execution, 'configured_execution', return_value=(runtime, 'cpu')))
        self.contexts.enter_context(patch.dict(os.environ, {
            'TASK_FAMILY': 'structured', 'TASK_TYPE': 'tabular-regression',
            'RUNTIME_BACKEND': 'onnxruntime', 'MODEL_ID': 'local/async-fixture',
            'MODEL_REVISION': 'fixture', 'MODEL_LOCAL_PATH': '', 'USE_GPU': '0',
            'ACPROF_MODEL_ADAPTER': 'family-default', 'ACPROF_REQUEST_TIMEOUT_S': '0.1',
        }))

    @staticmethod
    def run_request():
        return validate({'input_scale': 1, 'features': [[1., 2.]]})

    def test_request_does_not_complete_until_its_work_finishes(self):
        future, events, entered = Future(), [], Event()
        self.setup_backend(future, events, entered)
        with ThreadPoolExecutor(max_workers=1) as executor:
            request = executor.submit(self.run_request)
            try:
                self.assertTrue(entered.wait(1), 'execution completion hook was not called')
                self.assertFalse(request.done())
                self.assertEqual(events, ['submitted'])
            finally:
                future.set_result([[3.]])
            self.assertEqual(request.result(timeout=2)['status'], 'ok')
        self.assertEqual(events, ['submitted', 'completed', 'postprocessed', 'validated'])

    def test_background_exception_is_not_converted_to_success(self):
        future, events = Future(), []
        future.set_exception(RuntimeError('background inference failed'))
        self.setup_backend(future, events, Event())
        with self.assertRaisesRegex(RuntimeError, 'background inference failed'):
            self.run_request()
        self.assertEqual(events, ['submitted'])

    def test_timeout_never_reaches_output_validation(self):
        events = []
        self.setup_backend(Future(), events, Event())
        with self.assertRaises(TimeoutError):
            self.run_request()
        self.assertEqual(events, ['submitted'])

    def test_undeclared_or_unresolved_async_output_is_rejected(self):
        from acprof.container.execution import complete_prediction
        for runtime in (types.SimpleNamespace(), types.SimpleNamespace(
                wait_for_completion=lambda context, output, **kwargs: output)):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(TypeError, 'wait_for_completion'):
                complete_prediction(runtime, {}, Future())

    def test_synchronous_existing_runtime_keeps_raw_output(self):
        from acprof.container.execution import complete_prediction
        output = [[3.]]
        self.assertIs(complete_prediction(types.SimpleNamespace(), {}, output), output)


if __name__ == '__main__':
    unittest.main()

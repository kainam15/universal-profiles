"""真实本地 HTTP 服务的完成边界；可控异步替身，不宣称 GPU 验证。"""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
import importlib.util
import json
import os
import runpy
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


@unittest.skipUnless(importlib.util.find_spec('flask'), 'requires Flask in the runtime container')
class ONNXServerCompletionTests(unittest.TestCase):
    def start_server(self, future):
        from acprof.container.handlers import BaseHandler
        from werkzeug.serving import make_server

        events, entered = [], Event()

        class AsyncHandler(BaseHandler):
            def load(self, *args):
                return {'task_type': 'tabular-regression', 'device': 'cpu'}

            def preprocess(self, context, payload):
                return {'_effective_input_scale': 1}

            def predict(self, context, processed):
                events.append('submitted')
                return future

            def postprocess(self, context, output):
                if isinstance(output, Future):
                    raise AssertionError('unfinished output reached postprocess')
                events.append('postprocessed')
                return {'task': 'tabular-regression', 'output_type': 'regression',
                        'output_shape': [1, 1], 'n_results': 1}

            def validate_output(self, *args):
                events.append('validated_inside_request')
                raise AssertionError('formal HTTP request must not invoke output validation')

        def wait_for_completion(context, output, *, timeout_s):
            entered.set()
            value = output.result(timeout=timeout_s)
            events.append('completed')
            return value

        runtime = SimpleNamespace(inference_context=nullcontext, metadata=lambda: {},
                                  wait_for_completion=wait_for_completion)
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(patch('acprof.container.handlers.HandlerRegistry.get', return_value=AsyncHandler()))
        contexts.enter_context(patch('acprof.container.execution.configured_execution', return_value=(runtime, 'cpu')))
        contexts.enter_context(patch.dict(os.environ, {
            'TASK_FAMILY': 'structured', 'TASK_TYPE': 'tabular-regression', 'RUNTIME_BACKEND': 'onnxruntime',
            'MODEL_ID': 'local/async-fixture', 'MODEL_REVISION': 'fixture', 'MODEL_LOCAL_PATH': '',
            'USE_GPU': '0', 'ACPROF_MODEL_ADAPTER': 'family-default', 'ACPROF_REQUEST_TIMEOUT_S': '0.2',
        }))
        namespace = runpy.run_module('acprof.container.server', run_name='acprof_test_http_server')
        server = make_server('127.0.0.1', 0, namespace['app'], threaded=True)
        thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(cleanup)
        base = f'http://127.0.0.1:{server.server_port}'
        with build_opener(ProxyHandler({})).open(base + '/ready', timeout=2) as response:
            self.assertEqual(json.load(response)['status'], 'ok')
        return base, events, entered

    @staticmethod
    def request(base):
        payload = {'input_scale': 1, 'batch_size': 1, 'features': [[1., 2.]]}
        request = Request(base + '/predict', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_http_waits_for_completion_and_does_not_validate_inside_request(self):
        future = Future()
        base, events, entered = self.start_server(future)
        with ThreadPoolExecutor(max_workers=1) as executor:
            request = executor.submit(self.request, base)
            try:
                self.assertTrue(entered.wait(1), 'server never entered completion hook')
                self.assertFalse(request.done())
                self.assertEqual(events, ['submitted'])
            finally:
                future.set_result([[3.]])
            status, body = request.result(timeout=2)
        self.assertEqual(status, 200, body)
        self.assertEqual(body['output_shape'], [1, 1])
        self.assertEqual(body['workload_contract']['input']['rows'], 1)
        self.assertEqual(events, ['submitted', 'completed', 'postprocessed'])

    def test_background_error_becomes_failed_http_response(self):
        future = Future()
        future.set_exception(RuntimeError('background execution failed'))
        base, events, _ = self.start_server(future)
        status, body = self.request(base)
        self.assertEqual(status, 500)
        self.assertIn('background execution failed', body['error'])
        self.assertNotIn('workload_contract', body)
        self.assertEqual(events, ['submitted'])

    def test_timeout_never_returns_success_or_runs_postprocess(self):
        base, events, entered = self.start_server(Future())
        status, body = self.request(base)
        self.assertTrue(entered.is_set())
        self.assertEqual(status, 500)
        self.assertTrue(body['error'], 'timeout needs a diagnostic instead of an empty error')
        self.assertNotIn('workload_contract', body)
        self.assertEqual(events, ['submitted'])


if __name__ == '__main__':
    unittest.main()

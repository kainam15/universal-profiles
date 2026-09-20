"""Real WordPiece, HTTP /probe and host scale planning without model downloads."""
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import runpy
import string
import sys
import tempfile
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                        ('onnx', 'onnxruntime', 'tokenizers', 'flask', 'httpx')),
                     'requires the no-Torch ONNX Runtime CPU container')
class ONNXWordPiecePlanningTests(unittest.TestCase):
    def setUp(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from werkzeug.serving import make_server
        from examples.onnxruntime.fixtures import create_text_fixture
        import httpx
        from acprof.host import input_plan
        from acprof.host.detect import TaskInfo

        self.assertIsNone(importlib.util.find_spec('torch'))
        self.assertIsNone(importlib.util.find_spec('transformers'))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'plan').mkdir()
        model = create_text_fixture(self.root / 'model', max_length=32)
        words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', *string.ascii_lowercase,
                 *('##' + char for char in string.ascii_lowercase)]
        tokenizer = Tokenizer(models.WordPiece({word: index for index, word in enumerate(words)},
                                               unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)],
        )
        tokenizer.save(str(model / 'tokenizer.json'))
        self.tokenizer = tokenizer
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.contexts.enter_context(patch.dict(os.environ, {
            'TASK_FAMILY': 'nlp', 'TASK_TYPE': 'text-classification', 'RUNTIME_BACKEND': 'onnxruntime',
            'MODEL_ID': 'local/wordpiece-fixture', 'MODEL_REVISION': 'fixture', 'MODEL_LOCAL_PATH': str(model),
            'USE_GPU': '0', 'ACPROF_MODEL_ADAPTER': 'family-default',
        }))
        namespace = runpy.run_module('acprof.container.server', run_name='acprof_test_wordpiece_server')
        server = make_server('127.0.0.1', 0, namespace['app'], threaded=True)
        thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(cleanup)
        self.base = f'http://127.0.0.1:{server.server_port}'
        self.client = self.contexts.enter_context(httpx.Client(trust_env=False, timeout=3))
        self.assertEqual(self.client.get(self.base + '/ready').status_code, 200)
        self.probe_results = []

        def post(url, **kwargs):
            result = self.client.post(url, **kwargs)
            if url.endswith('/probe'):
                self.probe_results.append((result.status_code, result.json()))
            return result

        # Runtime images deliberately omit the host requests dependency. Swap only
        # that HTTP library for installed httpx; production parser/planner and real
        # localhost HTTP remain in use, without mocking token counts or responses.
        self.contexts.enter_context(patch.dict(sys.modules, {'requests': SimpleNamespace(post=post)}))
        session = SimpleNamespace(name='wordpiece-test-session', base_url=self.base)
        self.contexts.enter_context(patch.object(input_plan, '_start_probe_session', return_value=session))
        self.stop = self.contexts.enter_context(patch.object(input_plan, '_stop_container_session'))
        self.task = TaskInfo('local/wordpiece-fixture', 'text-classification', 'nlp',
                             'onnxruntime', 'onnx', 'fixture', 'test')

    def plan(self, input_scales=None):
        from acprof.host.input_plan import plan_input_scales

        return plan_input_scales(self.task, SimpleNamespace(), [1], [1], ['off'], 1,
                                 str(self.root / 'plan'), input_scales=input_scales)

    def test_auto_planner_recovers_from_over_limit_wordpiece_probe_without_truncating(self):
        planned = self.plan()
        self.assertEqual(planned.scales, [3., 8., 16., 21., 25., 28.])
        self.assertTrue(any(body.get('limit_exceeded') for _, body in self.probe_results))
        for status, body in self.probe_results:
            self.assertEqual(status, 200)
            self.assertFalse(body['truncated_by_limit'])
        plan = json.loads(Path(planned.plan_file).read_text())
        for entry in plan['entries']:
            payload = dict(entry['payload'], input_scale=entry['input_scale'])
            tokens = self.tokenizer.encode(payload['text'])
            self.assertLessEqual(len(tokens.ids), 32)
            self.assertEqual(sum(not item for item in tokens.special_tokens_mask), entry['input_scale'])
            response = self.client.post(self.base + '/predict', json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            contract = response.json()['workload_contract']['input']['text']
            self.assertEqual(contract['tokens'], len(tokens.ids))
            self.assertEqual(contract['truncation'], 'reject')
        self.stop.assert_called_once()

    def test_manual_plan_rejects_only_over_limit_and_predict_never_accepts_it(self):
        legal = self.plan(input_scales='2')
        self.assertEqual(legal.scales, [8.])
        with self.assertRaisesRegex(RuntimeError, 'manual NLP input scales exceed'):
            self.plan(input_scales='30')
        payload = {'text': ' '.join(['hello'] * 10), 'batch_size': 1}
        probe = self.client.post(self.base + '/probe', json=payload)
        self.assertEqual(probe.status_code, 200, probe.text)
        self.assertTrue(probe.json()['limit_exceeded'])
        self.assertFalse(probe.json()['truncated_by_limit'])
        prediction = self.client.post(self.base + '/predict', json=payload)
        self.assertEqual(prediction.status_code, 500)
        self.assertNotIn('workload_contract', prediction.json())

    def test_probe_does_not_convert_configuration_errors_into_recoverable_limits(self):
        response = self.client.post(self.base + '/probe', json={'text': 'hello', 'batch_size': 2})
        self.assertEqual(response.status_code, 500)
        self.assertIn('batch_size', response.json()['error'])
        self.assertNotIn('limit_exceeded', response.json())


if __name__ == '__main__':
    unittest.main()

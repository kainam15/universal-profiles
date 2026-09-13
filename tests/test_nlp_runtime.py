"""在 NLP 镜像中执行已有的 12 种原生小模型示例。"""
import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in (
    "torch", "transformers", "sentence_transformers", "pandas",
)), "requires the NLP container")
class NLPRuntimeTests(unittest.TestCase):
    def test_native_tasks_offline(self):
        from examples.nlp.smoke import main
        self.assertEqual(main(), 0)

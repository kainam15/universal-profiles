"""在 structured 镜像中验证真实 TorchScript 和 skops 加载。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in (
    "torch", "skops", "sklearn",
)), "requires the structured container")
class StructuredRuntimeTests(unittest.TestCase):
    def test_torchscript_and_skops_offline(self):
        from examples.structured.export_models import export_examples
        from examples.structured.smoke import TASKS, check_task, check_skops
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_examples(root)
            for task in TASKS:
                with self.subTest(task=task):
                    check_task(root / task, task, "torchscript")
            check_skops(root)

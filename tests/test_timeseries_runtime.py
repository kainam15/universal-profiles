"""在 timeseries 镜像中验证真实 ChronosBolt 加载与预测。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in (
    "torch", "chronos", "transformers",
)), "requires the timeseries container")
class TimeseriesRuntimeTests(unittest.TestCase):
    def test_chronos_bolt_offline(self):
        from examples.structured.chronos_smoke import main
        with tempfile.TemporaryDirectory() as directory:
            main(Path(directory))

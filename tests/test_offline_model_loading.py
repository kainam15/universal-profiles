import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from acprof.container.handlers.audio import AudioHandler
from acprof.container.handlers.cv import CVHandler
from acprof.container.handlers.nlp import NLPHandler
from acprof.container.handlers.timeseries import (
    ChronosHandler,
    TimeseriesTransformersHandler,
)


class OfflineModelLoadingTests(unittest.TestCase):
    def test_transformers_handlers_load_local_snapshot_without_hub_revision(self) -> None:
        calls = []

        def fake_pipeline(**kwargs):
            calls.append(kwargs)
            return object()

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.pipeline = fake_pipeline

        handlers = [
            (NLPHandler(), "fill-mask", "transformers_pipeline"),
            (CVHandler(), "image-classification", "transformers_pipeline"),
            (AudioHandler(), "audio-classification", "transformers_pipeline"),
            (
                TimeseriesTransformersHandler(),
                "time-series-forecasting",
                "transformers_pipeline",
            ),
        ]

        with tempfile.TemporaryDirectory() as model_source, patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            for handler, task_type, backend in handlers:
                handler.load(
                    model_source,
                    task_type,
                    backend,
                    "cpu",
                    "0123456789abcdef",
                )

        self.assertEqual(len(calls), len(handlers))
        for call in calls:
            self.assertEqual(call["model"], model_source)
            self.assertNotIn("revision", call)

    def test_transformers_handler_keeps_revision_when_loading_by_model_id(self) -> None:
        calls = []

        def fake_pipeline(**kwargs):
            calls.append(kwargs)
            return object()

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.pipeline = fake_pipeline

        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            NLPHandler().load(
                "google-bert/bert-base-uncased",
                "fill-mask",
                "transformers_pipeline",
                "cpu",
                "0123456789abcdef",
            )

        self.assertEqual(calls[0]["revision"], "0123456789abcdef")

    def test_chronos_loads_local_snapshot_without_hub_revision(self) -> None:
        calls = []

        class FakeChronosBoltPipeline:
            @classmethod
            def from_pretrained(cls, model_source, **kwargs):
                calls.append((model_source, kwargs))
                return object()

        fake_chronos = types.ModuleType("chronos")
        fake_chronos.ChronosBoltPipeline = FakeChronosBoltPipeline
        fake_torch = types.ModuleType("torch")

        with tempfile.TemporaryDirectory() as model_source, patch.dict(
            sys.modules,
            {"torch": fake_torch, "chronos": fake_chronos},
        ):
            ChronosHandler().load(
                model_source,
                "time-series-forecasting",
                "chronos",
                "cpu",
                "0123456789abcdef",
            )

        self.assertEqual(calls[0][0], model_source)
        self.assertNotIn("revision", calls[0][1])
        self.assertTrue(calls[0][1]["local_files_only"])


if __name__ == "__main__":
    unittest.main()

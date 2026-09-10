import contextlib
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from acprof.container.handlers.timeseries import ChronosHandler, TimeseriesTransformersHandler
from acprof.workloads.timeseries import TimeseriesWorkloadGenerator


class Tensor:
    def __init__(self, value):
        self.values = np.asarray(value)
        self.ndim = self.values.ndim
        self.shape = self.values.shape
        self.device = "cpu"

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.values, axis))

    def to(self, device):
        self.device = device
        return self


class TimeseriesSemanticsTests(unittest.TestCase):
    def test_chronos_exposes_context_limit_and_rejects_hidden_truncation(self):
        handler = ChronosHandler()
        ctx = {"device": "cpu", "pipeline": types.SimpleNamespace(model_context_length=2)}
        metadata = handler.get_scale_metadata(ctx, {})
        self.assertEqual(metadata["max_effective_input_scale"], 2)
        self.assertEqual(metadata["input_scale_type"], "context_length")
        torch = types.SimpleNamespace(tensor=lambda value, **kwargs: Tensor(value), float32="float32")
        with patch.dict("sys.modules", {"torch": torch}), self.assertRaisesRegex(ValueError, "context.*2"):
            handler.preprocess(ctx, {"context": [[1, 2, 3]]})

    def test_chronos_validates_scale_and_moves_input_during_preprocess(self):
        torch = types.SimpleNamespace(tensor=lambda value, **kwargs: Tensor(value), float32="float32",
                                      inference_mode=contextlib.nullcontext)
        ctx = {"device": "cuda", "task_type": "time-series-forecasting", "pipeline": Mock()}
        handler = ChronosHandler()
        with patch.dict("sys.modules", {"torch": torch}):
            processed = handler.preprocess(ctx, {"context": [[1.0, 2.0, 3.0]], "prediction_length": 2})
            self.assertEqual(processed["context"].device, "cuda")
            self.assertEqual(processed["_effective_input_scale"], 3)
            processed["context"].to = Mock(side_effect=AssertionError("predict moved input"))
            handler.predict(ctx, processed)
            ctx["pipeline"].predict.assert_called_once_with(processed["context"], prediction_length=2)

    def test_invalid_context_and_prediction_length_fail(self):
        handler = ChronosHandler()
        torch = types.SimpleNamespace(tensor=lambda value, **kwargs: Tensor(value), float32="float32")
        ctx = {"device": "cpu", "task_type": "time-series-forecasting", "pipeline": Mock()}
        for payload in ({"context": []}, {"context": [[1], [1, 2]]}, {"context": [[float("nan")]]},
                        {"context": [[True]]}, {"context": [[1]], "prediction_length": 0},
                        {"context": [[1]], "prediction_length": 1.5}):
            with self.subTest(payload=payload), patch.dict("sys.modules", {"torch": torch}), self.assertRaises(ValueError):
                handler.preprocess(ctx, payload)

    def test_non_chronos_pipeline_fails_with_explicit_support_boundary(self):
        torch = types.ModuleType("torch")
        transformers = types.SimpleNamespace(pipeline=Mock())
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            with self.assertRaisesRegex(ValueError, "Chronos"):
                TimeseriesTransformersHandler().load("unsupported/model", "time-series-forecasting", "transformers_pipeline", "cpu")
        transformers.pipeline.assert_not_called()

    def test_workload_rejects_invalid_scale_instead_of_slicing(self):
        generator = TimeseriesWorkloadGenerator("demo", "time-series-forecasting", 1)
        for value in (0, -1, 1.5, True, 2049):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generator.generate(value)


if __name__ == "__main__":
    unittest.main()

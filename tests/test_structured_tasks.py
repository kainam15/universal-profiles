from __future__ import annotations

import contextlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from acprof.container.handlers.structured import StructuredHandler
from acprof.workloads.structured import StructuredWorkloadGenerator


TASKS = ("tabular-classification", "tabular-regression", "reinforcement-learning", "robotics", "graph-ml")


class StructuredWorkloadTests(unittest.TestCase):
    def test_dense_scale_counts_rows_and_keeps_feature_width_fixed(self):
        for task, key, width in (("tabular-classification", "features", 8),
                                 ("tabular-regression", "features", 8),
                                 ("reinforcement-learning", "observations", 4),
                                 ("robotics", "observations", 7)):
            with self.subTest(task=task):
                generator = StructuredWorkloadGenerator("demo", task, 2)
                small, large = generator.generate(2), generator.generate(3)
                self.assertEqual(len(small[key]), 4)
                self.assertEqual(len(large[key]), 6)
                self.assertEqual({len(row) for row in large[key]}, {width})
                self.assertEqual(small, generator.generate(2))
                self.assertEqual(small[key][:2], large[key][:2])
                self.assertEqual(generator.effective_input_scale(100, small), 2)
                self.assertEqual(generator.input_metadata(2, small)["input_num_samples"], 4)

    def test_graph_scale_is_per_graph_nodes_with_valid_ring_edges(self):
        generator = StructuredWorkloadGenerator("demo", "graph-ml", 2)
        payload = generator.generate(5)
        self.assertEqual(len(payload["graphs"]), 2)
        for graph in payload["graphs"]:
            self.assertEqual(len(graph["node_features"]), 5)
            self.assertEqual(len(graph["edge_index"][0]), 10)
            self.assertEqual(max(max(row) for row in graph["edge_index"]), 4)
        self.assertEqual(generator.effective_input_scale(100, payload), 5)
        metadata = generator.input_metadata(5, payload)
        self.assertEqual(metadata["node_count"], 10)
        self.assertEqual(metadata["edge_count"], 20)
        self.assertEqual(metadata["input_num_samples"], 2)

    def test_invalid_scales_fail_instead_of_silent_rounding_or_truncation(self):
        generator = StructuredWorkloadGenerator("demo", "tabular-classification", 1)
        for value in (0, -1, 1.5, True, float("nan"), float("inf"), 4097):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generator.generate(value)

    def test_spec_controls_shape_scales_seed_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps({"schema_version": 1, "task": "robotics", "feature_dim": 12,
                                        "input_scales": [2, 4], "seed": 42,
                                        "provenance": {"source": "test"}}))
            generator = StructuredWorkloadGenerator("demo", "robotics", 1, workload_spec_path=str(path))
            self.assertEqual(len(generator.generate(2)["observations"][0]), 12)
            self.assertEqual(generator.default_input_scales(), [2.0, 4.0])
            self.assertEqual(generator.plan_metadata()["feature_dim"], 12)
            self.assertEqual(generator.plan_metadata()["seed"], 42)
            self.assertEqual(generator.plan_metadata()["provenance"], {"source": "test"})
            self.assertEqual(len(generator.plan_metadata()["workload_spec_sha256"]), 64)


class StructuredHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = StructuredHandler()
        self.fake_torch = types.SimpleNamespace(
            tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
            float32=np.float32, int64=np.int64, inference_mode=contextlib.nullcontext,
        )

    def context(self, task="tabular-classification", backend="torchscript", feature_dim=2):
        return {"task_type": task, "backend": backend, "feature_dim": feature_dim,
                "device": "cpu", "model": Mock(return_value=np.ones((2, 1), np.float32))}

    def test_preprocess_validates_actual_scale_and_dimension(self):
        ctx = self.context()
        with patch.dict("sys.modules", {"torch": self.fake_torch}):
            processed = self.handler.preprocess(ctx, {"features": [[1, 2], [3, 4]],
                                                       "batch_size": 1, "input_scale": 2})
            self.assertEqual(processed["_effective_input_scale"], 2)
            np.testing.assert_array_equal(processed["args"][0], [[1, 2], [3, 4]])
            for payload in ({"features": [[1]], "input_scale": 1},
                            {"features": [[1, 2]], "input_scale": 2},
                            {"features": [[1, float("nan")]], "input_scale": 1},
                            {"features": [[True, 2]], "input_scale": 1},
                            {"features": [[1, 2]], "batch_size": 2, "input_scale": 1}):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    self.handler.preprocess(ctx, payload)

    def test_graph_preprocess_forms_disjoint_batch_and_offsets_edges(self):
        ctx = self.context("graph-ml")
        graph = {"node_features": [[1, 2], [3, 4]], "edge_index": [[0, 1], [1, 0]]}
        with patch.dict("sys.modules", {"torch": self.fake_torch}):
            processed = self.handler.preprocess(ctx, {"graphs": [graph, graph], "batch_size": 2, "input_scale": 2})
        self.assertEqual(processed["_effective_input_scale"], 2)
        np.testing.assert_array_equal(processed["args"][1], [[0, 1, 2, 3], [1, 0, 3, 2]])
        np.testing.assert_array_equal(processed["args"][2], [0, 0, 1, 1])

    def test_graph_rejects_invalid_indices_and_mixed_node_counts(self):
        ctx = self.context("graph-ml")
        for edges in ([[0], [2]], [[-1], [0]], [[0.5], [1]], [[True], [1]], [[0], []]):
            with self.subTest(edges=edges), patch.dict("sys.modules", {"torch": self.fake_torch}), self.assertRaises(ValueError):
                self.handler.preprocess(ctx, {"graphs": [{"node_features": [[1, 2], [3, 4]], "edge_index": edges}],
                                              "input_scale": 2})

    def test_predict_uses_preprocessed_inputs_and_disables_gradients(self):
        ctx = self.context()
        tensor = np.array([[1, 2]], np.float32)
        self.fake_torch.tensor = Mock(side_effect=AssertionError("predict rebuilt the input"))
        self.fake_torch.inference_mode = Mock(return_value=contextlib.nullcontext())
        with patch.dict("sys.modules", {"torch": self.fake_torch}):
            output = self.handler.predict(ctx, {"args": (tensor,)})
        ctx["model"].assert_called_once_with(tensor)
        self.fake_torch.inference_mode.assert_called_once_with()
        self.assertEqual(output.shape, (2, 1))

    def test_postprocess_reports_tensor_results_for_each_task(self):
        for task in TASKS:
            with self.subTest(task=task):
                result = self.handler.postprocess(self.context(task), np.ones((3, 2)))
                self.assertEqual(result["task"], task)
                self.assertEqual(result["output_shape"], [3, 2])
                self.assertEqual(result["n_results"], 3)
        with self.assertRaises(ValueError):
            self.handler.postprocess(self.context(), {"loss": 0})

    def test_torchscript_loads_only_declared_local_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.pt").write_bytes(b"test")
            manifest = {"schema_version": 1, "task": "robotics", "format": "torchscript",
                        "model_file": "model.pt", "feature_dim": 7}
            (path / "acprof_model.json").write_text(json.dumps(manifest))
            model = Mock()
            model.eval.return_value = model
            self.fake_torch.jit = types.SimpleNamespace(load=Mock(return_value=model))
            with patch.dict("sys.modules", {"torch": self.fake_torch}):
                ctx = self.handler.load(directory, "robotics", "torchscript", "cpu")
                self.assertEqual(ctx["feature_dim"], 7)
                self.fake_torch.jit.load.assert_called_once_with(str(path / "model.pt"), map_location="cpu")
                with self.assertRaisesRegex(ValueError, "task"):
                    self.handler.load(directory, "graph-ml", "torchscript", "cpu")
                manifest["model_file"] = "../model.pt"
                (path / "acprof_model.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "model_file"):
                    self.handler.load(directory, "robotics", "torchscript", "cpu")

    def test_missing_manifest_and_unknown_backend_fail_with_format_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "acprof_model.json"):
                self.handler.load(directory, "robotics", "torchscript", "cpu")
            with self.assertRaisesRegex(ValueError, "backend"):
                self.handler.load(directory, "robotics", "transformers_pipeline", "cpu")
            with self.assertRaisesRegex(ValueError, "CPU"):
                self.handler.load(directory, "tabular-classification", "skops", "cuda:0")

    def test_skops_loads_single_artifact_without_trusting_custom_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.skops"
            path.write_bytes(b"test")
            estimator = Mock(n_features_in_=2)
            loader = Mock(return_value=estimator)
            modules = {"skops": types.ModuleType("skops"), "skops.io": types.SimpleNamespace(load=loader),
                       "sklearn": types.SimpleNamespace(base=types.SimpleNamespace(
                           is_classifier=lambda model: True, is_regressor=lambda model: False))}
            with patch.dict("sys.modules", modules):
                ctx = self.handler.load(directory, "tabular-classification", "skops", "cpu")
            loader.assert_called_once_with(str(path), trusted=[])
            self.assertEqual(ctx["feature_dim"], 2)
            processed = self.handler.preprocess(ctx, {"features": [[1, 2]], "input_scale": 1})
            self.handler.predict(ctx, processed)
            np.testing.assert_array_equal(estimator.predict.call_args.args[0], [[1, 2]])


if __name__ == "__main__":
    unittest.main()

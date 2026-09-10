"""Verify real exported artifacts through AC-Prof's four handler phases.

    PYTHONPATH=. python examples/structured/smoke.py /tmp/structured-models

With --skops, also create two tiny deterministic sklearn fixtures (no fitting)
and verify their skops persistence and inference using the installed versions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from acprof.container.handlers.structured import StructuredHandler
from acprof.workloads.structured import StructuredWorkloadGenerator


TASKS = ("tabular-classification", "tabular-regression", "reinforcement-learning", "robotics", "graph-ml")


def check_task(directory: Path, task: str, backend: str) -> None:
    handler = StructuredHandler()
    context = handler.load(str(directory), task, backend, "cpu")
    generator = StructuredWorkloadGenerator("format-smoke", task, 2)
    for scale in (2, 5):
        payload = generator.generate(scale)
        processed = handler.preprocess(context, payload)
        output = handler.predict(context, processed)
        result = handler.postprocess(context, output)
        expected = 2 if task == "graph-ml" else 2 * scale
        assert result["n_results"] == expected, result
        assert processed["_effective_input_scale"] == scale, processed
        if backend == "torchscript":
            assert not output.requires_grad
        print(json.dumps({"backend": backend, "scale": scale, **result}))


def check_skops(output_dir: Path) -> None:
    from skops import io
    from sklearn.linear_model import LinearRegression, LogisticRegression

    # Assign deterministic fitted-state fixtures, without running training.
    classifier = LogisticRegression()
    classifier.classes_ = np.array([0, 1])
    classifier.coef_ = np.ones((1, 8), dtype=np.float64)
    classifier.intercept_ = np.zeros(1)
    classifier.n_features_in_ = 8
    regressor = LinearRegression()
    regressor.coef_ = np.ones(8, dtype=np.float64)
    regressor.intercept_ = np.float64(0)
    regressor.n_features_in_ = 8
    for task, estimator in (("tabular-classification", classifier), ("tabular-regression", regressor)):
        destination = output_dir / (task + "-skops")
        destination.mkdir(parents=True, exist_ok=True)
        io.dump(estimator, destination / "model.skops")
        check_task(destination, task, "skops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--skops", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"torch_version": torch.__version__, "device": "cpu"}))
    for selected_task in TASKS:
        check_task(args.model_dir / selected_task, selected_task, "torchscript")
    if args.skops:
        check_skops(args.model_dir)

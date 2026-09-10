"""Explicit adapters for dense tables, vector policies and graph inference.

TorchScript artifacts must provide acprof_model.json; skops accepts a manifest
or a single *.skops file. These are inference formats, not arbitrary task-tagged
Hub repositories. No environment steps, policy training or robot I/O occur.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from acprof.container.handlers import BaseHandler, HandlerRegistry


TASK_OUTPUT_TYPES = {
    "tabular-classification": "classification", "tabular-regression": "regression",
    "reinforcement-learning": "actions", "robotics": "actions", "graph-ml": "graph",
}


def _positive_integer(value: Any, name: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 or int(value) != value):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _local_snapshot(model_source: str, model_revision: str) -> Path:
    path = Path(model_source)
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_source, revision=model_revision or "main",
                                  local_files_only=True)).resolve()


def _artifact_path(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute() or ".." in Path(filename).parts:
        raise ValueError("model_file must be a relative file within the model snapshot")
    path = root / filename
    # Hub snapshots may contain symlinks into their content-addressed blob store.
    if not path.is_file():
        raise ValueError(f"model_file does not exist in the local snapshot: {filename}")
    return path


def _dense_matrix(value: Any, feature_dim: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty matrix of finite numbers")
    for row in value:
        if not isinstance(row, list) or len(row) != feature_dim:
            raise ValueError(f"{name} feature_dim must match model feature_dim={feature_dim}; set workload feature_dim")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in row):
            raise ValueError(f"{name} must contain finite numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains values outside float32 range")
    return result


class StructuredHandler(BaseHandler):
    def load(self, model_source: str, task_type: str, backend: str, device: str,
             model_revision: str = "main", load_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if task_type not in TASK_OUTPUT_TYPES:
            raise ValueError(f"unsupported structured task: {task_type!r}")
        if backend not in {"torchscript", "skops"}:
            raise ValueError("structured backend must be torchscript or skops")
        if load_options:
            raise ValueError("structured artifacts do not support attention/profiler load options; export the required model implementation")
        if backend == "skops" and (device != "cpu" or not task_type.startswith("tabular-")):
            raise ValueError("skops supports only tabular-classification/tabular-regression on CPU")
        root = _local_snapshot(model_source, model_revision)
        manifest_path = root / "acprof_model.json"
        manifest = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or isinstance(manifest.get("schema_version"), bool):
                raise ValueError("acprof_model.json schema_version must be 1")
            if manifest.get("task") != task_type:
                raise ValueError("acprof_model.json task must match selected task")
            if manifest.get("format") != backend:
                raise ValueError("acprof_model.json format must match runtime backend")
            feature_dim = _positive_integer(manifest.get("feature_dim"), "feature_dim")
            artifact = _artifact_path(root, manifest.get("model_file"))
        elif backend == "torchscript":
            raise ValueError("TorchScript task support requires acprof_model.json with schema_version, task, format, model_file and feature_dim; arbitrary Hub models are not supported")
        else:
            files = sorted(root.glob("*.skops"))
            if len(files) != 1:
                raise ValueError("skops requires one *.skops file or acprof_model.json selecting model_file")
            artifact = files[0]
            feature_dim = None
        if backend == "torchscript":
            import torch

            model = torch.jit.load(str(artifact), map_location=device).eval()
        else:
            from skops import io as skops_io
            from sklearn import base as sklearn_base

            # Never automatically trust serialized custom classes or functions.
            model = skops_io.load(str(artifact), trusted=[])
            inferred_dim = _positive_integer(getattr(model, "n_features_in_", None), "model n_features_in_")
            if feature_dim is not None and feature_dim != inferred_dim:
                raise ValueError("acprof_model.json feature_dim does not match model n_features_in_")
            feature_dim = inferred_dim
            check = sklearn_base.is_classifier if task_type == "tabular-classification" else sklearn_base.is_regressor
            if not check(model):
                raise ValueError(f"loaded skops estimator does not support task {task_type!r}")
        return {"model": model, "backend": backend, "task_type": task_type, "device": device,
                "feature_dim": feature_dim, "model_revision": model_revision or "main",
                "model_format": backend, "load_options": {}}

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
        task = model_ctx["task_type"]
        if not isinstance(raw_input, dict):
            raise ValueError("structured input must be an object")
        batch_size = _positive_integer(raw_input.get("batch_size", 1), "batch_size")
        scale = _positive_integer(raw_input.get("input_scale"), "input_scale")
        feature_dim = model_ctx["feature_dim"]
        if task == "graph-ml":
            graphs = raw_input.get("graphs")
            if not isinstance(graphs, list) or len(graphs) != batch_size:
                raise ValueError("graphs count must match batch_size")
            features, edge_parts, batch_parts = [], [], []
            offset = 0
            for index, graph in enumerate(graphs):
                if not isinstance(graph, dict):
                    raise ValueError("each graph must be an object")
                x = _dense_matrix(graph.get("node_features"), feature_dim, "node_features")
                if len(x) != scale:
                    raise ValueError("input_scale must match node_count of every graph")
                edge_index = graph.get("edge_index")
                if (not isinstance(edge_index, list) or len(edge_index) != 2
                        or any(not isinstance(row, list) for row in edge_index)
                        or len(edge_index[0]) != len(edge_index[1])):
                    raise ValueError("edge_index must have shape [2, edge_count]")
                if any(isinstance(node, bool) or not isinstance(node, int) or node < 0 or node >= len(x)
                       for row in edge_index for node in row):
                    raise ValueError("edge_index must contain integer node indices within each graph")
                features.append(x)
                edge_parts.append(np.asarray(edge_index, dtype=np.int64) + offset)
                batch_parts.extend([index] * len(x))
                offset += len(x)
            arrays = (np.concatenate(features), np.concatenate(edge_parts, axis=1), np.asarray(batch_parts, dtype=np.int64))
        else:
            key = "features" if task.startswith("tabular-") else "observations"
            values = _dense_matrix(raw_input.get(key), feature_dim, key)
            if len(values) != batch_size * scale:
                raise ValueError(f"input_scale * batch_size must match actual {key} row count")
            arrays = (values,)
        if model_ctx["backend"] == "torchscript":
            import torch

            args = tuple(torch.tensor(value, dtype=torch.float32 if index == 0 else torch.int64,
                                      device=model_ctx["device"]) for index, value in enumerate(arrays))
        else:
            args = arrays
        return {"args": args, "_effective_input_scale": float(scale), "_truncated_by_limit": False,
                "_probe_reason": "validated structured input shape; feature dimension fixed"}

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        if model_ctx["backend"] == "skops":
            return model_ctx["model"].predict(processed_input["args"][0])
        import torch

        with torch.inference_mode():
            return model_ctx["model"](*processed_input["args"])

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        if not hasattr(raw_output, "shape") or len(raw_output.shape) < 1:
            raise ValueError("structured model must return one tensor/array with a result dimension")
        shape = list(raw_output.shape)
        return {"task": model_ctx["task_type"], "output_type": TASK_OUTPUT_TYPES[model_ctx["task_type"]],
                "output_shape": shape, "n_results": shape[0]}

    def get_scale_metadata(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
        return {"feature_dim": model_ctx["feature_dim"], "model_format": model_ctx["backend"]}


HandlerRegistry.register("structured", "torchscript", StructuredHandler)
HandlerRegistry.register("structured", "skops", StructuredHandler)

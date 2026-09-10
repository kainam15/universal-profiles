"""Reproducible dense rows, independent observations and graph batches.

The scale changes rows/observations per batch item or nodes per graph. Feature
width stays fixed. Policy inputs are independent observations, not a rollout.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from acprof.workloads import WorkloadGenerator, register_generator


TASK_FEATURE_DIMS = {
    "tabular-classification": 8, "tabular-regression": 8,
    "reinforcement-learning": 4, "robotics": 7, "graph-ml": 16,
}
TASK_SCALE_TYPES = {
    "tabular-classification": "table_rows", "tabular-regression": "table_rows",
    "reinforcement-learning": "observation_count", "robotics": "observation_count",
    "graph-ml": "node_count",
}
SPEC_KEYS = {"schema_version", "task", "feature_dim", "input_scales", "seed", "provenance"}


def positive_integer(value: Any, name: str, maximum: Optional[int] = None) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 or int(value) != value
            or (maximum is not None and value > maximum)):
        bound = f" <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{bound}")
    return int(value)


class StructuredWorkloadGenerator(WorkloadGenerator):
    def __init__(self, model_id: str, task_type: str, batch_size: int,
                 workload_spec_path: Optional[str] = None):
        if task_type not in TASK_FEATURE_DIMS:
            raise ValueError(f"unsupported structured task: {task_type!r}")
        super().__init__(model_id, task_type, positive_integer(batch_size, "batch_size"))
        self._max_scale = 2048 if task_type == "graph-ml" else 4096
        self.feature_dim = TASK_FEATURE_DIMS[task_type]
        self.seed = 12345
        self._scales = [8.0, 32.0, 128.0, 512.0] if task_type == "graph-ml" else [1.0, 8.0, 32.0, 128.0]
        self._spec_sha256 = None
        self._provenance: Dict[str, Any] = {"source": "synthetic", "generator": "python_random_uniform"}
        if workload_spec_path:
            raw = Path(workload_spec_path).read_bytes()
            spec = json.loads(raw)
            if not isinstance(spec, dict) or set(spec) - SPEC_KEYS:
                raise ValueError("structured workload spec must be an object with supported keys: " + ", ".join(sorted(SPEC_KEYS)))
            if spec.get("schema_version") != 1 or isinstance(spec.get("schema_version"), bool):
                raise ValueError("structured workload schema_version must be 1")
            if spec.get("task", task_type) != task_type:
                raise ValueError("structured workload task does not match selected task")
            self.feature_dim = positive_integer(spec.get("feature_dim", self.feature_dim), "feature_dim", 65536)
            seed = spec.get("seed", self.seed)
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")
            self.seed = seed
            if "input_scales" in spec:
                scales = spec["input_scales"]
                if not isinstance(scales, list) or not scales:
                    raise ValueError("input_scales must be a non-empty list")
                self._scales = [float(positive_integer(value, "input_scales", self._max_scale)) for value in scales]
                if self._scales != sorted(set(self._scales)):
                    raise ValueError("input_scales must be strictly increasing")
            provenance = spec.get("provenance", self._provenance)
            if not isinstance(provenance, dict):
                raise ValueError("provenance must be an object")
            self._provenance = copy.deepcopy(provenance)
            self._spec_sha256 = hashlib.sha256(raw).hexdigest()

    def _rows(self, count: int, batch_index: int) -> List[List[float]]:
        rng = random.Random(self.seed + batch_index)
        return [[rng.uniform(-1.0, 1.0) for _ in range(self.feature_dim)] for _ in range(count)]

    def generate(self, scale_value: float) -> Dict[str, Any]:
        count = positive_integer(scale_value, "input_scale", self._max_scale)
        payload: Dict[str, Any] = {"input_scale": count, "batch_size": self.batch_size}
        if self.task_type == "graph-ml":
            graphs = []
            for index in range(self.batch_size):
                # A bidirectional ring has exactly 2 * nodes edges, including
                # two self-loop entries for a one-node graph.
                source = list(range(count)) + [(node + 1) % count for node in range(count)]
                target = [(node + 1) % count for node in range(count)] + list(range(count))
                graphs.append({"node_features": self._rows(count, index), "edge_index": [source, target]})
            payload["graphs"] = graphs
        else:
            key = "features" if self.task_type.startswith("tabular-") else "observations"
            payload[key] = [row for index in range(self.batch_size) for row in self._rows(count, index)]
        return payload

    def scale_label(self, scale_value: float) -> str:
        count = positive_integer(scale_value, "input_scale", self._max_scale)
        return f"{TASK_SCALE_TYPES[self.task_type]}{count}"

    def effective_input_scale(self, scale_value: float, payload: Optional[Dict[str, Any]] = None) -> float:
        if payload is None:
            return float(positive_integer(scale_value, "input_scale", self._max_scale))
        if self.task_type == "graph-ml":
            graphs = payload.get("graphs", [])
            if not isinstance(graphs, list) or len(graphs) != self.batch_size:
                raise ValueError("graphs must match batch_size")
            counts = [len(graph["node_features"]) for graph in graphs]
            if len(set(counts)) != 1:
                raise ValueError("all graphs must have the same node count")
            return float(positive_integer(counts[0], "node_count", self._max_scale))
        key = "features" if self.task_type.startswith("tabular-") else "observations"
        rows = payload.get(key, [])
        if not isinstance(rows, list) or len(rows) % self.batch_size:
            raise ValueError(f"{key} rows must be divisible by batch_size")
        return float(positive_integer(len(rows) / self.batch_size, "input_scale", self._max_scale))

    def max_input_scale(self) -> float:
        return float(self._max_scale)

    def default_input_scales(self) -> Optional[List[float]]:
        return list(self._scales)

    def plan_metadata(self) -> Dict[str, Any]:
        return {"task": self.task_type, "feature_dim": self.feature_dim, "seed": self.seed,
                "input_scale_type": TASK_SCALE_TYPES[self.task_type],
                "construction": "bidirectional_ring" if self.task_type == "graph-ml" else "independent_dense_rows",
                "workload_spec_sha256": self._spec_sha256, "provenance": copy.deepcopy(self._provenance)}

    def input_metadata(self, scale_value: float, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if payload is None:
            payload = self.generate(scale_value)
        count = int(self.effective_input_scale(scale_value, payload))
        metadata = {"feature_dim": self.feature_dim, "seed": self.seed,
                    "input_scale_type": TASK_SCALE_TYPES[self.task_type], "batch_size": self.batch_size}
        if self.task_type == "graph-ml":
            metadata.update(input_num_samples=self.batch_size, node_count=count * self.batch_size,
                            edge_count=sum(len(graph["edge_index"][0]) for graph in payload["graphs"]))
        else:
            metadata.update(input_num_samples=count * self.batch_size,
                            **{TASK_SCALE_TYPES[self.task_type]: count * self.batch_size})
        return metadata


register_generator("structured", StructuredWorkloadGenerator)

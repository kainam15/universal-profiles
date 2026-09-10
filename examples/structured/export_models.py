"""Export small runnable examples of the structured inference contract.

Run in an environment containing torch:
    python examples/structured/export_models.py --output-dir /tmp/structured-models

These deterministic fixtures demonstrate packaging and tensor signatures. They
are not trained policies and do not control an environment or a real robot.
Use your own trained weights and matching feature_dim for actual measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TASKS = {
    "tabular-classification": (8, 3), "tabular-regression": (8, 1),
    "reinforcement-learning": (4, 2), "robotics": (7, 7), "graph-ml": (16, 2),
}


class DenseExample(torch.nn.Module):
    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(feature_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)


class GraphExample(torch.nn.Module):
    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(feature_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # One message-passing step followed by sum pooling per graph.
        messages = torch.zeros_like(x)
        messages.index_add_(0, edge_index[1], x[edge_index[0]])
        node_output = self.projection(x + messages)
        graph_count = int(batch.max().item()) + 1
        output = torch.zeros((graph_count, node_output.size(1)), dtype=node_output.dtype, device=x.device)
        return output.index_add_(0, batch, node_output)


def export_examples(output_dir: Path) -> None:
    torch.manual_seed(12345)
    for task, (feature_dim, output_dim) in TASKS.items():
        destination = output_dir / task
        destination.mkdir(parents=True, exist_ok=True)
        cls = GraphExample if task == "graph-ml" else DenseExample
        model = cls(feature_dim, output_dim).eval()
        torch.jit.script(model).save(str(destination / "model.pt"))
        manifest = {"schema_version": 1, "task": task, "format": "torchscript",
                    "model_file": "model.pt", "feature_dim": feature_dim,
                    "export_torch_version": torch.__version__}
        (destination / "acprof_model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        spec = {"schema_version": 1, "task": task, "feature_dim": feature_dim,
                "seed": 12345, "input_scales": [2, 4, 8],
                "provenance": {"source": "synthetic", "purpose": "format smoke test"}}
        (destination / "workload.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    export_examples(parser.parse_args().output_dir)

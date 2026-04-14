"""Time-series workload generator - synthetic float sequences."""

from __future__ import annotations

import random
from typing import Any, Dict

from workloads import WorkloadGenerator, register_generator

BASE_SEED = 12345


class TimeseriesWorkloadGenerator(WorkloadGenerator):
    """Generate synthetic time-series context sequences.

    Identical approach to example-code/client.py: fixed seed, random floats,
    small batch-specific perturbation.
    """

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        super().__init__(model_id, task_type, batch_size)
        # Pre-generate a long base sequence for slicing
        rng = random.Random(BASE_SEED)
        self._max_len = 2048
        self._base_seq = [rng.random() for _ in range(self._max_len)]

    def generate(self, scale_value: float) -> Dict[str, Any]:
        context_len = int(scale_value)
        seq = self._base_seq[:context_len]

        context_batch = []
        for b in range(self.batch_size):
            # Small perturbation per batch element
            context_batch.append([x + 1e-6 * (b + 1) for x in seq])

        return {
            "context": context_batch,
            "prediction_length": 64,
        }

    def scale_label(self, scale_value: float) -> str:
        return f"ctx{int(scale_value)}"


register_generator("timeseries", TimeseriesWorkloadGenerator)

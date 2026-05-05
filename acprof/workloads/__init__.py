"""AC-Prof workload generator registry and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class WorkloadGenerator(ABC):
    """Base class for task-family-specific input generators."""

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        self.model_id = model_id
        self.task_type = task_type
        self.batch_size = batch_size

    @abstractmethod
    def generate(self, scale_value: float) -> Dict[str, Any]:
        """Return a JSON-serializable payload for /predict."""

    @abstractmethod
    def scale_label(self, scale_value: float) -> str:
        """Return a label for sniff_group_id, e.g., 'seq256', 'res0.5'."""

    def effective_input_scale(
        self,
        scale_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Return the effective input scale represented by the generated payload."""
        return float(scale_value)

    def max_input_scale(self) -> Optional[float]:
        """Return the default maximum input scale for auto-planning when known."""
        return None


_generators: Dict[str, type] = {}


def register_generator(task_family: str, cls: type):
    _generators[task_family] = cls


def get_generator(task_family: str, model_id: str, task_type: str, batch_size: int) -> WorkloadGenerator:
    cls = _generators.get(task_family)
    if cls is None:
        raise ValueError(f"No workload generator for task_family='{task_family}'. Available: {list(_generators.keys())}")
    return cls(model_id, task_type, batch_size)


# Auto-import all generator modules
def _auto_register():
    try:
        from workloads import nlp  # noqa: F401
    except ImportError:
        pass
    try:
        from workloads import cv  # noqa: F401
    except ImportError:
        pass
    try:
        from workloads import audio  # noqa: F401
    except ImportError:
        pass
    try:
        from workloads import timeseries  # noqa: F401
    except ImportError:
        pass


_auto_register()

"""AC-Prof workload generator registry and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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

    def default_input_scales(self) -> Optional[List[float]]:
        """Return workload-defined default scales, or ``None`` for legacy planning."""
        return None

    def plan_metadata(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata to persist with a materialized plan."""
        return {}

    def input_metadata(
        self,
        scale_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return JSON-serializable metadata for one materialized input."""
        return {}


_generators: Dict[str, type] = {}


def register_generator(task_family: str, cls: type):
    _generators[task_family] = cls


def get_generator(
    task_family: str,
    model_id: str,
    task_type: str,
    batch_size: int,
    workload_spec_path: Optional[str] = None,
) -> WorkloadGenerator:
    cls = _generators.get(task_family)
    if cls is None:
        raise ValueError(f"No workload generator for task_family='{task_family}'. Available: {list(_generators.keys())}")
    if workload_spec_path is not None:
        return cls(
            model_id,
            task_type,
            batch_size,
            workload_spec_path=workload_spec_path,
        )
    return cls(model_id, task_type, batch_size)


# Auto-import all generator modules
def _auto_register():
    try:
        from acprof.workloads import nlp  # noqa: F401
    except ImportError:
        pass
    try:
        from acprof.workloads import cv  # noqa: F401
    except ImportError:
        pass
    try:
        from acprof.workloads import audio  # noqa: F401
    except ImportError:
        pass
    try:
        from acprof.workloads import timeseries  # noqa: F401
    except ImportError:
        pass
    try:
        from acprof.workloads import diffusion  # noqa: F401
    except ImportError:
        pass


_auto_register()

"""AC-Prof Handler Registry - BaseHandler interface and task routing."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


def resolve_model_source(model_id: str, model_path: Optional[str] = None) -> str:
    """Prefer a baked local snapshot while retaining model-ID cache compatibility."""
    candidate = model_path
    if candidate is None:
        candidate = os.getenv("MODEL_LOCAL_PATH", "")
    candidate = candidate.strip()
    if candidate and os.path.isdir(candidate):
        return candidate
    return model_id


def model_revision_kwargs(model_source: str, model_revision: str) -> Dict[str, str]:
    """Only pass Hub revision metadata when loading by repository ID."""
    if os.path.isdir(model_source):
        return {}
    return {"revision": model_revision or "main"}


class BaseHandler(ABC):
    """Standard four-phase handler interface for all task families."""

    @abstractmethod
    def load(
        self,
        model_source: str,
        task_type: str,
        backend: str,
        device: str,
        model_revision: str = "main",
    ) -> Dict[str, Any]:
        """Load model, return model_ctx dict containing model, tokenizer, processor, device, etc."""

    @abstractmethod
    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        """Convert raw request data into model input format."""

    @abstractmethod
    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        """Run inference, return raw output."""

    @abstractmethod
    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        """Convert model output to standardized response (shape/metadata only)."""

    def get_scale_metadata(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Return optional scale-planning metadata for the current model."""
        return {}


class HandlerRegistry:
    """Registry mapping (task_family, runtime_backend) -> Handler instance."""

    _handlers: Dict[str, BaseHandler] = {}

    @classmethod
    def register(cls, task_family: str, backend: str, handler_cls: type):
        key = f"{task_family}:{backend}"
        cls._handlers[key] = handler_cls()

    @classmethod
    def get(cls, task_family: str, backend: str) -> BaseHandler:
        key = f"{task_family}:{backend}"
        handler = cls._handlers.get(key)
        if handler is None:
            # Fallback: try family with default backend
            for k, v in cls._handlers.items():
                if k.startswith(f"{task_family}:"):
                    return v
            raise ValueError(
                f"No handler registered for task_family='{task_family}', backend='{backend}'. "
                f"Available: {list(cls._handlers.keys())}"
            )
        return handler


def _auto_register():
    """Import all handler modules to trigger registration."""
    try:
        from acprof.container.handlers import nlp  # noqa: F401
    except ImportError:
        pass
    try:
        from acprof.container.handlers import cv  # noqa: F401
    except ImportError:
        pass
    try:
        from acprof.container.handlers import audio  # noqa: F401
    except ImportError:
        pass
    try:
        from acprof.container.handlers import timeseries  # noqa: F401
    except ImportError:
        pass


_auto_register()

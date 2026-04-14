"""AC-Prof Handler Registry - BaseHandler interface and task routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseHandler(ABC):
    """Standard four-phase handler interface for all task families."""

    @abstractmethod
    def load(self, model_id: str, task_type: str, backend: str, device: str) -> Dict[str, Any]:
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
        from handlers import nlp  # noqa: F401
    except ImportError:
        pass
    try:
        from handlers import cv  # noqa: F401
    except ImportError:
        pass
    try:
        from handlers import audio  # noqa: F401
    except ImportError:
        pass
    try:
        from handlers import timeseries  # noqa: F401
    except ImportError:
        pass


_auto_register()

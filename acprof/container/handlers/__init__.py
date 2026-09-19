"""AC-Prof Handler Registry - BaseHandler interface and task routing."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any, Dict, Optional


def resolve_model_source(model_id: str, model_path: Optional[str] = None) -> str:
    """Use the configured baked snapshot, or an explicit Hub source for manual callers."""
    candidate = model_path
    if candidate is None:
        candidate = os.getenv("MODEL_LOCAL_PATH", "")
    candidate = candidate.strip()
    if candidate:
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"configured model snapshot is missing: {candidate}")
        return candidate
    return model_id


def model_revision_kwargs(model_source: str, model_revision: str) -> Dict[str, str]:
    """Only pass Hub revision metadata when loading by repository ID."""
    if os.path.isdir(model_source):
        return {}
    return {"revision": model_revision or "main"}


def transformers_pipeline_load_kwargs(
    load_options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Translate isolated profiler load options to Transformers pipeline args."""
    options = dict(load_options or {})
    attention_implementation = options.pop("attention_implementation", None)
    if options:
        unsupported = ", ".join(sorted(options))
        raise ValueError(f"unsupported handler load options: {unsupported}")
    if attention_implementation is None:
        return {}
    if str(attention_implementation) != "eager":
        raise ValueError(
            "attention_implementation must be 'eager' for compute profiling"
        )
    return {
        "model_kwargs": {
            "attn_implementation": "eager",
        },
    }


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
        load_options: Optional[Dict[str, Any]] = None,
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

    def validate_output(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any],
                        processed_input: Any, raw_output: Any, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate protocol and task output outside every formal measurement window."""
        entrypoint = model_ctx.get("_validation_entrypoint", "acprof.container.validation:validate_output")
        module_name, separator, name = entrypoint.partition(":")
        if not separator or not module_name or not name:
            raise ValueError("validation entrypoint must have module:callable form")
        validator = getattr(import_module(module_name), name)
        if not callable(validator):
            raise TypeError(f"validation entrypoint is not callable: {entrypoint}")
        return validator(model_ctx, raw_input, processed_input, raw_output, response)

    def get_scale_metadata(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Return optional scale-planning metadata for the current model."""
        return {}


class HandlerRegistrationError(ValueError):
    """Base error for registration and selected handler loading."""


class DuplicateHandlerRegistrationError(HandlerRegistrationError):
    pass


class HandlerNotRegisteredError(HandlerRegistrationError):
    pass


class HandlerDependencyMissingError(HandlerRegistrationError):
    pass


class HandlerModuleImportError(HandlerRegistrationError):
    pass


class HandlerInitializationError(HandlerRegistrationError):
    pass


def load_handler(handler: BaseHandler, model_source: str, task_type: str, backend: str,
                 device: str, model_revision: str = "main", **load_kwargs: Any) -> Dict[str, Any]:
    """Keep dependency failures during model loading as explicit as lazy import failures."""
    module_name = type(handler).__module__
    try:
        context = handler.load(model_source, task_type, backend, device, model_revision, **load_kwargs)
    except HandlerRegistrationError:
        raise
    except ModuleNotFoundError as exc:
        raise HandlerDependencyMissingError(
            f"backend: {backend}; module: {module_name}; missing dependency: {exc.name or 'unknown'}; "
            f"original exception: {type(exc).__name__}: {exc}"
        ) from exc
    except ImportError as exc:
        raise HandlerModuleImportError(
            f"backend: {backend}; module: {module_name}; "
            f"original exception: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        from acprof.extensions import UnsupportedExtensionError

        if isinstance(exc, UnsupportedExtensionError):
            raise
        raise HandlerInitializationError(
            f"backend: {backend}; handler: {module_name}:{type(handler).__qualname__}; "
            f"original exception: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(context, dict):
        raise HandlerInitializationError(f"backend: {backend}; module: {module_name}; handler.load must return a dict")
    return context


@dataclass(frozen=True)
class _LazyHandler:
    entrypoint: str


def _handler_source(value: Any) -> str:
    if isinstance(value, _LazyHandler):
        return value.entrypoint
    target = value if isinstance(value, type) else type(value)
    return f"{target.__module__}:{target.__qualname__}"


class HandlerRegistry:
    """Route declarations are cheap; import and instantiate only the selected handler."""

    _handlers: Dict[str, Any] = {}
    _adapters: Dict[tuple[str, str, str], Any] = {}
    _lock = RLock()

    @classmethod
    def _register(cls, key: Any, value: Any, *, adapter: bool, override: bool):
        target = cls._adapters if adapter else cls._handlers
        with cls._lock:
            if key in target and not override:
                raise DuplicateHandlerRegistrationError(
                    f"duplicate handler key={key!r}; original={_handler_source(target[key])}; "
                    f"new={_handler_source(value)}; source module={_handler_source(value).split(':')[0]}"
                )
            target[key] = value

    @classmethod
    def register(cls, task_family: str, backend: str, handler_cls: type, *, override: bool = False):
        cls._register(f"{task_family}:{backend}", handler_cls, adapter=False, override=override)

    @classmethod
    def register_adapter(cls, adapter: str, task_family: str, backend: str, handler_cls: type,
                         *, override: bool = False):
        cls._register((adapter, task_family, backend), handler_cls, adapter=True, override=override)

    @classmethod
    def register_lazy(cls, task_family: str, backend: str, entrypoint: str, *,
                      adapter: str = "family-default", override: bool = False):
        module, separator, name = entrypoint.partition(":")
        if not separator or not module or not name:
            raise ValueError("handler entrypoint must have module:Class form")
        key = f"{task_family}:{backend}" if adapter == "family-default" else (adapter, task_family, backend)
        cls._register(key, _LazyHandler(entrypoint), adapter=adapter != "family-default", override=override)

    @classmethod
    def get(cls, task_family: str, backend: str) -> BaseHandler:
        adapter = os.getenv("ACPROF_MODEL_ADAPTER", "family-default")
        key = f"{task_family}:{backend}" if adapter == "family-default" else (adapter, task_family, backend)
        target = cls._handlers if adapter == "family-default" else cls._adapters
        with cls._lock:
            selected = target.get(key)
            if selected is None:
                raise HandlerNotRegisteredError(
                    f"Handler not registered: family={task_family}, backend={backend}, adapter={adapter}; "
                    f"available={list(target)}"
                )
            if isinstance(selected, _LazyHandler):
                module_name, _, class_name = selected.entrypoint.partition(":")
                try:
                    handler_cls = getattr(import_module(module_name), class_name)
                except ModuleNotFoundError as exc:
                    # A missing declared module is a broken entrypoint, not an optional dependency.
                    missing_module = exc.name or "unknown"
                    error_cls = HandlerModuleImportError if (
                        module_name == missing_module or module_name.startswith(missing_module + ".")
                    ) else HandlerDependencyMissingError
                    raise error_cls(
                        f"backend: {backend}; module: {module_name}; missing dependency: {missing_module}; "
                        f"original exception: {type(exc).__name__}: {exc}"
                    ) from exc
                except Exception as exc:
                    raise HandlerModuleImportError(
                        f"backend: {backend}; module: {module_name}; "
                        f"original exception: {type(exc).__name__}: {exc}"
                    ) from exc
            elif isinstance(selected, type):
                handler_cls = selected
            else:
                return selected
            try:
                instance = handler_cls()
            except Exception as exc:
                raise HandlerInitializationError(
                    f"backend: {backend}; handler: {_handler_source(handler_cls)}; "
                    f"original exception: {type(exc).__name__}: {exc}"
                ) from exc
            target[key] = instance
            return instance


def _auto_register():
    """Register manifest entrypoints without importing optional implementations."""
    from acprof.extensions import CATALOG

    for declaration in CATALOG.extensions.values():
        for backend in declaration.backends:
            HandlerRegistry.register_lazy(
                declaration.family, backend, declaration.handler_entrypoint,
                adapter=declaration.adapter,
            )


_auto_register()

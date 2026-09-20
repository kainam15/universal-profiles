"""AC-Prof workload generator registry and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any, Dict, List, Optional

from acprof.extensions import CATALOG, ExtensionCatalog, validate_entrypoint


class WorkloadGenerator(ABC):
    """Base class for task-family-specific input generators."""

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        self.model_id = model_id
        self.task_type = task_type
        self.batch_size = batch_size

    @classmethod
    def from_config(
        cls, model_id: str, task_type: str, batch_size: int, *,
        workload_spec_path: Optional[str] = None, model_adapter: Optional[str] = None,
        **options: Any,
    ) -> "WorkloadGenerator":
        """Keep legacy constructors; adapter-specific families override this factory."""
        if workload_spec_path is not None:
            options["workload_spec_path"] = workload_spec_path
        return cls(model_id, task_type, batch_size, **options)

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


class DuplicateWorkloadRegistrationError(ValueError):
    """Different implementations claim the same workload family."""


class WorkloadNotRegisteredError(ValueError):
    """The requested family has no declared workload generator."""


class WorkloadDependencyMissingError(ImportError):
    """A selected generator requires an unavailable optional dependency."""


class WorkloadModuleImportError(ImportError):
    """The selected entrypoint is missing or its module could not initialize."""


class WorkloadConfigurationError(ValueError):
    """The selected generator rejected its construction parameters."""


class WorkloadInitializationError(RuntimeError):
    """The selected generator failed after its module loaded."""


@dataclass(frozen=True)
class _LazyGenerator:
    entrypoint: str
    implementation: Optional[type] = None


_generators: Dict[str, Any] = {}
_lock = RLock()


def _source(value: Any) -> str:
    if isinstance(value, _LazyGenerator):
        return value.entrypoint
    return f"{value.__module__}:{value.__qualname__}"


def _register(task_family: str, value: Any) -> None:
    if not isinstance(task_family, str) or not task_family.strip():
        raise WorkloadConfigurationError("workload task_family must be a non-empty string")
    with _lock:
        previous = _generators.get(task_family)
        if previous is not None:
            if previous is value:
                return
            if isinstance(previous, _LazyGenerator) and previous.implementation is value:
                return
            # A lazy declaration and that module's legacy self-registration are identical.
            unresolved = isinstance(previous, _LazyGenerator) and previous.implementation is None
            if (unresolved or isinstance(value, _LazyGenerator)) and _source(previous) == _source(value):
                # Import-time registration is not evidence that the module finished.
                return
            raise DuplicateWorkloadRegistrationError(
                f"duplicate workload family={task_family!r}; "
                f"original={_source(previous)}; new={_source(value)}"
            )
        _generators[task_family] = value


def register_generator(task_family: str, cls: type):
    if not isinstance(cls, type):
        raise WorkloadConfigurationError("workload generator must be a class")
    _register(task_family, cls)


def register_generator_lazy(task_family: str, entrypoint: str) -> None:
    validate_entrypoint(entrypoint, kind="workload")
    _register(task_family, _LazyGenerator(entrypoint))


def register_catalog(catalog: ExtensionCatalog) -> None:
    """Discover workload routes from the same declarations as host and container."""
    for family, entrypoint in catalog.workloads.items():
        register_generator_lazy(family, entrypoint)


def _resolve_generator(task_family: str) -> type:
    with _lock:
        selected = _generators.get(task_family)
        if selected is None:
            raise WorkloadNotRegisteredError(
                f"No workload generator for task_family={task_family!r}. Available: {sorted(_generators)}"
            )
        if not isinstance(selected, _LazyGenerator):
            return selected
        if selected.implementation is not None:
            return selected.implementation
        module_name, _, class_name = selected.entrypoint.partition(":")
        try:
            generator_cls = getattr(import_module(module_name), class_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            error = WorkloadModuleImportError if (
                module_name == missing or module_name.startswith(missing + ".")
            ) else WorkloadDependencyMissingError
            raise error(
                f"workload family={task_family!r}; source={selected.entrypoint}; "
                f"missing module={missing}; {type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:
            raise WorkloadModuleImportError(
                f"workload family={task_family!r}; source={selected.entrypoint}; "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(generator_cls, type):
            raise WorkloadConfigurationError(f"workload entrypoint is not a class: {selected.entrypoint}")
        _generators[task_family] = _LazyGenerator(selected.entrypoint, generator_cls)
        return generator_cls


def get_generator(
    task_family: str,
    model_id: str,
    task_type: str,
    batch_size: int,
    workload_spec_path: Optional[str] = None,
    model_adapter: Optional[str] = None,
    **options: Any,
) -> WorkloadGenerator:
    cls = _resolve_generator(task_family)
    try:
        # Older externally registered classes need not inherit WorkloadGenerator.
        factory = getattr(cls, "from_config", None)
        config = {"workload_spec_path": workload_spec_path, "model_adapter": model_adapter, **options}
        if factory is None:
            return WorkloadGenerator.from_config.__func__(cls, model_id, task_type, batch_size, **config)
        return factory(model_id, task_type, batch_size, **config)
    except ModuleNotFoundError as exc:
        raise WorkloadDependencyMissingError(
            f"workload family={task_family!r}; source={_source(cls)}; missing dependency={exc.name}: {exc}"
        ) from exc
    except ImportError as exc:
        raise WorkloadModuleImportError(
            f"workload family={task_family!r}; source={_source(cls)}; {type(exc).__name__}: {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise WorkloadConfigurationError(
            f"workload family={task_family!r}; source={_source(cls)}; invalid configuration: {exc}"
        ) from exc
    except Exception as exc:
        raise WorkloadInitializationError(
            f"workload family={task_family!r}; source={_source(cls)}; {type(exc).__name__}: {exc}"
        ) from exc


register_catalog(CATALOG)

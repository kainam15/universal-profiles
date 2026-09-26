"""Typed, standard-library-only validation for internal extension manifests."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
import math
import types
from typing import Any, Union, get_args, get_origin, get_type_hints


def validate_entrypoint(entrypoint: str, *, kind: str) -> None:
    if not isinstance(entrypoint, str):
        raise ValueError(f"{kind} entrypoint must have module:Class form")
    module, separator, name = entrypoint.partition(":")
    if (not separator or not module or not name.isidentifier()
            or not all(part.isidentifier() for part in module.split("."))):
        raise ValueError(f"{kind} entrypoint must have module:Class form: {entrypoint!r}")


def _decode(value: Any, expected: Any, path: str) -> Any:
    """Convert JSON collections according to annotations, rejecting coercions."""
    origin, args = get_origin(expected), get_args(expected)
    if expected is Any:
        return deepcopy(value)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _decode(value, option, path)
            except ValueError:
                pass
        raise ValueError(f"{path} has invalid type")
    if is_dataclass(expected):
        return declaration_from_dict(expected, value, path)
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        return {_decode(k, args[0], path): _decode(v, args[1], f"{path}.{k}") for k, v in value.items()}
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be an array")
        return tuple(_decode(item, args[0], path) for item in value)
    if expected is float and type(value) in (float, int) and math.isfinite(value):
        return float(value)
    if type(value) is not expected:
        raise ValueError(f"{path} must be {expected.__name__}")
    return value


def declaration_from_dict(cls, raw: dict, path: str = "declaration"):
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be an object")
    annotations = get_type_hints(cls)
    unknown = set(raw) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
    try:
        return cls(**{key: _decode(value, annotations[key], f"{path}.{key}") for key, value in raw.items()})
    except TypeError as exc:
        raise ValueError(f"invalid {path}: {exc}") from exc


def merge_template(base: dict, override: dict) -> dict:
    """JSON merge patch semantics: objects merge, lists replace, null removes."""
    result = deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict):
            result[key] = merge_template(result.get(key, {}) if isinstance(result.get(key), dict) else {}, value)
        else:
            result[key] = deepcopy(value)
    return result


def _validate_io_format(formats: dict) -> None:
    if not formats:
        return
    for direction in ("input", "output"):
        value = formats.get(direction)
        schema = value.get("json_schema") if isinstance(value, dict) else None
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"io_format.{direction} must declare an object json_schema")
        required, properties = schema.get("required", []), schema.get("properties", {})
        if (not isinstance(required, list) or not isinstance(properties, dict)
                or any(not isinstance(key, str) or key not in properties for key in required)):
            raise ValueError(f"io_format.{direction}.required must name declared properties")


@dataclass(frozen=True)
class ScalingDeclaration:
    param_name: str
    values: tuple[int | float, ...]
    description: str = ""
    csv_field: str = "input_scale"

    def __post_init__(self):
        if not self.param_name or not self.values or any(value <= 0 or not math.isfinite(value) for value in self.values):
            raise ValueError("scaling requires a name and finite positive values")


@dataclass(frozen=True)
class InputPlanCapabilities:
    accepts_workload_spec: bool = False
    text_payload: bool = False
    audio_payload: bool = False
    requires_scale_meta: bool = False
    max_scale_probe: bool = False
    workload_scales: bool = False
    fractional_scales: bool = False
    model_feature_dim: bool = False
    text_requires_workload_spec: bool = False


@dataclass(frozen=True)
class ExtensionDeclaration:
    extension_id: str
    family: str
    runtime: str
    backends: tuple[str, ...]
    tasks: tuple[str, ...]
    handler_entrypoint: str
    validation_entrypoint: str
    execution: dict[str, str]
    dtypes: tuple[str, ...]
    input_modalities: tuple[str, ...]
    execution_entrypoint: str = ""
    adapter: str = "family-default"
    environment: str = ""
    profile: str = ""
    environments: dict[str, str] = field(default_factory=dict)
    measurement: dict[str, str] = field(default_factory=dict)
    backend_tasks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    task_execution: dict[str, dict[str, str]] = field(default_factory=dict)
    architecture_constraints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    model_types: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    require_registered_custom_architecture: bool = False
    unsupported_reason: str = ""
    workload_entrypoint: str = ""
    scaling: ScalingDeclaration | None = None
    task_params: dict[str, Any] = field(default_factory=dict)
    io_format: dict[str, Any] = field(default_factory=dict)
    task_io_format: dict[str, dict[str, Any]] = field(default_factory=dict)
    precision: dict[str, str] = field(default_factory=dict)
    task_precision: dict[str, dict[str, str]] = field(default_factory=dict)
    backend_precision: dict[str, dict[str, str]] = field(default_factory=dict)
    input_plan: InputPlanCapabilities = field(default_factory=InputPlanCapabilities)
    task_input_plan: dict[str, dict[str, bool]] = field(default_factory=dict)
    handler_options: dict[str, Any] = field(default_factory=dict)
    backend_handler_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_handler_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    workload_defaults: dict[str, Any] = field(default_factory=dict)
    profile_options: dict[str, Any] = field(default_factory=dict)
    dependency_environment: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.extension_id or not self.family or not self.runtime or not self.backends or not self.dtypes:
            raise ValueError("extension requires non-empty ID, family, runtime, backends and dtypes")
        for kind in ("handler", "validation"):
            validate_entrypoint(getattr(self, f"{kind}_entrypoint"), kind=kind)
        if self.workload_entrypoint:
            validate_entrypoint(self.workload_entrypoint, kind="workload")
        for statuses in (self.execution, self.measurement, *self.task_execution.values()):
            if any(status not in {"available", "unsupported"} for status in statuses.values()):
                raise ValueError("manifest status must declare available or unsupported")
        for backend, tasks in self.backend_tasks.items():
            if backend not in self.backends or not set(tasks) <= set(self.tasks):
                raise ValueError(f"backend_tasks references undeclared backend/tasks: {backend}")
        for overrides in self.task_input_plan.values():
            declaration_from_dict(InputPlanCapabilities, overrides, "task_input_plan")
        for key in self.profile_options:
            if key not in {"gpu_dtype", "trust_remote_code", "runtime_line"}:
                raise ValueError(f"unknown profile option: {key}")
        for key, value in self.profile_options.items():
            _decode(value, bool if key == "trust_remote_code" else str, f"profile_options.{key}")

    def validate_templates(self) -> None:
        """Validate once during manifest load, outside request execution."""
        _validate_io_format(self.io_format)
        for task, override in self.task_io_format.items():
            if task in self.tasks:
                _validate_io_format(merge_template(self.io_format, override))


@dataclass(frozen=True)
class BackendRule:
    backend: str
    priority: int
    task: str = ""
    family: str = ""
    libraries: tuple[str, ...] = ()
    when_config: dict[str, Any] = field(default_factory=dict)
    inferred_task: str = ""
    artifact: bool = False

    def matches(self, task: str, family: str, library: str, config: dict) -> bool:
        if self.task and self.task != task or self.family and self.family != family:
            return False
        if self.libraries and library not in self.libraries:
            return False
        for key, expected in self.when_config.items():
            value = config.get(key)
            if (not value if expected is True else value != expected):
                return False
        return True

    @property
    def rank(self) -> tuple[int, int]:
        return (4 if self.artifact else 3 if self.task else 2 if self.libraries or self.when_config else 1, self.priority)

"""标准库扩展声明：主机与容器共享能力、路由；这里不加载推理实现。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Iterable


class UnsupportedExtensionError(ValueError):
    """A declared extension does not support the requested task or execution."""


def validate_entrypoint(entrypoint: str, *, kind: str) -> None:
    """Check a module:Class declaration without importing the implementation."""
    if not isinstance(entrypoint, str):
        raise ValueError(f"{kind} entrypoint must have module:Class form")
    module, separator, name = entrypoint.partition(":")
    if (not separator or not module or not name.isidentifier()
            or not all(part.isidentifier() for part in module.split("."))):
        raise ValueError(f"{kind} entrypoint must have module:Class form: {entrypoint!r}")


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


class ExtensionCatalog:
    def __init__(self):
        self.extensions: dict[str, ExtensionDeclaration] = {}
        self.routes: dict[tuple[str, str, str], ExtensionDeclaration] = {}
        self.task_families: dict[str, str] = {}
        self.architecture_tasks: dict[str, str] = {}
        self.model_type_tasks: dict[str, str] = {}
        self.library_backends: dict[str, str] = {}
        self.backend_rules: list[dict[str, Any]] = []
        self.workloads: dict[str, str] = {}
        self.workload_sources: dict[str, str] = {}

    def add(self, declaration: ExtensionDeclaration) -> None:
        if declaration.extension_id in self.extensions:
            if self.extensions[declaration.extension_id] == declaration:
                return
            raise ValueError(f"duplicate extension ID: {declaration.extension_id}")
        if declaration.workload_entrypoint != "":
            validate_entrypoint(declaration.workload_entrypoint, kind="workload")
            previous = self.workloads.get(declaration.family)
            if previous and previous != declaration.workload_entrypoint:
                raise ValueError(
                    f"conflicting workload family={declaration.family!r}; "
                    f"original={previous} ({self.workload_sources[declaration.family]}); "
                    f"new={declaration.workload_entrypoint} ({declaration.extension_id})"
                )
        for backend in declaration.backends:
            key = (declaration.family, backend, declaration.adapter)
            if key in self.routes:
                raise ValueError(f"duplicate extension route {key}: {self.routes[key].extension_id} / {declaration.extension_id}")
        for task in declaration.tasks:
            previous = self.task_families.get(task)
            if previous and previous != declaration.family:
                raise ValueError(f"conflicting task family for {task}: {previous} / {declaration.family}")
        self.extensions[declaration.extension_id] = declaration
        if declaration.workload_entrypoint:
            self.workloads[declaration.family] = declaration.workload_entrypoint
            self.workload_sources.setdefault(declaration.family, declaration.extension_id)
        for backend in declaration.backends:
            self.routes[(declaration.family, backend, declaration.adapter)] = declaration
        for task in declaration.tasks:
            self.task_families[task] = declaration.family

    def get_extension(self, task_family: str, backend: str, adapter: str = "family-default",
                      task: str | None = None) -> ExtensionDeclaration:
        declaration = self.routes.get((task_family, backend, adapter))
        if declaration is None:
            alternatives = sorted({b for (f, b, a), item in self.routes.items()
                                   if f == task_family and a == adapter and (task is None or task in item.tasks)})
            raise UnsupportedExtensionError(
                f"未登记 {task_family}/{backend} adapter={adapter}；"
                f"请设置 --backend 为 {' / '.join(alternatives) or '已登记后端'}。"
            )
        allowed = declaration.backend_tasks.get(backend, declaration.tasks)
        if task is not None and task not in allowed:
            raise UnsupportedExtensionError(declaration.unsupported_reason or
                f"后端 {backend} 不支持任务 {task}；请用 --backend 选择该任务的已登记后端。")
        if task in declaration.task_execution:
            return replace(declaration, execution={**declaration.execution, **declaration.task_execution[task]})
        return declaration

    def select_extension(self, task_info: Any) -> ExtensionDeclaration:
        config = getattr(task_info, "model_config", {}) or {}
        model_type = str(config.get("model_type") or "")
        model_id = task_info.model_id.lower()
        architecture_matches = [e for e in self.extensions.values() if
                   (model_type and model_type in e.model_types) or model_id in {m.lower() for m in e.model_ids}]
        matches = [e for e in architecture_matches if e.family == task_info.task_family
                   and task_info.runtime_backend in e.backends
                   and task_info.pipeline_tag in e.backend_tasks.get(task_info.runtime_backend, e.tasks)]
        if architecture_matches and not matches:
            raise UnsupportedExtensionError(
                f"No registered runtime/adapter for model architecture {model_type or model_id!r} "
                f"and {task_info.task_family}/{task_info.pipeline_tag}/{task_info.runtime_backend}"
            )
        if len(matches) > 1:
            raise UnsupportedExtensionError(f"ambiguous model extension: {[e.extension_id for e in matches]}")
        adapter = matches[0].adapter if matches else getattr(task_info, "model_adapter", "family-default")
        declaration = self.get_extension(task_info.task_family, task_info.runtime_backend, adapter, task_info.pipeline_tag)
        if matches and declaration.extension_id != matches[0].extension_id:
            raise UnsupportedExtensionError("Model metadata does not match its registered runtime architecture")
        if declaration.model_types and model_type and model_type not in declaration.model_types:
            raise UnsupportedExtensionError("Model metadata does not match its registered runtime architecture")
        supported = declaration.architecture_constraints.get(task_info.pipeline_tag)
        if supported is not None and model_type and model_type not in supported:
            raise UnsupportedExtensionError(
                f"No registered runtime/adapter for {task_info.pipeline_tag} architecture {model_type!r}"
            )
        if declaration.require_registered_custom_architecture and not supported and (config.get("auto_map") or {}).get("AutoConfig"):
            from acprof.model_spec import declared_multimodal_pipeline
            if not declared_multimodal_pipeline(task_info):
                raise UnsupportedExtensionError(
                    "Custom multimodal architecture requires a registered runtime/adapter or "
                    "--model-spec declaring its multimodal custom pipeline protocol"
                )
        return declaration

    def default_backend(self, task: str, family: str, library: str, current: str) -> str:
        # Specific library declarations take precedence over family fallback rules.
        matching = [rule for rule in self.backend_rules if
                    ("task" not in rule or rule["task"] == task) and
                    ("family" not in rule or rule["family"] == family) and
                    ("libraries" not in rule or library in rule["libraries"]) and
                    ("current" not in rule or rule["current"] == current)]
        matching.sort(key=lambda rule: "libraries" in rule, reverse=True)
        return matching[0]["backend"] if matching else current

    def infer_task(self, architectures: Iterable[str], model_type: str = "") -> str | None:
        if model_type in self.model_type_tasks:
            return self.model_type_tasks[model_type]
        # New extension-specific names must beat existing generic class suffixes.
        suffixes = sorted(self.architecture_tasks.items(), key=lambda item: len(item[0]), reverse=True)
        for architecture in architectures:
            for suffix, task in suffixes:
                if architecture.endswith(suffix):
                    return task
        return None


def load_catalog(paths: Iterable[Path] | None = None) -> ExtensionCatalog:
    """Discover manifests under this package; no plugin code or optional packages imported."""
    catalog = ExtensionCatalog()
    for path in sorted(paths if paths is not None else Path(__file__).parent.glob("*/manifest.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported extension manifest schema: {path}")
        for raw in data.get("extensions", []):
            values = dict(raw)
            for name in ("backends", "tasks", "dtypes", "input_modalities", "model_types", "model_ids"):
                if name in values:
                    values[name] = tuple(values[name])
            for name in ("backend_tasks", "architecture_constraints"):
                values[name] = {key: tuple(value) for key, value in values.get(name, {}).items()}
            status_maps = [values.get("execution", {}), values.get("measurement", {}), *values.get("task_execution", {}).values()]
            for statuses in status_maps:
                if any(status not in {"available", "unsupported"} for status in statuses.values()):
                    raise ValueError(f"manifest status must declare available or unsupported: {path}")
            try:
                catalog.add(ExtensionDeclaration(**values))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid extension declaration in {path}: {exc}") from exc
        for name in ("architecture_tasks", "model_type_tasks", "library_backends"):
            target = getattr(catalog, name)
            for key, value in data.get(name, {}).items():
                if key in target and target[key] != value:
                    raise ValueError(f"conflicting {name}: {key}")
                target[key] = value
        for rule in data.get("backend_rules", []):
            if rule not in catalog.backend_rules:
                catalog.backend_rules.append(rule)
    return catalog


CATALOG = load_catalog()
get_extension = CATALOG.get_extension
select_extension = CATALOG.select_extension

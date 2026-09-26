"""标准库扩展声明：主机与容器共享能力、路由；这里不加载推理实现。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable


from acprof.extensions.schema import (
    BackendRule, ExtensionDeclaration, declaration_from_dict, merge_template,
    validate_entrypoint,
)


class UnsupportedExtensionError(ValueError):
    """A declared extension does not support the requested task or execution."""


@dataclass(frozen=True)
class Resolution:
    task: str
    family: str
    backend: str
    declaration: ExtensionDeclaration
    precision: dict[str, str]
    io_format: dict[str, Any]


class ExtensionCatalog:
    def __init__(self):
        self.extensions: dict[str, ExtensionDeclaration] = {}
        self.routes: dict[tuple[str, str, str], ExtensionDeclaration] = {}
        self.task_families: dict[str, str] = {}
        self.architecture_tasks: dict[str, str] = {}
        self.model_type_tasks: dict[str, str] = {}
        self.library_backends: dict[str, str] = {}
        self.backend_rules: list[BackendRule] = []
        self.families: dict[str, dict[str, Any]] = {}
        self.platforms: dict[str, dict[str, Any]] = {}
        self.dependency_environments: dict[str, dict[str, Any]] = {}
        self.runtime_profiles: dict[str, dict[str, Any]] = {}
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
                f"未登记后端 {task_family}/{backend} adapter={adapter}；"
                f"请设置 --backend 为 {' / '.join(alternatives) or '已登记后端'}。"
            )
        allowed = declaration.backend_tasks.get(backend, declaration.tasks)
        if task is not None and task not in allowed:
            raise UnsupportedExtensionError(declaration.unsupported_reason or
                f"后端 {backend} 不支持任务 {task}；请用 --backend 选择该任务的已登记后端。")
        return replace(
            declaration,
            execution={**declaration.execution, **declaration.task_execution.get(task, {})},
            precision=deepcopy(declaration.task_precision.get(task,
                               declaration.backend_precision.get(backend, declaration.precision))),
            io_format=merge_template(declaration.io_format, declaration.task_io_format.get(task, {})),
            input_plan=replace(declaration.input_plan, **declaration.task_input_plan.get(task, {})),
            handler_options=merge_template(
                merge_template(declaration.handler_options, declaration.backend_handler_options.get(backend, {})),
                declaration.task_handler_options.get(task, {})),
        )

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

    def resolve(self, task: str | None = None, family: str | None = None,
                library: str = "", config: dict | None = None, *, backend: str = "",
                model_id: str = "", adapter: str = "family-default",
                precision_dtype: str | None = None, files: Iterable[str] = (),
                metadata: dict | None = None) -> Resolution:
        """Resolve only declared routes; explicit backend overrides source-library hints.

        Task rules precede library/config rules, then family defaults. Priority
        orders rules within a tier; a tied, different backend is an error.
        Architecture inference supplies missing task semantics, never replaces
        an explicit task. Implementations are not imported here.
        """
        config = dict(config or {})
        files = tuple(files)
        config["export_only_onnx"] = (any(name.lower().endswith(".onnx") for name in files)
                                      and not any(name.endswith((".safetensors", ".bin")) for name in files))
        config["sentence_embedding_artifact"] = "modules.json" in (metadata or {})
        if not task or task == "unknown":
            inferred = {rule.inferred_task for rule in self.backend_rules if rule.inferred_task
                        and rule.matches("", rule.family, library, config)}
            if len(inferred) > 1:
                raise UnsupportedExtensionError(f"ambiguous inferred tasks: {sorted(inferred)}")
            task = next(iter(inferred), None) or self.infer_task(
                config.get("architectures") or (), str(config.get("model_type") or ""))
        expected_family = self.task_families.get(task)
        if not expected_family:
            raise UnsupportedExtensionError(f"未声明任务 {task!r}；请设置 --task 或补齐 extension 声明")
        if family and family != "unknown" and family != expected_family:
            raise UnsupportedExtensionError(f"task {task!r} belongs to {expected_family}, not {family}")
        family = expected_family
        if not backend:
            matches = [rule for rule in self.backend_rules if rule.matches(task, family, library, config)]
            if library not in ("", "unknown") and library not in self.library_backends and not any(
                    library in rule.libraries for rule in matches):
                raise UnsupportedExtensionError(f"未声明 library {library!r}；请设置 --backend 或补齐 extension 声明")
            if matches:
                rank = max(rule.rank for rule in matches)
                choices = {rule.backend for rule in matches if rule.rank == rank}
                if len(choices) != 1:
                    raise UnsupportedExtensionError(f"ambiguous backend rules for {task}/{library}: {sorted(choices)}")
                backend = choices.pop()
            else:
                backend = self.library_backends.get(library, "")
            if not backend:
                raise UnsupportedExtensionError(f"未声明 {task}/{family}/{library} 的 backend；请设置 --backend")
        model_type = str(config.get("model_type") or "")
        models = [e for e in self.extensions.values() if
                  (model_type and model_type in e.model_types or model_id.lower() in {m.lower() for m in e.model_ids})
                  and e.family == family and backend in e.backends
                  and task in e.backend_tasks.get(backend, e.tasks)]
        if len(models) > 1:
            raise UnsupportedExtensionError(f"ambiguous model extension: {[e.extension_id for e in models]}")
        declaration = self.get_extension(family, backend, models[0].adapter if models else adapter, task)
        precision = dict(declaration.precision)
        if not precision and precision_dtype:
            precision = {"cpu": precision_dtype, "gpu": precision_dtype}
        return Resolution(task, family, backend, declaration, precision, deepcopy(declaration.io_format))

    def describe(self, task_info: Any) -> Resolution:
        return self.resolve(
            task_info.pipeline_tag, task_info.task_family, task_info.library_name,
            getattr(task_info, "model_config", {}), backend=task_info.runtime_backend,
            model_id=task_info.model_id, adapter=getattr(task_info, "model_adapter", "family-default"),
            precision_dtype=getattr(task_info, "precision_dtype", None),
        )

    def _family_values(self, name: str) -> dict:
        result = {}
        for declaration in self.extensions.values():
            value = getattr(declaration, name)
            if value is None:
                continue
            if declaration.family in result and result[declaration.family] != value:
                raise ValueError(f"conflicting family {name}: {declaration.family}")
            result[declaration.family] = deepcopy(value)
        return result

    def family_scaling(self) -> dict:
        return {family: asdict(value) for family, value in self._family_values("scaling").items()}

    def family_task_params(self) -> dict:
        return self._family_values("task_params")

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
    """Load package manifests in two passes so defaults do not depend on file order."""
    catalog = ExtensionCatalog()
    documents = []
    for path in sorted(paths if paths is not None else Path(__file__).parent.glob("*/manifest.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 2:
            raise ValueError(f"unsupported extension manifest schema; schema_version must be 2: {path}")
        allowed = {"schema_version", "extensions", "families", "architecture_tasks", "model_type_tasks",
                   "library_backends", "backend_rules", "platforms", "dependency_environments", "runtime_profiles"}
        if set(data) - allowed:
            raise ValueError(f"unknown manifest fields in {path}: {sorted(set(data) - allowed)}")
        for name in allowed - {"schema_version", "extensions", "backend_rules"}:
            values = data.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"{name} must be an object: {path}")
            target = getattr(catalog, name)
            for key, value in values.items():
                if key in target and target[key] != value:
                    raise ValueError(f"conflicting {name}: {key}")
                target[key] = value
        documents.append((path, data))
    for path, data in documents:
        try:
            for raw in data.get("extensions", []):
                values = merge_template(catalog.families.get(raw.get("family"), {}), raw)
                declaration = declaration_from_dict(ExtensionDeclaration, values)
                declaration.validate_templates()
                catalog.add(declaration)
            for raw in data.get("backend_rules", []):
                rule = declaration_from_dict(BackendRule, raw, "backend_rule")
                if rule not in catalog.backend_rules:
                    catalog.backend_rules.append(rule)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid extension declaration in {path}: {exc}") from exc
    catalog.family_scaling()
    catalog.family_task_params()
    backends = {backend for declaration in catalog.extensions.values() for backend in declaration.backends}
    for name in ("architecture_tasks", "model_type_tasks"):
        for key, task in getattr(catalog, name).items():
            if not isinstance(key, str) or not isinstance(task, str) or task not in catalog.task_families:
                raise ValueError(f"{name} references undeclared task: {key!r} / {task!r}")
    for library, backend in catalog.library_backends.items():
        if not isinstance(library, str) or not isinstance(backend, str) or backend not in backends:
            raise ValueError(f"library_backends references undeclared backend: {library!r} / {backend!r}")
    for rule in catalog.backend_rules:
        if (rule.backend not in backends or rule.task and rule.task not in catalog.task_families
                or rule.family and rule.family not in catalog.task_families.values()
                or rule.inferred_task and rule.inferred_task not in catalog.task_families):
            raise ValueError(f"backend_rule references undeclared task/family/backend: {rule}")
    return catalog


CATALOG = load_catalog()
get_extension = CATALOG.get_extension
select_extension = CATALOG.select_extension

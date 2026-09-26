"""JSON-only provenance for static model contracts, separate from execution specs."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any


RESOLVER_VERSION = "pipeline-contract-v3"
FIELD_STATES = {"declared", "derived", "verified", "ambiguous", "unresolved"}


def pinned_revision(revision: str) -> bool:
    return isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None


def content_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def resolution_provenance(task_info, candidates: list[dict], *, hub_task: str | None,
                          selected: str | None, status: str, explicit_task: str | None,
                          explicit_backend: str | None, overridden_conflicts: list[str],
                          loader_hint: bool) -> dict:
    """Capture correlated observations; no counting of fields as independent votes."""
    from acprof.model_spec import task_model_spec
    revision = task_info.model_revision
    snapshot = {"model_id": task_info.model_id, "revision": revision,
                "files_sha256": content_digest(sorted(task_info.repository_files or ())),
                "derived_from": []}
    sources = {"repository_snapshot": snapshot}
    hub = {**getattr(task_info, "hub_metadata", {}), "pipeline_tag": hub_task}
    values = {
        "hub": (hub, "repository_metadata", ["repository_snapshot"]),
        "repository_config": (task_info.model_config or {}, "repository_metadata", ["repository_snapshot"]),
        "repository_spec": ((task_info.repository_metadata or {}).get("acprof_model.json", {}), "declaration", ["repository_snapshot"]),
        "local_spec": (getattr(task_info, "model_spec", {}), "user", []),
        "explicit": ({"task": explicit_task, "backend": explicit_backend}, "user", []),
        "generated_contract": (task_model_spec(task_info), "derived", ["repository_config"]),
    }
    for name, (value, family, parents) in values.items():
        sources[name] = {"sha256": content_digest(value), "revision": revision,
                         "family": family, "derived_from": parents}
    observations = []
    for candidate in candidates:
        for field_name in candidate["evidence"]:
            source = ("explicit" if field_name == "explicit_task" else
                      "local_spec" if field_name == "local_model_spec" else
                      "repository_spec" if field_name == "repository_model_spec" else
                      "generated_contract" if field_name == "generated_contract" else
                      "hub" if field_name.startswith("hub") else "repository_config")
            observations.append({"field": field_name, "source_id": source,
                                 "task": candidate["task"], "backend": candidate["backend"]})
    for field_name, value in hub.get("transformers_info", {}).items():
        if field_name in {"processor", "custom_class"}:
            observations.append({"field": f"hub.transformers_info.{field_name}", "source_id": "hub",
                                 "task": selected, "value": value, "backend": task_info.runtime_backend})
        elif field_name == "pipeline_tag" and loader_hint:
            observations.append({"field": "hub.transformers_info.pipeline_tag", "source_id": "hub",
                                 "task": None, "value": value, "kind": "loader_hint",
                                 "reason": "AutoModel is a generic loader for config.custom_pipelines, not a task declaration"})
    identity = {"schema_version": 1, "resolver_version": "model-selection-v2",
                "model_id": task_info.model_id, "revision": revision, "sources": sources,
                "observations": observations, "selected_task": selected, "status": status,
                "overridden_conflicts": overridden_conflicts}
    return {**identity, "identity_sha256": content_digest(identity),
            "decision": "abstain" if status in {"ambiguous", "needs_configuration"} else
                        "explicit" if explicit_task else "consistent_evidence"}


@dataclass
class ModelEvidence:
    model_id: str
    revision: str
    fields: dict[str, dict] = field(default_factory=dict)
    sources: dict[str, dict] = field(default_factory=dict)
    dependency_candidates: list[dict] = field(default_factory=list)

    def add(self, name: str, value: Any, state: str, source: str, *, reason: str = "") -> None:
        if state not in FIELD_STATES:
            raise ValueError(f"unknown evidence state: {state}")
        item = {"value": value, "state": state, "sources": [source]}
        if reason:
            item["reason"] = reason
        if name in self.fields:
            previous = self.fields[name]
            if previous["value"] == value and previous["state"] == state:
                item["sources"] = sorted(set(previous["sources"] + [source]))
            else:
                item = {"value": None, "state": "ambiguous",
                        "sources": sorted(set(previous["sources"] + [source])),
                        "alternatives": [previous, item], "reason": f"conflicting evidence for {name}"}
        self.fields[name] = item

    def unresolved(self, name: str, reason: str, source: str) -> None:
        self.add(name, None, "unresolved", source, reason=reason)

    def source(self, filename: str, data: bytes) -> None:
        self.sources[filename] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                  "revision": self.revision}

    def report(self, *, draft_spec: dict, transformers_version: str | None) -> dict:
        pending = [name for name, value in self.fields.items() if value["state"] in {"unresolved", "ambiguous"}]
        identity = {"model_id": self.model_id, "revision": self.revision, "resolver_version": RESOLVER_VERSION,
                    "transformers_version": transformers_version, "sources": self.sources, "fields": self.fields,
                    "dependency_candidates": self.dependency_candidates}
        return {"schema_version": 1, **identity, "cache_key": content_digest(identity),
                "status": "needs_confirmation" if pending else "resolved", "unresolved_fields": pending,
                "draft_spec": draft_spec, "dependency_candidates": self.dependency_candidates,
                "runtime_validation": "not_run"}

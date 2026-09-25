"""JSON-only provenance for static model contracts, separate from execution specs."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any


RESOLVER_VERSION = "pipeline-contract-v2"
FIELD_STATES = {"declared", "derived", "verified", "ambiguous", "unresolved"}


def pinned_revision(revision: str) -> bool:
    return isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None


def content_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


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
                    "transformers_version": transformers_version, "sources": self.sources, "fields": self.fields}
        return {"schema_version": 1, **identity, "cache_key": content_digest(identity),
                "status": "needs_confirmation" if pending else "resolved", "unresolved_fields": pending,
                "draft_spec": draft_spec, "dependency_candidates": self.dependency_candidates,
                "runtime_validation": "not_run"}

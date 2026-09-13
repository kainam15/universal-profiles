"""Collection and repair provenance kept separate from static metadata."""
from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


COLLECTION_HISTORY_NAME = "collection_history.json"
COLLECTION_HISTORY_SCHEMA_VERSION = 1
COLLECTION_HISTORY_FIELDS = (
    "posthoc_profile_history",
    "timeout_retry_history",
    "quality_retry_history",
    "static_meta_backfill_history",
)


def empty_collection_history() -> Dict[str, Any]:
    """Return a new empty collection-history document."""
    return {
        "schema_version": COLLECTION_HISTORY_SCHEMA_VERSION,
        **{field: [] for field in COLLECTION_HISTORY_FIELDS},
    }


def normalize_collection_history(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Validate known fields while preserving forward-compatible extra fields."""
    if payload is None:
        return empty_collection_history()
    if not isinstance(payload, Mapping):
        raise ValueError("collection history must be a JSON object")

    normalized = copy.deepcopy(dict(payload))
    version = normalized.get("schema_version")
    if type(version) is not int or version != COLLECTION_HISTORY_SCHEMA_VERSION:
        raise ValueError(
            "unsupported collection history schema_version: "
            f"{version!r} (expected {COLLECTION_HISTORY_SCHEMA_VERSION})"
        )
    normalized["schema_version"] = COLLECTION_HISTORY_SCHEMA_VERSION

    for field in COLLECTION_HISTORY_FIELDS:
        history = normalized.get(field, [])
        if not isinstance(history, list):
            raise ValueError(f"{field} must be a JSON array")
        if any(not isinstance(record, Mapping) for record in history):
            raise ValueError(f"{field} entries must be JSON objects")
        normalized[field] = [copy.deepcopy(dict(record)) for record in history]
    return normalized


def append_collection_record(
    payload: Mapping[str, Any] | None,
    history_field: str,
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Append one provenance record to a known history."""
    if history_field not in COLLECTION_HISTORY_FIELDS:
        raise ValueError(f"unsupported collection history field: {history_field}")
    if not isinstance(record, Mapping):
        raise ValueError("collection history record must be a JSON object")
    updated = normalize_collection_history(payload)
    updated[history_field].append(copy.deepcopy(dict(record)))
    return updated


def write_collection_history_json(
    payload: Mapping[str, Any],
    output_path: str | os.PathLike[str],
) -> None:
    """Validate and atomically write ``collection_history.json``."""
    normalized = normalize_collection_history(payload)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        mode = (
            stat.S_IMODE(destination.stat().st_mode)
            if destination.exists()
            else 0o644
        )
        temporary_path.chmod(mode)
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass

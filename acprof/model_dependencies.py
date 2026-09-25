"""Turn static loader candidates into bounded, pinned offline download declarations."""
from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath
import re
from typing import Callable

from acprof.model_evidence import pinned_revision
from acprof.model_spec import validate_dependencies


_TOKENIZER = ("config.json", "tokenizer*.json", "special_tokens_map.json", "added_tokens.json",
              "vocab.json", "vocab.txt", "merges.txt", "*.model", "*.tiktoken", "chat_template.jinja", "chat_templates/*.jinja")
_ROLES = {"tokenizer": _TOKENIZER, "processor": (*_TOKENIZER, "processor*.json", "preprocessor*.json"),
          "metadata": ("config.json",), "generation_metadata": ("generation_config.json",), "weights": ("config.json",)}


def dependency_files(files: list[str], roles: set[str]) -> list[str]:
    """Select known loader files; a processor declaration cannot select weights."""
    if roles - _ROLES.keys():
        raise ValueError("dependency loader role is unresolved")
    if len(files) > 100000 or any(not isinstance(name, str) or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts or "\\" in name or str(PurePosixPath(name)) != name for name in files):
        raise ValueError("invalid or excessive dependency file listing")
    selected = set()
    for role in sorted(roles):
        # fnmatch '*' also crosses directory boundaries. Default loaders read
        # root assets and one chat_templates directory, never nested snapshots.
        matches = {name for name in files
                   if ("/" not in name or name.startswith("chat_templates/") and name.count("/") == 1)
                   and any(fnmatch.fnmatchcase(name, pattern) for pattern in _ROLES[role])}
        if role == "weights":
            # Select a single standard serialization, never all variants/checkpoints.
            if "model.safetensors" in files:
                weights = {"model.safetensors"}
            elif "model.safetensors.index.json" in files:
                weights = {name for name in files if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)}
                if weights:
                    weights.add("model.safetensors.index.json")
            elif "pytorch_model.bin" in files:
                weights = {"pytorch_model.bin"}
            elif "pytorch_model.bin.index.json" in files:
                weights = {name for name in files if re.fullmatch(r"pytorch_model-\d{5}-of-\d{5}\.bin", name)}
                if weights:
                    weights.add("pytorch_model.bin.index.json")
            else:
                weights = set()
            if not weights or "config.json" not in matches:
                raise ValueError("dependency has no unambiguous standard model weights/config")
            matches |= weights
        if not matches:
            raise ValueError(f"dependency has no files for {role}")
        selected |= matches
    if len(selected) > 128:
        raise ValueError("dependency file selection exceeds 128 files")
    # Hub uses fnmatch, even for filenames; preserve literal special characters.
    escape = {"[": "[[]", "*": "[*]", "?": "[?]"}
    return ["".join(escape.get(char, char) for char in name) for name in sorted(selected)]


def resolve_dependencies(candidates: list[dict], resolve_repository: Callable | None) -> tuple[list[dict], list[str]]:
    groups: dict[str, list[dict]] = {}
    errors = []
    for item in candidates:
        if item.get("required") == "inactive":
            continue
        repo = item.get("repo_id")
        if not repo or item["role"] not in _ROLES or item.get("conditional"):
            errors.append(f"{item['source']}: unresolved dependency {item.get('expression') or repo} ({item['role']})")
            continue
        groups.setdefault(repo, []).append(item)
    if len(groups) > 16:
        return [], ["dependency candidates exceed 16 repositories"]
    dependencies = []
    for repo, items in sorted(groups.items()):
        try:
            validate_dependencies([{"repo_id": repo, "revision": "0" * 40}])
            revisions = {item.get("requested_revision", "main") for item in items}
            if len(revisions) != 1:
                raise ValueError("conflicting dependency revisions")
            requested = revisions.pop()
            if requested != "main" and not pinned_revision(requested):
                raise ValueError("dynamic/branch dependency revision requires an explicit declaration")
            if resolve_repository is None:
                raise ValueError("dependency metadata resolver is unavailable")
            info = resolve_repository(repo, requested)
            revision = info["revision"]
            if not pinned_revision(revision) or pinned_revision(requested) and revision != requested:
                raise ValueError("dependency Hub response must match a fixed commit SHA")
            patterns = dependency_files(info["files"], {item["role"] for item in items})
            declaration = {"repo_id": repo, "revision": revision, "allow_patterns": patterns}
            validate_dependencies([declaration])
            dependencies.append(declaration)
            for item in items:
                item["resolution"] = {**declaration, "status": "pinned", "offline_ready": False}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{repo}: {exc}")
    return dependencies, errors

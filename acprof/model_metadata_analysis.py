"""Structured, revision-bound evidence and bounded source collection."""
from __future__ import annotations

import ast
import json
from pathlib import PurePosixPath
from typing import Callable

from acprof.model_evidence import ModelEvidence
from acprof.model_source_analysis import dependency_candidates, model_card_examples, parse_source
from acprof.model_spec import custom_code_files


MAX_SOURCE_FILES = 32
MAX_TOTAL_SOURCE_BYTES = 2 * 1024 * 1024


def collect_structured_evidence(task_info, evidence: ModelEvidence) -> dict:
    metadata = task_info.repository_metadata or {}
    config = metadata.get("config.json", task_info.model_config or {})
    selection = (task_info.model_resolution or {}).get("selection", {})
    task = selection.get("task") or task_info.pipeline_tag
    source = "explicit.task" if selection.get("source") == "explicit" else (
        "hub.pipeline_tag" if task_info.detection_method == "hub_api" else "config.architectures")
    evidence.add("task", task, "derived" if source == "config.architectures" else "declared", source)
    evidence.add("library", task_info.library_name, "declared", "hub.library_name")
    for filename, value in sorted(metadata.items()):
        # Values are already parsed by detection; explicitly distinguish this
        # canonical hash from hashes of the original source-file bytes.
        evidence.source(filename, json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
        evidence.sources[filename]["encoding"] = "canonical-json"
        if filename != "config.json":
            evidence.add(f"metadata.{filename}", value, "declared", filename)
    for key in ("model_type", "architectures", "auto_map", "custom_pipelines", "_name_or_path",
                "transformers_version", "text_config", "audio_config", "vision_config"):
        if key in config:
            evidence.add(f"config.{key}", config[key], "declared", f"config.json:/{key}")
    for key, value in config.items():
        if key.endswith(("_model_id", "_model_name_or_path")):
            evidence.add(f"config.{key}", value, "declared", f"config.json:/{key}")
    return config


def collect_source_evidence(task_info, evidence: ModelEvidence, config: dict,
                            read_text: Callable[[str], str]) -> dict[str, str]:
    files = set(task_info.repository_files)
    queue = custom_code_files(config)
    for name, metadata in task_info.repository_metadata.items():
        if name != "config.json" and isinstance(metadata, dict):
            queue.extend(custom_code_files(metadata))
    sources, total = {}, 0
    while queue:
        filename = queue.pop(0)
        if filename in sources:
            continue
        if filename not in files:
            raise ValueError(f"custom code is absent from pinned snapshot: {filename}")
        if len(sources) >= MAX_SOURCE_FILES:
            raise ValueError(f"custom source graph exceeds {MAX_SOURCE_FILES} files")
        source = read_text(filename)
        total += len(source.encode())
        if total > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("custom source graph exceeds 2 MiB")
        tree = parse_source(source, filename)
        sources[filename] = source
        evidence.source(filename, source.encode())
        evidence.dependency_candidates.extend(dependency_candidates(source, filename, config, task_info.model_id))
        # Follow local imports only. This is syntax inspection, never importlib.
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = PurePosixPath(filename).parent
            for _ in range(node.level - 1):
                if base == PurePosixPath("."):
                    raise ValueError(f"{filename}: relative import escapes pinned snapshot")
                base = base.parent
            modules = [node.module] if node.module else [alias.name for alias in node.names]
            for module in modules:
                module_path = base / module.replace(".", "/")
                choices = (str(module_path) + ".py", str(module_path / "__init__.py"))
                existing = [name for name in choices if name in files]
                if not existing:
                    raise ValueError(f"{filename}: relative source module is missing: {module}")
                queue.extend(existing)
    # Config references are candidates, not evidence that entire repos must be
    # downloaded. Include those hidden behind helpers for later dependency work.
    def config_references(value, path=""):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            location = f"{path}/{key}"
            if isinstance(item, dict):
                config_references(item, location)
            elif key.endswith(("_model_id", "_model_name_or_path")) or key == "_name_or_path" and path:
                if isinstance(item, str) and "/" in item and not item.startswith(("/", ".")) and item != task_info.model_id:
                    if not any(candidate["repo_id"] == item for candidate in evidence.dependency_candidates):
                        evidence.dependency_candidates.append({"repo_id": item, "role": "unknown", "loader": None,
                            "required": "candidate", "state": "declared", "source": f"config.json:{location}"})
    config_references(config)
    if "README.md" in files:
        try:
            card = read_text("README.md")
            evidence.source("README.md", card.encode())
            for index, example in enumerate(model_card_examples(card)):
                evidence.add(f"documentation.example.{index}", example, "derived", f"README.md:{example['line']}")
        except (OSError, ValueError) as exc:
            # Supplementary documentation cannot invalidate structured facts.
            evidence.sources["README.md"] = {"revision": evidence.revision, "error": str(exc)}
    return sources

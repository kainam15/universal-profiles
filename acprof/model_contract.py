"""Synthesize conservative v1 Pipeline specs without executing repository code.

Runtime verification and automatic dependency pinning are separate stages.
Only complete, conflict-free static contracts participate in runtime routing.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

from acprof.artifacts import atomic_write_json
from acprof.model_evidence import ModelEvidence, pinned_revision
from acprof.model_metadata_analysis import collect_source_evidence, collect_structured_evidence
from acprof.model_source_analysis import analyze_pipeline
from acprof.model_spec import MULTIMODAL_PIPELINE_INPUTS, task_model_spec, validate_model_spec


# These are canonical protocol names, never checkpoint-specific rules. Rename
# text to prompt only with a literal string default; structured chats need a DSL.
_INPUT_NAMES = {"text": ("text", "prompt"), "audio": ("audio",), "sampling_rate": ("sampling_rate",),
                "image": ("image",), "video": ("video",), "fps": ("fps",)}


def _compatible_input(canonical: str, item: dict) -> bool:
    default: Any = item["default"]
    if canonical == "text":
        return isinstance(default, str)
    if canonical in {"sampling_rate", "fps"} and item["default_known"] and default is not None:
        return type(default) in {int, float} and default > 0
    if canonical in {"audio", "image", "video"} and item["default_known"] and default is not None:
        return False
    return True


def _synthesize(config: dict, sources: dict[str, str], evidence: ModelEvidence, task: str) -> dict:
    name, pipeline = next(iter(config["custom_pipelines"].items()))
    evidence.add("pipeline_task", name, "declared", f"config.json:/custom_pipelines/{name}")
    evidence.add("format", "transformers-pipeline", "derived", "config.json:/custom_pipelines")
    module, _, class_name = pipeline["impl"].rpartition(".")
    filename = module.replace(".", "/") + ".py"
    analysis = analyze_pipeline(sources[filename], filename, class_name)
    for method, line in analysis["methods"].items():
        evidence.add(f"pipeline.methods.{method}", line, "derived", f"{filename}:{line}:{method}")
    evidence.add("pipeline.forward_signature", analysis["signature"], "derived", f"{filename}:_forward")
    if analysis["sanitized_keys"] is None:
        evidence.unresolved("pipeline.sanitized_keys", "forward parameter routing is dynamic", f"{filename}:_sanitize_parameters")
    else:
        evidence.add("pipeline.sanitized_keys", analysis["sanitized_keys"], "derived", f"{filename}:_sanitize_parameters")
    for index, issue in enumerate(analysis["issues"]):
        evidence.unresolved(f"pipeline.analysis.{index}", issue, filename)
    if analysis["registrations"] and name not in analysis["registrations"]:
        evidence.add("pipeline_task", analysis["registrations"], "derived", f"{filename}:register_pipeline")

    inputs = {}
    required_inputs: set[str] = MULTIMODAL_PIPELINE_INPUTS[task]
    for canonical in sorted(required_inputs):
        choices = []
        for target in _INPUT_NAMES[canonical]:
            item = analysis["inputs"].get(target)
            if item is None:
                continue
            if not _compatible_input(canonical, item):
                continue
            choices.append(target)
        if len(choices) == 1:
            target = choices[0]
            inputs[target] = canonical
            evidence.add(f"multimodal.inputs.{target}", canonical, "derived",
                         f"{filename}:{analysis['inputs'][target]['line']}:preprocess")
        else:
            evidence.add(f"multimodal.inputs.{canonical}", None, "ambiguous" if choices else "unresolved",
                         f"{filename}:preprocess", reason=f"input {canonical}: expected one compatible field, found {choices}")
    for name, item in analysis["inputs"].items():
        evidence.add(f"pipeline.inputs.{name}", item, "derived", f"{filename}:{item['line']}:preprocess")
        if item["required"] and name not in inputs:
            evidence.unresolved(f"multimodal.inputs.{name}", f"required input {name} has no canonical mapping", filename)
    kwargs = analysis["forward_kwargs"] or {}
    if not kwargs:
        evidence.unresolved("multimodal.forward_kwargs", "bounded deterministic generation is unresolved", f"{filename}:_forward")
    for name, value in kwargs.items():
        evidence.add(f"multimodal.forward_kwargs.{name}", value, "derived", f"{filename}:_forward")
    draft = {"schema_version": 1, "format": "transformers-pipeline", "task": task,
             "pipeline_task": evidence.fields["pipeline_task"]["value"],
             "multimodal": {"inputs": inputs, "forward_kwargs": kwargs}}
    # A draft with unresolved fields remains reviewable but cannot be executed.
    try:
        validate_model_spec(draft)
    except (ValueError, TypeError) as exc:
        evidence.unresolved("model_spec", str(exc), "schema-v1")
    return draft


def resolve_model_contract(task_info, read_text: Callable[[str], str], *, resolve_repository=None,
                           selected_pipeline: str | None = None) -> dict | None:
    """Attachable provenance; existing author specs bypass source synthesis."""
    config = (task_info.repository_metadata or {}).get("config.json", task_info.model_config or {})
    spec = task_info.model_spec or (task_info.repository_metadata or {}).get("acprof_model.json", {})
    # Native loaders and non-Pipeline formats keep their established resolution.
    if not config.get("custom_pipelines") and not spec:
        return None
    if not spec and task_info.pipeline_tag not in MULTIMODAL_PIPELINE_INPUTS:
        return None
    evidence = ModelEvidence(task_info.model_id, task_info.model_revision)
    config = collect_structured_evidence(task_info, evidence)
    draft = {}
    if not pinned_revision(task_info.model_revision):
        evidence.unresolved("revision", "model revision must be a full commit SHA before contract analysis", "hub.sha")
    elif spec:
        try:
            validate_model_spec(spec)
            draft = copy.deepcopy(spec)
            source = "local_model_spec" if task_info.model_spec else "acprof_model.json"
            for name, value in spec.items():
                # Discovery already handles Hub/spec conflicts and overrides.
                evidence.fields[name] = {"value": value, "state": "declared", "sources": [source]}
            if task_info.model_resolution.get("conflicts"):
                evidence.add("selection", None, "ambiguous", "candidate-selection",
                             reason="; ".join(task_info.model_resolution["conflicts"]))
        except (ValueError, TypeError) as exc:
            evidence.unresolved("model_spec", str(exc), "acprof_model.json")
    elif task_info.model_resolution.get("status") in {"ambiguous", "needs_configuration"}:
        evidence.unresolved("selection", "; ".join(task_info.model_resolution.get("conflicts", []) +
                                                   task_info.model_resolution.get("missing", [])), "candidate-selection")
    else:
        task = task_info.pipeline_tag
        pipelines = config.get("custom_pipelines")
        if not isinstance(pipelines, dict) or not pipelines:
            evidence.unresolved("pipeline_task", "custom_pipelines must be a nonempty object", "config.json")
        elif selected_pipeline is not None and selected_pipeline not in pipelines:
            evidence.unresolved("pipeline_task", "selected Pipeline is not declared", "user.review")
        elif len(pipelines) != 1 and selected_pipeline is None:
            evidence.add("pipeline_task", None, "ambiguous", "config.json:/custom_pipelines",
                         reason=f"multiple custom pipelines: {sorted(pipelines)}; select one with --model-spec")
        else:
            try:
                selected = {**config, "custom_pipelines": {selected_pipeline: pipelines[selected_pipeline]}} if selected_pipeline else config
                sources = collect_source_evidence(task_info, evidence, selected, read_text)
                evidence.add("custom_code", {"required": True, "files": sorted(sources),
                                             "revision": task_info.model_revision}, "declared", "config.json")
                draft = _synthesize(selected, sources, evidence, task)
                if selected_pipeline:
                    evidence.add("pipeline_task", selected_pipeline, "declared", "user.review")
                if evidence.dependency_candidates:
                    from acprof.model_dependencies import resolve_dependencies
                    dependencies, errors = resolve_dependencies(evidence.dependency_candidates, resolve_repository)
                    if dependencies:
                        draft["dependencies"] = dependencies
                    evidence.add("dependencies", dependencies, "unresolved" if errors else "derived",
                                 "source/config analysis + pinned Hub file listing", reason="; ".join(errors))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                evidence.unresolved("pipeline.source", str(exc), "pinned source analysis")
    from acprof.runtime_profiles import ENVIRONMENTS, _transformers_version
    version = _transformers_version(ENVIRONMENTS["custom-multimodal-cpu"]) if config.get("custom_pipelines") else None
    return evidence.report(draft_spec=draft, transformers_version=version)


def apply_model_contract(task_info, read_text: Callable[[str], str], *, override_tag=None, override_backend=None,
                         resolve_repository=None, selected_pipeline=None) -> None:
    report = resolve_model_contract(task_info, read_text, resolve_repository=resolve_repository,
                                    selected_pipeline=selected_pipeline)
    task_info.model_resolution.pop("generated_spec", None)
    if report is None:
        return
    task_info.model_resolution["contract"] = report
    if report["status"] == "resolved":
        if not task_model_spec(task_info):
            from acprof.model_resolution import discover_model_candidates
            task_info.model_resolution["generated_spec"] = report["draft_spec"]
            selection = discover_model_candidates(task_info, override_tag=override_tag, override_backend=override_backend)
            task_info.model_resolution.update(selection)
    else:
        resolution = task_info.model_resolution
        for name in report["unresolved_fields"]:
            item = report["fields"][name]
            destination = "conflicts" if item["state"] == "ambiguous" else "missing"
            resolution.setdefault(destination, []).append(f"{name}: {item.get('reason', item['state'])}")
        resolution["status"] = "ambiguous" if resolution.get("conflicts") else "needs_configuration"


def write_model_resolution(task_info, output_dir: str | Path) -> Path:
    """Export provenance atomically; unresolved drafts never become executable files."""
    root = Path(output_dir)
    path = root / "model_resolution.json"
    atomic_write_json(path, task_info.model_resolution)
    return path


def record_runtime_validation(task_info, report: dict) -> None:
    """Keep runtime facts scoped to the actual image, mode, payload and devices."""
    from acprof.model_evidence import content_digest
    contract = task_info.model_resolution["contract"]
    mode = report["mode"]
    passed = report["status"] == "ok"
    status = ("verified" if mode == "full" else "basic_verified") if passed else report["status"]
    contract["runtime_validation"] = {
        "status": status, "mode": mode, "image_id": report["image_id"],
        "build_fingerprint": report["build_fingerprint"], "payload_sha256": report["payload_sha256"],
        "report_sha256": content_digest(report), "devices": copy.deepcopy(report["devices"]),
    }
    # Runtime evidence does not change the static cache identity or erase gaps.
    contract["fields"]["runtime." + mode] = {
        "state": "verified" if passed else "unresolved", "value": status,
        "sources": ["runtime_validation.json"], "image_id": report["image_id"],
    }

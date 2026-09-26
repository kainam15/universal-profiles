"""Explicit field-level decisions over a pinned, reviewable draft contract."""
from __future__ import annotations

import copy

from acprof.model_evidence import content_digest
from acprof.model_spec import validate_model_spec


def review_questions(task_info) -> list[dict]:
    report = task_info.model_resolution.get("contract", {})
    fields = report.get("fields", {})
    pending = report.get("unresolved_fields", [])
    draft = report.get("draft_spec", {})
    inputs = draft.get("multimodal", {}).get("inputs", {})
    input_gaps = [name for name in pending if name.startswith("multimodal.inputs.")]
    questions = []
    unassigned = {name.removeprefix("pipeline.inputs."): item["value"] for name, item in fields.items()
                  if name.startswith("pipeline.inputs.") and name.removeprefix("pipeline.inputs.") not in inputs}
    targets = [name for name, item in unassigned.items() if item.get("required")]
    text_names = {"prompt", "text", "messages", "turns"}
    if "multimodal.inputs.text" in input_gaps and not text_names.intersection(targets):
        targets.extend(name for name in unassigned if name in text_names and name not in targets)
    if input_gaps:
        for target in targets:
            questions.append({"path": "multimodal.inputs." + target, "value": None,
                              "reason": "; ".join(fields[name].get("reason", name) for name in input_gaps)})
        if not targets:
            questions.append({"path": "multimodal.inputs", "value": inputs,
                              "reason": "canonical input mapping is unresolved"})
    for name in pending:
        if name in input_gaps or name == "model_spec" and questions:
            continue
        if name == "pipeline_task" and fields[name]["state"] == "ambiguous":
            config = task_info.repository_metadata.get("config.json", task_info.model_config)
            questions.append({"path": name, "value": None, "reason": fields[name]["reason"],
                              "options": sorted(config.get("custom_pipelines", {}))})
        elif name in {"dependencies", "multimodal.forward_kwargs"}:
            if name == "dependencies":
                value = []
                for candidate in report.get("dependency_candidates", []):
                    if candidate.get("activation") == "inactive" or candidate.get("dependency_kind") == "main_model":
                        continue
                    choice = {"repo_id": candidate["repo_id"], "role": candidate["role"], "required": True}
                    if choice not in value:
                        value.append(choice)
            else:
                value = draft.get("multimodal", {}).get("forward_kwargs", {})
            questions.append({"path": name, "value": value, "reason": fields[name].get("reason", name)})
        else:
            # Arbitrary source/control-flow failures cannot be fixed by accepting
            # a checkbox. They require an explicit adapter/declaration.
            questions.append({"path": name, "value": None, "reason": fields[name].get("reason", name), "read_only": True})
    return questions


def apply_review(task_info, answers: dict):
    """Return a new task; failed/partial decisions leave the original untouched."""
    from acprof.model_resolution import discover_model_candidates
    from acprof.host.task_support import require_task_support

    questions = review_questions(task_info)
    if any(item.get("read_only") for item in questions):
        raise ValueError("source or selection gaps require an explicit model spec/adapter")
    expected = {item["path"] for item in questions}
    if set(answers) != expected:
        raise ValueError("answers must cover exactly the unresolved fields")
    task = copy.deepcopy(task_info)
    if set(answers) == {"pipeline_task"}:
        from acprof.model_contract import apply_model_contract
        from acprof.host.detect import read_model_source, dependency_metadata
        if answers["pipeline_task"] not in questions[0].get("options", []):
            raise ValueError("select a declared Pipeline")
        previous = task.model_resolution["contract"]
        task.model_resolution = discover_model_candidates(task)
        apply_model_contract(task, lambda name: read_model_source(task.model_id, name, task.model_revision),
                             resolve_repository=dependency_metadata, selected_pipeline=answers["pipeline_task"])
        task.model_resolution["contract"]["reviews"] = [*previous.get("reviews", []),
            {"revision": task.model_revision, "answers": answers, "previous_cache_key": previous["cache_key"]}]
        if task.model_resolution["contract"]["status"] == "resolved":
            require_task_support(task)
        return task
    report = task.model_resolution["contract"]
    draft = copy.deepcopy(report["draft_spec"])
    for path, value in answers.items():
        if path == "dependencies":
            from acprof.model_dependencies import resolve_dependencies
            from acprof.host.detect import dependency_metadata
            if not isinstance(value, list) or len(value) > 16 or any(
                not isinstance(item, dict) or set(item) - {"repo_id", "role", "required"}
                or type(item.get("required", True)) is not bool
                or item.get("required", True) and (not isinstance(item.get("repo_id"), str)
                                                    or not isinstance(item.get("role"), str)) for item in value
            ):
                raise ValueError("dependency decisions require a bounded list of repo_id/role/required objects")
            candidates = []
            for item in value:
                if not item.get("required", True):
                    continue
                # Confirming a conditional loader doesn't change the revision
                # its source will request. Conflicts still need a declaration.
                original = [candidate for candidate in report.get("dependency_candidates", [])
                            if candidate.get("repo_id") == item["repo_id"] and candidate.get("role") in {item["role"], "unknown"}
                            and candidate.get("activation") != "inactive"
                            and candidate.get("dependency_kind") != "main_model"] or [{}]
                candidates.extend({**item, "required": "candidate", "source": "user.review",
                                   "activation": "active", "dependency_kind": "external", "loader": candidate.get("loader"),
                                   "requested_revision": candidate.get("requested_revision", "main")}
                                  for candidate in original)
            value, errors = resolve_dependencies(candidates, dependency_metadata)
            if errors:
                raise ValueError("; ".join(errors))
            report["reviewed_dependency_candidates"] = candidates
        target = draft
        *parts, key = path.split(".")
        for part in parts:
            target = target.setdefault(part, {})
        target[key] = copy.deepcopy(value)
    validate_model_spec(draft)
    previous = {name: report["fields"][name] for name in report["unresolved_fields"]}
    report.setdefault("reviews", []).append({"revision": task.model_revision, "answers": copy.deepcopy(answers),
                                              "previous_fields": previous})
    for name in report["unresolved_fields"]:
        report["fields"].pop(name)
    for name, value in answers.items():
        report["fields"][name] = {"value": copy.deepcopy(draft["dependencies"] if name == "dependencies" else value),
                                  "state": "declared", "sources": ["user.review"]}
    report.update(draft_spec=draft, status="resolved", unresolved_fields=[], runtime_validation="not_run")
    report["cache_key"] = content_digest({"previous": report["cache_key"], "answers": answers, "draft_spec": draft})
    task.model_spec = draft
    task.model_resolution = discover_model_candidates(task)
    task.model_resolution["contract"] = report
    require_task_support(task)
    return task

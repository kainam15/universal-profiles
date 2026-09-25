"""Frozen model samples and coverage reports, independent of formal measurements."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from acprof.artifacts import atomic_write_json
from acprof.model_evidence import content_digest, pinned_revision


def validate_sample(sample: dict) -> None:
    from huggingface_hub.utils import validate_repo_id
    if not isinstance(sample, dict) or sample.get("schema_version") != 1:
        raise ValueError("unsupported coverage sample schema")
    if not all(isinstance(sample.get(key), str) and sample[key] for key in ("sampling", "weight_basis")):
        raise ValueError("sample requires sampling and weight_basis")
    if not isinstance(sample.get("models"), list) or not sample["models"]:
        raise ValueError("coverage sample must contain models")
    seen = set()
    for item in sample["models"]:
        if not isinstance(item, dict):
            raise ValueError("model entries must be objects")
        validate_repo_id(item["model_id"])
        if not pinned_revision(item.get("revision")):
            raise ValueError("coverage requires full commit SHAs")
        identity = (item["model_id"], item["revision"])
        if identity in seen:
            raise ValueError("duplicate model/revision in sample")
        seen.add(identity)
        weight = item.get("weight")
        if type(weight) not in {float, int} or not math.isfinite(weight) or weight < 0:
            raise ValueError("coverage weights must be finite nonnegative numbers")
        reference = item.get("semantic_reference")
        if reference is not None and (not isinstance(reference, dict) or not isinstance(reference.get("task"), str)
                                      or not reference["task"] or not reference.get("source")):
            raise ValueError("semantic_reference needs an independently reviewed task and source")


def snapshot_sample(strata: list[str], limit: int) -> dict:
    from huggingface_hub import HfApi
    from acprof.hf_endpoints import hf_endpoints
    if limit <= 0:
        raise ValueError("limit must be positive")
    api = HfApi(endpoint=hf_endpoints()[0])
    models = {}
    for stratum in strata:
        task, separator, library = stratum.partition(":")
        if not separator or not task or not library:
            raise ValueError("stratum must be TASK:LIBRARY")
        selected = 0
        for info in api.list_models(pipeline_tag=task, filter=library, sort="downloads", limit=limit * 20, full=True):
            # Hub tag filtering can include wrappers tagged with multiple libraries.
            if info.library_name != library or info.pipeline_tag != task:
                continue
            revision = info.sha
            if not pinned_revision(revision):
                revision = api.model_info(info.id).sha
            models[(info.id, revision)] = {"model_id": info.id, "revision": revision,
                                          "weight": info.downloads or 0, "stratum": stratum}
            selected += 1
            if selected == limit:
                break
    sample = {"schema_version": 1, "sampling": "head_per_task_library; selected_sample_only",
              "captured_at": datetime.now(timezone.utc).isoformat(),
              "weight_basis": "Hub downloads at snapshot time; not a whole-Hub estimate",
              "strata": strata, "limit_per_stratum": limit, "models": list(models.values())}
    validate_sample(sample)
    return sample


def summarize(rows: list[dict], *, probe_requested: bool, sample_weight: float) -> dict:
    def weight(predicate):
        return sum(row["weight"] for row in rows if predicate(row))

    def rate(value, denominator=sample_weight):
        return value / denominator if denominator else None

    reviewed = [row for row in rows if row.get("semantic_correct") is not None]
    reviewed_weight = sum(row["weight"] for row in reviewed)
    return {
        "sample_weight": sample_weight, "completed_count": len(rows),
        "weighted_resolution_rate": rate(weight(lambda row: row["resolved"])),
        "weighted_support_rate": rate(weight(lambda row: row["supported"])),
        "weighted_abstain_rate": rate(weight(lambda row: row["resolution_status"] == "abstained")),
        "weighted_runtime_success_rate": rate(weight(lambda row: row["runtime_status"] == "ok")) if probe_requested else None,
        "weighted_resource_limited_rate": rate(weight(lambda row: row["runtime_status"] == "resource_limited")) if probe_requested else None,
        "weighted_access_denied_rate": rate(weight(lambda row: row["resolution_status"] == "access_denied")) if probe_requested else None,
        "semantic_reviewed_weight": reviewed_weight,
        "semantic_accuracy_on_reviewed_selections": rate(sum(row["weight"] for row in reviewed if row["semantic_correct"]), reviewed_weight),
        "semantic_wrong_selections": sum(row["semantic_correct"] is False for row in reviewed),
        "runtime_counts": dict(Counter(row["runtime_status"] for row in rows)),
        "failure_stages": dict(Counter(row["failed_stage"] for row in rows if row.get("failed_stage"))),
    }


def run_sample(sample: dict, root: Path, *, probe: str = "none", cpus: int = 2, memory_gb: int = 4,
               gpu: bool = False, timeout_seconds: float = 300) -> dict:
    from acprof.host.detect import detect_task
    from acprof.host.model_inspection import probe_model_contract
    from acprof.host.task_support import require_task_support
    from acprof.model_contract import write_model_resolution

    validate_sample(sample)
    if probe not in {"none", "full"} or type(cpus) is not int or cpus <= 0 or type(memory_gb) is not int or memory_gb <= 0:
        raise ValueError("invalid coverage probe/resources")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / "sample.json", sample)
    report = {"schema_version": 1, "scope": "selected_sample_only; no_formal_measurement",
              "sample_sha256": content_digest(sample), "sample_count": len(sample["models"]),
              "probe": probe, "resources": {"cpus": cpus, "memory_gb": memory_gb, "gpu": gpu,
                                             "timeout_seconds": timeout_seconds}, "rows": [], "status": "running"}
    sample_weight = sum(item["weight"] for item in sample["models"])

    def save():
        report["summary"] = summarize(report["rows"], probe_requested=probe == "full", sample_weight=sample_weight)
        atomic_write_json(root / "coverage.json", report)

    save()
    for index, item in enumerate(sample["models"]):
        row = {"model_id": item["model_id"], "revision": item["revision"], "weight": item["weight"],
               "resolved": False, "supported": False, "resolution_status": "error",
               "runtime_status": "not_requested" if probe == "none" else "not_run", "semantic_correct": None}
        output = root / f"model-{index:04d}"
        stage = "resolution"
        try:
            task = detect_task(item["model_id"], revision=item["revision"])
            if task.model_revision != item["revision"]:
                raise ValueError("resolver returned a different snapshot")
            write_model_resolution(task, output)
            row["selected_task"] = task.pipeline_tag
            row["resolved"] = task.model_resolution.get("status") not in {"ambiguous", "needs_configuration"}
            row["resolution_status"] = "resolved" if row["resolved"] else "abstained"
            if row["resolved"] and item.get("semantic_reference"):
                row["semantic_correct"] = task.pipeline_tag == item["semantic_reference"]["task"]
            stage = "support"
            require_task_support(task)
            row["supported"] = True
            if probe == "full":
                from acprof.host.automation import check_repository_access
                from acprof.model_spec import task_model_spec
                stage = "access"
                for repo_id in [task.model_id, *(dep["repo_id"] for dep in task_model_spec(task).get("dependencies", []))]:
                    check_repository_access(repo_id)
                stage = "runtime"
                validation = probe_model_contract(task, output, mode="full", cpus=cpus, memory_gb=memory_gb,
                                                   gpu=gpu, timeout_seconds=timeout_seconds, reuse_existing=True)
                row["runtime_status"] = validation["status"]
        except (Exception, SystemExit) as exc:
            row.update(failed_stage=getattr(exc, "stage", stage), error_type=type(exc).__name__)
            if stage == "access":
                row["resolution_status"] = "access_denied" if "GatedRepoError" in str(exc) or "RepositoryNotFoundError" in str(exc) else "access_error"
            if stage == "runtime":
                path = output / "runtime_validation.json"
                row["runtime_status"] = "error"
                if path.is_file():
                    validation = json.loads(path.read_text())
                    row["runtime_status"] = validation["status"]
                    if any("runtime_validation_timeout" in device.get("error", "") for device in validation.get("devices", {}).values()):
                        row["runtime_status"] = "timeout"
                    failures = [device.get("failed_stage") for device in validation.get("devices", {}).values() if device.get("failed_stage")]
                    row["failed_stage"] = ",".join(failures) or stage
        if row["runtime_status"] == "resource_limited":
            row["failed_stage"] = "resource"
        report["rows"].append(row)
        save()
    report["status"] = "complete"
    save()
    return report

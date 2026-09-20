"""只读比较已有实验条件；不替代严格的续跑身份或模型质量验收。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from acprof.analysis.audit import audit_result
from acprof.result_csv import measurement_key, read_result_csv
from acprof.runtime_settings import RUNTIME_ENV_NAMES


MEASUREMENT_OPTIONS = (
    "profiling_mode", "warmup", "repeat", "repeat_in_window", "repeat_window_seconds",
    "request_timeout_seconds", "sample_hz", "idle_seconds", "idle_cooldown_seconds",
    "compute_profile_tool", "execution_profile_tool", "prune_startup_oom",
)
RUNTIME_ENVIRONMENT = (set(RUNTIME_ENV_NAMES) - {"ACPROF_REQUEST_TIMEOUT_S"}) | {"OMP_NUM_THREADS", "MKL_NUM_THREADS"}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _unknown(value) -> bool:
    if value is None or value == "unknown":
        return True
    if isinstance(value, dict):
        return not value or any(_unknown(item) for item in value.values())
    if isinstance(value, list):
        return not value or any(_unknown(item) for item in value)
    return False


def _condition(left, right) -> dict:
    if isinstance(left, dict) and isinstance(right, dict) and left and right:
        statuses = {_condition(left.get(key), right.get(key))["status"] for key in left.keys() | right.keys()}
        status = ("incompatible" if "incompatible" in statuses
                  else "unknown" if "unknown" in statuses else "compatible")
    elif _unknown(left) or _unknown(right):
        status = "unknown"
    else:
        status = "compatible" if left == right else "incompatible"
    return {"status": status, "left": left, "right": right}


def _actual_workload(rows: list[dict]) -> dict | None:
    """Compare per-request facts, not auto-window throughput-dependent counts."""
    facts: dict[str, set[str]] = {}
    for row in rows:
        row_key = measurement_key(row)
        if row.get("status") not in {"ok", "warn"} or row_key[4] != "0":
            continue
        raw = row.get("workload_contract")
        if not raw or raw.lower() in {"nan", "null", "none"}:
            return None
        summary = json.loads(raw)
        if not isinstance(summary, dict) or summary.get("schema_version") != 1:
            return None
        variants = summary.get("variants")
        if not isinstance(variants, list) or not variants:
            return None
        key = _canonical(row_key[:4])
        for variant in variants:
            contract = variant.get("contract") if isinstance(variant, dict) else None
            if not isinstance(contract, dict) or contract.get("schema_version") != 1:
                return None
            # Backend parameters are reported separately; they are not workload semantics.
            contract = {name: item for name, item in contract.items() if name != "runtime"}
            if _unknown(contract):
                return None
            facts.setdefault(key, set()).add(_canonical(contract))
    return {key: sorted(values) for key, values in sorted(facts.items())} or None


def _snapshot(source: str | Path) -> dict:
    source = Path(source)
    directory = source if source.is_dir() else source.parent
    audit = audit_result(source)
    issues = []

    def read_json(name):
        path = directory / name
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("JSON 顶层应为对象")
            return value
        except (OSError, ValueError) as error:
            issues.append(f"{name}: {error}")
            return {}

    state = read_json("run_state.json")
    metadata = read_json("static_meta.json")
    plan = read_json("input_scale_plan.json")

    def object_field(parent, name):
        value = parent.get(name, {})
        if not isinstance(value, dict):
            issues.append(f"{name}: 应为 JSON 对象")
            return {}
        return value

    options = object_field(state, "options")
    environment = object_field(options, "measurement_environment") if "measurement_environment" in options else None
    runtime_environment = {key: value for key, value in (environment or {}).items() if key in RUNTIME_ENVIRONMENT}
    measurement_environment = (_digest({key: value for key, value in environment.items()
                                       if key not in RUNTIME_ENVIRONMENT})
                               if isinstance(environment, dict) else None)
    entries = plan.get("entries")
    inputs = None
    if plan.get("schema_version") == 2 and isinstance(entries, list) and entries:
        if all(isinstance(entry, dict) and isinstance(entry.get("payload"), dict)
               and entry.get("input_scale") is not None for entry in entries):
            try:
                inputs = _digest([{key: entry[key] for key in ("input_scale", "payload")} for entry in entries])
            except (ValueError, TypeError) as error:
                issues.append(f"input_scale_plan.json: {error}")
    threads = {}
    validation = object_field(metadata, "runtime_validation")
    devices = object_field(validation, "devices")
    # The independent probe injects quota-derived TORCH_NUM_THREADS, while the
    # ordinary server preserves runtime defaults. Its effective count only
    # establishes a shared setting when this run explicitly requested threads.
    thread_names = (("ACPROF_ONNX_INTRA_OP_THREADS",) if metadata.get("runtime_backend") == "onnxruntime" else ())
    thread_names += ("ACPROF_RUNTIME_THREADS", "TORCH_NUM_THREADS")
    thread_request = next(((environment or {})[name] for name in thread_names if name in (environment or {})), None)
    try:
        explicit_threads = not isinstance(thread_request, bool) and int(thread_request) > 0
    except (TypeError, ValueError):
        explicit_threads = False
    for device in devices:
        result = object_field(devices, device)
        parameters = object_field(result, "runtime_parameters")
        effective = object_field(parameters, "effective")
        observed = effective.get("threads")
        threads[device] = (observed if explicit_threads and result.get("status") == "ok"
                           and type(observed) is int and observed > 0 else None)
    try:
        _, rows = read_result_csv(directory / "result_all.csv" if source.is_dir() else source)
        actual = _actual_workload(rows)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        actual = None
        issues.append(f"actual workload: {error}")
    # Only the input payload and semantic conditions are compared. Model/export,
    # packages and backend hashes continue to belong to the existing run identity.
    host = object_field(state, "host")
    return {
        "run_id": state.get("run_id"), "result_csv": audit["result_csv"],
        "valid": audit["valid"] and not issues, "issues": issues,
        "conditions": {
            "task_semantics": {key: plan.get(key) for key in ("task_family", "pipeline_tag", "scenario")},
            "planned_inputs": inputs,
            "resources": {**{key: options.get(key) for key in ("cpus", "mems", "gpus", "batch_size")},
                          **{key: metadata.get(key) for key in ("cgroup_version", "cgroup_collection_mode")}},
            "runtime_threads": threads or None,
            "measurement_protocol": {**{key: options.get(key) for key in MEASUREMENT_OPTIONS},
                                     "measurement_environment": measurement_environment,
                                     "static_schema_version": metadata.get("schema_version")},
            "quality_constraints": plan.get("quality_constraints"),
            "actual_workload": actual,
        },
        "identity": {**{key: metadata.get(key) for key in (
            "model_name", "model_revision", "runtime_backend", "image_id", "runtime_environment")},
            "runtime_requested_environment": runtime_environment,
            **{key: host.get(key) for key in ("source_sha256", "packages_sha256")}},
    }


def compare_results(left: str | Path, right: str | Path) -> dict:
    """Return compatible/incompatible/unknown for recorded comparison conditions.

    Quality constraints describe a shared target, not proof that either model
    meets it. Missing legacy evidence is unknown and never reconstructed.
    """
    snapshots = {"left": _snapshot(left), "right": _snapshot(right)}
    lhs, rhs = snapshots.values()
    conditions = {name: _condition(value, rhs["conditions"][name])
                  for name, value in lhs["conditions"].items()}
    statuses = {item["status"] for item in conditions.values()}
    valid = lhs["valid"] and rhs["valid"]
    status = ("incompatible" if not valid or "incompatible" in statuses
              else "unknown" if "unknown" in statuses else "compatible")
    differences = {name: {"left": value, "right": rhs["identity"][name]}
                   for name, value in lhs["identity"].items() if value != rhs["identity"][name]}
    return {"schema_version": 1, "status": status, "valid": valid,
            "scope": "recorded_comparison_conditions_not_model_quality_or_resume_identity",
            "limitations": ["hardware equivalence and CPU affinity are not verified",
                            "thread counts require an explicit positive request and independent runtime probe evidence; defaults remain unknown",
                            "matching quality constraints do not prove either model meets them"],
            "conditions": conditions, "expected_differences": differences,
            "experiments": {side: {key: snapshot[key] for key in ("run_id", "result_csv", "valid", "issues")}
                            for side, snapshot in snapshots.items()}}

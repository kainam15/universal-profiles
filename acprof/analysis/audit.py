"""只读结果审计；保持历史字段和不可用原因，不回填测量值。"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from acprof.metric_registry import METRICS, NUMERIC_FIELDS
from acprof.result_csv import expected_measurements, measurement_key, read_result_csv


MISSING = {"", "nan", "none", "null", "n/a"}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def missing_reason(field: str, row: dict, metadata: dict) -> str:
    if field not in row:
        return "not_recorded"
    metric = METRICS[field]
    if metric.applicability in {"gpu", "gpu_idle_debug"} and row.get("gpu_mode") == "off":
        return "not_applicable"
    if metric.applicability == "cpu" and row.get("gpu_mode") == "on":
        return "not_applicable"
    if str(row.get("status", "")).strip().lower() == "error":
        return "measurement_failed"
    if metric.tool:
        suffix = "torch_profiler_eager" if metric.tool == "torch" else metric.tool
        error = str(row.get(f"compute_profile_error_{suffix}", "")).strip()
        if error.lower() not in MISSING:
            return "profiler_reported_error"
        key = "compute_profile_tools" if metric.tool in {"torch", "ncu"} else "execution_profile_tools"
        enabled = metadata.get(key)
        if isinstance(enabled, str):
            try:
                enabled = json.loads(enabled)
            except ValueError:
                enabled = enabled.split(",")
        if isinstance(enabled, list) and not any(metric.tool in str(tool) for tool in enabled):
            return "tool_not_enabled"
    return "unavailable_unspecified"


def _formula_checks(row: dict) -> dict[str, float]:
    """仅检查可由同一行确定的关系，不重建 RAPL 分段归因或补猜缺失来源。"""
    values = {name: number(value) for name, value in row.items()}
    expected = {}
    cpu = values.get("vcpu_energy_eff_j")
    gpu = values.get("gpu_energy_eff_j") if row.get("gpu_mode") == "on" else 0.0
    if cpu is not None and cpu >= 0 and gpu is not None and gpu >= 0:
        expected["container_attributed_energy_eff_j"] = cpu + gpu
    request = values.get("packet_request_wire_bytes_per_request")
    response = values.get("packet_response_wire_bytes_per_request")
    if request is not None and response is not None:
        expected["packet_total_wire_bytes_per_request"] = request + response
    total = values.get("packet_total_wire_bytes_per_request")
    payload = values.get("packet_tcp_payload_bytes_per_request")
    if total is not None and payload is not None:
        expected["packet_protocol_overhead_bytes_per_request"] = total - payload
        if total > 0:
            expected["packet_protocol_overhead_ratio"] = (total - payload) / total
    return expected


def audit_result(source: str | Path) -> dict:
    path = Path(source)
    if path.is_dir():
        path /= "result_all.csv"
    report = {"schema_version": 1, "result_csv": str(path.resolve()),
              "valid": True, "issues": [], "missing_metrics": {},
              "counts": {"rows": 0, "formal_ok": 0, "warmup": 0, "warn": 0, "error": 0},
              "completion": "unknown", "coverage": None}

    def issue(code, message, *, severity="error", **detail):
        report["issues"].append({"code": code, "message": str(message), "severity": severity, **detail})
        if severity == "error":
            report["valid"] = False

    def read_json(name):
        artifact = path.parent / name
        if not artifact.exists():
            return {}
        try:
            payload = json.loads(artifact.read_text())
            if not isinstance(payload, dict):
                raise ValueError("JSON 顶层应为对象")
            return payload
        except (ValueError, OSError) as error:
            issue("invalid_metadata", f"{name}: {error}")
            return {}

    metadata = read_json("static_meta.json")
    state = read_json("run_state.json")
    if state:
        report["run_status"] = state.get("status")
        report["run_id"] = state.get("run_id")
        report["completion"] = "complete" if state.get("status") == "complete" else "incomplete"
    try:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        fields, rows = read_result_csv(path)
        report["result_sha256"] = before
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            issue("changing_snapshot", "读取期间 CSV 发生变化，请采集结束后重新审计")
    except (ValueError, OSError, csv.Error) as error:
        issue("invalid_csv", error)
        return report
    report["counts"]["rows"] = len(rows)
    report["unknown_columns"] = [name for name in fields if name not in METRICS]
    report["missing_columns"] = [name for name in METRICS if name not in fields]
    expected_hash = metadata.get("input_scale_plan_sha256")
    if expected_hash and str(expected_hash).lower() not in MISSING:
        plan = path.parent / "input_scale_plan.json"
        if not plan.is_file() or hashlib.sha256(plan.read_bytes()).hexdigest() != expected_hash:
            issue("input_plan_hash", "输入计划与 static_meta.json 中的 SHA256 不一致")
    if state.get("runtime"):
        try:
            options = state["options"]
            expected = expected_measurements(
                [int(value) for value in options["cpus"].split(",")],
                [int(value) for value in options["mems"].split(",")], options["gpus"].split(","),
                state["runtime"]["planned"]["scales"], options["warmup"], options["repeat"],
            )
            actual = {measurement_key(row) for row in rows}
            report["coverage"] = {"expected": len(expected), "actual": len(actual),
                                  "missing": len(expected - actual), "unexpected": len(actual - expected)}
            if expected != actual:
                issue("plan_coverage", "结果行与原实验计划不完全一致")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            issue("invalid_run_state", error)
    missing = {}
    for index, row in enumerate(rows, 2):
        status = row.get("status", "").strip().lower()
        warmup = number(row["warmup"]) == 1
        report["counts"]["warmup"] += warmup
        if status not in {"ok", "warn", "error"}:
            issue("invalid_status", "status 应为 ok、warn 或 error", row=index)
        if status in {"warn", "error"}:
            report["counts"][status] += 1
        formal = status == "ok" and not warmup
        report["counts"]["formal_ok"] += formal
        for field in NUMERIC_FIELDS:
            raw = str(row.get(field, "nan")).strip().lower()
            if raw not in MISSING and number(raw) is None:
                issue("invalid_number", f"{field} 不是有限数值或 nan", row=index, field=field)
            elif formal and raw in MISSING:
                missing.setdefault(field, Counter())[missing_reason(field, row, metadata)] += 1
        if formal:
            for field, expected_value in _formula_checks(row).items():
                actual_value = number(row.get(field))
                # CSV 保存到 6 位小数，容忍加减的累计舍入，不把缺失分母解释为错误。
                if actual_value is not None and not math.isclose(actual_value, expected_value, rel_tol=2e-5, abs_tol=2e-6):
                    issue("formula_mismatch", f"{field}: 实际 {actual_value:g}，应为 {expected_value:g}", row=index, field=field)
    report["missing_metrics"] = {name: dict(reasons) for name, reasons in missing.items()}
    return report

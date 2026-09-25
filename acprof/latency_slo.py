"""延迟 SLO 的纯配置解析；运行前冻结，采集与合并只读取 metadata。"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _positive_threshold(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("latency SLO threshold must be a finite number > 0 seconds") from None
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError("latency SLO threshold must be a finite number > 0 seconds")
    return number


def parse_latency_slo_rules(
    rules: Sequence[str], *, environment_threshold: str | None = None,
) -> dict[str, float]:
    """解析 default/task/profile 规则；不为任何任务猜测默认 SLO。"""
    thresholds = {}
    for rule in rules:
        selector, separator, value = rule.partition("=")
        selector = selector.strip()
        kind, colon, name = selector.partition(":")
        if not separator or not (selector == "default" or (
            kind in {"task", "profile"} and colon and name.strip() and name == name.strip()
        )):
            raise ValueError("--latency-slo requires default=SECONDS, task:TAG=SECONDS or profile:ID=SECONDS")
        if selector in thresholds:
            raise ValueError(f"duplicate --latency-slo selector: {selector}")
        thresholds[selector] = _positive_threshold(value)
    if environment_threshold is not None and "default" not in thresholds:
        thresholds["environment"] = _positive_threshold(environment_threshold)
    return thresholds


def resolve_latency_slo(
    thresholds: Mapping[str, float], *, pipeline_tag: str, runtime_profile_id: str,
) -> dict[str, Any]:
    for selector in (f"task:{pipeline_tag}", f"profile:{runtime_profile_id}", "default", "environment"):
        if selector in thresholds:
            return {
                "threshold_s": thresholds[selector],
                "source": "env:SLOW_LATENCY_THRESHOLD_S" if selector == "environment" else selector,
                "pipeline_tag": pipeline_tag,
                "runtime_profile_id": runtime_profile_id,
            }
    return {"threshold_s": None, "source": "unconfigured",
            "pipeline_tag": pipeline_tag, "runtime_profile_id": runtime_profile_id}


def latency_slo_threshold(metadata: Mapping[str, Any]) -> float | None:
    """旧 v7 缺字段表示未知；已记录的非法阈值直接拒绝，绝不回退到环境。"""
    if "latency_slo" not in metadata:
        return None
    slo = metadata["latency_slo"]
    if not isinstance(slo, dict) or "threshold_s" not in slo:
        raise ValueError("static metadata latency_slo requires a threshold_s field")
    return None if slo["threshold_s"] is None else _positive_threshold(slo["threshold_s"])

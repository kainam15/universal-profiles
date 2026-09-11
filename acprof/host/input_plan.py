"""输入规模规划、探测与可复现负载计划。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from acprof.config import SCALING_DIMENSIONS
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import (
    ImageInfo,
    RunningContainer,
    _sanitize_model_id,
    _normalize_gpu_mode,
    _start_container_session,
    _stop_container_session,
)


@dataclass
class PlannedInputScales:
    scales: List[float]
    source: str
    plan_file: Optional[str] = None
    workload: Dict[str, Any] = field(default_factory=dict)
    plan_sha256: str = ""


AUTO_INPUT_SCALE_COUNT = 6


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _format_scale_value(scale: float) -> str:
    value = float(scale)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def serialize_input_scales(scales: List[float]) -> str:
    return ",".join(_format_scale_value(scale) for scale in scales)


def resolve_input_scales(task_family: str, input_scales: Optional[str] = None) -> List[float]:
    if input_scales:
        return sorted(set(_parse_float_list(input_scales)))

    scaling_cfg = SCALING_DIMENSIONS.get(task_family)
    if scaling_cfg:
        return sorted(set(float(v) for v in scaling_cfg.values))
    return [1.0]


def _scale_plan_file_path(output_dir: str) -> str:
    return os.path.join(output_dir, "input_scale_plan.json")


def _clear_scale_plan_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _write_scale_plan_file(
    path: str,
    task_info: TaskInfo,
    entries: List[Dict[str, Any]],
    *,
    workload: Optional[Dict[str, Any]] = None,
    model_constraints: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "schema_version": 2,
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "workload": dict(workload or {}),
        "model_constraints": dict(model_constraints or {}),
        "entries": entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _materialize_scale_plan(
    *,
    task_info: TaskInfo,
    scales: List[float],
    batch_size: int,
    output_dir: str,
    source: str,
    workload_spec_path: Optional[str] = None,
    model_constraints: Optional[Dict[str, Any]] = None,
) -> PlannedInputScales:
    """Generate one reusable payload plan for non-NLP task families."""
    from acprof.workloads import get_generator

    workload_gen = get_generator(
        task_info.task_family,
        task_info.model_id,
        task_info.pipeline_tag,
        batch_size,
        workload_spec_path=workload_spec_path,
    )
    entries: List[Dict[str, Any]] = []
    effective_scales: List[float] = []

    for scale in scales:
        requested_scale = float(scale)
        payload = workload_gen.generate(requested_scale)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"workload generator returned a non-object payload for "
                f"task_family={task_info.task_family}, "
                f"input_scale={_format_scale_value(requested_scale)}"
            )

        effective_scale = workload_gen.effective_input_scale(
            requested_scale,
            payload,
        )
        if effective_scale is None:
            raise RuntimeError(
                f"cannot determine effective input scale for "
                f"task_family={task_info.task_family}, "
                f"input_scale={_format_scale_value(requested_scale)}"
            )

        actual_scale = float(effective_scale)
        if effective_scales and actual_scale <= effective_scales[-1]:
            raise RuntimeError(
                f"input scale plan is not strictly increasing for "
                f"task_family={task_info.task_family}: "
                f"{_format_scale_value(effective_scales[-1])}, "
                f"{_format_scale_value(actual_scale)}"
            )

        effective_scales.append(actual_scale)
        input_metadata: Dict[str, Any] = {}
        metadata_fn = getattr(workload_gen, "input_metadata", None)
        if callable(metadata_fn):
            candidate_metadata = metadata_fn(actual_scale, payload)
            if candidate_metadata is not None:
                if not isinstance(candidate_metadata, dict):
                    raise RuntimeError(
                        "workload generator returned non-object input metadata "
                        f"for input_scale={_format_scale_value(actual_scale)}"
                    )
                input_metadata = candidate_metadata

        entries.append({
            "input_scale": actual_scale,
            "scale_label": workload_gen.scale_label(actual_scale),
            "input_metadata": input_metadata,
            "payload": payload,
        })

    workload_metadata: Dict[str, Any] = {}
    metadata_fn = getattr(workload_gen, "plan_metadata", None)
    if callable(metadata_fn):
        candidate_metadata = metadata_fn()
        if candidate_metadata is not None:
            if not isinstance(candidate_metadata, dict):
                raise RuntimeError("workload generator returned non-object plan metadata")
            workload_metadata = candidate_metadata

    plan_file = _scale_plan_file_path(output_dir)
    constraints = dict(model_constraints or {})
    plan_sha256 = _write_scale_plan_file(
        plan_file,
        task_info,
        entries,
        workload=workload_metadata,
        model_constraints=constraints,
    )
    static_workload = dict(workload_metadata)
    if constraints:
        static_workload["model_constraints"] = constraints
    return PlannedInputScales(
        scales=effective_scales,
        source=source,
        plan_file=plan_file,
        workload=static_workload,
        plan_sha256=plan_sha256,
    )


def _integer_auto_scales(max_value: int, count: int = AUTO_INPUT_SCALE_COUNT) -> List[float]:
    if max_value < count:
        raise RuntimeError(
            f"cannot generate {count} unique integer input scales from max_value={max_value}"
        )

    scales = [max(1, math.floor(max_value * idx / count)) for idx in range(1, count + 1)]
    scales[-1] = int(max_value)
    if len(set(scales)) != count:
        raise RuntimeError(
            f"failed to derive {count} unique integer input scales from max_value={max_value}: {scales}"
        )
    return [float(v) for v in scales]


def _float_auto_scales(max_value: float, count: int = AUTO_INPUT_SCALE_COUNT) -> List[float]:
    if max_value <= 0:
        raise RuntimeError(f"invalid max_value for float scales: {max_value}")
    scales = [round(float(max_value) * idx / count, 6) for idx in range(1, count + 1)]
    scales[-1] = round(float(max_value), 6)
    return scales


def _start_probe_session(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
) -> RunningContainer:
    probe_cpu = max(cpu_list)
    probe_mem = max(mem_list)
    normalized_gpu = [_normalize_gpu_mode(gpu) for gpu in gpu_list]
    probe_gpu = "on" if "on" in normalized_gpu else "off"
    model_tag = _sanitize_model_id(task_info.model_id)
    container_name = f"probe_{model_tag}_{probe_cpu}c_{probe_mem}g_{probe_gpu}"

    print(
        f"[scale] Starting probe container with CPU={probe_cpu}, "
        f"MEM={probe_mem}GB, GPU={probe_gpu}"
    )
    return _start_container_session(
        task_info=task_info,
        cpu=probe_cpu,
        mem=probe_mem,
        gpu=probe_gpu,
        image_info=image_info,
        container_name=container_name,
        log_prefix="[probe]",
    )


def _parse_probe_response(data: Dict[str, Any], desc: str) -> Dict[str, Any]:
    required = {"effective_input_scale", "truncated_by_limit", "reason"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise RuntimeError(f"/probe response missing fields for {desc}: {missing}")

    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError(f"/probe returned empty reason for {desc}")

    if data.get("effective_input_scale") is None or data.get("truncated_by_limit") is None:
        raise RuntimeError(f"cannot reliably determine effective input scale for {desc}: {reason}")

    try:
        effective_scale = float(data["effective_input_scale"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"/probe returned invalid effective_input_scale for {desc}: "
            f"{data.get('effective_input_scale')!r}"
        ) from exc

    truncated_by_limit = data.get("truncated_by_limit")
    if not isinstance(truncated_by_limit, bool):
        raise RuntimeError(
            f"/probe returned non-boolean truncated_by_limit for {desc}: {truncated_by_limit!r}"
        )

    return {
        "effective_input_scale": effective_scale,
        "truncated_by_limit": truncated_by_limit,
        "reason": reason,
    }


def _post_probe_payload(
    session: RunningContainer,
    payload: Dict[str, Any],
    desc: str,
) -> Dict[str, Any]:
    import requests

    response = requests.post(
        session.base_url + "/probe",
        json=payload,
        timeout=300,
        headers={"Connection": "close"},
    )
    if response.status_code >= 400:
        stale_image_hint = ""
        if response.status_code == 404:
            stale_image_hint = " /probe endpoint is missing; rebuild the image without --skip-build."
        raise RuntimeError(
            f"/probe HTTP {response.status_code} for {desc}: {response.text[:300]}{stale_image_hint}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"/probe returned non-JSON response for {desc}") from exc

    parsed = _parse_probe_response(data, desc)
    parsed["payload"] = payload
    return parsed


def _request_scale_meta(
    session: RunningContainer,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    import requests

    response = requests.post(
        session.base_url + "/scale_meta",
        json=payload,
        timeout=300,
        headers={"Connection": "close"},
    )
    if response.status_code >= 400:
        stale_image_hint = ""
        if response.status_code == 404:
            stale_image_hint = " /scale_meta endpoint is missing; rebuild the image without --skip-build."
        raise RuntimeError(
            f"/scale_meta HTTP {response.status_code}: {response.text[:300]}{stale_image_hint}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError("/scale_meta returned non-JSON response") from exc

    if not isinstance(data, dict):
        raise RuntimeError("/scale_meta returned a non-object response")
    return data


def _request_nlp_scale_meta(
    session: RunningContainer,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    data = _request_scale_meta(session, payload)
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError("/scale_meta returned empty reason")

    raw_max = data.get("max_effective_input_scale")
    if raw_max is None:
        raise RuntimeError(f"cannot reliably determine NLP max input scale: {reason}")

    try:
        max_effective_input_scale = int(float(raw_max))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"/scale_meta returned invalid max_effective_input_scale: {raw_max!r}"
        ) from exc

    if max_effective_input_scale <= 0:
        raise RuntimeError(
            f"/scale_meta returned non-positive max_effective_input_scale={max_effective_input_scale}: {reason}"
        )

    return {
        "input_scale_type": str(data.get("input_scale_type", "")).strip() or "input_scale",
        "max_effective_input_scale": max_effective_input_scale,
        "reason": reason,
    }


def _request_audio_scale_meta(
    session: RunningContainer,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    data = _request_scale_meta(session, payload)
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError(
            "cannot determine audio input constraints from /scale_meta; "
            "rebuild the audio image without --skip-build"
        )

    result: Dict[str, Any] = {
        "input_scale_type": str(data.get("input_scale_type") or "duration_s"),
        "reason": reason,
    }
    numeric_fields = (
        "required_sampling_rate",
        "source_sampling_rate",
        "max_short_form_duration_s",
        "model_input_num_samples",
        "model_input_frames",
        "fixed_frontend_num_samples",
        "fixed_frontend_num_frames",
        "frontend_feature_bins",
        "encoder_positions",
        "decoder_output_token_limit",
    )
    for field_name in numeric_fields:
        raw_value = data.get(field_name)
        if raw_value is None:
            result[field_name] = None
            continue
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"/scale_meta returned invalid {field_name}: {raw_value!r}"
            ) from exc
        if not math.isfinite(number) or number <= 0:
            raise RuntimeError(
                f"/scale_meta returned non-positive/non-finite "
                f"{field_name}: {raw_value!r}"
            )
        result[field_name] = int(number) if number.is_integer() else number

    result["model_type"] = str(data.get("model_type") or "unknown")
    raw_fixed_padding = data.get("short_form_fixed_padding")
    if not isinstance(raw_fixed_padding, bool):
        raise RuntimeError(
            "/scale_meta returned invalid short_form_fixed_padding: "
            f"{raw_fixed_padding!r}"
        )
    result["short_form_fixed_padding"] = raw_fixed_padding
    resampling_policy = data.get("resampling_policy")
    if resampling_policy is not None:
        if resampling_policy != "scipy.signal.resample_poly_if_required_in_preprocess":
            raise RuntimeError(f"/scale_meta returned unsupported resampling_policy: {resampling_policy!r}")
        result["resampling_policy"] = resampling_policy
    return result


def _assert_manual_timeseries_scales_legal(
    task_info: TaskInfo,
    scales: List[float],
    batch_size: int,
) -> List[float]:
    from acprof.workloads import get_generator

    workload_gen = get_generator(task_info.task_family, task_info.model_id, task_info.pipeline_tag, batch_size)
    invalid: Dict[float, str] = {}

    for scale in scales:
        payload = workload_gen.generate(scale)
        effective = workload_gen.effective_input_scale(scale, payload)
        if effective is None:
            raise RuntimeError(
                f"cannot determine effective context length for requested scale {_format_scale_value(scale)}"
            )
        if float(effective) + 1e-9 < float(scale):
            invalid[scale] = f"context_length_clamped_to_{_format_scale_value(float(effective))}"

    if invalid:
        details = ", ".join(
            f"{_format_scale_value(scale)} ({reason})" for scale, reason in invalid.items()
        )
        raise RuntimeError(f"manual input scales exceed the usable timeseries context length: {details}")

    print(f"[scale] Using manual input scales: {serialize_input_scales(scales)}")
    return scales


def _assert_manual_nlp_scales_legal(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    scales: List[float],
    batch_size: int,
) -> List[float]:
    from acprof.workloads import get_generator

    session: Optional[RunningContainer] = None
    try:
        session = _start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list)
        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
        )
        invalid: Dict[float, str] = {}
        for scale in scales:
            payload = workload_gen.generate(scale)
            result = _post_probe_payload(
                session,
                payload,
                f"manual scale {_format_scale_value(scale)}",
            )
            if result["truncated_by_limit"]:
                invalid[scale] = (
                    f"{result['reason']}; effective_input_scale="
                    f"{_format_scale_value(result['effective_input_scale'])}"
                )

        if invalid:
            details = ", ".join(
                f"{_format_scale_value(scale)} ({reason})" for scale, reason in invalid.items()
            )
            raise RuntimeError(f"manual NLP input scales exceed the usable tokenizer limit: {details}")
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")

    print(f"[scale] Using manual input scales: {serialize_input_scales(scales)}")
    return scales


def _plan_manual_nlp_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    scales: List[float],
    batch_size: int,
    output_dir: str,
    workload_spec_path: Optional[str] = None,
) -> PlannedInputScales:
    from acprof.workloads import get_generator

    session: Optional[RunningContainer] = None
    try:
        session = _start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list)
        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
            workload_spec_path=workload_spec_path,
        )
        invalid: Dict[float, str] = {}
        entries: List[Dict[str, Any]] = []
        actual_scales: List[float] = []

        for scale in scales:
            payload = workload_gen.generate(scale)
            result = _post_probe_payload(
                session,
                payload,
                f"manual scale {_format_scale_value(scale)}",
            )
            if result["truncated_by_limit"]:
                invalid[scale] = (
                    f"{result['reason']}; effective_input_scale="
                    f"{_format_scale_value(result['effective_input_scale'])}"
                )
                continue

            actual_scale = float(result["effective_input_scale"])
            actual_scales.append(actual_scale)
            entries.append({
                "input_scale": actual_scale,
                "scale_label": workload_gen.scale_label(actual_scale),
                "input_metadata": workload_gen.input_metadata(actual_scale, result["payload"]),
                "payload": result["payload"],
            })

        if invalid:
            details = ", ".join(
                f"{_format_scale_value(scale)} ({reason})" for scale, reason in invalid.items()
            )
            raise RuntimeError(f"manual NLP input scales exceed the usable tokenizer limit: {details}")
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")

    plan_file = _scale_plan_file_path(output_dir)
    workload_metadata = workload_gen.plan_metadata()
    plan_sha256 = _write_scale_plan_file(plan_file, task_info, entries, workload=workload_metadata)

    print(
        "[scale] Using manual input scales: "
        f"{serialize_input_scales(scales)}; effective scales: "
        f"{serialize_input_scales(actual_scales)}"
    )
    return PlannedInputScales(
        scales=actual_scales,
        source="manual",
        plan_file=plan_file,
        plan_sha256=plan_sha256,
        workload=workload_metadata,
    )


def _plan_nlp_auto_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    batch_size: int,
    output_dir: str,
    workload_spec_path: Optional[str] = None,
) -> PlannedInputScales:
    from acprof.workloads import get_generator

    session: Optional[RunningContainer] = None
    try:
        session = _start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list)
        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
            workload_spec_path=workload_spec_path,
        )
        metadata_payload = workload_gen.generate(1.0)
        scale_meta = _request_nlp_scale_meta(session, metadata_payload)
        max_effective = int(scale_meta["max_effective_input_scale"])
        target_scales = _integer_auto_scales(max_effective, count=AUTO_INPUT_SCALE_COUNT)

        probe_cache: Dict[int, Dict[str, Any]] = {}

        def probe_word_count(word_count: int) -> Dict[str, Any]:
            word_count = max(1, int(word_count))
            cached = probe_cache.get(word_count)
            if cached is not None:
                return cached

            if hasattr(workload_gen, "generate_for_word_count"):
                payload = workload_gen.generate_for_word_count(word_count)
            else:
                payload = workload_gen.generate(float(word_count))
            result = _post_probe_payload(
                session,
                payload,
                f"word_count={word_count}",
            )
            probe_cache[word_count] = result
            return result

        def find_best_candidate(target_scale: int) -> Dict[str, Any]:
            left = 1
            right = max(1, int(target_scale))
            best_below_wc: Optional[int] = None
            best_below_result: Optional[Dict[str, Any]] = None

            while left <= right:
                mid = (left + right) // 2
                result = probe_word_count(mid)
                condition = (not result["truncated_by_limit"]) and (
                    result["effective_input_scale"] <= float(target_scale)
                )
                if condition:
                    best_below_wc = mid
                    best_below_result = result
                    left = mid + 1
                else:
                    right = mid - 1

            candidates: List[Dict[str, Any]] = []
            if best_below_result is not None:
                candidates.append(best_below_result)

            if best_below_wc is None:
                upper_wc = 1
            else:
                upper_wc = min(max(1, int(target_scale)), best_below_wc + 1)

            if upper_wc >= 1:
                upper_result = probe_word_count(upper_wc)
                if not upper_result["truncated_by_limit"]:
                    candidates.append(upper_result)

            if not candidates:
                raise RuntimeError(
                    f"failed to build a non-truncated NLP payload near target scale {target_scale}"
                )

            best = min(
                candidates,
                key=lambda item: (
                    abs(item["effective_input_scale"] - float(target_scale)),
                    -item["effective_input_scale"],
                ),
            )
            return best

        entries: List[Dict[str, Any]] = []
        actual_scales: List[float] = []
        previous_scale = float("-inf")
        for target_scale in target_scales:
            candidate = find_best_candidate(int(target_scale))
            actual_scale = float(candidate["effective_input_scale"])
            if actual_scale <= previous_scale:
                raise RuntimeError(
                    "failed to derive 6 strictly increasing NLP input scales: "
                    f"target={_format_scale_value(target_scale)} produced "
                    f"{_format_scale_value(actual_scale)} after "
                    f"{_format_scale_value(previous_scale)}"
                )
            previous_scale = actual_scale
            actual_scales.append(actual_scale)
            entries.append({
                "input_scale": actual_scale,
                "scale_label": workload_gen.scale_label(actual_scale),
                "input_metadata": workload_gen.input_metadata(actual_scale, candidate["payload"]),
                "payload": candidate["payload"],
            })

        plan_file = _scale_plan_file_path(output_dir)
        workload_metadata = workload_gen.plan_metadata()
        plan_sha256 = _write_scale_plan_file(plan_file, task_info, entries,
                                           workload=workload_metadata, model_constraints=scale_meta)

        print(
            "[scale] Auto-planned NLP scales from tokenizer limit "
            f"{max_effective}: {serialize_input_scales(actual_scales)}"
        )
        print(f"[scale] Scale metadata: {scale_meta['reason']}")
        return PlannedInputScales(
            scales=actual_scales,
            source="auto",
            plan_file=plan_file,
            plan_sha256=plan_sha256,
            workload=workload_metadata,
        )
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")


def _default_family_max_scale(task_info: TaskInfo, batch_size: int) -> float:
    if task_info.task_family == "timeseries":
        from acprof.workloads import get_generator

        workload_gen = get_generator(task_info.task_family, task_info.model_id, task_info.pipeline_tag, batch_size)
        if hasattr(workload_gen, "max_input_scale"):
            max_scale = workload_gen.max_input_scale()
            if max_scale is not None:
                return float(max_scale)

    scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
    if scaling_cfg and scaling_cfg.values:
        return float(max(scaling_cfg.values))

    raise RuntimeError(f"cannot determine default max input scale for task_family={task_info.task_family}")


def _plan_audio_scales(
    *,
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    scales: List[float],
    batch_size: int,
    output_dir: str,
    source: str,
    workload_spec_path: Optional[str],
) -> PlannedInputScales:
    """Validate real audio payloads against the loaded model before a sweep."""
    from acprof.workloads import get_generator

    workload_gen = get_generator(
        task_info.task_family,
        task_info.model_id,
        task_info.pipeline_tag,
        batch_size,
        workload_spec_path=workload_spec_path,
    )
    normalized_scales = sorted(set(float(scale) for scale in scales))
    if not normalized_scales or normalized_scales[0] <= 0:
        raise RuntimeError("audio input scales must be finite positive durations")
    if any(not math.isfinite(scale) for scale in normalized_scales):
        raise RuntimeError("audio input scales must be finite positive durations")

    # Validate manifest/asset limits before paying the cost of loading a large
    # model in the probe container (for example, reject Whisper 31s directly).
    materialized_payloads = [
        (scale, workload_gen.generate(scale)) for scale in normalized_scales
    ]

    session: Optional[RunningContainer] = None
    model_constraints: Dict[str, Any] = {}
    try:
        session = _start_probe_session(
            task_info,
            image_info,
            cpu_list,
            mem_list,
            gpu_list,
        )
        first_payload = materialized_payloads[0][1]
        model_constraints = _request_audio_scale_meta(session, first_payload)

        expected_rate = model_constraints.get("required_sampling_rate")
        payload_rate = first_payload.get("sample_rate")
        if expected_rate is not None and payload_rate is not None:
            permits_resampling = (
                task_info.pipeline_tag not in {"automatic-speech-recognition", "asr", "speech-recognition"}
                and model_constraints.get("resampling_policy")
                == "scipy.signal.resample_poly_if_required_in_preprocess"
                and model_constraints.get("source_sampling_rate") == int(payload_rate)
            )
            if int(expected_rate) != int(payload_rate) and not permits_resampling:
                raise RuntimeError(
                    "audio workload sampling rate does not match the model: "
                    f"payload={payload_rate}Hz, model={expected_rate}Hz"
                )

        max_short = model_constraints.get("max_short_form_duration_s")
        if max_short is not None:
            invalid = [scale for scale in normalized_scales if scale > float(max_short) + 1e-9]
            if invalid:
                raise RuntimeError(
                    "audio input scales exceed the model short-form limit "
                    f"of {_format_scale_value(float(max_short))} seconds: "
                    f"{serialize_input_scales(invalid)}. Use a separate long-form workload."
                )

        for scale, payload in materialized_payloads:
            result = _post_probe_payload(
                session,
                payload,
                f"audio scale {_format_scale_value(scale)}s",
            )
            if result["truncated_by_limit"]:
                raise RuntimeError(
                    "audio input scale is not valid for short-form inference: "
                    f"{_format_scale_value(scale)}s ({result['reason']})"
                )
            if not math.isclose(
                float(result["effective_input_scale"]),
                float(scale),
                rel_tol=0.0,
                abs_tol=(0.5 / max(1, int(payload.get("sample_rate", 16000)))),
            ):
                raise RuntimeError(
                    "audio effective duration differs from the requested scale: "
                    f"requested={scale}, effective={result['effective_input_scale']}"
                )
    finally:
        if session is not None:
            _stop_container_session(session.name, log_prefix="[probe]")

    print(
        f"[scale] Using {source} audio scales: "
        f"{serialize_input_scales(normalized_scales)}"
    )
    return _materialize_scale_plan(
        task_info=task_info,
        scales=normalized_scales,
        batch_size=batch_size,
        output_dir=output_dir,
        source=source,
        workload_spec_path=workload_spec_path,
        model_constraints=model_constraints,
    )


def _plan_timeseries_scales(
    task_info: TaskInfo, image_info: ImageInfo,
    cpu_list: List[int], mem_list: List[int], gpu_list: List[str],
    batch_size: int, output_dir: str, input_scales: Optional[str],
) -> PlannedInputScales:
    """Use the loaded forecasting model's context limit before materialization."""
    from acprof.workloads import get_generator

    generator = get_generator(task_info.task_family, task_info.model_id, task_info.pipeline_tag, batch_size)
    session = _start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list)
    try:
        constraints = _request_scale_meta(session, generator.generate(1))
    finally:
        _stop_container_session(session.name, log_prefix="[probe]")
    limit = constraints.get("max_effective_input_scale")
    if (isinstance(limit, bool) or not isinstance(limit, (float, int))
            or not math.isfinite(limit) or limit < 1 or int(limit) != limit):
        raise RuntimeError("cannot determine forecasting model context limit from /scale_meta; rebuild the timeseries image")
    maximum = min(int(limit), int(generator.max_input_scale()))
    if input_scales:
        scales = resolve_input_scales(task_info.task_family, input_scales)
        if any(not math.isfinite(scale) or scale < 1 or int(scale) != scale or scale > maximum for scale in scales):
            raise RuntimeError(f"manual input scales must be positive integer context lengths <= {maximum}")
        source = "manual"
    else:
        scales = _integer_auto_scales(maximum, count=min(AUTO_INPUT_SCALE_COUNT, maximum))
        source = "auto"
    return _materialize_scale_plan(
        task_info=task_info, scales=scales, batch_size=batch_size, output_dir=output_dir,
        source=source, model_constraints=constraints,
    )


def plan_input_scales(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    batch_size: int,
    output_dir: str,
    input_scales: Optional[str] = None,
    workload_spec_path: Optional[str] = None,
) -> PlannedInputScales:
    if workload_spec_path and task_info.task_family not in {"cv", "audio", "multimodal", "diffusion", "structured"}:
        raise ValueError(
            "--workload-spec is implemented for cv, audio, multimodal, diffusion and structured tasks"
        )
    plan_file = _scale_plan_file_path(output_dir)
    _clear_scale_plan_file(plan_file)
    text_audio = task_info.pipeline_tag in {"text-to-speech", "text-to-audio"}
    table_qa = task_info.pipeline_tag == "table-question-answering"
    if task_info.task_family == "timeseries":
        return _plan_timeseries_scales(
            task_info, image_info, cpu_list, mem_list, gpu_list, batch_size, output_dir, input_scales,
        )

    if input_scales:
        manual_scales = resolve_input_scales(task_info.task_family, input_scales=input_scales)
        if task_info.task_family == "audio" and not text_audio:
            return _plan_audio_scales(
                task_info=task_info,
                image_info=image_info,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                scales=manual_scales,
                batch_size=batch_size,
                output_dir=output_dir,
                source="manual",
                workload_spec_path=workload_spec_path,
            )
        if (task_info.task_family == "nlp" and not table_qa) or text_audio:
            return _plan_manual_nlp_scales(
                task_info=task_info,
                image_info=image_info,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                scales=manual_scales,
                batch_size=batch_size,
                output_dir=output_dir,
                **({"workload_spec_path": workload_spec_path} if text_audio else {}),
            )
        if task_info.task_family == "timeseries":
            manual_scales = _assert_manual_timeseries_scales_legal(
                task_info=task_info,
                scales=manual_scales,
                batch_size=batch_size,
            )
        else:
            print(f"[scale] Using manual input scales: {serialize_input_scales(manual_scales)}")

        return _materialize_scale_plan(
            task_info=task_info,
            scales=manual_scales,
            batch_size=batch_size,
            output_dir=output_dir,
            source="manual",
            workload_spec_path=workload_spec_path,
        )

    if (task_info.task_family == "nlp" and not table_qa) or text_audio:
        return _plan_nlp_auto_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=batch_size,
            output_dir=output_dir,
            **({"workload_spec_path": workload_spec_path} if text_audio else {}),
        )

    if task_info.task_family == "audio":
        from acprof.workloads import get_generator

        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
            workload_spec_path=workload_spec_path,
        )
        default_scales = workload_gen.default_input_scales()
        if not default_scales:
            raise RuntimeError(
                "audio workload manifest did not provide default input scales"
            )
        return _plan_audio_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            scales=[float(scale) for scale in default_scales],
            batch_size=batch_size,
            output_dir=output_dir,
            source="workload_spec",
            workload_spec_path=workload_spec_path,
        )

    if table_qa or task_info.task_family in {"diffusion", "multimodal", "structured"} or (
        task_info.task_family == "cv" and workload_spec_path
    ):
        from acprof.workloads import get_generator

        workload_gen = get_generator(
            task_info.task_family,
            task_info.model_id,
            task_info.pipeline_tag,
            batch_size,
            workload_spec_path=workload_spec_path,
        )
        default_scales = workload_gen.default_input_scales()
        if not default_scales and task_info.task_family == "cv":
            default_scales = _float_auto_scales(
                _default_family_max_scale(task_info, batch_size), count=AUTO_INPUT_SCALE_COUNT,
            )
        if not default_scales:
            raise RuntimeError(
                f"{task_info.task_family} workload did not provide default scales"
            )
        scales = [float(scale) for scale in default_scales]
        print(
            f"[scale] Using {task_info.task_family} workload scales: "
            f"{serialize_input_scales(scales)}"
        )
        return _materialize_scale_plan(
            task_info=task_info,
            scales=scales,
            batch_size=batch_size,
            output_dir=output_dir,
            source="workload_default",
            workload_spec_path=workload_spec_path,
        )

    max_scale = _default_family_max_scale(task_info, batch_size)
    if task_info.task_family == "cv":
        scales = _float_auto_scales(max_scale, count=AUTO_INPUT_SCALE_COUNT)
    else:
        scales = _integer_auto_scales(int(max_scale), count=AUTO_INPUT_SCALE_COUNT)

    print(
        f"[scale] Auto-planned {task_info.task_family} scales: "
        f"{serialize_input_scales(scales)}"
    )
    return _materialize_scale_plan(
        task_info=task_info,
        scales=scales,
        batch_size=batch_size,
        output_dir=output_dir,
        source="auto",
        workload_spec_path=workload_spec_path,
    )

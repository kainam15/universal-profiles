"""AC-Prof 资源矩阵、单项测量及结果整合。"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from acprof.config import (
    CLIENT_REQUEST_TIMEOUT_EXIT_CODE,
    CSV_FIELDS,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    IDLE_DIAG_DIRNAME,
)
from acprof.host.detect import TaskInfo
from acprof.host.compute_profile_plan import NCU_ERROR_FIELD, TORCH_ERROR_FIELD
from acprof.pixel_metrics import PIXEL_COUNT_FIELDS, PIXEL_RATE_SOURCES
from acprof.monitors.perf_mips import MIPS_EXIT_CODE
from acprof.capabilities import measurement_requested, require_profiling_mode
from acprof.installation import module_command
from acprof.host.docker_runtime import (
    ImageInfo,
    _sanitize_model_id,
    _run,
    _container_runtime_oom_error,
    _normalize_gpu_mode,
    _parse_csv_float,
    _cold_start_client_env,
    _host_port,
    _start_container_session,
    _stop_container_session,
)
from acprof.host.input_plan import (
    _format_scale_value,
    serialize_input_scales,
    resolve_input_scales,
)
from acprof.host.packet_capture import _packet_latency_error, _resolve_packet_latency_runtime


@dataclass(frozen=True)
class MatrixProgress:
    """One safely completed resource case in a matrix sweep."""

    completed_cases: int
    total_cases: int
    cpu: int
    mem: int
    gpu: str
    result_csv: Optional[str]


class EnergyProfilingError(RuntimeError):
    """Raised when energy profiling cannot continue reliably."""


class MIPSProfilingError(RuntimeError):
    """Raised when required MIPS profiling cannot continue reliably."""


IDLE_POWER_RELATIVE_RANGE_THRESHOLD = 0.05


def _format_watts(values: List[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def _row_has_error_status(row: Dict[str, Any]) -> bool:
    return str(row.get("status") or "").strip().lower() == "error"


def _check_idle_power_values_stable(
    *,
    csv_path: str,
    metric_name: str,
    idle_values: List[float],
    invalid_rows: int,
    row_count: int,
    threshold: float,
    remediation: str,
    skip_when_no_rows: bool = False,
) -> None:
    if row_count == 0 and skip_when_no_rows:
        return
    if invalid_rows or not idle_values:
        raise EnergyProfilingError(
            f"{metric_name} case validation failed: "
            f"csv={csv_path}, valid_rows={len(idle_values)}, invalid_rows={invalid_rows}. "
            f"This case's energy data is not reliable. {remediation}"
        )

    mean_idle = sum(idle_values) / len(idle_values)
    if mean_idle <= 0.0:
        raise EnergyProfilingError(
            f"{metric_name} case validation failed: idle baseline mean is not positive. "
            f"csv={csv_path}. This case's energy data is not reliable. {remediation}"
        )

    relative_range = (max(idle_values) - min(idle_values)) / mean_idle
    if relative_range >= threshold:
        print(
            f"[energy][WARN] {metric_name} case check warning: "
            f"csv={csv_path}, {metric_name}={_format_watts(idle_values)} W, "
            f"min={min(idle_values):.3f} W, max={max(idle_values):.3f} W, "
            f"mean={mean_idle:.3f} W, relative_range={relative_range * 100.0:.1f}%, "
            f"threshold={threshold * 100.0:.1f}%. This case's energy data may be "
            f"noisy; experiment will continue. {remediation}"
        )
        return

    print(
        f"[energy] {metric_name} case check passed: "
        f"rows={len(idle_values)}, min={min(idle_values):.3f} W, "
        f"max={max(idle_values):.3f} W, mean={mean_idle:.3f} W, "
        f"relative_range={relative_range * 100.0:.1f}%",
    )


def _check_case_gpu_idle_power_stable(
    csv_path: str,
    threshold: float = IDLE_POWER_RELATIVE_RANGE_THRESHOLD,
    ignore_error_rows: bool = False,
) -> None:
    """Validate that GPU idle baseline did not drift across a finished case CSV."""
    if not os.path.exists(csv_path):
        raise EnergyProfilingError(
            f"gpu_idle_power_w case validation failed: result CSV does not exist: {csv_path}"
        )

    gpu_rows = 0
    invalid_rows = 0
    idle_values: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if ignore_error_rows and _row_has_error_status(row):
                continue
            if _normalize_gpu_mode(row.get("gpu_mode", "off")) != "on":
                continue
            gpu_rows += 1
            gpu_idle_power_w = _parse_csv_float(
                row.get("gpu_idle_power_w", row.get("idle_power_w"))
            )
            if not math.isfinite(gpu_idle_power_w) or gpu_idle_power_w <= 0.0:
                invalid_rows += 1
                continue
            idle_values.append(gpu_idle_power_w)

    _check_idle_power_values_stable(
        csv_path=csv_path,
        metric_name="gpu_idle_power_w",
        idle_values=idle_values,
        invalid_rows=invalid_rows,
        row_count=gpu_rows,
        threshold=threshold,
        remediation=(
            "Increase --idle-seconds, close other GPU processes, wait for GPU "
            "clocks/power to stabilize, then rerun."
        ),
        skip_when_no_rows=True,
    )


def _check_case_cpu_idle_power_stable(
    csv_path: str,
    threshold: float = IDLE_POWER_RELATIVE_RANGE_THRESHOLD,
    ignore_error_rows: bool = False,
) -> None:
    """Validate that CPU package idle baseline did not drift across a finished case CSV."""
    if not os.path.exists(csv_path):
        raise EnergyProfilingError(
            f"cpu_idle_power_w case validation failed: result CSV does not exist: {csv_path}"
        )

    rows = 0
    invalid_rows = 0
    idle_values: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if ignore_error_rows and _row_has_error_status(row):
                continue
            rows += 1
            cpu_idle_power_w = _parse_csv_float(row.get("cpu_idle_power_w"))
            if not math.isfinite(cpu_idle_power_w) or cpu_idle_power_w <= 0.0:
                invalid_rows += 1
                continue
            idle_values.append(cpu_idle_power_w)

    _check_idle_power_values_stable(
        csv_path=csv_path,
        metric_name="cpu_idle_power_w",
        idle_values=idle_values,
        invalid_rows=invalid_rows,
        row_count=rows,
        threshold=threshold,
        remediation=(
            "Increase --idle-seconds, close host background processes, wait for "
            "CPU package power to stabilize, then rerun."
        ),
        skip_when_no_rows=ignore_error_rows,
    )


def _assert_packet_latency_csv_complete(
    csv_path: str,
    *,
    ignore_error_rows: bool = False,
) -> None:
    import csv

    if not os.path.exists(csv_path):
        raise _packet_latency_error(f"result CSV does not exist: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise _packet_latency_error(f"result CSV is empty: {csv_path}")
        if "latency_s" not in reader.fieldnames:
            raise _packet_latency_error(f"latency_s column is missing in {csv_path}")

        total = 0
        missing = 0
        for row in reader:
            if ignore_error_rows and _row_has_error_status(row):
                continue
            total += 1
            raw_value = (row.get("latency_s") or "").strip()
            try:
                value = float(raw_value)
            except Exception:
                missing += 1
                continue

            if not math.isfinite(value) or value <= 0:
                missing += 1

    if total == 0 and not ignore_error_rows:
        raise _packet_latency_error(f"result CSV has no rows: {csv_path}")

    if missing:
        raise _packet_latency_error(
            f"latency_s is missing for {missing}/{total} row(s) in {csv_path}"
        )


def run_single_case(
    task_info: TaskInfo,
    cpu: int,
    mem: int,
    gpu: str,
    image_info: ImageInfo,
    output_dir: str,
    project_dir: str,
    batch_size: int = 1,
    warmup: int = 2,
    repeat: int = 5,
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW,
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    sample_hz: float = 20.0,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS,
    idle_debug: bool = False,
    sniff_iface: str = "docker0",
    input_scales: Optional[str] = None,
    input_scale_plan_file: Optional[str] = None,
    compute_profile_plan_file: Optional[str] = None,
    execution_profile_plan_file: Optional[str] = None,
    require_packet_latency: bool = True,
    profiling_mode: str = "full",
    input_scale_order: Optional[List[float]] = None,
    dram_energy: str = "auto",
) -> str:
    """Run one profiling case and return result CSV path."""
    require_profiling_mode(profiling_mode)
    request_timeout_seconds = float(request_timeout_seconds)
    if (
        request_timeout_seconds <= 0.0
        or not math.isfinite(request_timeout_seconds)
    ):
        raise ValueError("request_timeout_seconds must be a finite value > 0")
    model_tag = _sanitize_model_id(task_info.model_id)
    case_name = f"case_{model_tag}_{cpu}c_{mem}g_{gpu}"
    container_name = case_name

    host_port = _host_port(cpu, mem)
    out_csv = os.path.join(output_dir, f"result_{case_name}.csv")
    idle_diag_path = os.path.join(
        output_dir,
        IDLE_DIAG_DIRNAME,
        f"{os.path.basename(out_csv)}.idle_diag.jsonl",
    )
    pcap_file = os.path.join(output_dir, f"sniff_{case_name}.pcap")
    lat_json = os.path.join(output_dir, f"lat_{case_name}.json")
    client_error_path = f"{out_csv}.client_error.json"

    print(f"\n{'='*60}")
    print(f"[case] {case_name}")
    print(f"  CPU={cpu}, MEM={mem}GB, GPU={gpu}")
    print(f"  Port: {host_port}")
    print(f"{'='*60}")

    try:
        os.remove(client_error_path)
    except FileNotFoundError:
        pass

    try:
        session = _start_container_session(
            task_info=task_info,
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            image_info=image_info,
            container_name=container_name,
            log_prefix="[case]",
            request_timeout_seconds=request_timeout_seconds,
        )
    except RuntimeError as exc:
        error = f"container_start_failed: {exc}"
        print(f"[case] {error}", file=sys.stderr)
        _write_case_error_csv(
            task_info=task_info,
            out_csv=out_csv,
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            warmup=warmup,
            repeat=repeat,
            repeat_in_window=repeat_in_window,
            input_scales=input_scales,
            error=error,
        )
        return out_csv

    base_url = session.base_url
    tcpdump_proc = None
    case_incomplete = False
    completed_rows_before_failure = 0
    incomplete_case_reason = ""
    require_packet_latency = require_packet_latency and measurement_requested(profiling_mode, "packet_latency")
    sniff_runtime = _resolve_packet_latency_runtime(
        project_dir=project_dir,
        pcap_file=pcap_file,
        sniff_iface=sniff_iface,
    ) if measurement_requested(profiling_mode, "packet_latency") else None

    try:
        if sniff_runtime is not None:
            print(f"[sniff] Starting tcpdump on {sniff_iface} via {sniff_runtime.mode}")
            try:
                tcpdump_proc = subprocess.Popen(
                    sniff_runtime.tcpdump_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                raise _packet_latency_error("failed to start tcpdump", repr(exc)) from exc
            time.sleep(0.3)
            if require_packet_latency and tcpdump_proc.poll() is not None:
                raise _packet_latency_error(
                    "tcpdump exited before workload started",
                    f"command={' '.join(sniff_runtime.tcpdump_cmd)}",
                )
        else:
            if require_packet_latency:
                raise _packet_latency_error(
                    "tcpdump/tshark runtime could not be resolved"
                )
            if measurement_requested(profiling_mode, "packet_latency"):
                print("[sniff] tcpdump/tshark unavailable, skipping packet-level latency")
            else:
                print(f"[sniff] packet_latency=not_requested (profiling_mode={profiling_mode})")

        print("[case] Running workload...")
        scales_str = input_scales or serialize_input_scales(
            resolve_input_scales(task_info.task_family, input_scales=None)
        )

        client_env = {
            **os.environ,
            "MODEL_ID": task_info.model_id,
            "MODEL_REVISION": task_info.model_revision,
            "TASK_FAMILY": task_info.task_family,
            "PIPELINE_TAG": task_info.pipeline_tag,
            "RUNTIME_BACKEND": task_info.runtime_backend,
            "IMAGE_TAG": image_info.tag,
            "CPU_CORES": str(cpu),
            "MEM_CAP_GB": str(mem),
            "GPU_MODE": gpu,
            "BASE_URL": base_url,
            "ENDPOINT": "/predict",
            "BATCH_SIZE": str(batch_size),
            "WARMUP": str(warmup),
            "REPEAT": str(repeat),
            "REPEAT_IN_WINDOW": str(repeat_in_window),
            "REPEAT_WINDOW_SECONDS": str(repeat_window_seconds),
            "REQUEST_TIMEOUT_SECONDS": f"{request_timeout_seconds:g}",
            **_cold_start_client_env(session),
            "OUT_CSV": out_csv,
            "CASE_NAME": case_name,
            "CONTAINER_NAME": container_name,
            "USE_MIPS": "1" if measurement_requested(profiling_mode, "cpu_instructions") else "0",
            "PROFILING_MODE": profiling_mode,
            "DRAM_ENERGY": dram_energy,
            "INPUT_SCALE_ORDER": json.dumps(input_scale_order) if input_scale_order is not None else "",
            "SAMPLE_HZ": str(sample_hz),
            "IDLE_SECONDS": str(idle_seconds),
            "IDLE_COOLDOWN_SECONDS": str(idle_cooldown_seconds),
            "IDLE_DEBUG": "1" if idle_debug else "0",
            "IDLE_DIAG_PATH": idle_diag_path if idle_debug else "",
            "CLIENT_ERROR_PATH": client_error_path,
            "DEVICE_INDEX": str(session.gpu_device.get("index", 0)),
            "GPU_DEVICE_UUID": str(session.gpu_device.get("uuid", "")),
            "INPUT_SCALES": scales_str,
            "INPUT_SCALE_PLAN_FILE": input_scale_plan_file or "",
            "COMPUTE_PROFILE_PLAN_FILE": compute_profile_plan_file or "",
            "EXECUTION_PROFILE_PLAN_FILE": execution_profile_plan_file or "",
        }
        # The host-side client never sends notifications.  Keep the webhook
        # credential out of the measured subprocess environment entirely.
        client_env.pop("ACPROF_WECOM_WEBHOOK_URL", None)

        client_result = _run(
            module_command("acprof.host.client"),
            check=False,
            capture=False,
            env=client_env,
        )

        if client_result.returncode != 0:
            runtime_oom_error = _container_runtime_oom_error(
                container_name,
                mem,
                client_result.returncode,
            )
            if runtime_oom_error is not None:
                case_incomplete = True
                incomplete_case_reason = "runtime OOM"
                print(f"[case][WARN] {runtime_oom_error}", file=sys.stderr)
                completed_rows_before_failure, _ = _write_case_error_csv(
                    task_info=task_info,
                    out_csv=out_csv,
                    cpu=cpu,
                    mem=mem,
                    gpu=gpu,
                    warmup=warmup,
                    repeat=repeat,
                    repeat_in_window=repeat_in_window,
                    input_scales=input_scales,
                    error=runtime_oom_error,
                    preserve_existing=True,
                    annotate_existing_error_rows=True,
                )
            elif client_result.returncode == CLIENT_REQUEST_TIMEOUT_EXIT_CODE:
                case_incomplete = True
                incomplete_case_reason = "request timeout"
                timeout_context = _load_client_error_context(client_error_path)
                timeout_context.setdefault(
                    "request_timeout_s",
                    request_timeout_seconds,
                )
                timeout_s = _timeout_context_float(
                    timeout_context.get("request_timeout_s"),
                    request_timeout_seconds,
                )
                error = (
                    "client_request_timeout: triggering request exceeded "
                    f"{timeout_s:g}s; incomplete and later rows will be marked "
                    "individually"
                )
                print(f"[case][WARN] {error}", file=sys.stderr)
                completed_rows_before_failure, _ = _write_case_error_csv(
                    task_info=task_info,
                    out_csv=out_csv,
                    cpu=cpu,
                    mem=mem,
                    gpu=gpu,
                    warmup=warmup,
                    repeat=repeat,
                    repeat_in_window=repeat_in_window,
                    input_scales=input_scales,
                    error=error,
                    preserve_existing=True,
                    timeout_context=timeout_context,
                )
            elif client_result.returncode == MIPS_EXIT_CODE:
                raise MIPSProfilingError(
                    "client.py exited because MIPS profiling failed; review the "
                    "[mips][ERROR] output above for the perf remediation steps."
                )
            else:
                raise EnergyProfilingError(
                    "client.py exited with code "
                    f"{client_result.returncode}; aborting profiling matrix. "
                    "Review the client output above for the energy stability diagnostic."
                )

        if tcpdump_proc is not None:
            time.sleep(1.0)
            tcpdump_proc.terminate()
            try:
                tcpdump_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tcpdump_proc.kill()
            time.sleep(0.2)

            if case_incomplete and completed_rows_before_failure == 0:
                print(
                    "[sniff] No completed measurements before "
                    f"{incomplete_case_reason}; skipping packet-latency merge"
                )
            elif not os.path.exists(pcap_file) or os.path.getsize(pcap_file) <= 0:
                if require_packet_latency:
                    raise _packet_latency_error(
                        f"pcap file is missing or empty: {pcap_file}"
                    )
            else:
                print("[sniff] Parsing pcap -> packet latencies...")
                assert sniff_runtime is not None
                parse_result = _run(sniff_runtime.parse_cmd, check=False)
                parse_output = parse_result.stdout.strip()
                if parse_result.returncode != 0:
                    raise _packet_latency_error(
                        "pcap parsing failed",
                        (parse_result.stderr or parse_result.stdout or "").strip(),
                    )
                if not parse_output:
                    raise _packet_latency_error("pcap parser produced no output")
                try:
                    latency_payload = json.loads(parse_output)
                except json.JSONDecodeError as exc:
                    raise _packet_latency_error(
                        "pcap parser produced invalid JSON",
                        repr(exc),
                    ) from exc
                latency_records = (
                    latency_payload.get("requests")
                    if isinstance(latency_payload, dict)
                    and "requests" in latency_payload
                    else latency_payload
                )
                if not isinstance(latency_records, dict) or not latency_records:
                    raise _packet_latency_error(
                        "pcap parser did not find matching request latency records"
                    )
                with open(lat_json, "w", encoding="utf-8") as lf:
                    json.dump(latency_payload, lf, ensure_ascii=True, indent=2)

                if not os.path.exists(out_csv):
                    raise _packet_latency_error(
                        f"client did not produce result CSV: {out_csv}"
                    )

                print("[sniff] Merging packet latency into CSV...")
                merged_csv = out_csv + ".merged"
                merge_result = _run(
                    [
                        *module_command("acprof.packet.merge_packet_latency"),
                        out_csv,
                        lat_json,
                        merged_csv,
                    ],
                    check=False,
                )
                if merge_result.returncode != 0:
                    raise _packet_latency_error(
                        "packet latency merge failed",
                        (merge_result.stderr or merge_result.stdout or "").strip(),
                    )
                if not os.path.exists(merged_csv):
                    raise _packet_latency_error(
                        f"packet latency merge did not produce {merged_csv}"
                    )
                os.replace(merged_csv, out_csv)
                if require_packet_latency:
                    _assert_packet_latency_csv_complete(
                        out_csv,
                        ignore_error_rows=case_incomplete,
                    )

        if measurement_requested(profiling_mode, "cpu_energy") and (not case_incomplete or completed_rows_before_failure > 0):
            _check_case_cpu_idle_power_stable(
                out_csv,
                ignore_error_rows=case_incomplete,
            )
            if _normalize_gpu_mode(gpu) == "on":
                _check_case_gpu_idle_power_stable(
                    out_csv,
                    ignore_error_rows=case_incomplete,
                )
    finally:
        if tcpdump_proc is not None and tcpdump_proc.poll() is None:
            tcpdump_proc.terminate()
            try:
                tcpdump_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tcpdump_proc.kill()
        _stop_container_session(container_name, log_prefix="[case]")

    print(f"[case] Done. Output: {out_csv}")
    return out_csv


def _write_case_error_csv(
    *,
    task_info: TaskInfo,
    out_csv: str,
    cpu: int,
    mem: int,
    gpu: str,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    input_scales: Optional[str],
    error: str,
    preserve_existing: bool = False,
    annotate_existing_error_rows: bool = False,
    timeout_context: Optional[Dict[str, Any]] = None,
    result_origin: str = "formal_attempt",
) -> Tuple[int, int]:
    """Write missing error rows and return successful-preserved/added counts."""
    if not str(error or "").strip():
        raise ValueError("error rows require a non-empty diagnostic")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    scales = resolve_input_scales(task_info.task_family, input_scales)

    def make_row(scale: float, repeat_idx: int, is_warmup: bool) -> Dict[str, Any]:
        row = {field: "nan" for field in CSV_FIELDS}
        row.update({
            "cpu_cores": str(cpu),
            "mem_cap_gb": str(mem),
            "gpu_mode": gpu,
            "input_scale": _format_scale_value(scale),
            "task_param": "",
            "repeat_idx": str(repeat_idx),
            "warmup": "1" if is_warmup else "0",
            "repeat_in_window": str(repeat_in_window),
            TORCH_ERROR_FIELD: "not_run",
            "status": "error",
            "error": error,
            "result_origin": result_origin,
        })
        if gpu == "on":
            row[NCU_ERROR_FIELD] = "not_run"
            row["compute_profile_error_nsys"] = "not_run"
        else:
            row["compute_profile_error_massif"] = "not_run"
        return row

    planned_rows: List[Dict[str, Any]] = []
    for scale in scales:
        for idx in range(warmup):
            planned_rows.append(make_row(scale, idx, True))
        for idx in range(repeat):
            planned_rows.append(make_row(scale, idx, False))

    def measurement_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
        try:
            scale = _format_scale_value(float(row.get("input_scale", "nan")))
            repeat_idx = int(row.get("repeat_idx", ""))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid measurement key in partial case CSV: {row!r}"
            ) from exc
        warmup_flag = "1" if str(row.get("warmup") or "0").strip() == "1" else "0"
        return scale, warmup_flag, repeat_idx

    planned_by_key = {
        measurement_key(row): row
        for row in planned_rows
    }
    existing_rows: List[Dict[str, Any]] = []
    existing_keys = set()
    if preserve_existing and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            optional_pixel_fields = {*PIXEL_COUNT_FIELDS, *PIXEL_RATE_SOURCES}
            missing_fields = set(CSV_FIELDS) - set(reader.fieldnames or []) - optional_pixel_fields
            if missing_fields:
                raise RuntimeError(
                    "partial case CSV is missing required fields: "
                    + ", ".join(sorted(missing_fields))
                )
            for row in reader:
                key = measurement_key(row)
                if key not in planned_by_key:
                    raise RuntimeError(
                        f"partial case CSV contains an unplanned measurement: {key!r}"
                    )
                if key in existing_keys:
                    raise RuntimeError(
                        f"partial case CSV contains a duplicate measurement: {key!r}"
                    )
                existing_keys.add(key)
                normalized_row = {
                    field: row.get(field, "nan")
                    for field in CSV_FIELDS
                }
                if (
                    annotate_existing_error_rows
                    and _row_has_error_status(normalized_row)
                ):
                    existing_error = str(normalized_row.get("error") or "").strip()
                    if error not in existing_error:
                        normalized_row["error"] = (
                            f"{existing_error}; {error}"
                            if existing_error
                            else error
                        )
                existing_rows.append(normalized_row)

    missing_rows = [
        row
        for row in planned_rows
        if measurement_key(row) not in existing_keys
    ]
    if timeout_context is not None and missing_rows:
        _annotate_timeout_placeholder_rows(
            missing_rows,
            measurement_key=measurement_key,
            timeout_context=timeout_context,
        )
    rows = [*existing_rows, *missing_rows]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    preserved_success_rows = sum(
        not _row_has_error_status(row)
        for row in existing_rows
    )
    if preserve_existing:
        print(
            f"[case] Preserved existing rows: {len(existing_rows)} "
            f"(successful: {preserved_success_rows}); "
            f"wrote error rows: {len(missing_rows)}; total: {len(rows)}"
        )
    else:
        print(f"[case] Wrote error rows: {out_csv} ({len(rows)} rows)")
    return preserved_success_rows, len(missing_rows)


def _timeout_context_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else float(fallback)


def _load_client_error_context(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[case][WARN] could not read structured client error context {path}: {exc}",
            file=sys.stderr,
        )
        return {}
    if not isinstance(payload, dict):
        print(
            f"[case][WARN] ignored non-object client error context: {path}",
            file=sys.stderr,
        )
        return {}
    return payload


def _annotate_timeout_placeholder_rows(
    rows: List[Dict[str, Any]],
    *,
    measurement_key,
    timeout_context: Dict[str, Any],
) -> None:
    first_missing_scale = measurement_key(rows[0])[0]
    try:
        triggering_scale = _format_scale_value(
            float(timeout_context.get("input_scale", first_missing_scale))
        )
    except (TypeError, ValueError):
        triggering_scale = first_missing_scale

    timeout_s = _timeout_context_float(
        timeout_context.get("request_timeout_s"),
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    request_id = str(timeout_context.get("request_id") or "unknown")
    request_phase = str(timeout_context.get("request_phase") or "unknown")
    repeat_idx = timeout_context.get("measurement_repeat_idx")
    attempted_key = None
    if request_phase == "measurement_warmup" and repeat_idx is not None:
        attempted_key = (triggering_scale, "1", int(repeat_idx))
    elif request_phase == "measurement_repeat" and repeat_idx is not None:
        attempted_key = (triggering_scale, "0", int(repeat_idx))

    context_source = (
        "client_timeout_sidecar"
        if timeout_context.get("input_scale") is not None
        else "inferred_from_first_missing_scale"
    )
    trigger_fields = (
        f"trigger_error=client_request_timeout; triggering_input_scale={triggering_scale}; "
        f"request_timeout_s={timeout_s:g}; triggering_request_latency_s>{timeout_s:g}; "
        f"trigger_phase={request_phase}; trigger_request_id={request_id}; "
        f"context_source={context_source}"
    )

    for row in rows:
        key = measurement_key(row)
        planned_scale = key[0]
        if attempted_key is not None and key == attempted_key:
            row["error"] = (
                "client_request_timeout: planned_request_attempted=true; "
                "measurement_row_completed=false; "
                f"planned_input_scale={planned_scale}; {trigger_fields}"
            )
            continue

        reason = (
            "triggering_scale_probe_timed_out"
            if planned_scale == triggering_scale
            else "skipped_after_prior_scale_timeout"
        )
        row["error"] = (
            "not_measured_after_timeout: planned_request_attempted=false; "
            "measurement_row_completed=false; "
            f"planned_input_scale={planned_scale}; reason={reason}; {trigger_fields}"
        )


def _write_startup_oom_pruned_case_csv(
    *,
    task_info: TaskInfo,
    output_dir: str,
    cpu: int,
    mem: int,
    gpu: str,
    reference_cpu: int,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    input_scales: Optional[str],
) -> str:
    model_tag = _sanitize_model_id(task_info.model_id)
    case_name = f"case_{model_tag}_{cpu}c_{mem}g_{gpu}"
    out_csv = os.path.join(output_dir, f"result_{case_name}.csv")
    error = (
        "not_measured_after_startup_oom_pruning: "
        "planned_request_attempted=false; measurement_row_completed=false; "
        "reason=confirmed_startup_oom_at_reference_cpu; "
        f"reference_cpu_cores={reference_cpu}; reference_mem_cap_gb={mem}; "
        f"gpu_mode={_normalize_gpu_mode(gpu)}; "
        "pruning_scope=container_startup_only; "
        "result_origin=inferred_not_measured"
    )
    _write_case_error_csv(
        task_info=task_info,
        out_csv=out_csv,
        cpu=cpu,
        mem=mem,
        gpu=gpu,
        warmup=warmup,
        repeat=repeat,
        repeat_in_window=repeat_in_window,
        input_scales=input_scales,
        error=error,
        result_origin="inferred_not_measured",
    )
    return out_csv


def run_matrix(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    output_dir: str,
    project_dir: str,
    batch_size: int = 1,
    warmup: int = 2,
    repeat: int = 5,
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW,
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    sample_hz: float = 20.0,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS,
    idle_debug: bool = False,
    sniff_iface: str = "docker0",
    input_scales: Optional[str] = None,
    input_scale_plan_file: Optional[str] = None,
    compute_profile_plan_file: Optional[str] = None,
    execution_profile_plan_file: Optional[str] = None,
    progress_callback: Optional[Callable[[MatrixProgress], None]] = None,
    prune_startup_oom: bool = False,
    run_state=None,
    profiling_mode: str = "full",
    matrix_order: str = "seeded",
    matrix_seed: int = 0,
    dram_energy: str = "auto",
) -> List[str]:
    """Run a frozen resource plan after independent readiness-only probes."""
    from pathlib import Path
    from acprof.host.matrix_plan import MATRIX_PLAN_NAME, matrix_identity, load_matrix_plan, freeze_matrix_plan
    from acprof.host.startup_probe import PROBE_NAME, run_startup_probes, startup_oom_prefixes

    request_timeout_seconds = float(request_timeout_seconds)
    require_profiling_mode(profiling_mode)
    if request_timeout_seconds <= 0 or not math.isfinite(request_timeout_seconds):
        raise ValueError("request_timeout_seconds must be a finite value > 0")
    os.makedirs(output_dir, exist_ok=True)
    scales = resolve_input_scales(task_info.task_family, input_scales)
    identity = matrix_identity(task_info, image_info, cpu_list, mem_list, gpu_list, scales,
                               order=matrix_order, seed=matrix_seed, prune=prune_startup_oom,
                               input_plan_file=input_scale_plan_file)
    plan_path = Path(output_dir) / MATRIX_PLAN_NAME
    if plan_path.exists():
        plan = load_matrix_plan(plan_path, identity)
        if prune_startup_oom:
            evidence = json.loads((Path(output_dir) / PROBE_NAME).read_text())
            if (evidence.get("schema_version") != 2 or evidence.get("status") != "complete"
                    or evidence.get("identity") != identity
                    or startup_oom_prefixes(evidence) != plan["startup_oom_prefixes"]):
                raise ValueError("matrix plan startup probe evidence mismatch")
    else:
        if run_state is not None and run_state.data.get("cases"):
            raise ValueError("missing frozen matrix plan for existing cases; start a new experiment")
        prefixes = {}
        if prune_startup_oom:
            evidence = run_startup_probes(output_dir, identity, task_info, image_info,
                                          request_timeout_seconds=request_timeout_seconds)
            prefixes = startup_oom_prefixes(evidence)
        plan = freeze_matrix_plan(plan_path, identity, prefixes)
    if run_state is not None:
        run_state.bind_matrix_plan(plan)
    print(f"[matrix] Frozen plan: {plan_path}; order={matrix_order}, seed={matrix_seed}, hash={plan['plan_sha256']}")
    result_csvs = []
    total = len(plan["cases"])
    for current, case in enumerate(plan["cases"], 1):
        cpu, mem, gpu = case["cpu_cores"], case["mem_cap_gb"], case["gpu_mode"]
        print(f"[matrix] Case {current}/{total}: CPU={cpu}, MEM={mem}GB, GPU={gpu}")
        filename = f"result_case_{_sanitize_model_id(task_info.model_id)}_{cpu}c_{mem}g_{gpu}.csv"
        cached_case = run_state.prepare_case(filename, cpu, mem, gpu) if run_state else None
        if cached_case:
            csv_path = cached_case
            print(f"[resume] 已完成，复用 case：{filename}")
        elif case["result_origin"] == "inferred_not_measured":
            csv_path = _write_startup_oom_pruned_case_csv(
                task_info=task_info, output_dir=output_dir, cpu=cpu, mem=mem, gpu=gpu,
                reference_cpu=min(cpu_list), warmup=warmup, repeat=repeat,
                repeat_in_window=repeat_in_window, input_scales=input_scales)
        else:
            csv_path = run_single_case(
                task_info=task_info, cpu=cpu, mem=mem, gpu=gpu, image_info=image_info,
                output_dir=output_dir, project_dir=project_dir, batch_size=batch_size,
                warmup=warmup, repeat=repeat, repeat_in_window=repeat_in_window,
                repeat_window_seconds=repeat_window_seconds, request_timeout_seconds=request_timeout_seconds,
                sample_hz=sample_hz, idle_seconds=idle_seconds, idle_cooldown_seconds=idle_cooldown_seconds,
                idle_debug=idle_debug, sniff_iface=sniff_iface, input_scales=input_scales,
                input_scale_plan_file=input_scale_plan_file, compute_profile_plan_file=compute_profile_plan_file,
                execution_profile_plan_file=execution_profile_plan_file, profiling_mode=profiling_mode,
                input_scale_order=case["input_scales"], dram_energy=dram_energy)
        if run_state is not None and csv_path and not cached_case:
            run_state.finish_case(csv_path, cpu, mem, gpu)
        if csv_path:
            result_csvs.append(csv_path)
        if progress_callback is not None:
            try:
                progress_callback(MatrixProgress(current, total, cpu, mem, gpu, csv_path or None))
            except Exception as exc:
                print(f"[progress][WARN] Progress callback failed: {type(exc).__name__}", file=sys.stderr)
    return result_csvs


def merge_all_csvs(csv_paths: List[str], output_path: str, *, expected=None) -> None:
    """Validate all cases before atomically publishing the final result CSV."""
    from acprof.result_csv import merge_result_csvs

    row_count = merge_result_csvs(csv_paths, output_path, expected=expected)
    print(f"[merge] Final CSV: {output_path} ({row_count} rows)")

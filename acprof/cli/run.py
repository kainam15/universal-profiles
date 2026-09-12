#!/usr/bin/env python3
"""AC-Prof Universal Profiler - One-click HuggingFace model profiling.

Usage:
    python run.py --model bert-base-uncased
    python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
    python run.py --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from acprof.config import (
    DEFAULT_COMPUTE_PROFILE_TOOL,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    SCALING_DIMENSIONS,
)
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.preflight import (
    NATIVE_DOCKER_SOCKET,
    _docker_info_is_docker_desktop,
    _docker_context_is_docker_desktop,
    _process_is_wsl,
    _exit_unsupported_host,
    require_native_linux_host,
    detect_cgroup_version,
    require_cgroup_prerequisites,
    require_result_cgroup_compatibility,
    _docker_host_is_native_socket,
    _exit_nonlocal_docker,
    _exit_docker_desktop,
    require_native_docker,
    require_cpu_energy_prerequisites,
)
from acprof.cli.run_args import build_parser as _build_parser
from acprof.host.collection_history import (
    COLLECTION_HISTORY_NAME,
    empty_collection_history,
    write_collection_history_json,
)
from acprof.host.orchestrator import (
    EnergyProfilingError,
    MatrixProgress,
    MIPSProfilingError,
)
from acprof.host.packet_capture import (
    PacketLatencyError,
    require_packet_latency_prerequisites,
)
from acprof.host.profiler_progress import ProfilerProgress
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.monitors.perf_mips import require_mips_prerequisites
from acprof.notifications import (
    NotificationConfigError,
    NotificationError,
    NotificationEvent,
    WeComWebhookNotifier,
)

PROJECT_DIR = str(Path(__file__).resolve().parents[2])
TMUX_TERMINAL_LOG_FILENAME = "tmux_all.log"
DEFAULT_NOTIFY_PROVIDER = "auto"
_ACTIVE_TMUX_TERMINAL_LOG: tuple[str, str, str] | None = None


@dataclass
class _RunNotificationContext:
    """Mutable lifecycle state; the webhook itself is never logged or persisted."""

    notifier: WeComWebhookNotifier
    model_id: str
    output_dir: str
    started_at: float
    run_command: str = ""
    total_cases: int | None = None
    event: NotificationEvent | None = None


_ACTIVE_RUN_NOTIFICATION: _RunNotificationContext | None = None


def _parse_int_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_run_command(argv: list[str]) -> str:
    """Return a shell-safe command string matching the run.py invocation."""
    if not argv:
        return "python run.py"
    return shlex.join(["python", *argv])


def _start_tmux_terminal_log(
    output_dir: str,
    argv: list[str],
) -> tuple[str, str, str] | None:
    """Pipe all future output from the current tmux pane to a temporary log."""
    pane_id = os.environ.get("TMUX_PANE", "").strip()
    if not os.environ.get("TMUX") or not pane_id:
        return None

    try:
        pipe_status = subprocess.run(
            [
                "tmux",
                "display-message",
                "-p",
                "-t",
                pane_id,
                "#{pane_pipe}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[terminal-log][WARN] Cannot inspect tmux pane {pane_id}: {exc}")
        return None

    if pipe_status.returncode != 0:
        detail = (pipe_status.stderr or pipe_status.stdout or "").strip()
        print(
            f"[terminal-log][WARN] Cannot inspect tmux pane {pane_id}: "
            f"{detail or f'exit {pipe_status.returncode}'}"
        )
        return None
    if pipe_status.stdout.strip().lower() in {"1", "on", "true", "yes"}:
        print(
            f"[terminal-log][WARN] tmux pane {pane_id} already has an active "
            "pipe; leaving it unchanged and skipping automatic tmux_all.log"
        )
        return None

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, TMUX_TERMINAL_LOG_FILENAME)
    partial_path = f"{log_path}.part"
    try:
        with open(partial_path, "w", encoding="utf-8") as f:
            f.write(f"$ {_format_run_command(argv)}\n")
    except OSError as exc:
        print(f"[terminal-log][WARN] Cannot initialize {partial_path}: {exc}")
        return None

    pipe_command = f"cat >> {shlex.quote(partial_path)}"
    try:
        pipe_result = subprocess.run(
            [
                "tmux",
                "pipe-pane",
                "-O",
                "-t",
                pane_id,
                pipe_command,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[terminal-log][WARN] Cannot start tmux pane logging: {exc}")
        return None

    if pipe_result.returncode != 0:
        detail = (pipe_result.stderr or pipe_result.stdout or "").strip()
        print(
            "[terminal-log][WARN] Cannot start tmux pane logging: "
            f"{detail or f'exit {pipe_result.returncode}'}"
        )
        return None

    print(f"[terminal-log] Recording tmux pane {pane_id}: {log_path}")
    return pane_id, partial_path, log_path


def _stop_tmux_terminal_log(
    terminal_log: tuple[str, str, str],
) -> bool:
    """Stop the pane pipe and atomically publish the completed terminal log."""
    pane_id, partial_path, log_path = terminal_log
    try:
        close_result = subprocess.run(
            ["tmux", "pipe-pane", "-t", pane_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(
            f"[terminal-log][WARN] Cannot stop tmux pane logging; "
            f"partial log remains at {partial_path}: {exc}"
        )
        return False

    if close_result.returncode != 0:
        detail = (close_result.stderr or close_result.stdout or "").strip()
        print(
            f"[terminal-log][WARN] Cannot stop tmux pane logging; "
            f"partial log remains at {partial_path}: "
            f"{detail or f'exit {close_result.returncode}'}"
        )
        return False

    try:
        os.replace(partial_path, log_path)
    except OSError as exc:
        print(
            f"[terminal-log][WARN] Cannot finalize {log_path}; "
            f"partial log remains at {partial_path}: {exc}"
        )
        return False

    print(f"[terminal-log] Saved terminal display: {log_path}")
    return True


def _activate_run_notification(
    *,
    provider: str,
    model_id: str,
    output_dir: str,
    started_at: float,
    run_command: str,
) -> None:
    """Configure a notifier without making any network request."""
    global _ACTIVE_RUN_NOTIFICATION

    _ACTIVE_RUN_NOTIFICATION = None
    if provider == "none":
        return
    if provider not in {"auto", "wecom"}:
        print(f"[notify][WARN] Unsupported notification provider: {provider}")
        return

    try:
        notifier = WeComWebhookNotifier.from_env()
    except NotificationConfigError as exc:
        if provider == "wecom":
            print(f"[notify][WARN] 企业微信通知未启用：{exc}", file=sys.stderr)
        return

    _ACTIVE_RUN_NOTIFICATION = _RunNotificationContext(
        notifier=notifier,
        model_id=model_id,
        output_dir=output_dir,
        started_at=started_at,
        run_command=run_command,
    )
    print(
        "[notify] 企业微信通知已启用；实验开始、每个 profiler 阶段和 case "
        "结束后、实验结束时发送通知"
    )


def _notify_run_started() -> None:
    """Report the command before preflight/build work and any measurements."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return

    event = NotificationEvent(
        status="started",
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        run_command=context.run_command,
        detail="命令已启动，正在执行环境预检",
    )
    _send_notification_event(
        context,
        event,
        success_message="[notify] 企业微信实验开始通知已发送",
    )


def _update_run_notification_plan(
    *,
    model_id: str,
    output_dir: str,
    total_cases: int,
) -> None:
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return
    context.model_id = model_id
    context.output_dir = output_dir
    context.total_cases = total_cases


def _result_status_counts(result_csv: str) -> tuple[int, int]:
    result_rows = 0
    error_rows = 0
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            result_rows += 1
            if str(row.get("status") or "").strip().lower() == "error":
                error_rows += 1
    return result_rows, error_rows


def _record_run_completion(
    *,
    final_csv: str | None,
    completed_cases: int,
) -> None:
    """Create a completion event after all measurements and result writes."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return

    result_rows: int | None = None
    error_rows: int | None = None
    detail: str | None = None
    if final_csv:
        try:
            result_rows, error_rows = _result_status_counts(final_csv)
        except (OSError, csv.Error, UnicodeError) as exc:
            detail = f"结果已生成，但通知摘要读取失败（{type(exc).__name__}）"

    if not final_csv or result_rows == 0:
        status = "no_results"
        if detail is None:
            detail = "本次运行未产生结果数据行"
    elif (
        (error_rows or 0) > 0
        or (
            context.total_cases is not None
            and completed_cases < context.total_cases
        )
    ):
        status = "partial"
    else:
        status = "success"

    context.event = NotificationEvent(
        status=status,
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        total_cases=context.total_cases,
        completed_cases=completed_cases,
        result_rows=result_rows,
        error_rows=error_rows,
        final_csv=final_csv,
        detail=detail,
    )


def _record_run_termination(status: str, detail: str) -> None:
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return
    context.event = NotificationEvent(
        status=status,
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        total_cases=context.total_cases,
        detail=detail,
    )


def _send_notification_event(
    context: _RunNotificationContext,
    event: NotificationEvent,
    *,
    success_message: str,
) -> None:
    """Deliver one event without allowing notification errors to escape."""
    try:
        context.notifier.send(event)
    except NotificationError as exc:
        print(f"[notify][WARN] 企业微信通知发送失败：{exc}", file=sys.stderr)
    except Exception as exc:
        # Unknown provider errors may embed request details.  Print only the
        # exception type so credentials can never be copied into terminal logs.
        print(
            "[notify][WARN] 企业微信通知发送失败："
            f"{type(exc).__name__}",
            file=sys.stderr,
        )
    else:
        print(success_message)


def _notify_profiler_completion(progress: ProfilerProgress) -> None:
    """Report one finished profiler stage before any next probe or CSV sweep."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return

    event = NotificationEvent(
        status=f"profiler_{progress.status}",
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        profiler=progress.profiler,
        profile_elapsed_seconds=progress.elapsed_seconds,
        profile_samples=progress.total_samples,
        profile_error_samples=progress.error_samples,
        detail=progress.detail or None,
    )
    _send_notification_event(
        context,
        event,
        success_message=(
            f"[notify] 企业微信 Profiler 通知已发送：{progress.profiler} "
            f"({progress.status})"
        ),
    )


def _notify_case_progress(progress: MatrixProgress) -> None:
    """Send progress only after run_single_case has stopped its resources."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return

    result_rows: int | None = None
    error_rows: int | None = None
    summary_note = ""
    if progress.result_csv:
        try:
            result_rows, error_rows = _result_status_counts(progress.result_csv)
        except (OSError, csv.Error, UnicodeError) as exc:
            summary_note = f"；结果摘要读取失败（{type(exc).__name__}）"

    detail = (
        f"刚完成：CPU={progress.cpu}, MEM={progress.mem}GB, GPU={progress.gpu}"
        f"{summary_note}"
    )
    event = NotificationEvent(
        status="progress",
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        total_cases=progress.total_cases,
        completed_cases=progress.completed_cases,
        result_rows=result_rows,
        error_rows=error_rows,
        detail=detail,
    )
    _send_notification_event(
        context,
        event,
        success_message=(
            "[notify] 企业微信进度通知已发送："
            f"case {progress.completed_cases}/{progress.total_cases}"
        ),
    )


def _deliver_run_notification(terminal_log: str | None = None) -> None:
    """Send the recorded event best-effort without changing command outcome."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None or context.event is None:
        return

    event = context.event
    if terminal_log:
        event = replace(event, terminal_log=terminal_log)
    _send_notification_event(
        context,
        event,
        success_message="[notify] 企业微信最终通知已发送",
    )


def _cleanup_intermediate_results(csv_paths: list[str], output_dir: str, final_csv: str) -> None:
    """Delete per-run intermediate artifacts after the merged CSV is safely written."""
    if not csv_paths:
        return

    if not os.path.exists(final_csv):
        print(f"[cleanup][WARN] Skip cleanup because merged CSV is missing: {final_csv}")
        return

    if os.path.getsize(final_csv) <= 0:
        print(f"[cleanup][WARN] Skip cleanup because merged CSV is empty: {final_csv}")
        return

    targets = set()
    for csv_path in csv_paths:
        targets.add(csv_path)
        targets.add(f"{csv_path}.sniff_groups.jsonl")

        base_name = os.path.basename(csv_path)
        if not (base_name.startswith("result_case_") and base_name.endswith(".csv")):
            print(f"[cleanup][WARN] Skip derived cleanup for unexpected CSV name: {csv_path}")
            continue

        case_name = base_name[len("result_"):-len(".csv")]
        targets.add(os.path.join(output_dir, f"lat_{case_name}.json"))
        targets.add(os.path.join(output_dir, f"sniff_{case_name}.pcap"))

    removed = 0
    missing = 0
    failed = 0

    for path in sorted(targets):
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            os.remove(path)
            removed += 1
            print(f"[cleanup] Removed: {path}")
        except OSError as exc:
            failed += 1
            print(f"[cleanup][WARN] Failed to remove {path}: {exc}")

    print(f"[cleanup] Done. removed={removed}, missing={missing}, failed={failed}")


def _run_main():
    global _ACTIVE_TMUX_TERMINAL_LOG

    start_time = time.perf_counter()
    bootstrap_project_env(PROJECT_DIR)

    parser = _build_parser(default_notify_provider=DEFAULT_NOTIFY_PROVIDER)

    args = parser.parse_args()
    run_command = _format_run_command(sys.argv)
    compute_profile_disabled = (
        args.no_compute_profile or args.compute_profile_tool == "none"
    )
    if args.repeat_in_window < 0:
        parser.error("--repeat-in-window must be >= 0")
    if args.repeat_window_seconds <= 0.0:
        parser.error("--repeat-window-seconds must be > 0")
    if (
        args.request_timeout_seconds <= 0.0
        or not math.isfinite(args.request_timeout_seconds)
    ):
        parser.error("--request-timeout-seconds must be a finite value > 0")
    if args.torch_profiler_repeat <= 0:
        parser.error("--torch-profiler-repeat must be > 0")
    if args.ncu_repeat <= 0:
        parser.error("--ncu-repeat must be > 0")
    if args.massif_repeat <= 0:
        parser.error("--massif-repeat must be > 0")
    if args.nsys_repeat <= 0:
        parser.error("--nsys-repeat must be > 0")
    for option, value in (
        ("--massif-reference-cpu", args.massif_reference_cpu),
        ("--massif-reference-mem", args.massif_reference_mem),
        ("--nsys-reference-cpu", args.nsys_reference_cpu),
        ("--nsys-reference-mem", args.nsys_reference_mem),
    ):
        if value is not None and value <= 0:
            parser.error(f"{option} must be > 0")

    terminal_output_dir = os.path.join(
        PROJECT_DIR,
        args.output_dir,
        args.model.replace("/", "--"),
    )
    _activate_run_notification(
        provider=args.notify,
        model_id=args.model,
        output_dir=terminal_output_dir,
        started_at=start_time,
        run_command=run_command,
    )
    _ACTIVE_TMUX_TERMINAL_LOG = _start_tmux_terminal_log(
        terminal_output_dir,
        sys.argv,
    )
    _notify_run_started()

    require_native_linux_host()
    require_native_docker()
    cgroup_version = require_cgroup_prerequisites(
        allow_cgroup_v1=args.allow_cgroup_v1,
    )
    cgroup_collection_mode = (
        "legacy_compatible" if args.allow_cgroup_v1 else "strict_v2"
    )

    try:
        require_packet_latency_prerequisites(
            project_dir=PROJECT_DIR,
            sniff_iface=args.sniff_iface,
        )
    except PacketLatencyError as exc:
        print(f"\n[sniff][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    require_cpu_energy_prerequisites()
    require_mips_prerequisites()

    # ── Step 1: Detect task ──
    print("=" * 60)
    print("AC-Prof Universal Profiler")
    print("=" * 60)

    from acprof.host.detect import detect_task

    task_info = detect_task(
        model_id=args.model,
        override_tag=args.task,
        override_family=args.task_family,
        override_backend=args.backend,
    )
    require_task_support(task_info, batch_size=args.batch_size)

    print(f"\n  Model:    {task_info.model_id}")
    print(f"  Task:     {task_info.pipeline_tag} (family={task_info.task_family})")
    print(f"  Backend:  {task_info.runtime_backend}")
    print(f"  Library:  {task_info.library_name}")
    print(f"  Revision: {task_info.model_revision}")
    print(f"  Detected: {task_info.detection_method}")
    print(f"  Cgroup:   {cgroup_version} (mode={cgroup_collection_mode})")

    output_dir = os.path.join(
        PROJECT_DIR,
        args.output_dir,
        task_info.model_id.replace("/", "--"),
    )
    require_result_cgroup_compatibility(
        output_dir,
        cgroup_version=cgroup_version,
    )

    # ── Step 2: Build Docker image ──
    from acprof.host.docker_runtime import prepare_image
    from acprof.host.input_plan import plan_input_scales, serialize_input_scales
    from acprof.host.orchestrator import merge_all_csvs, run_matrix
    from acprof.host.static_metadata import (
        collect_static_meta,
        enrich_static_meta_from_input_plan,
        enrich_static_meta_from_compute_plan,
        enrich_static_meta_from_execution_plan,
        write_static_meta_json,
    )

    try:
        image_info = prepare_image(
            task_info, PROJECT_DIR, reuse_existing=args.skip_build,
        )
    except (RuntimeError, OSError) as exc:
        print(f"\n[build][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Collect static metadata ──
    cpu_list = _parse_int_list(args.cpus)
    mem_list = _parse_int_list(args.mems)
    gpu_list = _parse_str_list(args.gpus)
    if args.prune_startup_oom:
        if not cpu_list or not mem_list or not gpu_list:
            parser.error("--prune-startup-oom requires non-empty resource lists")
        if len(set(cpu_list)) != len(cpu_list):
            parser.error("--prune-startup-oom requires unique --cpus values")
        if len(set(mem_list)) != len(mem_list):
            parser.error("--prune-startup-oom requires unique --mems values")
        normalized_gpu_modes = [
            "on" if str(gpu).lower() == "on" else "off"
            for gpu in gpu_list
        ]
        if len(set(normalized_gpu_modes)) != len(normalized_gpu_modes):
            parser.error("--prune-startup-oom requires unique --gpus modes")
        if any(cpu <= 0 for cpu in cpu_list):
            parser.error("--prune-startup-oom requires positive --cpus values")
        if any(mem <= 0 for mem in mem_list):
            parser.error("--prune-startup-oom requires positive --mems values")
    massif_selected = args.execution_profile_tool in {"massif", "both"}
    nsys_selected = args.execution_profile_tool in {"nsys", "both"}
    reference_checks = []
    if massif_selected and args.massif_sampling == "per-scale":
        reference_checks.extend(
            [
                ("--massif-reference-cpu", args.massif_reference_cpu, cpu_list),
                ("--massif-reference-mem", args.massif_reference_mem, mem_list),
            ]
        )
    if nsys_selected and args.nsys_sampling != "full":
        reference_checks.append(
            ("--nsys-reference-mem", args.nsys_reference_mem, mem_list)
        )
    if nsys_selected and args.nsys_sampling == "per-scale":
        reference_checks.append(
            ("--nsys-reference-cpu", args.nsys_reference_cpu, cpu_list)
        )
    for option, value, resources in reference_checks:
        if value is not None and value not in resources:
            parser.error(
                f"{option}={value} must be present in the selected resource "
                f"matrix {resources}"
            )

    os.makedirs(output_dir, exist_ok=True)

    static_meta_json = os.path.join(output_dir, "static_meta.json")
    collection_history_json = os.path.join(output_dir, COLLECTION_HISTORY_NAME)
    scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
    input_scale_type = scaling_cfg.param_name if scaling_cfg else ""

    static_meta = collect_static_meta(
        task_info=task_info,
        image_info=image_info,
        batch_size=args.batch_size,
        input_scale_type=input_scale_type,
        run_command=run_command,
        cgroup_version=cgroup_version,
        cgroup_collection_mode=cgroup_collection_mode,
        compute_profile_enabled=not compute_profile_disabled,
        execution_profile_enabled=args.execution_profile_tool != "none",
    )
    write_static_meta_json(static_meta, static_meta_json)
    write_collection_history_json(
        empty_collection_history(),
        collection_history_json,
    )

    # ── Step 4: Run profiling matrix ──
    try:
        planned_input_scales = plan_input_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=args.batch_size,
            output_dir=output_dir,
            input_scales=args.input_scales,
            workload_spec_path=args.workload_spec,
        )
    except Exception as exc:
        print(f"\n[scale][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    input_scales_arg = serialize_input_scales(planned_input_scales.scales)
    static_meta = enrich_static_meta_from_input_plan(
        static_meta,
        planned_input_scales,
    )
    write_static_meta_json(static_meta, static_meta_json)
    compute_profile_plan_file = ""
    if getattr(image_info, "runtime_environment", {}):
        from acprof.host.runtime_validation import validate_runtime
        from acprof.host.static_metadata import enrich_static_meta

        try:
            validation = validate_runtime(
                task_info=task_info, image_info=image_info, planned=planned_input_scales,
                cpu_list=cpu_list, mem_list=mem_list, gpu_list=gpu_list, output_dir=output_dir,
                timeout_seconds=args.request_timeout_seconds,
            )
            static_meta = enrich_static_meta(static_meta, {"runtime_validation": validation})
            write_static_meta_json(static_meta, static_meta_json)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"[runtime-check][ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
    total_cases = len(cpu_list) * len(mem_list) * len(gpu_list)
    _update_run_notification_plan(
        model_id=task_info.model_id,
        output_dir=output_dir,
        total_cases=total_cases,
    )
    if compute_profile_disabled:
        reason = (
            "--no-compute-profile (compatibility alias)"
            if args.no_compute_profile
            else "--compute-profile-tool none"
        )
        print(f"[compute] Compute profiling disabled by {reason}")
    else:
        try:
            from acprof.host.compute_profile import collect_compute_profile_plan

            compute_profile_plan_file = collect_compute_profile_plan(
                task_info=task_info,
                image_tag=image_info.tag,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                output_dir=output_dir,
                input_scale_plan_file=planned_input_scales.plan_file,
                advisor_root=args.advisor_root,
                ncu_root=args.ncu_root,
                advisor_repeat=args.advisor_repeat,
                torch_profiler_repeat=args.torch_profiler_repeat,
                ncu_repeat=args.ncu_repeat,
                keep_profiles=args.keep_compute_profiles,
                compute_profile_cpus=args.compute_profile_cpus,
                compute_profile_mem=args.compute_profile_mem,
                compute_profile_tool=args.compute_profile_tool,
                progress_callback=(
                    _notify_profiler_completion
                    if _ACTIVE_RUN_NOTIFICATION is not None
                    else None
                ),
            )
            static_meta = enrich_static_meta_from_compute_plan(
                static_meta,
                compute_profile_plan_file,
            )
            write_static_meta_json(static_meta, static_meta_json)
        except Exception as exc:
            print(f"[compute][WARN] Compute profiling unavailable: {exc}")

    execution_profile_plan_file = ""
    if args.execution_profile_tool == "none":
        print(
            "[execution-profile] Massif/Nsight Systems profiling disabled "
            "(enable with --execution-profile-tool)"
        )
    else:
        try:
            from acprof.host.execution_profile import (
                collect_execution_profile_plan,
            )

            execution_profile_plan_file = collect_execution_profile_plan(
                task_info=task_info,
                image_tag=image_info.tag,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                output_dir=output_dir,
                input_scale_plan_file=planned_input_scales.plan_file,
                project_dir=PROJECT_DIR,
                tool_mode=args.execution_profile_tool,
                massif_sampling=args.massif_sampling,
                massif_reference_cpu=args.massif_reference_cpu,
                massif_reference_mem=args.massif_reference_mem,
                massif_repeat=args.massif_repeat,
                nsys_sampling=args.nsys_sampling,
                nsys_reference_cpu=args.nsys_reference_cpu,
                nsys_reference_mem=args.nsys_reference_mem,
                nsys_repeat=args.nsys_repeat,
                nsys_root=args.nsys_root,
                keep_profiles=args.keep_execution_profiles,
                progress_callback=(
                    _notify_profiler_completion
                    if _ACTIVE_RUN_NOTIFICATION is not None
                    else None
                ),
            )
            static_meta = enrich_static_meta_from_execution_plan(
                static_meta,
                execution_profile_plan_file,
            )
            write_static_meta_json(static_meta, static_meta_json)
        except Exception as exc:
            print(
                "[execution-profile][WARN] Execution profiling unavailable: "
                f"{exc}"
            )

    n_scales = len(planned_input_scales.scales)
    total_iters = total_cases * n_scales * (args.warmup + args.repeat)

    print(f"\n  Resource matrix: {len(cpu_list)} CPUs x {len(mem_list)} MEMs x {len(gpu_list)} GPUs = {total_cases} cases")
    print(f"  Input scales: {n_scales} levels")
    print(f"  Scale source: {planned_input_scales.source}")
    print(f"  Validated scales: {input_scales_arg}")
    print(f"  Iterations per case: {args.warmup} warmup + {args.repeat} repeat")
    print(
        "  Startup OOM pruning: "
        + (
            f"enabled (reference CPU={min(cpu_list)}, startup-only)"
            if args.prune_startup_oom
            else "disabled"
        )
    )
    if args.repeat_in_window > 0:
        repeat_desc = str(args.repeat_in_window)
    else:
        repeat_desc = f"auto target {args.repeat_window_seconds:.1f}s"
    print(f"  Requests per iteration: {repeat_desc}")
    print(f"  Request timeout: {args.request_timeout_seconds:g}s per /predict")
    print(f"  Total iterations: {total_iters}")
    print(f"  Output: {output_dir}")
    print()

    try:
        csv_paths = run_matrix(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            output_dir=output_dir,
            project_dir=PROJECT_DIR,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeat=args.repeat,
            repeat_in_window=args.repeat_in_window,
            repeat_window_seconds=args.repeat_window_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            sample_hz=args.sample_hz,
            idle_seconds=args.idle_seconds,
            idle_cooldown_seconds=args.idle_cooldown_seconds,
            idle_debug=args.idle_debug,
            sniff_iface=args.sniff_iface,
            input_scales=input_scales_arg,
            input_scale_plan_file=planned_input_scales.plan_file,
            compute_profile_plan_file=compute_profile_plan_file,
            execution_profile_plan_file=execution_profile_plan_file,
            progress_callback=(
                _notify_case_progress
                if _ACTIVE_RUN_NOTIFICATION is not None
                else None
            ),
            prune_startup_oom=args.prune_startup_oom,
        )
    except PacketLatencyError as exc:
        print(f"\n[sniff][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except EnergyProfilingError as exc:
        print(f"\n[energy][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except MIPSProfilingError as exc:
        print(f"\n[mips][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Step 5: Merge all CSVs ──
    if csv_paths:
        final_csv = os.path.join(output_dir, "result_all.csv")
        merge_all_csvs(csv_paths, final_csv)
        _cleanup_intermediate_results(csv_paths, output_dir, final_csv)
        elapsed = _format_elapsed(time.perf_counter() - start_time)
        print(f"\n{'='*60}")
        print(f"Profiling complete!")
        print(f"  Static meta:      {static_meta_json}")
        print(f"  Collection log:   {collection_history_json}")
        if args.prune_startup_oom:
            print(
                "  OOM pruning plan:  "
                f"{os.path.join(output_dir, 'startup_oom_pruning.json')}"
            )
        print(f"  Merged results:   {final_csv}")
        print(f"  Total elapsed:    {elapsed}")
        print(f"  Intermediate files from this run were cleaned up.")
        print(f"{'='*60}")
        _record_run_completion(
            final_csv=final_csv,
            completed_cases=len(csv_paths),
        )
    else:
        elapsed = _format_elapsed(time.perf_counter() - start_time)
        print(f"\n[WARN] No results produced after {elapsed}. Static meta is still available: {static_meta_json}")
        _record_run_completion(final_csv=None, completed_cases=0)


def main():
    """Run profiling, finalize terminal logging, then notify best-effort."""
    global _ACTIVE_TMUX_TERMINAL_LOG, _ACTIVE_RUN_NOTIFICATION

    _ACTIVE_TMUX_TERMINAL_LOG = None
    _ACTIVE_RUN_NOTIFICATION = None
    try:
        return _run_main()
    except TaskSupportError as exc:
        print(str(exc), file=sys.stderr)
        _record_run_termination("failed", str(exc))
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        _record_run_termination("cancelled", "用户中断了采集")
        raise
    except SystemExit as exc:
        if exc.code not in (None, 0):
            _record_run_termination(
                "failed",
                f"采集命令以退出码 {exc.code!r} 结束",
            )
        raise
    except Exception as exc:
        detail = str(exc).strip()
        if detail:
            detail = f"{type(exc).__name__}: {detail}"
        else:
            detail = type(exc).__name__
        _record_run_termination("failed", detail)
        raise
    finally:
        terminal_log = _ACTIVE_TMUX_TERMINAL_LOG
        _ACTIVE_TMUX_TERMINAL_LOG = None
        finalized_log_path = None
        if terminal_log is not None:
            if _stop_tmux_terminal_log(terminal_log):
                finalized_log_path = terminal_log[2]
        _deliver_run_notification(finalized_log_path)
        _ACTIVE_RUN_NOTIFICATION = None


if __name__ == "__main__":
    main()

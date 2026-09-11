"""Orchestration of profiler-only collection and backfill."""
from __future__ import annotations

import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from acprof.host.posthoc.backfill import (
    backfill_rows,
    csv_tool_complete,
    update_collection_history,
    update_static_meta,
)
from acprof.host.posthoc.context import (
    POSTHOC_DIRNAME,
    PosthocError,
    PosthocSummary,
    STATIC_META_NAME,
    SUPPORTED_TOOLS,
    TOOL_GPU_MODES,
    _read_plan,
    load_result_context,
)
from acprof.host.posthoc.plans import (
    _collect_compute_plan,
    _collect_execution_plan,
    _find_reusable_compute_plan,
    _find_reusable_execution_plan,
    _validate_profiler_runtime,
    applicable_tools,
    merge_compute_plans,
    merge_execution_plans,
    parse_tools,
)
from acprof.host.posthoc.storage import (
    PosthocLock,
    _atomic_write_json,
    commit_result_files,
    create_backup,
    find_active_processes,
)


def run_posthoc(
    result_dir: str | os.PathLike[str],
    *,
    tools: str | Iterable[str] = SUPPORTED_TOOLS,
    massif_sampling: str = "per-scale",
    massif_reference_cpu: Optional[int] = None,
    massif_reference_mem: Optional[int] = None,
    nsys_sampling: str = "per-cpu-scale",
    nsys_reference_cpu: Optional[int] = None,
    nsys_reference_mem: Optional[int] = None,
    ncu_root: Optional[str] = None,
    nsys_root: Optional[str] = None,
    torch_repeat: int = 1,
    ncu_repeat: int = 1,
    nsys_repeat: int = 1,
    massif_repeat: int = 1,
    compute_profile_cpus: Optional[int] = None,
    compute_profile_mem: Optional[int] = None,
    force_reprofile: bool = False,
    dry_run: bool = False,
) -> PosthocSummary:
    if massif_sampling not in {"per-scale", "full"}:
        raise PosthocError("massif_sampling must be 'per-scale' or 'full'")
    if nsys_sampling not in {"per-cpu-scale", "per-scale", "full"}:
        raise PosthocError(
            "nsys_sampling must be 'per-cpu-scale', 'per-scale', or 'full'"
        )
    for name, value in (
        ("torch_repeat", torch_repeat),
        ("ncu_repeat", ncu_repeat),
        ("nsys_repeat", nsys_repeat),
        ("massif_repeat", massif_repeat),
    ):
        if int(value) <= 0:
            raise PosthocError(f"{name} must be > 0")
    for name, value in (
        ("massif_reference_cpu", massif_reference_cpu),
        ("massif_reference_mem", massif_reference_mem),
        ("nsys_reference_cpu", nsys_reference_cpu),
        ("nsys_reference_mem", nsys_reference_mem),
    ):
        if value is not None and int(value) <= 0:
            raise PosthocError(f"{name} must be > 0")

    selected = parse_tools(tools)
    directory = Path(result_dir).expanduser().resolve()
    early_meta = _read_plan(directory / STATIC_META_NAME) or {}
    model_id = str(early_meta.get("model_name") or "").strip()
    active = find_active_processes(directory, model_id=model_id)
    if active:
        detail = "\n".join(f"  pid={pid}: {command}" for pid, command in active)
        raise PosthocError(
            "run.py/profiler processes are still using this result directory; "
            f"wait for them to finish:\n{detail}"
        )

    context = load_result_context(directory)
    applicable, inapplicable = applicable_tools(context, selected)
    for tool in inapplicable:
        mode = "/".join(TOOL_GPU_MODES[tool])
        print(
            f"[profile][{tool}] Skipped: result_all.csv has no gpu_mode={mode} rows"
        )
    if not applicable:
        print("[profile] No requested profiler applies to this result CSV.")
        return PosthocSummary(
            result_csv=str(context.result_csv),
            static_meta=str(context.static_meta_path),
            backup_dir=None,
            collected_tools=(),
            reused_tools=(),
            skipped_tools=tuple(inapplicable),
            updated_rows_by_tool={},
        )

    already_complete = tuple(
        tool
        for tool in applicable
        if not force_reprofile and csv_tool_complete(context, tool)
    )
    needed = tuple(tool for tool in applicable if tool not in already_complete)
    for tool in already_complete:
        print(f"[profile][{tool}] Skipped: CSV already has successful values")

    print(f"[profile] Result directory: {context.result_dir}")
    print(f"[profile] Model: {context.task_info.model_id}")
    print(
        "[profile] Resource cases: "
        f"CPU-only={len(context.cases_for_mode('off'))}, "
        f"GPU={len(context.cases_for_mode('on'))}"
    )
    print(f"[profile] Tools requiring work: {', '.join(needed) or 'none'}")
    if dry_run or not needed:
        if dry_run:
            print("[profile] Dry run: no profiler was started and no file was changed.")
        return PosthocSummary(
            result_csv=str(context.result_csv),
            static_meta=str(context.static_meta_path),
            backup_dir=None,
            collected_tools=(),
            reused_tools=(),
            skipped_tools=tuple((*inapplicable, *already_complete)),
            updated_rows_by_tool={tool: 0 for tool in needed},
        )

    workspace = context.result_dir / POSTHOC_DIRNAME
    workspace.mkdir(parents=True, exist_ok=True)
    collected: List[str] = []
    reused: List[str] = []
    plans_by_tool: Dict[str, Dict[str, Any]] = {}
    runtime_validated = False

    with PosthocLock(context.result_dir):
        for tool in needed:
            reusable: Optional[Dict[str, Any]] = None
            if not force_reprofile:
                reusable = (
                    _find_reusable_compute_plan(context, workspace, tool)
                    if tool in {"torch", "ncu"}
                    else _find_reusable_execution_plan(context, workspace, tool)
                )
            if reusable is not None:
                plans_by_tool[tool] = reusable
                reused.append(tool)
                continue

            if not runtime_validated:
                _validate_profiler_runtime(context)
                runtime_validated = True
            print(f"[profile][{tool}] Collecting isolated profiler probes...")
            if tool in {"torch", "ncu"}:
                plan = _collect_compute_plan(
                    context,
                    workspace / tool,
                    tool=tool,
                    ncu_root=ncu_root,
                    torch_repeat=torch_repeat,
                    ncu_repeat=ncu_repeat,
                    compute_profile_cpus=compute_profile_cpus,
                    compute_profile_mem=compute_profile_mem,
                    resume_existing_ncu_profiles=not force_reprofile,
                )
            else:
                plan = _collect_execution_plan(
                    context,
                    workspace / tool,
                    tool=tool,
                    massif_sampling=massif_sampling,
                    massif_reference_cpu=massif_reference_cpu,
                    massif_reference_mem=massif_reference_mem,
                    nsys_sampling=nsys_sampling,
                    nsys_reference_cpu=nsys_reference_cpu,
                    nsys_reference_mem=nsys_reference_mem,
                    massif_repeat=massif_repeat,
                    nsys_repeat=nsys_repeat,
                    nsys_root=nsys_root,
                    resume_existing_profiles=not force_reprofile,
                )
            plans_by_tool[tool] = plan
            collected.append(tool)

        compute_tool_plans = {
            tool: plan
            for tool, plan in plans_by_tool.items()
            if tool in {"torch", "ncu"}
        }
        compute_plan = (
            merge_compute_plans(context, compute_tool_plans)
            if compute_tool_plans
            else None
        )
        execution_tool_plans = {
            tool: plan
            for tool, plan in plans_by_tool.items()
            if tool in {"massif", "nsys"}
        }
        execution_plan = (
            merge_execution_plans(context, execution_tool_plans)
            if execution_tool_plans
            else None
        )
        if compute_plan is not None:
            _atomic_write_json(workspace / "compute_profile_plan.json", compute_plan)
        if execution_plan is not None:
            _atomic_write_json(
                workspace / "execution_profile_plan.json", execution_plan
            )

        fieldnames, rows, updated_rows = backfill_rows(
            context,
            tools=needed,
            compute_plan=compute_plan,
            execution_plan=execution_plan,
            force=force_reprofile,
        )
        backup_dir = create_backup(context)
        static_meta = update_static_meta(
            context,
            tools=needed,
            compute_plan=compute_plan,
            execution_plan=execution_plan,
        )
        collection_history = update_collection_history(
            context,
            tools=needed,
            backup_dir=backup_dir,
            massif_sampling=massif_sampling,
            nsys_sampling=nsys_sampling,
        )
        commit_result_files(
            context,
            fieldnames=fieldnames,
            rows=rows,
            static_meta=static_meta,
            collection_history=collection_history,
            backup_dir=backup_dir,
        )

    print(f"[profile] Updated in place: {context.result_csv}")
    print(f"[profile] Updated in place: {context.static_meta_path}")
    print(f"[profile] Updated in place: {context.collection_history_path}")
    print(f"[profile] Original files backed up to: {backup_dir}")
    for tool, count in updated_rows.items():
        print(f"[profile][{tool}] Backfilled rows: {count}")
    return PosthocSummary(
        result_csv=str(context.result_csv),
        static_meta=str(context.static_meta_path),
        backup_dir=str(backup_dir),
        collected_tools=tuple(collected),
        reused_tools=tuple(reused),
        skipped_tools=tuple((*inapplicable, *already_complete)),
        updated_rows_by_tool=updated_rows,
    )

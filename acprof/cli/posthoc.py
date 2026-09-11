"""Compatibility entry point for profiler-only collection and backfill."""
from __future__ import annotations

import argparse

from typing import Optional, Sequence

from acprof.host.collection_history import (
    COLLECTION_HISTORY_NAME,
    LEGACY_STATIC_META_COLLECTION_FIELDS,
    append_collection_record,
    migrate_legacy_static_meta_history,
    normalize_collection_history,
)
from acprof.host.compute_profile_plan import (
    INPUT_SCALE_ABS_TOLERANCE,
    NCU_ERROR_FIELD,
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_PROFILE_KEY,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    NCU_TOTAL_MFLOP_FIELD,
    TORCH_ERROR_FIELD,
    TORCH_LOGICAL_MFLOP_FIELD,
    TORCH_PROFILE_KEY,
    compute_mflops,
    find_compute_profile_entry,
)
from acprof.host.detect import TaskInfo
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.execution_profile_plan import (
    MASSIF_ERROR_FIELD,
    MASSIF_HEAP_EXTRA_PEAK_FIELD,
    MASSIF_HEAP_PEAK_FIELD,
    MASSIF_HEAP_PEAK_TOTAL_FIELD,
    MASSIF_METRIC_FIELDS,
    MASSIF_PEAK_AT_MS_FIELD,
    MASSIF_STACK_PEAK_FIELD,
    NSYS_CUDA_API_CALL_COUNT_FIELD,
    NSYS_CUDA_API_TIME_FIELD,
    NSYS_ERROR_FIELD,
    NSYS_GPU_KERNEL_LAUNCH_COUNT_FIELD,
    NSYS_GPU_KERNEL_TIME_FIELD,
    NSYS_GPU_MEMCPY_BYTES_FIELD,
    NSYS_GPU_MEMCPY_COUNT_FIELD,
    NSYS_GPU_MEMCPY_TIME_FIELD,
    NSYS_HOST_WALL_TIME_FIELD,
    NSYS_METRIC_FIELDS,
    find_execution_profile_entry,
)
from acprof.host.posthoc.backfill import (
    _backfill_execution_row,
    _backfill_ncu_row,
    _backfill_torch_row,
    _merge_provenance,
    _normalize_tool_metadata,
    _packet_mflops,
    _row_tool_complete,
    _static_flops_from_compute_plan,
    backfill_rows,
    csv_tool_complete,
    update_collection_history,
    update_static_meta,
)
from acprof.host.posthoc.context import (
    BACKUP_DIRNAME,
    COMPUTE_PLAN_METRIC_FIELDS,
    INPUT_SCALE_PLAN_NAME,
    LOCK_FILENAME,
    MASSIF_FIELDS,
    NCU_DERIVED_APP_FIELD,
    NCU_DERIVED_PACKET_FIELD,
    NCU_FIELDS,
    NSYS_FIELDS,
    POSTHOC_DIRNAME,
    PROJECT_DIR,
    PosthocError,
    PosthocSummary,
    RESULT_CSV_NAME,
    ResultContext,
    STATIC_META_NAME,
    SUPPORTED_TOOLS,
    TOOL_ERROR_FIELD,
    TOOL_FIELDS,
    TOOL_GPU_MODES,
    TOOL_METRIC_FIELDS,
    TORCH_FIELDS,
    _csv_encoding,
    _finite_float,
    _fmt_float,
    _input_plan_scales,
    _integer,
    _load_json_object,
    _load_result_csv,
    _read_plan,
    _scale_is_present,
    _unique_scales,
    load_result_context,
)
from acprof.host.posthoc.plans import (
    _collect_compute_plan,
    _collect_execution_plan,
    _compute_plan_candidates,
    _execution_plan_candidates,
    _find_reusable_compute_plan,
    _find_reusable_execution_plan,
    _plan_matches_context,
    _resolve_massif_reference,
    _validate_profiler_runtime,
    applicable_tools,
    compute_plan_covers_ncu,
    compute_plan_covers_tool,
    execution_plan_covers_tool,
    expand_representative_massif_plan,
    merge_compute_plans,
    merge_execution_plans,
    parse_tools,
)
from acprof.host.posthoc.service import run_posthoc
from acprof.host.posthoc.storage import (
    PosthocLock,
    _atomic_write_json,
    _option_value,
    _read_process_cmdline,
    _restore_from_backup,
    _timestamp_token,
    _write_csv_temporary,
    _write_json_temporary,
    commit_result_files,
    create_backup,
    find_active_processes,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect missing Torch/NCU/Nsys/Massif metrics for an existing AC-Prof "
            "result directory and safely backfill result_all.csv in place."
        )
    )
    parser.add_argument(
        "result_dir",
        help=(
            "Completed model result directory containing result_all.csv, "
            "static_meta.json, and input_scale_plan.json"
        ),
    )
    parser.add_argument(
        "--tools",
        default=",".join(SUPPORTED_TOOLS),
        help="Comma-separated tools (default: torch,ncu,nsys,massif)",
    )
    parser.add_argument(
        "--massif-sampling",
        choices=("per-scale", "full"),
        default="per-scale",
        help=(
            "per-scale profiles one representative CPU/memory case per input "
            "scale and reuses it across CPU rows (default); full profiles the "
            "entire CPU/memory matrix"
        ),
    )
    parser.add_argument("--massif-reference-cpu", type=int, default=None)
    parser.add_argument("--massif-reference-mem", type=int, default=None)
    parser.add_argument(
        "--nsys-sampling",
        choices=("per-cpu-scale", "per-scale", "full"),
        default="per-cpu-scale",
        help=(
            "per-cpu-scale profiles every CPU at one representative memory "
            "per input scale (default); per-scale profiles one representative "
            "CPU/memory case; full profiles the entire CPU/memory matrix"
        ),
    )
    parser.add_argument("--nsys-reference-cpu", type=int, default=None)
    parser.add_argument("--nsys-reference-mem", type=int, default=None)
    parser.add_argument("--ncu-root", default=None)
    parser.add_argument("--nsys-root", default=None)
    parser.add_argument(
        "--torch-profiler-repeat",
        "--torch-repeat",
        dest="torch_repeat",
        type=int,
        default=1,
    )
    parser.add_argument("--ncu-repeat", type=int, default=1)
    parser.add_argument("--nsys-repeat", type=int, default=1)
    parser.add_argument("--massif-repeat", type=int, default=1)
    parser.add_argument("--compute-profile-cpus", type=int, default=None)
    parser.add_argument("--compute-profile-mem", type=int, default=None)
    parser.add_argument(
        "--force-reprofile",
        action="store_true",
        help="Collect again and replace even already-successful profiler fields",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and show which tools need work without collecting",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    bootstrap_project_env(PROJECT_DIR)
    try:
        run_posthoc(
            args.result_dir,
            tools=args.tools,
            massif_sampling=args.massif_sampling,
            massif_reference_cpu=args.massif_reference_cpu,
            massif_reference_mem=args.massif_reference_mem,
            nsys_sampling=args.nsys_sampling,
            nsys_reference_cpu=args.nsys_reference_cpu,
            nsys_reference_mem=args.nsys_reference_mem,
            ncu_root=args.ncu_root,
            nsys_root=args.nsys_root,
            torch_repeat=args.torch_repeat,
            ncu_repeat=args.ncu_repeat,
            nsys_repeat=args.nsys_repeat,
            massif_repeat=args.massif_repeat,
            compute_profile_cpus=args.compute_profile_cpus,
            compute_profile_mem=args.compute_profile_mem,
            force_reprofile=args.force_reprofile,
            dry_run=args.dry_run,
        )
    except (OSError, PosthocError) as exc:
        parser.exit(2, f"[profile][ERROR] {exc}\n")


if __name__ == "__main__":
    main()

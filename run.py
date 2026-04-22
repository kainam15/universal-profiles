#!/usr/bin/env python3
"""AC-Prof Universal Profiler - One-click HuggingFace model profiling.

Usage:
    python run.py --model bert-base-uncased
    python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
    python run.py --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from config import SCALING_DIMENSIONS
from env_utils import bootstrap_project_env

PROJECT_DIR = str(Path(__file__).resolve().parent)


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


def main():
    start_time = time.perf_counter()
    bootstrap_project_env(PROJECT_DIR)

    parser = argparse.ArgumentParser(
        description="AC-Prof: Universal HuggingFace Model Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --model bert-base-uncased
  python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
  python run.py --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
        """,
    )

    # Required
    parser.add_argument("--model", required=True, help="HuggingFace model ID")

    # Detection overrides
    parser.add_argument("--task", default=None, help="Override pipeline_tag (e.g., text-generation)")
    parser.add_argument("--task-family", default=None, help="Override task family (nlp/cv/audio/timeseries)")
    parser.add_argument("--backend", default=None, help="Override runtime backend (transformers_pipeline/chronos/...)")

    # Resource matrix
    parser.add_argument("--cpus", default="1,2,4,8", help="CPU core counts (comma-separated)")
    parser.add_argument("--mems", default="2,4,8,16", help="Memory caps in GB (comma-separated)")
    parser.add_argument("--gpus", default="off,on", help="GPU modes (comma-separated: off,on)")

    # Experiment parameters
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="Measurement repeat count")
    parser.add_argument("--repeat-in-window", type=int, default=20, help="Requests per energy window")
    parser.add_argument("--sample-hz", type=float, default=20.0, help="GPU energy sampling rate")
    parser.add_argument("--idle-seconds", type=float, default=3.0, help="GPU idle baseline duration")
    parser.add_argument("--input-scales", default=None, help="Override input scale values (comma-separated)")

    # Compute profiling
    parser.add_argument("--no-compute-profile", action="store_true", help="Disable Intel Advisor / ncu MFLOPS profiling")
    parser.add_argument("--advisor-root", default=None, help="Host Intel Advisor install root or advisor executable")
    parser.add_argument("--ncu-root", default=None, help="Host Nsight Compute install root or ncu executable")
    parser.add_argument("--advisor-repeat", type=int, default=20, help="Intel Advisor profiled inference repetitions")
    parser.add_argument("--ncu-repeat", type=int, default=1, help="ncu profiled inference repetitions")
    parser.add_argument("--compute-profile-cpus", type=int, default=None, help="CPU cores for temporary compute profiler containers (default: host logical CPUs)")
    parser.add_argument("--compute-profile-mem", type=int, default=None, help="Memory GB for temporary compute profiler containers (default: 75%% of host memory)")
    parser.add_argument("--keep-compute-profiles", action="store_true", help="Keep raw Advisor/ncu profiler artifacts")

    # Infrastructure
    parser.add_argument("--sniff-iface", default="docker0", help="Network interface for tcpdump")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker image build (use existing)")

    args = parser.parse_args()

    # ── Step 1: Detect task ──
    print("=" * 60)
    print("AC-Prof Universal Profiler")
    print("=" * 60)

    from detect import detect_task

    task_info = detect_task(
        model_id=args.model,
        override_tag=args.task,
        override_family=args.task_family,
        override_backend=args.backend,
    )

    print(f"\n  Model:    {task_info.model_id}")
    print(f"  Task:     {task_info.pipeline_tag} (family={task_info.task_family})")
    print(f"  Backend:  {task_info.runtime_backend}")
    print(f"  Library:  {task_info.library_name}")
    print(f"  Revision: {task_info.model_revision}")
    print(f"  Detected: {task_info.detection_method}")

    # ── Step 2: Build Docker image ──
    from orchestrator import (
        build_image,
        collect_static_meta,
        merge_all_csvs,
        plan_input_scales,
        run_matrix,
        serialize_input_scales,
        write_static_meta_csv,
    )

    if args.skip_build:
        from orchestrator import ImageInfo
        model_tag = task_info.model_id.replace("/", "--").replace(".", "_").lower()
        tag = f"acprof-{task_info.task_family}-{model_tag}:latest"
        image_info = ImageInfo(tag=tag)
        print(f"\n[build] Skipping build, using: {tag}")
    else:
        image_info = build_image(task_info, PROJECT_DIR)

    # ── Step 3: Collect static metadata ──
    cpu_list = _parse_int_list(args.cpus)
    mem_list = _parse_int_list(args.mems)
    gpu_list = _parse_str_list(args.gpus)

    output_dir = os.path.join(PROJECT_DIR, args.output_dir, task_info.model_id.replace("/", "--"))
    os.makedirs(output_dir, exist_ok=True)

    static_meta_csv = os.path.join(output_dir, "static_meta.csv")
    scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
    input_scale_type = scaling_cfg.param_name if scaling_cfg else ""

    static_meta = collect_static_meta(
        task_info=task_info,
        image_info=image_info,
        batch_size=args.batch_size,
        input_scale_type=input_scale_type,
    )
    write_static_meta_csv(static_meta, static_meta_csv)

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
        )
    except Exception as exc:
        print(f"\n[scale][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    input_scales_arg = serialize_input_scales(planned_input_scales.scales)
    compute_profile_plan_file = ""
    if args.no_compute_profile:
        print("[compute] Compute profiling disabled by --no-compute-profile")
    else:
        try:
            from compute_profile import collect_compute_profile_plan

            compute_profile_plan_file = collect_compute_profile_plan(
                task_info=task_info,
                image_tag=image_info.tag,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                batch_size=args.batch_size,
                output_dir=output_dir,
                input_scale_plan_file=planned_input_scales.plan_file,
                input_scales=input_scales_arg,
                advisor_root=args.advisor_root,
                ncu_root=args.ncu_root,
                advisor_repeat=args.advisor_repeat,
                ncu_repeat=args.ncu_repeat,
                keep_profiles=args.keep_compute_profiles,
                compute_profile_cpus=args.compute_profile_cpus,
                compute_profile_mem=args.compute_profile_mem,
            )
        except Exception as exc:
            print(f"[compute][WARN] Compute profiling unavailable: {exc}")

    total_cases = len(cpu_list) * len(mem_list) * len(gpu_list)
    n_scales = len(planned_input_scales.scales)
    total_iters = total_cases * n_scales * (args.warmup + args.repeat)

    print(f"\n  Resource matrix: {len(cpu_list)} CPUs x {len(mem_list)} MEMs x {len(gpu_list)} GPUs = {total_cases} cases")
    print(f"  Input scales: {n_scales} levels")
    print(f"  Scale source: {planned_input_scales.source}")
    print(f"  Validated scales: {input_scales_arg}")
    print(f"  Iterations per case: {args.warmup} warmup + {args.repeat} repeat")
    print(f"  Requests per iteration: {args.repeat_in_window}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Output: {output_dir}")
    print()

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
        sample_hz=args.sample_hz,
        idle_seconds=args.idle_seconds,
        sniff_iface=args.sniff_iface,
        input_scales=input_scales_arg,
        input_scale_plan_file=planned_input_scales.plan_file,
        compute_profile_plan_file=compute_profile_plan_file,
    )

    # ── Step 5: Merge all CSVs ──
    if csv_paths:
        final_csv = os.path.join(output_dir, "result_all.csv")
        merge_all_csvs(csv_paths, final_csv)
        _cleanup_intermediate_results(csv_paths, output_dir, final_csv)
        elapsed = _format_elapsed(time.perf_counter() - start_time)
        print(f"\n{'='*60}")
        print(f"Profiling complete!")
        print(f"  Static meta:      {static_meta_csv}")
        print(f"  Merged results:   {final_csv}")
        print(f"  Total elapsed:    {elapsed}")
        print(f"  Intermediate files from this run were cleaned up.")
        print(f"{'='*60}")
    else:
        elapsed = _format_elapsed(time.perf_counter() - start_time)
        print(f"\n[WARN] No results produced after {elapsed}. Static meta is still available: {static_meta_csv}")


if __name__ == "__main__":
    main()

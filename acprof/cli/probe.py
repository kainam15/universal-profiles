#!/usr/bin/env python3
"""CLI for a one-request largest-input-scale timing probe."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Sequence

from acprof.cli.run import (
    require_cgroup_prerequisites,
    require_native_docker,
    require_native_linux_host,
)
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.largest_scale_probe import (
    create_probe_output_dir,
    run_largest_scale_probe,
    write_probe_summary,
)
from acprof.host.orchestrator import (
    ImageInfo,
    build_image,
    plan_input_scales,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]


def _int_list(value: str, label: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label}必须是逗号分隔的整数"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{label}必须全部大于 0")
    return values


def _gpu_list(value: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values or any(item not in {"off", "on"} for item in values):
        raise argparse.ArgumentTypeError("GPU 模式只能包含 off 或 on")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan requested memory caps from low to high, then measure the "
            "first cap that completes one largest-scale cold request."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--task", default=None, help="Override pipeline_tag")
    parser.add_argument("--task-family", default=None, help="Override task family")
    parser.add_argument("--backend", default=None, help="Override runtime backend")
    parser.add_argument("--cpus", default="1,2,4,8", help="CPU list")
    parser.add_argument(
        "--mems",
        default="2,4,8,16",
        help="Memory GB candidates scanned from low to high",
    )
    parser.add_argument("--gpus", default="off,on", help="GPU modes")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--input-scales",
        default=None,
        help="Optional input scales; the largest value is probed",
    )
    parser.add_argument(
        "--workload-spec",
        default=None,
        help="Optional audio workload manifest",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help=(
            "Optional timeout for the one /predict request; omitted means "
            "wait indefinitely"
        ),
    )
    parser.add_argument("--output-dir", default="results", help="Output root")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--allow-cgroup-v1", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.timeout_seconds is not None and (
        args.timeout_seconds <= 0.0 or not math.isfinite(args.timeout_seconds)
    ):
        parser.error("--timeout-seconds must be a finite value > 0")
    try:
        cpu_list = _int_list(args.cpus, "CPU 列表")
        mem_list = _int_list(args.mems, "内存列表")
        gpu_list = _gpu_list(args.gpus)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    command_started = time.perf_counter()
    bootstrap_project_env(str(PROJECT_DIR))
    require_native_linux_host()
    require_native_docker()
    cgroup_version = require_cgroup_prerequisites(
        allow_cgroup_v1=args.allow_cgroup_v1
    )

    from acprof.host.detect import detect_task

    task_info = detect_task(
        model_id=args.model,
        override_tag=args.task,
        override_family=args.task_family,
        override_backend=args.backend,
    )
    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root
    output_dir = create_probe_output_dir(output_root, task_info.model_id)

    print("=" * 60)
    print("AC-Prof Minimum Viable Memory + Largest Input Scale Probe")
    print("=" * 60)
    print(f"  Model:    {task_info.model_id}")
    print(
        f"  Task:     {task_info.pipeline_tag} "
        f"(family={task_info.task_family})"
    )
    print(f"  Backend:  {task_info.runtime_backend}")
    print(f"  Cgroup:   {cgroup_version}")
    print(f"  Output:   {output_dir}")

    if args.skip_build:
        model_tag = task_info.model_id.replace("/", "--").replace(".", "_").lower()
        image_info = ImageInfo(
            tag=f"acprof-{task_info.task_family}-{model_tag}:latest"
        )
        print(f"[build] Skipping build, using: {image_info.tag}")
    else:
        image_info = build_image(task_info, str(PROJECT_DIR))

    try:
        planned = plan_input_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=args.batch_size,
            output_dir=str(output_dir),
            input_scales=args.input_scales,
            workload_spec_path=args.workload_spec,
        )
        summary = run_largest_scale_probe(
            task_info=task_info,
            image_info=image_info,
            planned_input_scales=planned,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=args.batch_size,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"[largest-probe][ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary["timing"]["command_s"] = time.perf_counter() - command_started
    write_probe_summary(summary["artifacts"]["summary"], summary)
    print(
        "[largest-probe] Total command elapsed: "
        f"{summary['timing']['command_s']:.3f}s"
    )
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

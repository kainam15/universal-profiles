"""Compatibility entry point for profiler-only collection and backfill."""
from __future__ import annotations

import argparse

from typing import Optional, Sequence

from acprof.host.env_utils import bootstrap_project_env
from acprof.host.posthoc.context import PROJECT_DIR, PosthocError, SUPPORTED_TOOLS
from acprof.host.posthoc.service import run_posthoc


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

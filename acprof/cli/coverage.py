"""Freeze a Hub sample or report resolution and isolated execution coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="Freeze revisions and download weights for a rolling sample")
    snapshot.add_argument("--stratum", action="append", required=True, help="TASK:LIBRARY; repeat for multiple strata")
    snapshot.add_argument("--limit", type=int, default=5, help="Head models per stratum")
    snapshot.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Evaluate an existing frozen sample")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output-dir", type=Path, required=True, help="New report directory")
    run.add_argument("--probe", choices=("none", "full"), default="none")
    run.add_argument("--cpus", type=int, default=2)
    run.add_argument("--mems", type=int, default=4)
    run.add_argument("--gpus", choices=("off", "on"), default="off")
    run.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.model_coverage import run_sample, snapshot_sample
    bootstrap_project_env(Path.cwd())
    try:
        if args.command == "snapshot":
            if args.output.exists():
                raise ValueError("sample already exists; choose a new output path")
            sample = snapshot_sample(args.stratum, args.limit)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(sample, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            print(f"Frozen sample: {args.output}")
        else:
            report = run_sample(json.loads(args.manifest.read_text()), args.output_dir, probe=args.probe,
                                 cpus=args.cpus, memory_gb=args.mems, gpu=args.gpus == "on", timeout_seconds=args.timeout_seconds)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"[coverage][ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

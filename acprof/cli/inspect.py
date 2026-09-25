"""Explain model resolution and optionally validate it before any measurement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model ID")
    parser.add_argument("--model-spec", help="Explicit local model declaration")
    parser.add_argument("--expected-revision", help="Refuse a model commit that changed since review")
    parser.add_argument("--task", help="Explicit task selection")
    parser.add_argument("--backend", help="Explicit backend selection")
    parser.add_argument("--explain", action="store_true", help="Show field values and evidence sources")
    parser.add_argument("--output-dir", type=Path, help="Export model_resolution.json and probe evidence")
    parser.add_argument("--probe", choices=("none", "basic", "full"), default="none",
                        help="basic imports/checks signatures; full runs one minimal inference in an isolated container")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--mems", type=int, default=4, help="Probe memory limit in GiB")
    parser.add_argument("--gpus", choices=("off", "on"), default="off")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--skip-build", action="store_true", help="Reuse a matching image when available")
    args = parser.parse_args(argv)
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.detect import detect_task
    from acprof.host.model_inspection import explain_resolution, probe_model_contract
    from acprof.model_contract import write_model_resolution

    bootstrap_project_env(Path.cwd())
    task = detect_task(args.model, model_spec_path=args.model_spec, override_tag=args.task, override_backend=args.backend)
    if args.expected_revision and task.model_revision != args.expected_revision:
        print("[contract-probe][ERROR] Model revision changed; resolve and review it again", file=sys.stderr)
        return 2
    output = args.output_dir
    if args.probe != "none" and output is None:
        output = Path("results/inspection") / task.model_id.replace("/", "--") / uuid.uuid4().hex[:12]
    if args.probe != "none" and (output / "runtime_validation.json").exists():
        print("[contract-probe][ERROR] Choose a new output directory for this probe", file=sys.stderr)
        return 2
    if args.expected_revision and args.model_spec and output is not None:
        # The TUI hands its reviewed declaration and provenance to a subprocess.
        # Preserve that explanation only for the same pinned model and exact spec;
        # a prior runtime observation never becomes evidence for this new probe.
        from acprof.model_evidence import RESOLVER_VERSION
        from acprof.model_spec import task_model_spec
        previous = output / "model_resolution.json"
        if previous.is_file() and previous.stat().st_size <= 4 * 1024 * 1024:
            try:
                contract = json.loads(previous.read_text()).get("contract", {})
                if (contract.get("model_id") == task.model_id and contract.get("revision") == task.model_revision
                        and contract.get("resolver_version") == RESOLVER_VERSION and contract.get("status") == "resolved"
                        and contract.get("draft_spec") == task_model_spec(task)):
                    contract["runtime_validation"] = "not_run"
                    contract["fields"] = {key: value for key, value in contract["fields"].items() if not key.startswith("runtime.")}
                    task.model_resolution["contract"] = contract
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass
    if output is not None:
        write_model_resolution(task, output)
    print(explain_resolution(task, explain=args.explain), flush=True)
    if task.model_resolution.get("status") in {"ambiguous", "needs_configuration"}:
        return 2
    if args.probe != "none":
        try:
            report = probe_model_contract(task, output, mode=args.probe, cpus=args.cpus, memory_gb=args.mems,
                                          gpu=args.gpus == "on", timeout_seconds=args.timeout_seconds,
                                          reuse_existing=args.skip_build)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"[contract-probe][ERROR] {exc}", file=sys.stderr)
            return 1
        print(explain_resolution(task), flush=True)
        print(f"Evidence: {output}", flush=True)
        return 0 if report["status"] == "ok" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

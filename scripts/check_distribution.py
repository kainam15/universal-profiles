"""Smoke-test an installed wheel or standalone binary from an empty directory."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, help="默认使用当前 Python 的已安装 acprof")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    prefix = [str(args.binary.resolve())] if args.binary else [sys.executable, "-m", "acprof"]
    evidence = []
    with tempfile.TemporaryDirectory(prefix="acprof-install-check-") as temporary:
        workspace = Path(temporary)
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        environment.update({"XDG_CONFIG_HOME": str(workspace / "config"), "MPLBACKEND": "Agg"})

        def run(arguments, *, accepted=(0,), env=None):
            result = subprocess.run([*prefix, *arguments], cwd=workspace,
                                    env=env or environment, text=True, capture_output=True, timeout=90)
            evidence.append({"arguments": arguments, "returncode": result.returncode})
            if result.returncode not in accepted:
                raise RuntimeError(f"{arguments}: exit={result.returncode}\n{result.stdout}\n{result.stderr}")
            return result

        run(["--version"])
        run(["--help"])
        for command in ("run", "probe", "plot", "tui", "doctor", "profile", "audit", "stats", "inspect", "auto", "coverage"):
            run([command, "--help"])
        run(["invalid-command"], accepted=(2,))
        # Simulate a machine without Docker; JSON must still include valid bundled resources.
        doctor = run(["doctor", "--profiling-mode", "basic", "--json"], accepted=(1,),
                     env={**environment, "PATH": ""})
        report = json.loads(doctor.stdout)
        assert not report["ready"] and report["scope"] == "prerequisites_only", report
        assert next(item for item in report["checks"] if item["name"] == "resources")["status"] == "available", report
        assert all(path.name == "config" for path in workspace.iterdir()), "Help/doctor created result files"

        # Exercise the actual child-process dispatcher, not just its command construction.
        source = workspace / "case.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sniff_group_id", "latency_s", "batch_size"])
            writer.writeheader()
            writer.writerow({"sniff_group_id": "req", "latency_s": "nan", "batch_size": "1"})
        (workspace / "static_meta.json").write_text(json.dumps({"schema_version": 7, "batch_size": 1}))
        metrics = workspace / "packet.json"
        metrics.write_text(json.dumps({"schema_version": 2, "requests": {"req:1": {"latency_s": 0.25}}}))
        merged = workspace / "merged.csv"
        run(["_worker", "acprof.packet.merge_packet_latency", str(source), str(metrics), str(merged)])
        with merged.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and float(rows[0]["latency_s"]) == 0.25, rows
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"successful": True, "checks": evidence}, indent=2) + "\n")
    print(f"Distribution smoke passed: {len(evidence)} checks from an empty workspace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

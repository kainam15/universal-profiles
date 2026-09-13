"""在专用 Linux/Docker 主机执行小型真实采集，保存命令、日志和审计结果。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.analysis.audit import audit_result, number
from acprof.artifacts import atomic_write_json
from acprof.tui.commands import RunConfig, build_run_command
from acprof.result_csv import read_result_csv


def run_config(config: RunConfig, output: Path, *, ui: str = "cli") -> dict:
    if ui == "terminal" and not sys.stdout.isatty():
        raise ValueError("terminal 对照须在终端中运行；自动化可用 script 分配 PTY 并保存终端输出")
    command = build_run_command(config, project_dir=ROOT, python_executable=sys.executable)
    environment = os.environ.copy()
    environment.update(ACPROF_WECOM_WEBHOOK_URL="", PYTHONUNBUFFERED="1")
    environment.pop("TMUX", None)
    environment.pop("TMUX_PANE", None)
    atomic_write_json(output / "command.json", {"command": command, "config": asdict(config), "ui": ui})
    launch = command if ui == "cli" else [sys.executable, str(ROOT / "scripts/run_tui_validation.py"),
                                          str(output / "command.json"), "--ui", ui]
    if ui == "terminal":
        (output / "run.log").write_text("TUI 渲染输出到当前终端；可用 script 保存会话。\n")
        code = subprocess.run(launch, cwd=ROOT, env=environment).returncode
    else:
        with (output / "run.log").open("w") as log:
            code = subprocess.run(launch, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    audit = audit_result(config.result_dir(ROOT))
    audit["exit_code"] = code
    audit["successful"] = code == 0 and audit["valid"] and audit["completion"] == "complete"
    counts = audit["counts"]
    audit["successful"] &= counts["formal_ok"] > 0 and counts["formal_ok"] == counts["rows"] - counts["warmup"]
    unavailable = []
    if audit["valid"]:
        _, rows = read_result_csv(config.result_csv(ROOT))
        for index, row in enumerate(rows, 2):
            if row["warmup"] == "0":
                required = ["latency_s", "latency_app_s", "cpu_energy_total_j", "cpu_instructions_per_request"]
                if row["gpu_mode"] == "on":
                    required.append("gpu_energy_total_j")
                unavailable.extend({"row": index, "field": name} for name in required if number(row.get(name)) is None)
    audit["required_metrics_missing"] = unavailable
    audit["successful"] &= not unavailable
    atomic_write_json(output / "audit.json", audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--gpus", default="off,on")
    parser.add_argument("--cpus", default="2")
    parser.add_argument("--mems", default="8")
    parser.add_argument("--input-scales", default="64")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--repeat-window-seconds", type=float, default=2.0)
    parser.add_argument("--idle-seconds", type=float, default=2.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--compute-profile-tool", choices=("none", "torch", "ncu", "both"), default="none")
    parser.add_argument("--execution-profile-tool", choices=("none", "massif", "nsys", "both"), default="none")
    parser.add_argument("--ui", choices=("cli", "headless", "terminal"), default="cli")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("硬件验证输出目录必须为空")
    config = RunConfig(model=args.model, task=args.task, cpus=args.cpus, mems=args.mems, gpus=args.gpus,
                       input_scales=args.input_scales, output_dir=str(output / "results"),
                       warmup=1, repeat=args.repeat, repeat_window_seconds=args.repeat_window_seconds,
                       idle_seconds=args.idle_seconds, idle_cooldown_seconds=1.0, sample_hz=args.sample_hz,
                       compute_profile_tool=args.compute_profile_tool, execution_profile_tool=args.execution_profile_tool,
                       notify="none", skip_build=True).validate(project_dir=ROOT)
    report = run_config(config, output, ui=args.ui)
    print(f"硬件验证{'通过' if report['successful'] else '未通过'}，证据：{output}")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

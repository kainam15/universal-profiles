"""以相同采集命令对照 CLI 和 TUI，每次使用独立实验目录。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.artifacts import atomic_write_json
from acprof.analysis.uncertainty import bootstrap_mean_interval
from acprof.result_csv import read_result_csv
from acprof.tui.commands import RunConfig
from scripts.check_hardware import run_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", type=Path, help="check_hardware.py 产生的 command.json")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ui", choices=("headless", "terminal"), default="headless")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rounds < 3:
        parser.error("需要至少 3 个独立实验对")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("对照输出目录必须为空")
    config = RunConfig(**json.loads(args.command.read_text())["config"])
    if any("," in str(value) for value in (config.cpus, config.mems, config.gpus, config.input_scales)):
        parser.error("UI 对照每次只选择一个 CPU/内存/设备/尺度 case，避免混合不同工作负载")
    config = replace(config, notify="none", skip_build=True, compute_profile_tool="none", execution_profile_tool="none")
    rng = random.Random(args.seed)
    results = {"schema_version": 1, "kind": "ui_overhead", "ui": args.ui, "seed": args.seed,
               "successful": False, "pairs": [],
               "scope": "不同容器运行的配对实验；headless 不包含终端渲染；terminal 包含 TUI 输出，终端模拟器须另行注明"}
    identity = None
    try:
        for index in range(args.rounds):
            order = ["cli", args.ui]
            rng.shuffle(order)
            pair = {"round": index, "order": order, "results": {}}
            for ui in order:
                destination = output / f"round-{index}-{ui}"
                destination.mkdir()
                candidate = replace(config, output_dir=str(destination / "results"))
                audit = run_config(candidate, destination, ui=ui)
                if not audit["successful"]:
                    raise RuntimeError(f"{ui} 采集未通过审计，停止对照：{destination}")
                result_dir = candidate.result_dir(ROOT)
                meta = json.loads((result_dir / "static_meta.json").read_text())
                observed = (meta["image_id"], meta["model_revision"], meta["input_scale_plan_sha256"])
                if identity is not None and observed != identity:
                    raise RuntimeError("UI 对照的镜像、revision 或输入计划不一致")
                identity = observed
                _, rows = read_result_csv(candidate.result_csv(ROOT))
                values = [float(row["latency_app_s"]) for row in rows if row["warmup"] == "0" and row["status"] == "ok"]
                pair["results"][ui] = {"mean_latency_app_s": statistics.fmean(values), "n_windows": len(values),
                                        "result_dir": str(result_dir)}
            pair["change_pct"] = (pair["results"][args.ui]["mean_latency_app_s"] / pair["results"]["cli"]["mean_latency_app_s"] - 1) * 100
            results["pairs"].append(pair)
            atomic_write_json(output / "comparison.json", results)
        changes = [pair["change_pct"] for pair in results["pairs"]]
        low, high = bootstrap_mean_interval(changes, seed=args.seed)
        results.update(successful=True, identity=identity, paired_mean_change_pct=statistics.fmean(changes),
                       ci_low_pct=low, ci_high_pct=high)
    except Exception as error:
        results["error"] = str(error)
        raise
    finally:
        atomic_write_json(output / "comparison.json", results)
    print(f"UI 对照完成：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

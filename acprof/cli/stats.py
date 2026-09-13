"""将结果按独立测量窗口汇总，输出均值、标准差和 bootstrap 置信区间。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from acprof.analysis.uncertainty import summarize_windows
from acprof.artifacts import atomic_write_json
from acprof.result_csv import read_result_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="CSV 或结果目录")
    parser.add_argument("--metric", action="append", help="可重复指定，默认两种延迟和归因能耗")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=1, help="连续窗口的循环移动块长度")
    parser.add_argument("--output", type=Path, help="保存 JSON；省略时输出到 stdout")
    args = parser.parse_args(argv)
    path = args.source / "result_all.csv" if args.source.is_dir() else args.source
    if args.output and (args.output.resolve() == path.resolve() or args.output.exists()):
        parser.error("统计输出必须为新的文件，不能覆盖输入或已有产物")
    try:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        _, rows = read_result_csv(path)
        report = summarize_windows(rows, args.metric or ["latency_app_s", "latency_s", "container_attributed_energy_eff_j"],
                                   confidence=args.confidence, resamples=args.resamples, seed=args.seed, block_size=args.block_size)
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise ValueError("统计期间结果 CSV 发生变化，请采集结束后重试")
        report["result_sha256"] = before
        report["result_csv"] = str(path.resolve())
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"统计失败：{error}\n")
    if args.output:
        atomic_write_json(args.output, report)
        print(f"已保存 {len(report['groups'])} 项窗口统计：{args.output}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

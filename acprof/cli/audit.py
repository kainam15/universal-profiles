"""审计实验结果、列协议和计划完整性，不修改输入产物。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acprof.analysis.audit import audit_result
from acprof.metric_registry import METRICS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="结果目录或 CSV")
    parser.add_argument("--json", action="store_true", help="将完整审计报告输出到 stdout")
    parser.add_argument("--metrics", action="store_true", help="输出指标登记表 JSON")
    parser.add_argument("--require-complete", action="store_true", help="要求有 complete 状态及完整计划")
    parser.add_argument("--require-ok", action="store_true", help="要求所有正式行成功且至少有一行")
    args = parser.parse_args(argv)
    if args.metrics:
        print(json.dumps([metric.to_dict() for metric in METRICS.values()], ensure_ascii=False, indent=2))
        return 0
    if args.source is None:
        parser.error("请提供结果目录或 CSV")
    report = audit_result(args.source)
    passed = report["valid"]
    if args.require_complete:
        passed = passed and report["completion"] == "complete" and report["coverage"] is not None
    if args.require_ok:
        counts = report["counts"]
        passed = passed and counts["formal_ok"] > 0 and counts["formal_ok"] == counts["rows"] - counts["warmup"]
    report["accepted"] = bool(passed)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        counts = report["counts"]
        print(f"审计{'通过' if passed else '未通过'}；完成状态={report['completion']}；"
              f"共 {counts['rows']} 行，正式有效 {counts['formal_ok']}，warmup {counts['warmup']}，"
              f"warn {counts['warn']}，error {counts['error']}")
        for issue in report["issues"]:
            print(f"[{issue['severity']}] {issue['message']}")
        if report["missing_metrics"]:
            print(f"{len(report['missing_metrics'])} 个数值字段存在缺失；使用 --json 查看有证据支持的原因。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Explain missing host prerequisites with human-readable or JSON output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acprof.host.doctor import collect_checks, report_dict
from acprof.host.env_utils import load_project_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读检查 AC-Prof 主机前置条件；不安装软件或更改权限")
    parser.add_argument("--profiling-mode", choices=("basic", "full"), default="full")
    parser.add_argument("--gpus", choices=("off", "on"), default="off")
    parser.add_argument("--sniff-iface", default="docker0")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="输出 JSON；必要条件缺失时 exit 1")
    args = parser.parse_args(argv)
    load_project_env(Path.cwd())
    checks = collect_checks(profiling_mode=args.profiling_mode, gpus=args.gpus,
                            sniff_iface=args.sniff_iface, output_dir=args.output_dir)
    report = report_dict(checks, profiling_mode=args.profiling_mode, gpus=args.gpus)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"AC-Prof doctor · {args.profiling_mode} · GPU {args.gpus}")
        for check in checks:
            print(f"[{check.status}] {check.name}: {check.detail}")
            if check.remedy:
                print(f"  建议：{check.remedy}")
        print("前置检查通过；请继续运行最小 smoke test。" if report["ready"] else
              "存在缺失条件；按上面的建议处理后重新运行 doctor。")
        print("此检查不下载模型、不启动容器；不证明真实推理或完整 profiling 已通过。")
    return 0 if report["ready"] else 1

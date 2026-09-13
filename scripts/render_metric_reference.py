"""由指标登记表生成字段速查；--check 用于发现文档漂移。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.metric_registry import METRICS


def render():
    lines = ["# 指标登记表速查", "",
             "由 `scripts/render_metric_reference.py` 生成，来源为 [metric_registry.py](../acprof/metric_registry.py)。",
             "字段解释和计算公式见 [指标](Metrics.md)、[能耗](Energy_Measurement.md) 和 [Profiler](Profilers.md)。", "",
             "`request_window` 表示本行请求窗口；单位中的 `/request` 表示已按实际请求数归一化。",
             "`cgroup_lifetime`、`container_startup`、`profiler_process_lifetime` 和 `independent_profiler`",
             "具有独立的生命周期，不能把复制到多行的值当作重复测量。`input_scale_type` 与",
             "`task_output_unit` 的具体单位由模型任务和物化输入计划定义。", "",
             "| 字段 | 单位 | 来源 | 窗口 | 适用范围 | 类型 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for metric in METRICS.values():
        values = (metric.name, metric.unit, metric.source, metric.window, metric.applicability, metric.kind)
        lines.append("| " + " | ".join("`" + value.replace("|", " / ") + "`" for value in values) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "docs/Metric_Reference.md"
    content = render()
    if args.check:
        if not path.exists() or path.read_text() != content:
            print("指标速查已过期，请运行 python scripts/render_metric_reference.py", file=sys.stderr)
            return 1
    else:
        path.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

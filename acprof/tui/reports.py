"""将现有统计/对照 JSON 转成只读表格；不加载 Textual 或测量依赖。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from acprof.tui.i18n import join_messages, message


@dataclass(frozen=True)
class ReportRow:
    cells: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class ReportView:
    source: Path
    title: str
    columns: tuple[str, ...]
    rows: tuple[ReportRow, ...]
    note: str


_METRIC_LABELS = {
    "latency_app_s": "应用延迟",
    "latency_s": "抓包延迟",
    "cpu_energy_total_j": "CPU 总能耗",
    "gpu_energy_total_j": "GPU 总能耗",
    "container_attributed_energy_eff_j": "容器归因有效能耗",
}
_REASONS = {
    "insufficient_windows": "窗口不足",
    "missing_windows_break_blocks": "缺失窗口打断连续块",
    "nonconsecutive_windows": "窗口序号不连续",
}


def _number(value, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(message("报告包含无效数值"))
    return float(value)


def _count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(message("报告包含无效样本数"))
    return value


def _interval(row: dict, low_key: str, high_key: str) -> tuple[float | None, float | None]:
    low = _number(row.get(low_key), nullable=True)
    high = _number(row.get(high_key), nullable=True)
    if (low is None) != (high is None) or (low is not None and low > high):
        raise ValueError(message("报告置信区间无效"))
    return low, high


def _objects(value) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(message("报告缺少有效的数据行"))
    return value


def _text(value, fallback: str = "") -> str:
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise ValueError(message("报告包含无效文本"))
    return value


def _windows(source: Path, data: dict) -> ReportView:
    confidence = _number(data.get("confidence"))
    if not 0 < confidence < 1 or data.get("filter") != "status=ok and warmup=0":
        raise ValueError(message("报告的置信水平或窗口筛选口径不受支持"))
    rows = []
    for group in _objects(data.get("groups")):
        name = _text(group.get("metric"))
        if not name:
            raise ValueError(message("报告缺少指标名称"))
        unit = _text(group.get("unit"))
        multiplier, display_unit = (1000, "ms") if unit == "s" else (1, unit)
        count = _count(group.get("n_windows"))
        missing = _count(group.get("missing_windows"))
        mean = _number(group.get("mean"), nullable=True)
        std = _number(group.get("std"), nullable=True)
        low, high = _interval(group, "ci_low", "ci_high")
        if (count == 0) != (mean is None) or (low is not None and count < 3):
            raise ValueError(message("报告的样本数与统计值不一致"))
        mode = group.get("gpu_mode")
        if mode not in {"on", "off"}:
            raise ValueError(message("报告包含无效设备配置"))
        config = "/".join((f"{_number(group.get('cpu_cores')):g}c",
                           f"{_number(group.get('mem_cap_gb')):g}G",
                           "GPU" if mode == "on" else "CPU",
                           f"{_number(group.get('input_scale')):g}"))
        mean_text = "—" if mean is None else f"{mean * multiplier:.6g} {display_unit}".strip()
        interval = "—" if low is None else f"[{low * multiplier:.6g}, {high * multiplier:.6g}] {display_unit}".strip()
        reason = _text(group.get("reason"))
        state = ("无有效窗口" if not count else _REASONS.get(reason, reason or
                 ("已估计" if low is not None else "区间不可用")))
        rows.append(ReportRow(
            (config, message(_METRIC_LABELS.get(name, name)), mean_text, interval,
             f"{count}/{missing}", message(state)),
            message("字段：{0} · 标准差：{1} · 有效窗口：{2} · 缺失：{3}",
                    name, "—" if std is None else f"{std * multiplier:.6g} {display_unit}", count, missing),
        ))
    origin = _text(data.get("result_csv"), str(source))
    note = join_messages("\n", (
        message("仅统计正式成功窗口；少量窗口的区间可能不稳定。"),
        message("数据来源：{0}", origin),
    ))
    return ReportView(source, message("窗口统计 · {0:g}% 置信区间 · {1} 项", confidence * 100, len(rows)),
                      ("资源/输入", "指标", "均值", message("{0:g}% 区间", confidence * 100),
                       "有效/缺失", "说明"), tuple(rows), note)


def _comparison_row(label: str, row: dict, count: int, detail: str) -> ReportRow:
    change = _number(row.get("paired_mean_change_pct"))
    low, high = _interval(row, "ci_low_pct", "ci_high_pct")
    if count < 3 or low is None:
        raise ValueError(message("对照报告缺少足够的配对轮次或区间"))
    state = "方向不确定" if low <= 0 <= high else "延迟增加" if low > 0 else "本次延迟降低"
    return ReportRow((label, str(count), f"{change:+.6g}%", f"[{low:.6g}%, {high:.6g}%]", message(state)), detail)


def _comparisons(source: Path, data: dict) -> ReportView:
    if data.get("successful") is not True:
        raise ValueError(message("对照实验未完成或失败：{0}", _text(data.get("error"), "—")))
    rows = []
    if data["kind"] == "monitor_overhead_diagnostic":
        mode = data.get("gpu_mode")
        if mode not in {"on", "off"}:
            raise ValueError(message("报告包含无效设备配置"))
        for row in _objects(data.get("comparisons")):
            label = _text(row.get("scenario"))
            if label.startswith("monitors-"):
                label = label.removeprefix("monitors-") + " Hz"
            rows.append(_comparison_row(label, row, _count(row.get("paired_rounds")),
                                        message("监测线程与关闭监测的基线对比；不包含抓包、perf 或 TUI。")))
        title = message("{0} 监测开销 · 95% 置信区间", "GPU" if mode == "on" else "CPU")
    else:
        pairs = _objects(data.get("pairs"))
        ui = data.get("ui")
        if ui not in {"terminal", "headless"}:
            raise ValueError(message("报告包含无效界面对照模式"))
        detail = message("包含终端绘制；具体终端环境以原实验记录为准。") if ui == "terminal" else message("仅 TUI 调度和日志路径，不包含终端绘制。")
        rows.append(_comparison_row("TUI / CLI", data, len(pairs), detail))
        title = message("CLI/TUI 对照 · 95% 置信区间")
    if not rows:
        raise ValueError(message("报告缺少有效的数据行"))
    note = message("正值表示更慢；跨零时方向不确定，负值不证明普遍加速。仅限本次对照。")
    return ReportView(source, title, ("对照", "配对轮数", "延迟变化", "95% 区间", "判断"), tuple(rows), note)


def read_report(path: str | Path) -> ReportView:
    """读取一次完整 JSON；失败/未知报告不能冒充成功结果。"""
    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        content = stream.read(32 * 1024 * 1024 + 1)
    if len(content) > 32 * 1024 * 1024:
        raise ValueError(message("报告超过 32 MiB，请先缩小报告范围"))
    try:
        data = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(message("JSON 报告损坏或编码无效")) from exc
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError(message("不支持的报告类型或版本；请选择 stats 或开销对照报告"))
    if data.get("resampling_unit") == "csv_request_window":
        return _windows(source, data)
    if data.get("kind") in {"monitor_overhead_diagnostic", "ui_overhead"}:
        return _comparisons(source, data)
    raise ValueError(message("不支持的报告类型或版本；请选择 stats 或开销对照报告"))

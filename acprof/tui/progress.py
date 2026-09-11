"""采集日志进度解析及正式测量窗口状态；不依赖 Textual。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from acprof.tui.i18n import message


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CASE_RE = re.compile(
    r"# Case\s+(?P<current>\d+)/(?P<total>\d+):\s+"
    r"CPU=(?P<cpu>\d+),\s+MEM=(?P<mem>\d+)GB,\s+GPU=(?P<gpu>\S+)"
)
MATRIX_RE = re.compile(r"Resource matrix:.*=\s*(?P<total>\d+)\s+cases")
FINAL_CSV_RE = re.compile(r"Merged results:\s+(?P<path>.+)$")
MERGE_CSV_RE = re.compile(r"\[merge\]\s+Final CSV:\s+(?P<path>.+?)\s+\(\d+\s+rows\)")
TASK_SUPPORT_ERROR_RE = re.compile(
    r"^\[task-support\]\[ERROR\] Unsupported collection task: (?P<task>.+)$"
)
LARGEST_PROBE_START_RE = re.compile(
    r"\[largest-probe\] Starting minimum configuration:\s+"
    r"CPU=(?P<cpu>\d+),\s+MEM=(?P<mem>\d+)GB,\s+"
    r"GPU=(?P<gpu>\S+),\s+input_scale=(?P<scale>\S+)"
)
LARGEST_PROBE_SCAN_RE = re.compile(
    r"\[largest-probe\] MEMORY_SCAN\s+"
    r"cpu=(?P<cpu>\d+)\s+gpu=(?P<gpu>\S+)\s+"
    r"candidates=(?P<candidates>[\d,]+)\s+"
    r"input_scale=(?P<scale>\S+)"
)
LARGEST_PROBE_MEMORY_TRY_RE = re.compile(
    r"\[largest-probe\] MEMORY_TRY\s+"
    r"current=(?P<current>\d+)\s+total=(?P<total>\d+)\s+"
    r"cpu=(?P<cpu>\d+)\s+mem=(?P<mem>\d+)\s+"
    r"gpu=(?P<gpu>\S+)\s+input_scale=(?P<scale>\S+)"
)
LARGEST_PROBE_MEMORY_RESULT_RE = re.compile(
    r"\[largest-probe\] MEMORY_RESULT\s+"
    r"mem=(?P<mem>\d+)\s+status=(?P<status>\S+)"
)
LARGEST_PROBE_RESULT_RE = re.compile(
    r"\[largest-probe\] RESULT\s+"
    r"status=(?P<status>\S+)\s+"
    r"input_scale=(?P<scale>\S+)\s+"
    r"cpu=(?P<cpu>\d+)\s+"
    r"mem=(?P<mem>\S+)\s+"
    r"gpu=(?P<gpu>\S+)\s+"
    r"cold_start_s=(?P<cold>\S+)\s+"
    r"request_s=(?P<request>\S+)\s+"
    r"ready_plus_request_s=(?P<total>\S+)"
)
LARGEST_PROBE_SUMMARY_RE = re.compile(
    r"\[largest-probe\] Summary JSON:\s+(?P<path>.+)$"
)


def _format_probe_duration(raw: str) -> str:
    try:
        value = float(raw)
    except ValueError:
        return message('不可用')
    return f"{value:.3f}s" if math.isfinite(value) else message('不可用')


@dataclass(frozen=True)
class ProgressSnapshot:
    stage: str = message('等待')
    detail: str = message('尚未启动')
    current_case: int = 0
    completed_cases: int = 0
    total_cases: int = 0
    cpu: str = "-"
    mem: str = "-"
    gpu: str = "-"
    measurement_active: bool = False
    warnings: int = 0
    errors: int = 0
    final_csv: str = ""
    probe_summary: str = ""


class RunProgressTracker:
    """Translate stable run.py log markers into low-frequency UI state."""

    def __init__(self) -> None:
        self.snapshot = ProgressSnapshot()

    def feed(self, raw_line: str) -> ProgressSnapshot:
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        state = self.snapshot
        updates: dict[str, object] = {}

        if "[WARN]" in line or "[warning]" in line.lower():
            updates["warnings"] = state.warnings + 1
        if "[ERROR]" in line or line.startswith("Traceback"):
            updates["errors"] = state.errors + 1

        task_support_error = TASK_SUPPORT_ERROR_RE.match(line)
        if task_support_error:
            updates.update(
                stage=message('任务不支持'),
                detail=message(
                    '解决办法见日志（F8）；任务：{0}',
                    task_support_error.group("task"),
                ),
                measurement_active=False,
            )

        probe_start = LARGEST_PROBE_START_RE.search(line)
        if probe_start:
            updates.update(
                stage=message('启动探测容器'),
                detail=(
                    message('最低配置 · 最大尺度 {0}', probe_start.group('scale'))
                ),
                current_case=1,
                completed_cases=0,
                total_cases=1,
                cpu=probe_start.group("cpu"),
                mem=probe_start.group("mem"),
                gpu=probe_start.group("gpu"),
                measurement_active=False,
            )

        probe_scan = LARGEST_PROBE_SCAN_RE.search(line)
        if probe_scan:
            candidates = probe_scan.group("candidates").split(",")
            updates.update(
                stage=message('准备内存探测'),
                detail=(
                    message('候选内存 {0}GB · 最大尺度 {1}', probe_scan.group('candidates'), probe_scan.group('scale'))
                ),
                current_case=0,
                completed_cases=0,
                total_cases=len(candidates),
                cpu=probe_scan.group("cpu"),
                mem="-",
                gpu=probe_scan.group("gpu"),
                measurement_active=False,
            )

        memory_try = LARGEST_PROBE_MEMORY_TRY_RE.search(line)
        if memory_try:
            current = int(memory_try.group("current"))
            updates.update(
                stage=message('探测最低可用内存'),
                detail=(
                    message('正在尝试 {0}GB · 最大尺度 {1}', memory_try.group('mem'), memory_try.group('scale'))
                ),
                current_case=current,
                completed_cases=max(state.completed_cases, current - 1),
                total_cases=int(memory_try.group("total")),
                cpu=memory_try.group("cpu"),
                mem=memory_try.group("mem"),
                gpu=memory_try.group("gpu"),
                measurement_active=False,
            )

        memory_result = LARGEST_PROBE_MEMORY_RESULT_RE.search(line)
        if memory_result:
            memory_status = memory_result.group("status")
            status_labels = {
                "ok": message('可运行，已找到最低内存'),
                "startup_oom": message('启动 OOM，继续下一档'),
                "runtime_oom": message('推理 OOM，继续下一档'),
                "cuda_oom": message('CUDA 显存 OOM，停止探测'),
                "timeout": message('请求超时'),
                "error": message('探测失败'),
            }
            updates.update(
                stage=(
                    message('找到最低可用内存')
                    if memory_status == "ok"
                    else message('内存可行性探测')
                ),
                detail=(
                    message("{0}GB · {1}", memory_result.group("mem"),
                            status_labels.get(memory_status, memory_status))
                ),
                completed_cases=max(state.completed_cases, state.current_case),
                measurement_active=False,
            )

        probe_result = LARGEST_PROBE_RESULT_RE.search(line)
        if probe_result:
            status = probe_result.group("status")
            completed = max(1, state.current_case)
            if status == "ok":
                result_detail = message(
                    "最低可用内存 {0}GB · 最大尺度 {1} · 单次请求 {2} · 冷启动 {3} · 就绪+请求 {4}",
                    probe_result.group("mem"), probe_result.group("scale"),
                    _format_probe_duration(probe_result.group("request")),
                    _format_probe_duration(probe_result.group("cold")),
                    _format_probe_duration(probe_result.group("total")),
                )
            else:
                result_detail = (
                    message('未找到最低可用内存 · 最后尝试 {0}GB · 状态 {1}', probe_result.group('mem'), status)
                )
            updates.update(
                stage=message('探测完成') if status == "ok" else message('探测失败'),
                detail=result_detail,
                current_case=completed,
                completed_cases=completed,
                total_cases=completed,
                cpu=probe_result.group("cpu"),
                mem=probe_result.group("mem"),
                gpu=probe_result.group("gpu"),
                measurement_active=False,
            )

        probe_summary = LARGEST_PROBE_SUMMARY_RE.search(line)
        if probe_summary:
            updates["probe_summary"] = probe_summary.group("path").strip()

        match = MATRIX_RE.search(line)
        if match:
            updates.update(
                stage=message('准备矩阵'),
                detail=message('资源矩阵与输入规模已确定'),
                total_cases=int(match.group("total")),
            )

        match = CASE_RE.search(line)
        if match:
            current = int(match.group("current"))
            updates.update(
                stage=message('启动容器'),
                detail=message('正在准备 case {0}/{1}', current, match.group('total')),
                current_case=current,
                completed_cases=max(state.completed_cases, current - 1),
                total_cases=int(match.group("total")),
                cpu=match.group("cpu"),
                mem=match.group("mem"),
                gpu=match.group("gpu"),
                measurement_active=False,
            )
        elif line.startswith("[largest-probe] Running one largest-scale request"):
            updates.update(
                stage=message('最大尺度探测'),
                detail=message('正在用 {0}GB 执行一次 /predict 请求', state.mem),
                measurement_active=True,
            )
        elif line.startswith("[largest-probe][ERROR]"):
            updates.update(
                stage=message('探测失败'),
                detail=line.partition("] ")[2] or line,
                measurement_active=False,
            )
        elif line.startswith("[build]"):
            updates.update(stage=message('构建镜像'), detail=line)
        elif line.startswith("[scale]") or line.startswith("[probe]"):
            updates.update(stage=message('规划输入'), detail=line)
        elif line.startswith("[compute-profile]"):
            updates.update(stage=message('计算分析'), detail=line)
        elif line.startswith("[execution-profile]"):
            updates.update(stage=message('执行分析'), detail=line)
        elif "[case] Server ready" in line:
            updates.update(stage=message('服务就绪'), detail=line)
        elif "[case] Running workload" in line:
            updates.update(
                stage=message('正式测量'),
                detail=message('采集窗口进行中；TUI 已停止常规重绘'),
                measurement_active=True,
            )
        elif "[case] Stopping container" in line:
            updates.update(
                stage=message('清理 case'),
                detail=line,
                measurement_active=False,
            )
        elif "[case] Done." in line:
            updates.update(
                stage=message('case 完成'),
                detail=line,
                completed_cases=max(state.completed_cases, state.current_case),
                measurement_active=False,
            )
        elif "[oom-prune] Skipping" in line:
            updates.update(
                stage=message('OOM 剪枝'),
                detail=line,
                completed_cases=max(state.completed_cases, state.current_case),
                measurement_active=False,
            )
        elif line.startswith("[merge]"):
            updates.update(stage=message('合并结果'), detail=line, measurement_active=False)
        elif "Profiling complete!" in line:
            updates.update(
                stage=message('已完成'),
                detail=message('采集与合并已完成'),
                completed_cases=state.total_cases or state.completed_cases,
                measurement_active=False,
            )

        final_match = FINAL_CSV_RE.search(line) or MERGE_CSV_RE.search(line)
        if final_match:
            updates["final_csv"] = final_match.group("path").strip()

        if updates:
            self.snapshot = replace(state, **updates)
        return self.snapshot

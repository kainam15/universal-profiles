"""Check known collection gaps before image preparation or measurement.

Task detection identifies a model; it does not establish collection support.
This check uses local metadata only and does not load models or runtime libraries.
"""

from __future__ import annotations

from acprof.config import PIPELINE_TAG_TO_FAMILY
from acprof.host.detect import TaskInfo


# Keep task-wide limitations separate from detection mappings. Remove a gap
# only after its workload, handler, output semantics and collection are verified.
_TASK_LIMITATIONS = {
    "image-to-text": (
        "图像描述生成尚未完成输出及生成长度指标适配；"
        "当前 CV 处理器仍按分类/检测解释输出。"
    ),
}


class TaskSupportError(ValueError):
    """An identified task cannot use the current collection implementation."""


def require_task_support(task_info: TaskInfo) -> None:
    """Reject known gaps and invalid task routing, without promising runtime compatibility."""
    task = task_info.pipeline_tag
    expected_family = PIPELINE_TAG_TO_FAMILY.get(task)
    reason = _TASK_LIMITATIONS.get(task)
    if reason is None and expected_family is None:
        reason = "当前项目尚未登记该任务类型的采集适配。"
    if reason is None and task_info.task_family != expected_family:
        reason = (
            f"任务 {task} 对应任务族 {expected_family}，"
            f"当前选择的任务族 {task_info.task_family} 与之不匹配。"
        )
    if reason is None:
        return

    lines = [
        f"[task-support][ERROR] Unsupported collection task: {task}",
        "  当前项目暂不支持此任务或所选任务族的采集，已停止本次任务。",
        f"  模型：{task_info.model_id}",
        f"  任务：{task}；任务族：{task_info.task_family}；后端：{task_info.runtime_backend}",
        f"  原因：{reason}",
        "  解决办法：",
        "    1. 如需立即采集，请选择已适配任务的模型；支持范围见 README.md。",
        "    2. 如需继续使用此任务，请等待支持该任务的项目版本，"
        "或按 README.md 的扩展说明补齐输入、推理、输出与指标适配并验证。",
        "    3. 若任务识别有误，请核对模型页后用 --task / --task-family / --backend 纠正；"
        "TUI 可在高级配置的“识别覆盖”中设置。仅在模型确实支持目标任务时使用覆盖。",
    ]
    if task == "image-to-text":
        lines.extend([
            "  图像描述任务还需处理依赖兼容性：Transformers v5 已移除 image-to-text pipeline。",
            "    开发适配时可直接加载模型，或固定兼容的 Transformers 4.x 后重建镜像；"
            "重建时取消 --skip-build / TUI 的“复用现有镜像”。",
            "    仅降级依赖不能补齐上述采集适配；增加内存或超时时间也不能解决任务不支持。",
        ])
    lines.append("  本次未进入资源矩阵或推理测量，不会生成新的测量 CSV；已有测量结果保留。")
    raise TaskSupportError("\n".join(lines))

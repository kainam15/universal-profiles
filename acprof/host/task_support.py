"""Check known collection gaps before image preparation or measurement.

Task detection identifies a model; it does not establish collection support.
This check uses local metadata only and does not load models or runtime libraries.
"""

from __future__ import annotations

from acprof.config import PIPELINE_TAG_TO_FAMILY
from acprof.host.detect import TaskInfo


class TaskSupportError(ValueError):
    """An identified task cannot use the current collection implementation."""


def require_task_support(task_info: TaskInfo, *, batch_size: int = 1) -> None:
    """Reject known gaps and invalid task routing, without promising runtime compatibility."""
    task = task_info.pipeline_tag
    expected_family = PIPELINE_TAG_TO_FAMILY.get(task)
    reason = None
    if expected_family is None:
        reason = "当前项目尚未登记该任务类型的采集适配。"
    if reason is None and task_info.task_family != expected_family:
        reason = (
            f"任务 {task} 对应任务族 {expected_family}，"
            f"当前选择的任务族 {task_info.task_family} 与之不匹配。"
        )
    if reason is None and task == "image-to-text" and batch_size != 1:
        reason = "图像描述采集每个请求只输入一张图片；请设置 --batch-size 1。"
    if reason is None and (
        expected_family == "multimodal"
        or task in {"image-text-to-image", "image-text-to-video", "image-to-image", "image-to-video"}
    ) and batch_size != 1:
        reason = "多模态采集每个请求使用一个样本；请设置 --batch-size 1。"
    if reason is None and expected_family == "multimodal" and task_info.runtime_backend not in {
        "transformers_model", "transformers_pipeline",
    }:
        reason = "多模态任务需要 Transformers 后端；请设置 --backend transformers_model。"
    if reason is None and expected_family == "diffusion" and task_info.runtime_backend != "diffusers":
        reason = "图像/视频生成任务需要 Diffusers 后端；请设置 --backend diffusers。"
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
    lines.append("  本次未进入资源矩阵或推理测量，不会生成新的测量 CSV；已有测量结果保留。")
    raise TaskSupportError("\n".join(lines))

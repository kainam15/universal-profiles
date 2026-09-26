"""任务 IO 与精度由扩展声明产生；返回副本供静态元数据补充。"""
from __future__ import annotations

from typing import Any

from acprof.extensions import CATALOG, UnsupportedExtensionError
from acprof.host.detect import TaskInfo


def _model_io_formats(task_info: TaskInfo) -> tuple[dict[str, Any], dict[str, Any]]:
    formats = CATALOG.describe(task_info).io_format
    if not {"input", "output"} <= formats.keys():
        raise UnsupportedExtensionError(f"IO format is not declared for {task_info.pipeline_tag}")
    return formats["input"], formats["output"]


def _inference_precision_by_device(task_info: TaskInfo) -> dict[str, str]:
    return CATALOG.describe(task_info).precision

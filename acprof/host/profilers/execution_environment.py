"""核验 profiler 镜像、运行库和工具版本；测量窗口外调用。"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from acprof.host.profiler_common import _run
from acprof.host.profilers.tool_discovery import _find_nsys_importer

MASSIF_TOOL = "massif"
NSYS_TOOL = "nsys"


EXECUTION_RUNTIME_LABEL_PREFIX = "org.acprof.execution-profile."


EXECUTION_RUNTIME_VERSION = "1"


def _command_detail(result: Any, limit: int = 2000) -> str:
    detail = str(
        getattr(result, "stderr", "")
        or getattr(result, "stdout", "")
        or f"exit_code={getattr(result, 'returncode', 'unknown')}"
    ).strip()
    return detail[:limit]


def _inspect_execution_image(image_ref: str) -> Optional[Dict[str, Any]]:
    result = _run(
        ["docker", "image", "inspect", image_ref, "--format", "{{json .}}"],
        check=False,
    )
    if result.returncode != 0:
        detail = _command_detail(result)
        if "no such image" in detail.lower() or "no such object" in detail.lower():
            return None
        raise RuntimeError(f"execution_image_inspect_failed:{detail}")
    try:
        image = json.loads(result.stdout)
        if not isinstance(image, dict) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(image.get("Id", ""))
        ):
            raise ValueError("missing immutable image ID")
        config = image.get("Config") or {}
        labels = config.get("Labels") or {}
        if not isinstance(labels, dict):
            raise ValueError("invalid image labels")
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"execution_image_inspect_failed:invalid_metadata:{image_ref}"
        ) from exc
    return {
        "id": image["Id"],
        "labels": labels,
    }


def require_execution_image(image_tag: str, tool: str) -> str:
    """核验当前镜像能力并固定 image ID，不为旧镜像安装运行库。"""
    base = _inspect_execution_image(image_tag)
    if base is None:
        raise RuntimeError(f"{tool}_runtime_unavailable: image not found: {image_tag}")
    if base["labels"].get(EXECUTION_RUNTIME_LABEL_PREFIX + tool) != EXECUTION_RUNTIME_VERSION:
        raise RuntimeError(f"{tool}_runtime_unavailable: rebuild the image with current AC-Prof; "
                           "legacy profiler image builds are no longer supported")
    return base["id"]


def _massif_version(derived_image: Optional[str]) -> str:
    if not derived_image:
        return "unknown"
    try:
        result = _run(
            ["docker", "run", "--rm", derived_image, "valgrind", "--version"],
            check=False,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    output = str(result.stdout or result.stderr or "").strip()
    return output.splitlines()[-1].strip() if output else "unknown"


def _nsys_version(nsys_bin: Optional[str]) -> str:
    if not nsys_bin:
        return "unknown"
    try:
        result = _run([nsys_bin, "--version"], check=False)
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    output = str(result.stdout or result.stderr or "").strip()
    return output.splitlines()[-1].strip() if output else "unknown"


def _validate_nsys_container_runtime(
    image_tag: str,
    nsys_mount_root: str,
) -> str:
    """Fail before the resource sweep if QdstrmImporter cannot run.

    Without this preflight, Nsys can leave a multi-gigabyte .qdstrm for every
    resource/scale pair while never producing the required .nsys-rep.
    """
    importer = _find_nsys_importer(nsys_mount_root)
    if not importer:
        raise RuntimeError(
            "nsys_importer_not_found:"
            f"root={os.path.abspath(os.fspath(nsys_mount_root))}"
        )
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{nsys_mount_root}:{nsys_mount_root}:ro",
            image_tag,
            importer,
            "--version",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "nsys_importer_unavailable:"
            f"{_command_detail(result)}"
        )
    output = str(result.stdout or result.stderr or "").strip()
    return output.splitlines()[-1].strip() if output else "unknown"

"""核验 profiler 镜像、运行库和工具版本；测量窗口外调用。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from uuid import uuid4
from typing import Any, Dict, Optional

from acprof.host.profiler_common import _run
from acprof.host.profilers.tool_discovery import _find_nsys_importer

MASSIF_TOOL = "massif"
NSYS_TOOL = "nsys"


EXECUTION_RUNTIME_LABEL_PREFIX = "org.acprof.execution-profile."


EXECUTION_RUNTIME_VERSION = "1"


EXECUTION_BASE_IMAGE_LABEL = EXECUTION_RUNTIME_LABEL_PREFIX + "base-image-id"


EXECUTION_DOCKERFILE_LABEL = EXECUTION_RUNTIME_LABEL_PREFIX + "dockerfile-sha256"


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
        "references": [*(image.get("RepoTags") or []), *(image.get("RepoDigests") or [])],
    }


def _ensure_execution_image(image_tag: str, project_dir: str, tool: str) -> str:
    """Reuse the model's shared runtime, or cache an old-image compatibility build.

    Always return an immutable image ID: a moved :latest tag must neither
    change the image between probes nor match a checkpoint from an older build.
    """
    runtime_label = EXECUTION_RUNTIME_LABEL_PREFIX + tool
    base = _inspect_execution_image(image_tag)
    if base is None:
        raise RuntimeError(f"{tool}_image_build_failed:base_image_not_found:{image_tag}")
    if base["labels"].get(runtime_label) == EXECUTION_RUNTIME_VERSION:
        print(
            f"[execution-profile][{tool}] Using runtime from model image "
            f"{image_tag}; no profiler image build needed"
        )
        return base["id"]

    dockerfile = os.path.join(
        os.path.abspath(os.fspath(project_dir)), "dockerfiles", f"{tool}.Dockerfile"
    )
    if not os.path.isfile(dockerfile):
        raise FileNotFoundError(
            f"{tool}_image_build_failed:dockerfile_not_found:{dockerfile}"
        )
    with open(dockerfile, "rb") as dockerfile_handle:
        recipe_hash = hashlib.sha256(dockerfile_handle.read()).hexdigest()
    expected_labels = {
        EXECUTION_BASE_IMAGE_LABEL: base["id"],
        EXECUTION_DOCKERFILE_LABEL: recipe_hash,
        runtime_label: EXECUTION_RUNTIME_VERSION,
    }
    digest = hashlib.sha256(str(image_tag).encode("utf-8")).hexdigest()[:12]
    derived_tag = f"acprof-{tool}-{digest}:latest"
    cached = _inspect_execution_image(derived_tag)
    if cached is not None and all(
        cached["labels"].get(key) == value for key, value in expected_labels.items()
    ):
        print(
            f"[execution-profile][{tool}] Reusing compatible image {derived_tag}; "
            "no build needed"
        )
        return cached["id"]

    print(f"[execution-profile][{tool}] Preparing legacy image runtime {derived_tag}")
    # BuildKit treats a bare sha256 image ID in FROM as a registry name. Give
    # the resolved local image a private temporary tag so a concurrent change
    # to the model's :latest cannot change this build's base.
    pinned_tag = f"acprof-execution-base:{uuid4().hex}"
    tagged = _run(
        ["docker", "image", "tag", base["id"], pinned_tag],
        check=False,
    )
    if tagged.returncode != 0:
        raise RuntimeError(f"{tool}_image_build_failed:{_command_detail(tagged)}")
    try:
        result = _run(
            [
                "docker",
                "build",
                "--file",
                dockerfile,
                "--build-arg",
                f"BASE_IMAGE={pinned_tag}",
                "--label",
                f"{EXECUTION_BASE_IMAGE_LABEL}={base['id']}",
                "--label",
                f"{EXECUTION_DOCKERFILE_LABEL}={recipe_hash}",
                "--tag",
                derived_tag,
                os.path.abspath(os.fspath(project_dir)),
            ],
            check=False,
        )
    finally:
        try:
            pinned_base = _inspect_execution_image(base["id"])
            if pinned_base is not None and any(
                reference != pinned_tag for reference in pinned_base["references"]
            ):
                removed = _run(
                    ["docker", "image", "rm", "--no-prune", pinned_tag], check=False
                )
                if removed.returncode != 0:
                    print(
                        f"[execution-profile][{tool}][WARN] Temporary base tag "
                        f"cleanup failed: {pinned_tag}: {_command_detail(removed)}"
                    )
            elif pinned_base is not None:
                # Removing the last reference would delete the original image
                # record too. Keep it if the caller used an untagged ID or the
                # original tag moved during the build.
                print(
                    f"[execution-profile][{tool}] Keeping {pinned_tag} to preserve "
                    f"the now-untagged source image {base['id']}"
                )
        except (OSError, RuntimeError) as exc:
            print(
                f"[execution-profile][{tool}][WARN] Temporary base tag cleanup "
                f"failed: {pinned_tag}: {exc}"
            )
    if result.returncode != 0:
        raise RuntimeError(
            f"{tool}_image_build_failed:{_command_detail(result)}"
        )
    built = _inspect_execution_image(derived_tag)
    if built is None or any(
        built["labels"].get(key) != value for key, value in expected_labels.items()
    ):
        raise RuntimeError(f"{tool}_image_build_failed:built_image_metadata_mismatch")
    return built["id"]


def _build_massif_image(image_tag: str, project_dir: str) -> str:
    return _ensure_execution_image(image_tag, project_dir, MASSIF_TOOL)


def _build_nsys_image(image_tag: str, project_dir: str) -> str:
    """Select a model image with the runtime libraries required by Nsys.

    The host Nsys installation is mounted into the profiler container.  Its
    QdstrmImporter still links against the container's elfutils runtime
    (notably libdw.so.1), which is absent from python:*slim images.
    """
    return _ensure_execution_image(image_tag, project_dir, NSYS_TOOL)


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

"""定位本机 profiler 可执行文件与完整挂载目录，不启动 profiler。"""
from __future__ import annotations

import glob
import os
import re
import shutil
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_TOOL_SEARCH_ROOTS = (
    "/opt/intel/oneapi/advisor",
    "/opt/intel/oneapi",
    "/opt/nvidia/nsight-compute",
    "/usr/local/NVIDIA-Nsight-Compute",
    "/usr/local/cuda",
    "/usr/local/cuda-*",
    "/usr/lib/nsight-compute",
)


def _candidate_executable_paths(root: str, names: Sequence[str]) -> Iterable[str]:
    for expanded_root in glob.glob(os.path.abspath(root)):
        if os.path.isfile(expanded_root) and os.path.basename(expanded_root) in names:
            yield expanded_root
            continue
        if not os.path.isdir(expanded_root):
            continue
        for name in names:
            for suffix in (
                name,
                os.path.join("bin", name),
                os.path.join("bin64", name),
                os.path.join("*", name),
                os.path.join("*", "bin", name),
                os.path.join("*", "bin64", name),
                os.path.join("latest", "bin", name),
                os.path.join("latest", "bin64", name),
                os.path.join("advisor", "latest", "bin64", name),
                os.path.join("target", "linux-desktop-glibc_2_11_3-x64", name),
            ):
                for candidate in glob.glob(os.path.join(expanded_root, suffix)):
                    yield candidate


def _executable_version_key(path: str, names: Sequence[str]) -> Tuple[Tuple[int, ...], int, str]:
    basename = os.path.basename(path)
    try:
        name_rank = len(names) - names.index(basename)
    except ValueError:
        name_rank = 0
    return tuple(int(part) for part in re.findall(r"\d+", path)), name_rank, path


def _best_existing_executable(roots: Sequence[str], names: Sequence[str]) -> Optional[str]:
    candidates = [
        os.path.realpath(candidate)
        for root in roots
        for candidate in _candidate_executable_paths(root, names)
        if os.path.isfile(candidate)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: _executable_version_key(path, names))


def _find_executable(root: Optional[str], names: Sequence[str]) -> Optional[str]:
    if root:
        best = _best_existing_executable([root], names)
        if best:
            return best
        for name in names:
            for candidate in glob.glob(os.path.join(root, "**", name), recursive=True):
                if os.path.isfile(candidate):
                    return os.path.realpath(candidate)
    best_default = _best_existing_executable(DEFAULT_TOOL_SEARCH_ROOTS, names)
    if best_default:
        return best_default
    for name in names:
        found = shutil.which(name)
        if found:
            return os.path.realpath(found)
    return None


def _tool_mount_root(tool_path: str, requested_root: Optional[str]) -> str:
    if requested_root:
        return os.path.abspath(requested_root)
    path = os.path.realpath(tool_path)
    parts = path.split(os.sep)

    if "advisor" in parts:
        idx = parts.index("advisor")
        if idx + 1 < len(parts):
            return os.sep + os.path.join(*parts[1:idx + 2])
    if "nsight-compute" in parts:
        idx = parts.index("nsight-compute")
        if idx + 1 < len(parts) and parts[idx + 1] not in {"ncu", "nv-nsight-cu-cli"}:
            return os.sep + os.path.join(*parts[1:idx + 2])
        return os.sep + os.path.join(*parts[1:idx + 1])
    if os.path.basename(path) in {"ncu", "nv-nsight-cu-cli"}:
        cuda_marker = next((part for part in parts if part.startswith("cuda")), None)
        if cuda_marker and "bin" in parts:
            idx = parts.index(cuda_marker)
            return os.sep + os.path.join(*parts[1:idx + 1])
    if os.path.isfile(path):
        return os.path.dirname(path)
    return path


def _tool_mount_roots(tool_path: str, requested_root: Optional[str]) -> List[str]:
    root = _tool_mount_root(tool_path, requested_root)
    roots = [root]
    for ncu_target in glob.glob(os.path.join(root, "target", "*")):
        real_target = os.path.realpath(ncu_target)
        if real_target == os.path.abspath(ncu_target):
            continue
        parts = real_target.split(os.sep)
        if "nsight-compute" not in parts:
            continue
        idx = parts.index("nsight-compute")
        real_root = os.sep + os.path.join(*parts[1:idx + 1])
        if real_root not in roots:
            roots.append(real_root)
    return roots


NSYS_DEFAULT_SEARCH_ROOTS = (
    "/opt/nvidia/nsight-systems",
    "/opt/nvidia/nsight-compute",
    "/usr/local/NVIDIA-Nsight-Systems",
    "/usr/local/cuda",
    "/usr/local/cuda-*",
    "/usr/lib/nsight-systems",
)


def _candidate_nsys_paths(root: str) -> Iterable[str]:
    for expanded in glob.glob(os.path.abspath(os.fspath(root))):
        if os.path.isfile(expanded) and os.path.basename(expanded) == "nsys":
            yield expanded
            continue
        if not os.path.isdir(expanded):
            continue
        for suffix in (
            "nsys",
            os.path.join("bin", "nsys"),
            os.path.join("bin64", "nsys"),
            os.path.join("target-linux-x64", "nsys"),
            os.path.join("*", "target-linux-x64", "nsys"),
            os.path.join("*", "host", "target-linux-x64", "nsys"),
        ):
            yield from glob.glob(os.path.join(expanded, suffix))
        yield from glob.iglob(
            os.path.join(expanded, "**", "nsys"),
            recursive=True,
        )


def _nsys_path_rank(path: str) -> Tuple[Tuple[int, ...], str]:
    return tuple(int(part) for part in re.findall(r"\d+", path)), path


def _find_nsys_executable(nsys_root: Optional[str]) -> Optional[str]:
    roots = [nsys_root] if nsys_root else list(NSYS_DEFAULT_SEARCH_ROOTS)
    candidates: List[str] = []
    seen = set()
    for root in roots:
        if not root:
            continue
        for candidate in _candidate_nsys_paths(root):
            real = os.path.realpath(candidate)
            if real in seen or not os.path.isfile(real):
                continue
            if not os.access(real, os.X_OK):
                continue
            seen.add(real)
            candidates.append(real)
    if candidates:
        return max(candidates, key=_nsys_path_rank)
    found = shutil.which("nsys")
    return os.path.realpath(found) if found else None


def _nsys_mount_root(nsys_bin: str) -> str:
    """Return an install root containing Nsys reports, Python, and libraries."""
    path = os.path.realpath(nsys_bin)
    parts = path.split(os.sep)
    target_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part.startswith("target-")
        ),
        None,
    )
    if target_index is not None and target_index > 0:
        root = os.sep + os.path.join(*parts[1:target_index])
        if os.path.basename(root) == "host":
            root = os.path.dirname(root)
        return root

    parent = os.path.dirname(path)
    if os.path.basename(parent) in {"bin", "bin64"} and any(
        marker in parent.lower()
        for marker in ("nsight", "nvidia", "cuda")
    ):
        return os.path.dirname(parent)
    return parent


def _find_nsys_importer(nsys_mount_root: str) -> Optional[str]:
    """Find the QDSTRM importer shipped beside the selected Nsys CLI."""
    root = os.path.realpath(os.path.abspath(os.fspath(nsys_mount_root)))
    candidates: List[str] = []
    for candidate in glob.iglob(
        os.path.join(root, "**", "QdstrmImporter"),
        recursive=True,
    ):
        real = os.path.realpath(candidate)
        try:
            inside_root = os.path.commonpath((root, real)) == root
        except ValueError:
            inside_root = False
        if (
            inside_root
            and os.path.isfile(real)
            and os.access(real, os.X_OK)
        ):
            candidates.append(real)
    return max(candidates, key=_nsys_path_rank) if candidates else None

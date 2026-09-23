"""Locations and child commands shared by source, wheel and frozen installations."""
from __future__ import annotations

from pathlib import Path
import sys


def resource_root() -> Path:
    """Read-only build context; never use this directory for user output/settings."""
    package = Path(__file__).resolve().parent
    bundled = package / "_bundle"
    return bundled if bundled.is_dir() else package.parent


def module_command(module: str, *, python_executable: str | Path | None = None) -> list[str]:
    executable = str(python_executable or sys.executable)
    if getattr(sys, "frozen", False):
        return [executable, "_worker", module]
    return [executable, "-m", module]


def cli_command(command: str, *, python_executable: str | Path | None = None) -> list[str]:
    executable = str(python_executable or sys.executable)
    if getattr(sys, "frozen", False):
        return [executable, command]
    return [executable, "-u", "-m", "acprof", command]

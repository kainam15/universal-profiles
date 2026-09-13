"""基于同目录临时文件发布产物；调用方负责在测量窗口之外写入。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, TextIO


def atomic_write(path: str | Path, write: Callable[[TextIO], None]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=destination.parent, prefix=f".{destination.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        if os.name == "posix":
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    def write(stream: TextIO) -> None:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    atomic_write(path, write)

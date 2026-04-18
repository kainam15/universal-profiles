"""Small helpers for local project environment bootstrap."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _iter_env_files(project_dir: str | os.PathLike[str]) -> Iterable[Path]:
    root = Path(project_dir)
    yield root / ".env"
    yield root / ".env.local"


def load_project_env(project_dir: str | os.PathLike[str]) -> None:
    """Load simple KEY=VALUE pairs from local env files if present."""
    for env_file in _iter_env_files(project_dir):
        if not env_file.exists():
            continue

        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            value = value.strip()
            if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
                value = value[1:-1]

            os.environ.setdefault(key, value)


def resolve_hf_token() -> str | None:
    """Populate HF_TOKEN from env or local Hugging Face login when available."""
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        try:
            from huggingface_hub.utils import get_token

            token = (get_token() or "").strip()
        except Exception:
            token = ""

    if not token:
        return None

    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return token


def bootstrap_project_env(project_dir: str | os.PathLike[str]) -> str | None:
    """Load project env files and normalize Hugging Face auth env vars."""
    load_project_env(project_dir)
    return resolve_hf_token()

"""Small helpers for local project environment bootstrap."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, MutableMapping

from acprof.hf_endpoints import hf_endpoints

from acprof.config import (
    CONTAINER_HF_HOME,
    CONTAINER_MODEL_LOCAL_PATH,
)


def _iter_env_files(project_dir: str | os.PathLike[str]) -> Iterable[Path]:
    root = Path(project_dir)
    yield root / ".env"
    yield root / ".env.local"


def load_project_env(
    project_dir: str | os.PathLike[str],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Load local KEY=VALUE pairs into the process or an isolated environment."""
    target_environ = os.environ if environ is None else environ
    retired_setting = "ACPROF_SUDO_PASSWORD"
    migration = f"{retired_setting} is no longer supported; remove it and follow docs/Getting_Started.md#最小权限安装"
    if retired_setting in target_environ:
        raise ValueError(migration)
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
            if key == retired_setting:
                raise ValueError(migration)

            value = value.strip()
            if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
                value = value[1:-1]

            target_environ.setdefault(key, value)


def _set_default_if_blank(key: str, value: str) -> None:
    if not os.environ.get(key, "").strip():
        os.environ[key] = value


def configure_hf_network() -> str:
    """Normalize the selected endpoint while preserving the user's proxy policy."""
    endpoint = hf_endpoints()[0]
    os.environ["HF_ENDPOINT"] = endpoint
    _set_default_if_blank("HF_HUB_ENDPOINT", endpoint)
    return endpoint


def resolve_hf_token() -> str | None:
    """Populate HF_TOKEN from env or local Hugging Face login when available."""
    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if not token:
        try:
            from huggingface_hub.utils import get_token

            token = (get_token() or "").strip()
        except Exception:
            token = ""

    if not token:
        return None

    _set_default_if_blank("HF_TOKEN", token)
    _set_default_if_blank("HUGGING_FACE_HUB_TOKEN", token)
    return token


def bootstrap_project_env(project_dir: str | os.PathLike[str]) -> str | None:
    """Load project env files and normalize Hugging Face auth env vars."""
    load_project_env(project_dir)
    configure_hf_network()
    return resolve_hf_token()


def hf_offline_docker_env_args() -> list[str]:
    """Return the shared Docker environment for network-free model loading."""
    return [
        "-e", "HF_HUB_DISABLE_TELEMETRY=1",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", f"HF_HOME={CONTAINER_HF_HOME}",
        "-e", f"HF_HUB_CACHE={CONTAINER_HF_HOME}",
        "-e", f"TRANSFORMERS_CACHE={CONTAINER_HF_HOME}",
        "-e", f"MODEL_LOCAL_PATH={CONTAINER_MODEL_LOCAL_PATH}",
    ]

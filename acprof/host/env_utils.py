"""Small helpers for local project environment bootstrap."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from acprof.config import (
    CONTAINER_HF_HOME,
    CONTAINER_MODEL_LOCAL_PATH,
    HF_MIRROR_ENDPOINT,
)


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


def _endpoint_host(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    host = parsed.netloc or parsed.path
    return host.strip().lower().rstrip("/")


def _append_no_proxy_host(host: str) -> None:
    if not host:
        return

    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        known = {part.lower() for part in parts}
        if host not in known:
            os.environ[key] = ",".join(parts + [host]) if parts else host


def _set_default_if_blank(key: str, value: str) -> None:
    if not os.environ.get(key, "").strip():
        os.environ[key] = value


def configure_hf_network() -> str:
    """Normalize host-side Hugging Face endpoint and proxy bypass for metadata calls."""
    endpoint = (os.environ.get("HF_ENDPOINT") or os.environ.get("HF_HUB_ENDPOINT") or "").strip()
    if not endpoint:
        endpoint = HF_MIRROR_ENDPOINT

    _set_default_if_blank("HF_ENDPOINT", endpoint)
    _set_default_if_blank("HF_HUB_ENDPOINT", endpoint)
    _append_no_proxy_host(_endpoint_host(endpoint))
    return endpoint


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

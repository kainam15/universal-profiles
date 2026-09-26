"""Small helpers for local project environment bootstrap."""

from __future__ import annotations

import os
import json
import re
import stat
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping
from urllib.parse import urlsplit

from acprof.hf_endpoints import hf_endpoints

from acprof.config import (
    CONTAINER_HF_HOME,
    CONTAINER_MODEL_LOCAL_PATH,
)


CONFIGURABLE_ENV_KEYS = (
    "HF_TOKEN", "HF_ENDPOINT", "HF_FALLBACK_ENDPOINTS",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "ACPROF_WECOM_WEBHOOK_URL",
)
_ENV_ALIASES = {
    "HF_TOKEN": "HUGGING_FACE_HUB_TOKEN", "HF_ENDPOINT": "HF_HUB_ENDPOINT",
    "HTTP_PROXY": "http_proxy", "HTTPS_PROXY": "https_proxy",
    "ALL_PROXY": "all_proxy", "NO_PROXY": "no_proxy",
}


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if line.startswith("export "):
        line = line[7:].lstrip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key, value = key.strip(), value.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        quote, value = value[0], value[1:-1]
        if quote == '"':
            value = re.sub(r'\\([\\"])', r'\1', value)
    return key, value


def _read_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(item for line in path.read_text(encoding="utf-8").splitlines()
                if (item := _parse_env_line(line)) is not None)


def project_env_values(project_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Read local values without changing the process; .env.local has priority."""
    values: dict[str, str] = {}
    for path in _iter_env_files(project_dir):
        values.update(_read_env_values(path))
    return values


def configurable_env_values(project_dir, *, environ=None) -> dict[str, str]:
    values = project_env_values(project_dir)
    values.update(os.environ if environ is None else environ)
    return {key: values.get(key, values.get(_ENV_ALIASES.get(key, ""), ""))
            for key in CONFIGURABLE_ENV_KEYS}


def _private_env_text(path: Path) -> str:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Local configuration must be a regular file, not a symbolic link")
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _atomic_private_text(path: Path, text: str) -> None:
    _private_env_text(path)
    fd, name = tempfile.mkstemp(prefix=".env.local.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), stat.S_IRUSR | stat.S_IWUSR)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_project_env(
    project_dir: str | os.PathLike[str], values: Mapping[str, str], *,
    environ: MutableMapping[str, str] | None = None, expected_text: str | None = None,
) -> Path:
    """Save explicit TUI edits privately, preserving other settings and a backup.

    No shell evaluation or variable interpolation is performed. Only an explicit
    save updates the current process and its children; ordinary bootstrap still
    gives inherited environment variables priority.
    """
    if set(values) - set(CONFIGURABLE_ENV_KEYS):
        raise ValueError("Unsupported environment setting")
    updates = {}
    for key, value in values.items():
        if not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f"{key}: enter a single line without control characters")
        updates[key] = value.strip()
    hf_endpoints(updates)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if updates.get(key):
            try:
                parsed = urlsplit(updates[key])
                valid = (parsed.scheme in {"http", "https", "socks5", "socks5h"}
                         and parsed.hostname and not parsed.query and not parsed.fragment)
                port = parsed.port
                valid = valid and (port is None or port > 0)
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"{key}: invalid proxy URL")
    if updates.get("ACPROF_WECOM_WEBHOOK_URL"):
        from acprof.notifications import NotificationConfigError, validate_wecom_webhook_url
        try:
            validate_wecom_webhook_url(updates["ACPROF_WECOM_WEBHOOK_URL"])
        except (ValueError, NotificationConfigError):
            raise ValueError("ACPROF_WECOM_WEBHOOK_URL: invalid WeCom webhook URL") from None
    for key, alias in _ENV_ALIASES.items():
        if key in updates:
            updates[alias] = updates[key]
    path = Path(project_dir) / ".env.local"
    original = _private_env_text(path)
    if expected_text is not None and original != expected_text:
        raise ValueError(".env.local changed since this form opened; close and reopen before saving")
    remaining = dict(updates)
    lines = []
    for line in original.splitlines(keepends=True):
        item = _parse_env_line(line)
        if item is not None and item[0] in updates:
            key = item[0]
            if key in remaining:
                lines.append(f"{key}={json.dumps(remaining.pop(key), ensure_ascii=False)}\n")
        else:
            lines.append(line)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.extend(f"{key}={json.dumps(value, ensure_ascii=False)}\n" for key, value in remaining.items())
    if path.exists():
        _atomic_private_text(path.with_name(".env.local.bak"), original)
    _atomic_private_text(path, "".join(lines))
    (os.environ if environ is None else environ).update(updates)
    return path


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
    values = project_env_values(project_dir)
    if retired_setting in values:
        raise ValueError(migration)
    for key, value in values.items():
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

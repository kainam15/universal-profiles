"""Download and cache Hugging Face model weights with retry/fallback logic.

Used both during Docker image build and host-side pre-warming.
"""

from __future__ import annotations

import inspect
import os
import time

from typing import Sequence

from huggingface_hub import snapshot_download

MODEL_ID = ""
MODEL_REVISION = "main"
CACHE_DIR = "/models/hf"
DEFAULT_LOCAL_MODEL_PATH = "/models/model-snapshot"
DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_WORKERS = "8,2,1"
DEFAULT_BACKOFF_S = 5.0
DEFAULT_ETAG_TIMEOUT_S = 30.0


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def _normalize_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def _candidate_endpoints() -> list[str]:
    endpoints: list[str] = []
    for endpoint in _split_csv(os.getenv("HF_ENDPOINT", "")):
        normalized = _normalize_endpoint(endpoint)
        if normalized not in endpoints:
            endpoints.append(normalized)

    for endpoint in _split_csv(os.getenv("HF_FALLBACK_ENDPOINTS", "")):
        normalized = _normalize_endpoint(endpoint)
        if normalized not in endpoints:
            endpoints.append(normalized)

    if DEFAULT_ENDPOINT not in endpoints:
        endpoints.append(DEFAULT_ENDPOINT)

    return endpoints


def _worker_plan() -> list[int]:
    workers: list[int] = []
    for part in _split_csv(os.getenv("HF_DOWNLOAD_WORKERS", DEFAULT_WORKERS)):
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0 and value not in workers:
            workers.append(value)

    return workers or [1]


def _build_snapshot_kwargs(endpoint: str, max_workers: int) -> dict:
    sig = inspect.signature(snapshot_download)
    params = sig.parameters

    kwargs = {
        "repo_id": MODEL_ID,
        "cache_dir": CACHE_DIR,
    }
    revision = os.getenv("MODEL_REVISION", MODEL_REVISION).strip()
    if revision and "revision" in params:
        kwargs["revision"] = revision

    if "etag_timeout" in params:
        kwargs["etag_timeout"] = float(
            os.getenv("HF_ETAG_TIMEOUT", str(DEFAULT_ETAG_TIMEOUT_S))
        )
    if "max_workers" in params:
        kwargs["max_workers"] = max_workers
    if "endpoint" in params:
        kwargs["endpoint"] = endpoint
    if "resume_download" in params:
        kwargs["resume_download"] = True

    return kwargs


def _download_once(endpoint: str, max_workers: int) -> str:
    os.environ["HF_ENDPOINT"] = endpoint
    kwargs = _build_snapshot_kwargs(endpoint, max_workers)
    print(
        f"[download] repo={MODEL_ID} endpoint={endpoint} "
        f"revision={kwargs.get('revision', 'default')} workers={max_workers} "
        f"cache={CACHE_DIR}",
        flush=True,
    )
    return snapshot_download(**kwargs)


def _publish_local_model_path(target_dir: str, local_path: str) -> None:
    """Expose the downloaded revision through one stable in-image path."""
    target = os.path.abspath(target_dir)
    link_path = os.path.abspath(local_path)
    if not os.path.isdir(target):
        raise RuntimeError(f"downloaded model snapshot is not a directory: {target}")

    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    if os.path.lexists(link_path):
        if os.path.islink(link_path) and os.path.realpath(link_path) == os.path.realpath(target):
            return
        raise RuntimeError(
            f"local model path already exists and does not reference this snapshot: {link_path}"
        )

    os.symlink(target, link_path, target_is_directory=True)
    print(f"[download] Local model path: {link_path} -> {target}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    del argv
    global MODEL_ID, MODEL_REVISION, CACHE_DIR

    MODEL_ID = os.getenv("MODEL_ID", "").strip()
    MODEL_REVISION = os.getenv("MODEL_REVISION", "main").strip() or "main"
    CACHE_DIR = os.getenv("HF_HOME", os.getenv("MODEL_CACHE_DIR", "/models/hf"))

    if not MODEL_ID:
        raise SystemExit("MODEL_ID environment variable is required")

    os.makedirs(CACHE_DIR, exist_ok=True)

    attempts = [
        (endpoint, workers)
        for endpoint in _candidate_endpoints()
        for workers in _worker_plan()
    ]
    backoff_s = float(os.getenv("HF_DOWNLOAD_RETRY_BACKOFF", str(DEFAULT_BACKOFF_S)))
    last_error: Exception | None = None
    target_dir: str | None = None

    for idx, (endpoint, workers) in enumerate(attempts, start=1):
        try:
            target_dir = _download_once(endpoint, workers)
            print(f"[download] Completed: {target_dir}", flush=True)
            break
        except Exception as exc:
            last_error = exc
            print(
                f"[download] Attempt {idx}/{len(attempts)} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if idx < len(attempts):
                sleep_s = backoff_s * idx
                print(f"[download] Backing off for {sleep_s:.1f}s before retry.", flush=True)
                time.sleep(sleep_s)
    else:
        raise SystemExit(
            f"Failed to download model '{MODEL_ID}' after {len(attempts)} attempts: {last_error}"
        )

    assert target_dir is not None
    local_path = (
        os.getenv("MODEL_LOCAL_PATH", DEFAULT_LOCAL_MODEL_PATH).strip()
        or DEFAULT_LOCAL_MODEL_PATH
    )
    _publish_local_model_path(target_dir, local_path)


if __name__ == "__main__":
    main()

"""Download and cache Hugging Face model weights with retry/fallback logic.

Used both during Docker image build and host-side pre-warming.
"""

from __future__ import annotations

import inspect
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import time
from pathlib import Path

from typing import Sequence

from huggingface_hub import snapshot_download

if __package__:
    from .model_files import ModelFilesError, PLAN_FILENAME, plan_download, seal_plan
else:
    from model_files import ModelFilesError, PLAN_FILENAME, plan_download, seal_plan

MODEL_ID = ""
MODEL_REVISION = "main"
CACHE_DIR = "/models/hf"
DEFAULT_LOCAL_MODEL_PATH = "/models/model-snapshot"
DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_WORKERS = "8,2,1"
DEFAULT_BACKOFF_S = 5.0
DEFAULT_ETAG_TIMEOUT_S = 30.0
_LAST_PLAN: dict | None = None


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
    accepts_kwargs = any(value.kind == inspect.Parameter.VAR_KEYWORD for value in params.values())

    def supports(name: str) -> bool:
        return name in params or accepts_kwargs

    kwargs = {
        "repo_id": MODEL_ID,
        "cache_dir": CACHE_DIR,
    }
    revision = os.getenv("MODEL_REVISION", MODEL_REVISION).strip()
    if revision and supports("revision"):
        kwargs["revision"] = revision

    if supports("etag_timeout"):
        kwargs["etag_timeout"] = float(
            os.getenv("HF_ETAG_TIMEOUT", str(DEFAULT_ETAG_TIMEOUT_S))
        )
    if supports("max_workers"):
        kwargs["max_workers"] = max_workers
    if supports("endpoint"):
        kwargs["endpoint"] = endpoint
    if "resume_download" in params:
        kwargs["resume_download"] = True

    return kwargs


def _native_model_types() -> set[str]:
    if importlib.util.find_spec("transformers") is None:
        return set()
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    return set(CONFIG_MAPPING_NAMES)


def _library_versions() -> dict[str, str]:
    result = {}
    for name in ("huggingface-hub", "transformers", "diffusers", "sentence-transformers", "chronos-forecasting"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _prepare_plan(endpoint: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    revision = os.getenv("MODEL_REVISION", MODEL_REVISION).strip() or "main"
    info = HfApi(endpoint=endpoint).model_info(MODEL_ID, revision=revision, files_metadata=True)
    if len(revision) == 40 and info.sha != revision:
        raise ModelFilesError("Hub response does not match the requested model commit")
    files = {}
    for item in info.siblings:
        lfs = getattr(item, "lfs", None)
        lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files[item.rfilename] = {
            "size": getattr(item, "size", None), "blob_id": getattr(item, "blob_id", None),
            "lfs_sha256": lfs_sha,
        }

    def read_json(name: str):
        path = hf_hub_download(MODEL_ID, name, revision=info.sha, cache_dir=CACHE_DIR, endpoint=endpoint)
        try:
            return json.loads(Path(path).read_text())
        except (ValueError, UnicodeError) as exc:
            raise ModelFilesError(f"invalid model metadata: {name}") from exc

    return plan_download(
        model_id=MODEL_ID, revision=info.sha, family=os.getenv("TASK_FAMILY", ""),
        backend=os.getenv("RUNTIME_BACKEND", ""), files=files, read_json=read_json,
        policy=os.getenv("MODEL_DOWNLOAD_POLICY", "auto"),
        adapter=os.getenv("MODEL_ADAPTER", "family-default"),
        native_model_types=_native_model_types(), library_versions=_library_versions(),
    )


def _download_once(endpoint: str, max_workers: int) -> str:
    global _LAST_PLAN
    os.environ["HF_ENDPOINT"] = endpoint
    plan = _prepare_plan(endpoint)
    kwargs = _build_snapshot_kwargs(endpoint, max_workers)
    kwargs["revision"] = plan["model_revision"]
    # Hub 的 allow_patterns 使用 fnmatch；文件名中的通配字符也必须按字面匹配。
    escape = {"[": "[[]", "*": "[*]", "?": "[?]"}
    kwargs["allow_patterns"] = ["".join(escape.get(char, char) for char in item["path"]) for item in plan["files"]]
    print(
        f"[download] repo={MODEL_ID} endpoint={endpoint} "
        f"revision={kwargs.get('revision', 'default')} workers={max_workers} "
        f"cache={CACHE_DIR}",
        flush=True,
    )
    print(f"[download] policy={plan['requested_policy']} -> {plan['effective_policy']} "
          f"({plan['reason']}); files={len(plan['files'])}, excluded={len(plan['excluded_files'])}, "
          f"selected_bytes={plan['selected_bytes']}", flush=True)
    target = snapshot_download(**kwargs)
    _LAST_PLAN = plan
    return target


def verify_download(target: str | Path, plan: dict) -> dict:
    """构建期间检查完整性并记录实际 SHA256；不会进入 server 启动路径。"""
    root = Path(target)
    hashes = {}
    for item in plan["files"]:
        path = root / item["path"]
        if not path.is_file():
            raise ModelFilesError(f"downloaded snapshot is missing: {item['path']}")
        stat = path.stat()
        if item.get("size") is not None and stat.st_size != item["size"]:
            raise ModelFilesError(f"model file size mismatch: {item['path']}")
        key = (stat.st_dev, stat.st_ino)
        if key not in hashes:
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(f"blob {stat.st_size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    sha256.update(chunk)
                    git_sha1.update(chunk)
            hashes[key] = (sha256.hexdigest(), git_sha1.hexdigest())
        sha256_hex, git_hex = hashes[key]
        if item.get("lfs_sha256"):
            if sha256_hex != item["lfs_sha256"]:
                raise ModelFilesError(f"model file SHA256 mismatch: {item['path']}")
        elif item.get("blob_id") and git_hex != item["blob_id"]:
            raise ModelFilesError(f"model file Git blob mismatch: {item['path']}")
        item.update(size=stat.st_size, sha256=sha256_hex)
    plan["selected_bytes"] = sum(item["size"] for item in plan["files"])
    plan["verification"] = "sha256"
    return seal_plan(plan)


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
    parser = argparse.ArgumentParser(description="Download a fixed model snapshot with a reproducible file plan")
    parser.add_argument("--plan-only", action="store_true", help="Resolve files and sizes without downloading weights")
    parser.add_argument("--output-plan", default=None, help="Path for model_download_plan.json")
    args = parser.parse_args(argv)
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
            if args.plan_only:
                plan = _prepare_plan(endpoint)
                destination = Path(args.output_plan or PLAN_FILENAME)
                destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
                print(f"[download] Plan: {destination}; selected_bytes={plan['selected_bytes']}", flush=True)
                return
            target_dir = _download_once(endpoint, workers)
            print(f"[download] Completed: {target_dir}", flush=True)
            break
        except (ModelFilesError, ImportError) as exc:
            raise SystemExit(f"Invalid model download plan: {exc}") from exc
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
    assert _LAST_PLAN is not None
    plan = verify_download(target_dir, _LAST_PLAN)
    destination = Path(args.output_plan or str(Path(local_path).parent / PLAN_FILENAME))
    destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(f"[download] Verified plan: {destination} ({plan['plan_sha256']})", flush=True)


if __name__ == "__main__":
    main()

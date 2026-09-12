"""在镜像构建期间记录真实依赖；不加载模型，也不启动测量。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


MANIFEST_PATH = "/app/runtime_environment.json"


def installed_packages() -> dict[str, str]:
    return dict(sorted(
        (dist.metadata["Name"].lower().replace("_", "-"), dist.version)
        for dist in importlib.metadata.distributions()
    ))


def collect_manifest() -> dict:
    packages = installed_packages()
    locked = Path("/opt/acprof/requirements.lock")
    lock_hash = ""
    if locked.is_file():
        lock_hash = hashlib.sha256(locked.read_bytes()).hexdigest()
        expected = os.environ.get("ACPROF_DEPENDENCY_LOCK_SHA256", "")
        if expected and lock_hash != expected:
            raise RuntimeError("dependency lock hash differs from selected runtime profile")
        for line in locked.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, version = line.split("==", 1)
            if packages.get(name.lower().replace("_", "-")) != version:
                raise RuntimeError(f"installed dependency does not match lock: {name}=={version}")
    source = Path(os.getenv("MODEL_LOCAL_PATH", "/models/model-snapshot"))
    snapshot_revision = source.resolve().name if source.is_symlink() else ""
    if snapshot_revision != os.environ.get("MODEL_REVISION"):
        raise RuntimeError("baked model snapshot differs from declared model revision")
    from acprof.container.model_files import PLAN_FILENAME, validate_plan

    download = json.loads((source.parent / PLAN_FILENAME).read_text())
    validate_plan(download)
    if download.get("verification") != "sha256":
        raise RuntimeError("model download plan has not been verified during build")
    for key, value in {
        "model_id": os.environ.get("MODEL_ID"), "model_revision": snapshot_revision,
        "requested_policy": os.environ.get("MODEL_DOWNLOAD_POLICY", "auto"),
        "adapter": os.environ.get("ACPROF_MODEL_ADAPTER"),
    }.items():
        if download.get(key) != value:
            raise RuntimeError(f"model download plan differs from runtime: {key}")
    if any(not (source / item["path"]).is_file() for item in download["files"]):
        raise RuntimeError("model snapshot is missing files from its download plan")
    return {
        "schema_version": 1,
        "profile_id": os.environ["ACPROF_RUNTIME_PROFILE"],
        "adapter": os.environ["ACPROF_MODEL_ADAPTER"],
        "build_fingerprint": os.environ["ACPROF_BUILD_FINGERPRINT"],
        "model_id": os.environ.get("MODEL_ID", ""),
        "model_revision": os.environ.get("MODEL_REVISION", ""),
        "model_snapshot_revision": snapshot_revision,
        "model_download": download,
        "python_version": platform.python_version(),
        "packages": packages,
        "dependency_lock_sha256": lock_hash,
        "resolved_packages_sha256": hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest(),
        "custom_code_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*.py"))
        },
    }


if __name__ == "__main__":
    Path(MANIFEST_PATH).write_text(json.dumps(collect_manifest(), indent=2) + "\n")

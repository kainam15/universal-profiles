"""在镜像构建期间记录真实依赖；不加载模型，也不启动测量。"""

from __future__ import annotations

import hashlib
import base64
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from acprof.dependency_locks import normalized_name, package_versions, read_python_lock, require_exact_packages


MANIFEST_PATH = "/app/runtime_environment.json"


def installed_packages() -> dict[str, str]:
    return dict(sorted(
        (normalized_name(dist.metadata["Name"]), dist.version)
        for dist in importlib.metadata.distributions()
    ))


def collect_manifest() -> dict:
    packages = installed_packages()
    dependency = json.loads(Path("/opt/acprof/environment-manifest.json").read_text())
    require_exact_packages(dependency["packages"], packages)
    output = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${Package}:${Architecture}\t${Version}\t${db:Status-Status}\n"], text=True,
    )
    actual_system = {name: version for line in output.splitlines()
                     for name, version, status in [line.split("\t")] if status == "installed"}
    require_exact_packages(dependency["system_packages"], actual_system, kind="system")
    locked = Path("/opt/acprof/requirements.lock")
    lock_hash = ""
    if locked.is_file():
        lock_hash = hashlib.sha256(locked.read_bytes()).hexdigest()
        expected = os.environ.get("ACPROF_DEPENDENCY_LOCK_SHA256", "")
        if expected and lock_hash != expected:
            raise RuntimeError("dependency lock hash differs from selected runtime profile")
        require_exact_packages(package_versions(read_python_lock(locked)), packages)
    else:
        raise RuntimeError("dependency lock is missing")
    if dependency["platform_image_id"] != os.environ["ACPROF_PLATFORM_IMAGE_ID"]:
        raise RuntimeError("platform image differs from dependency manifest")
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
    from acprof.model_spec import load_model_dependencies
    dependencies = download.get("dependencies", [])
    if [{key: value for key, value in item.items() if key != "download"} for item in dependencies] != load_model_dependencies():
        raise RuntimeError("model dependency plan differs from the baked declaration")
    cache_root = Path(os.getenv("HF_HUB_CACHE", str(Path(os.getenv("HF_HOME", "/models/hf")) / "hub")))
    for item in dependencies:
        cache = cache_root / ("models--" + item["repo_id"].replace("/", "--"))
        reference = cache / "refs/main"
        snapshot = cache / "snapshots" / item["revision"]
        if not reference.is_file() or reference.read_text().strip() != item["revision"]:
            raise RuntimeError("offline model dependency revision differs from its pinned plan")
        if any(not (snapshot / record["path"]).is_file() for record in item["download"]["files"]):
            raise RuntimeError("offline model dependency is missing files from its download plan")
    return {
        **dependency,
        "schema_version": 1,
        "request_fingerprint": os.environ["ACPROF_REQUEST_FINGERPRINT"],
        "platform_image_id": os.environ["ACPROF_PLATFORM_IMAGE_ID"],
        "environment_image_id": os.environ["ACPROF_ENVIRONMENT_IMAGE_ID"],
        "model_image_id": os.environ["ACPROF_MODEL_IMAGE_ID"],
        "profile_id": os.environ["ACPROF_RUNTIME_PROFILE"],
        "adapter": os.environ["ACPROF_MODEL_ADAPTER"],
        "build_fingerprint": os.environ["ACPROF_BUILD_FINGERPRINT"],
        "model_id": os.environ.get("MODEL_ID", ""),
        "model_revision": os.environ.get("MODEL_REVISION", ""),
        "model_snapshot_revision": snapshot_revision,
        "model_download": download,
        "model_spec": json.loads(base64.b64decode(os.environ["ACPROF_MODEL_SPEC_B64"], validate=True))
                      if os.getenv("ACPROF_MODEL_SPEC_B64") else {},
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

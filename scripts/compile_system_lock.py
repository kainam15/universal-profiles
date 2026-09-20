"""在固定基础容器中解析 Debian Snapshot，生成完整系统锁；普通镜像构建不调用此脚本。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.dependency_locks import read_system_lock  # noqa: E402 -- 脚本先设置仓库导入路径。
from acprof.runtime_profiles import PYTHON_BASE_IMAGE  # noqa: E402 -- 脚本先设置仓库导入路径。

DIRECT_PACKAGES = ("valgrind", "libdw1t64", "build-essential")
ARCHIVES = {"debian": ("trixie", "trixie-updates"), "debian-security": ("trixie-security",)}


def snapshot_url(archive: str, cutoff: str) -> str:
    """选取截止时间之前实际存在的快照，避免依赖 HTTP 隐式回退。"""
    limit = dt.datetime.strptime(cutoff, "%Y%m%dT%H%M%SZ")
    month = limit.replace(day=1)
    for _ in range(12):
        url = f"https://snapshot.debian.org/archive/{archive}/?year={month.year}&month={month.month}"
        with urllib.request.urlopen(url, timeout=60) as response:
            timestamps = re.findall(r"\d{8}T\d{6}Z", response.read().decode())
        candidates = [timestamp for timestamp in timestamps if timestamp <= cutoff]
        if candidates:
            return f"https://snapshot.debian.org/archive/{archive}/{max(candidates)}/"
        month = (month - dt.timedelta(days=1)).replace(day=1)
    raise ValueError(f"找不到截止 {cutoff} 的 {archive} 快照")


def resolve_in_container(cutoff: str, output: Path) -> None:
    if not Path("/.dockerenv").is_file() or os.environ.get("ACPROF_SYSTEM_LOCK_CONTAINER") != "1":
        raise RuntimeError("系统依赖解析只能在专用的一次性基础容器中执行")
    sources = [{"url": snapshot_url(archive, cutoff), "suites": list(suites)}
               for archive, suites in ARCHIVES.items()]
    for path in Path("/etc/apt/sources.list.d").glob("*"):
        path.unlink()
    Path("/etc/apt/sources.list").write_text("".join(
        f"deb [check-valid-until=no signed-by=/usr/share/keyrings/debian-archive-keyring.pgp] {source['url']} {suite} main\n"
        for source in sources for suite in source["suites"]
    ))
    # APT 核验 Debian 签名；仅关闭历史快照的时间过期检查。
    subprocess.run(["apt-get", "-o", "Acquire::Retries=3", "update"], check=True)
    command = ["apt-get", "install", "-y", "--no-install-recommends", *DIRECT_PACKAGES]
    printed = subprocess.check_output([*command, "--print-uris"], text=True)
    urls = {parts[1]: parts[0] for line in printed.splitlines() if line.startswith("'")
            for parts in [shlex.split(line)]}
    subprocess.run([*command, "--download-only"], check=True)
    artifacts = []
    for path in sorted(Path("/var/cache/apt/archives").glob("*.deb")):
        raw = subprocess.check_output(["dpkg-deb", "-f", str(path)], text=True)
        fields = dict(line.split(": ", 1) for line in raw.splitlines()
                      if ": " in line and not line.startswith(" "))
        artifacts.append({"name": fields["Package"], "version": fields["Version"],
                          "architecture": fields["Architecture"], "url": urls[path.name],
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size})
    subprocess.run([*command, "--no-download"], check=True)
    raw = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${Package}:${Architecture}\t${Version}\t${db:Status-Status}\n"], text=True)
    packages = {name: version for line in raw.splitlines()
                for name, version, status in [line.split("\t")] if status == "installed"}
    releases = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("/var/lib/apt/lists").glob("*InRelease"))}
    data = {"schema_version": 1, "base_image": PYTHON_BASE_IMAGE, "architecture": "amd64",
            "sources": sources, "release_sha256": releases, "packages": packages, "artifacts": artifacts}
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    read_system_lock(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="快照截止时间，格式 YYYYMMDDTHHMMSSZ")
    parser.add_argument("--output", type=Path, default=ROOT / "dockerfiles/locks/system-trixie-amd64.json")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        dt.datetime.strptime(args.snapshot, "%Y%m%dT%H%M%SZ")
    except ValueError:
        parser.error("--snapshot 必须是有效的 YYYYMMDDTHHMMSSZ 日期")
    if args.inside_container:
        resolve_in_container(args.snapshot, args.output)
        return 0
    # 不在主机安装系统包；候选锁完全解析及校验成功后才替换目标。
    with tempfile.TemporaryDirectory(prefix="acprof-system-lock-") as directory:
        output = Path(directory) / "system.json"
        subprocess.run([
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "-e", "DEBIAN_FRONTEND=noninteractive", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "ACPROF_SYSTEM_LOCK_CONTAINER=1", "-v", f"{ROOT}:/workspace:ro",
            "-v", f"{directory}:/evidence", PYTHON_BASE_IMAGE,
            "python", "/workspace/scripts/compile_system_lock.py", "--inside-container",
            "--snapshot", args.snapshot, "--output", "/evidence/system.json",
        ], check=True)
        lock = read_system_lock(output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, mode="w", delete=False) as target:
            target.write(output.read_text())
            temporary = Path(target.name)
        try:
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Locked {len(lock['packages'])} system packages and {len(lock['artifacts'])} deb artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

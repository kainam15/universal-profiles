"""完整依赖锁的标准库读取与校验；不解析依赖、不探测运行设备。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def content_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def read_python_lock(path: str | Path) -> list[dict[str, str]]:
    """仅接受本项目生成的单一 wheel URL + SHA256 锁，拒绝未锁定安装语法。"""
    records = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+) @ (https://\S+) --hash=sha256:([a-f0-9]{64})", line)
        if not match:
            raise ValueError(f"无效或未锁定的依赖 lock 条目：{path}: {line[:100]}")
        name, url, digest = match.groups()
        parsed = urlsplit(url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("依赖 lock URL 不允许凭据、查询参数或 fragment")
        filename = unquote(parsed.path.rsplit("/", 1)[-1])
        parts = filename.removesuffix(".whl").split("-")
        name = normalized_name(name)
        if not filename.endswith(".whl") or len(parts) not in (5, 6) or normalized_name(parts[0]) != name:
            raise ValueError(f"锁定 wheel 与包名不符：{name}")
        if name in records:
            raise ValueError(f"依赖 lock 包名重复：{name}")
        records[name] = {"name": name, "version": parts[1], "url": url, "sha256": digest}
    if not records:
        raise ValueError(f"依赖 lock 不能为空：{path}")
    return [records[name] for name in sorted(records)]


def python_lock_text(records: list[dict[str, str]]) -> str:
    lines = ["# Generated from uv's target-specific wheel resolution. Do not edit artifact URLs by hand."]
    for entry in sorted(records, key=lambda entry: entry["name"]):
        lines += [f"# {entry['name']}=={entry['version']}",
                  f"{entry['name']} @ {entry['url']} --hash=sha256:{entry['sha256']}"]
    return "\n".join(lines) + "\n"


def package_versions(records: list[dict[str, str]]) -> dict[str, str]:
    return {entry["name"]: entry["version"] for entry in records}


def require_parent_subset(parent: list[dict[str, str]], child: list[dict[str, str]]) -> None:
    packages = {entry["name"]: entry for entry in child}
    for entry in parent:
        if packages.get(entry["name"]) != entry:
            raise ValueError(f"依赖环境缺少或覆盖父层锁定包：{entry['name']}")


def require_exact_packages(expected: dict[str, str], actual: dict[str, str], *, kind="Python") -> None:
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = {key: {"expected": expected[key], "actual": actual[key]}
                   for key in sorted(expected.keys() & actual.keys()) if expected[key] != actual[key]}
        raise ValueError(f"{kind} package set differs from lock: missing={missing}, extra={extra}, changed={changed}")


def read_system_lock(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("无效 system lock schema_version")
    if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", data.get("base_image", "")):
        raise ValueError("system lock 必须固定基础镜像 digest")
    for source in data.get("sources", []):
        if not re.fullmatch(r"https://snapshot\.debian\.org/archive/(debian|debian-security)/\d{8}T\d{6}Z/", source["url"]):
            raise ValueError("system lock 必须使用固定 Debian snapshot")
    if len(data.get("sources", [])) != 2 or not data.get("packages") or not data.get("artifacts"):
        raise ValueError("system lock 缺少来源、完整包集合或 deb 制品")
    artifact_names = set()
    for artifact in data["artifacts"]:
        if not any(artifact["url"].startswith(source["url"]) for source in data["sources"]):
            raise ValueError("deb 制品不属于锁定 snapshot")
        if not re.fullmatch(r"[a-f0-9]{64}", artifact["sha256"]):
            raise ValueError("deb 制品缺少 SHA256")
        name = f"{artifact['name']}:{artifact['architecture']}"
        if name in artifact_names or data["packages"].get(name) != artifact["version"]:
            raise ValueError("deb 制品重复或与最终系统包集合不符")
        artifact_names.add(name)
    return data


def system_lock_identity(data: dict) -> dict:
    return {key: data[key] for key in ("schema_version", "base_image", "architecture", "sources",
                                     "release_sha256", "packages", "artifacts")}

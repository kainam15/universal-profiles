"""用固定 uv 生成目标 wheel 锁；--check 只读核验平台、环境及 profile 映射。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.dependency_locks import normalized_name, package_versions, python_lock_text, read_python_lock  # noqa: E402 -- 脚本先设置仓库导入路径。
from acprof.runtime_profiles import ENVIRONMENTS, PLATFORMS, PROFILES, environment_id, environment_identity  # noqa: E402 -- 脚本先设置仓库导入路径。

UV_VERSION = "0.12.13"


def target_markers(platform):
    return {"python_version": ".".join(platform.python_version.split(".")[:2]),
            "python_full_version": platform.python_version, "sys_platform": "linux",
            "platform_machine": "x86_64", "platform_system": "Linux", "os_name": "posix",
            "implementation_name": "cpython", "implementation_version": platform.python_version,
            "platform_python_implementation": "CPython", "extra": ""}


def target_tags(platform):
    from packaging.tags import compatible_tags, cpython_tags
    if platform.architecture != "linux/amd64" or platform.python_target != "x86_64-manylinux_2_28":
        raise ValueError("尚未注册此目标平台的 wheel 选择规则")
    version = tuple(int(value) for value in platform.python_version.split(".")[:2])
    interpreter = "cp" + "".join(str(value) for value in version)
    # 与声明的 manylinux 2.28 下限一致；不使用主机 Python/ABI/体系结构。
    platforms = [f"manylinux_2_{v}_x86_64" for v in range(28, 4, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    tags = list(cpython_tags(version, abis=[interpreter], platforms=platforms))
    tags += list(compatible_tags(version, interpreter=interpreter, platforms=platforms))
    return tags


def wheel_records(data, platform):
    from packaging.markers import Marker
    from packaging.utils import parse_wheel_filename
    rank = {tag: index for index, tag in enumerate(target_tags(platform))}
    records = []
    for package in data["packages"]:
        if package.get("marker") and not Marker(package["marker"]).evaluate(target_markers(platform)):
            continue
        ranked = []
        for wheel in package.get("wheels", []):
            parsed = parse_wheel_filename(unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1]))
            matches = [rank[tag] for tag in parsed[3] if tag in rank]
            if matches:
                ranked.append((min(matches), wheel["url"], wheel))
        if not ranked:
            raise ValueError(f"没有目标平台 wheel；不会自动从源码构建：{package['name']}")
        wheel = min(ranked, key=lambda item: (item[0], item[1]))[2]
        records.append({"name": package["name"], "version": package["version"],
                        "url": wheel["url"], "sha256": wheel["hashes"]["sha256"]})
    return records


def check_sources(paths, packages, root, visited=None):
    from packaging.requirements import Requirement
    visited = set() if visited is None else visited
    for relative in paths:
        path = (root / relative).resolve()
        if path in visited:
            continue
        visited.add(path)
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-r "):
                check_sources([str(path.parent / line[3:].strip())], packages, root, visited)
                continue
            requirement = Requirement(line)
            name = normalized_name(requirement.name)
            if name not in packages or not requirement.specifier.contains(packages[name], prereleases=True):
                raise ValueError(f"依赖输入与 lock 不符：{path}: {line}")


def check_catalog(root=ROOT, variants=None):
    from packaging.utils import parse_wheel_filename
    environments = {}
    selected = set(variants or PLATFORMS)
    for key, environment in ENVIRONMENTS.items():
        if environment.platform.platform_id not in selected:
            continue
        identity = environment_identity(environment, root)
        compatible = set(target_tags(environment.platform))
        for record in identity["packages"]:
            filename = unquote(urlsplit(record["url"]).path.rsplit("/", 1)[-1])
            if not compatible.intersection(parse_wheel_filename(filename)[3]):
                raise ValueError(f"wheel 与目标平台不兼容：{key}: {filename}")
        check_sources(environment.requirements_inputs, package_versions(identity["packages"]), root)
        identifier = environment_id(environment, root)
        if identifier in environments:
            raise ValueError(f"重复依赖环境应合并声明：{key} / {environments[identifier]['environment_key']}")
        environments[identifier] = {"environment_key": key, "platform_id": environment.platform.platform_id,
                                    "profiles": []}
    for profile in PROFILES.values():
        if profile.environment.platform.platform_id not in selected:
            continue
        if profile.environment not in ENVIRONMENTS.values():
            raise ValueError(f"profile 引用了未登记的依赖环境：{profile.profile_id}")
        environments[environment_id(profile.environment, root)]["profiles"].append(profile.profile_id)
    return {"profiles": sum(len(item["profiles"]) for item in environments.values()),
            "environments": environments, "platforms": sorted(selected)}


def resolve(uv, inputs, output, platform, pins):
    try:
        import tomllib
    except ImportError as error:
        raise RuntimeError("重新生成 wheel 锁需要 Python 3.11+；只读 --check 支持 Python 3.10+") from error
    with tempfile.TemporaryDirectory(prefix="acprof-resolve-") as directory:
        directory = Path(directory)
        constraint = directory / "constraints.txt"
        constraint.write_text("".join(f"{name}=={version}\n" for name, version in sorted(pins.items())))
        target = directory / "pylock.toml"
        command = [uv, "pip", "compile", *map(str, inputs), "--constraint", str(constraint),
                        "--python-version", platform.python_version, "--python-platform", platform.python_target,
                        "--only-binary", ":all:",
                        "--format", "pylock.toml", "--no-header", "--output-file", str(target)]
        if platform.torch_version:
            command += ["--torch-backend", platform.platform_id]
        subprocess.run(command,
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        records = wheel_records(tomllib.loads(target.read_text()), platform)
        for name, version in package_versions(records).items():
            if name in pins and pins[name] != version:
                raise ValueError(f"解析改变了锁定版本：{name}")
        text = python_lock_text(records)
        # 先核验生成格式再替换，失败时保留原锁。
        generated = directory / "requirements.txt"
        generated.write_text(text)
        read_python_lock(generated)
        output.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv", help=f"uv 可执行文件，要求 {UV_VERSION}")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--variant", action="append", choices=PLATFORMS)
    parser.add_argument("--check", action="store_true", help="不访问 Docker/网络、不写文件，核验容器锁及映射")
    parser.add_argument("--upgrade", action="store_true", help="显式更新环境包；平台 Torch 和安装工具仍保持锁定")
    args = parser.parse_args(argv)
    if args.host_only and args.runtime_only:
        parser.error("--host-only 与 --runtime-only 互斥")
    if args.check:
        if args.host_only or args.upgrade:
            parser.error("--check 不与 --host-only / --upgrade 同用")
        print(json.dumps(check_catalog(variants=args.variant), ensure_ascii=False, indent=2))
        return 0
    version = subprocess.check_output([args.uv, "--version"], text=True).split()
    if version[:2] != ["uv", UV_VERSION]:
        parser.error(f"锁生成工具必须是 uv {UV_VERSION}")
    if not args.runtime_only:
        command = [args.uv, "pip", "compile", "requirements-host.in", "--python-version", "3.10",
                   "--universal", "--generate-hashes", "--no-annotate", "--no-header", "-o", "requirements.lock"]
        subprocess.run(command + (["--upgrade"] if args.upgrade else []), cwd=ROOT, check=True)
    if args.host_only:
        return 0
    for key in args.variant or PLATFORMS:
        platform = PLATFORMS[key]
        platform_pins = package_versions(read_python_lock(ROOT / platform.requirements_lock))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "platform.in"
            base_packages = ("pip", "wheel", "packaging", "setuptools")
            if platform.torch_version:
                base_packages = ("torch", *base_packages)
            source.write_text("".join(f"{name}=={platform_pins[name]}\n" for name in base_packages))
            resolve(args.uv, [source], ROOT / platform.requirements_lock, platform, platform_pins)
        for environment in ENVIRONMENTS.values():
            if environment.platform.platform_id != key:
                continue
            pins = {} if args.upgrade else package_versions(read_python_lock(ROOT / environment.requirements_lock))
            pins.update(package_versions(read_python_lock(ROOT / platform.requirements_lock)))
            # 平台完整闭包与安装工具也是环境的显式依赖。
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "platform.in"
                source.write_text("".join(f"{name}=={version}\n" for name, version in platform_pins.items()))
                resolve(args.uv, [*(ROOT / name for name in environment.requirements_inputs), source],
                        ROOT / environment.requirements_lock, platform, pins)
    check_catalog(variants=args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

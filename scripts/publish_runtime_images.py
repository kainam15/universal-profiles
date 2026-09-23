"""Build and verify shared dependency images; publish only with explicit --push."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.host.dependency_images import (  # noqa: E402 -- 源码发布脚本先设置仓库导入路径。
    DEFAULT_RUNTIME_REGISTRY, prepare_environment_image, prepare_platform_image, registry_reference,
)
from acprof.runtime_profiles import ENVIRONMENTS, PLATFORMS  # noqa: E402 -- 同上。


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--matrix", action="store_true", help="只输出 CI matrix，不访问 Docker")
    selection.add_argument("--platform", choices=tuple(PLATFORMS))
    selection.add_argument("--environment", choices=tuple(ENVIRONMENTS))
    parser.add_argument("--registry", default=DEFAULT_RUNTIME_REGISTRY)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--pull-platform", action="store_true",
                        help="环境发布必须复用已发布平台的不可变 ID，不在每个 job 重建父层")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.matrix:
        print(json.dumps({"platforms": list(PLATFORMS), "environments": list(ENVIRONMENTS)}))
        return 0
    if args.platform:
        prepared = prepare_platform_image(PLATFORMS[args.platform], ROOT, image_source="auto",
                                           registry=args.registry)
        kind, fingerprint = "platform", prepared.expected["platform_build_fingerprint"]
    else:
        environment = ENVIRONMENTS[args.environment]
        if args.pull_platform:
            prepare_platform_image(environment.platform, ROOT, image_source="pull", registry=args.registry)
        prepared = prepare_environment_image(environment, ROOT, image_source="build")
        kind, fingerprint = "environment", prepared.manifest["environment_build_fingerprint"]
    reference = registry_reference(kind, fingerprint, args.registry)
    report = {"kind": kind, "image_id": prepared.image_id, "reference": reference,
              "verified": "dependency_manifest", "pushed": False}
    if args.push:
        subprocess.run(["docker", "tag", prepared.image_id, reference], check=True)
        subprocess.run(["docker", "push", reference], check=True)
        report["pushed"] = True
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

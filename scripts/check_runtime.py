"""构建指定锁定环境，离线运行真实小模型接口测试并拒绝跳过。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.artifacts import atomic_write_json
from acprof.host.dependency_images import prepare_environment_image
from acprof.runtime_profiles import DEFAULT_PROFILES, PROFILES, environment_id

PATTERNS = {
    "nlp": ("test_nlp_runtime.py",),
    "cv": ("test_cv_runtime.py",),
    "audio": ("test_audio_runtime_optional.py",),
    "diffusion": ("test_diffusion_runtime.py",),
    "structured": ("test_structured_runtime.py",),
    "timeseries": ("test_timeseries_runtime.py",),
    "multimodal": ("test_multimodal_runtime.py", "test_multimodal_generation_runtime.py"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--family", choices=PATTERNS)
    selection.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--variant", choices=("cpu", "cu124", "cu128"), default="cpu")
    parser.add_argument("--build-only", action="store_true", help="仅构建并核验完整依赖清单，不宣称模型接口验证通过")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("验证输出目录必须为空")
    name = args.profile or DEFAULT_PROFILES[(args.family, args.variant)]
    profile = PROFILES[name]
    if profile.adapter != "family-default" and not args.build_only:
        parser.error("自定义 adapter 请通过主流程独立 runtime validation 验证；此入口可用 --build-only 核验依赖")
    started = time.perf_counter()
    result = {"schema_version": 1, "family": profile.family, "profile": profile.to_dict(),
              "validation_scope": "dependencies" if args.build_only else "offline_cpu_interfaces",
              "device": None if args.build_only else "cpu", "successful": False}
    try:
        result["environment_id"] = environment_id(profile.environment, ROOT)
        image = prepare_environment_image(profile.environment, ROOT)
        image_id = image.image_id
        result["image_id"] = image_id
        result["image"] = image.name
        result["platform_image_id"] = image.platform_image_id
        result["environment_manifest"] = image.manifest
        if args.build_only:
            result["successful"] = True
            return 0
        command = [
            "docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "4g",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{ROOT}:/workspace:ro", "-v", f"{output}:/evidence", "-w", "/workspace",
        ]
        for env in ("HOME=/tmp", "USER=acprof", "LOGNAME=acprof", "HF_HOME=/tmp/hf", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1"):
            command += ["-e", env]
        subprocess.run([*command, image_id, "python", "-m", "pip", "check"], check=True)
        run_command = [*command, image_id, "python", "scripts/run_tests.py",
                       "--require-no-skips", "--report", "/evidence/tests.json"]
        for pattern in PATTERNS[profile.family]:
            run_command += ["--pattern", pattern]
        code = subprocess.run(run_command, cwd=ROOT).returncode
        result["successful"] = code == 0
        return code
    except subprocess.CalledProcessError as error:
        result["error"] = str(error)
        return error.returncode or 1
    except (RuntimeError, ValueError, OSError) as error:
        result["error"] = str(error)
        return 1
    finally:
        result["duration_s"] = time.perf_counter() - started
        atomic_write_json(output / "runtime.json", result)


if __name__ == "__main__":
    raise SystemExit(main())

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
from acprof.host.runtime_images import runtime_fingerprint
from acprof.runtime_profiles import PROFILES

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
    parser.add_argument("--family", choices=PATTERNS, required=True)
    parser.add_argument("--variant", choices=("cpu", "cu124", "cu128"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("验证输出目录必须为空")
    stem = "multimodal-transformers4576" if args.family == "multimodal" else args.family
    name = stem if args.family == "multimodal" and args.variant == "cu128" else f"{stem}-{args.variant}"
    profile = PROFILES[name]
    image = f"acprof-runtime-{name}:{runtime_fingerprint(profile)[:20]}"
    started = time.perf_counter()
    result = {"schema_version": 1, "family": args.family, "profile": profile.to_dict(),
              "device": "cpu", "successful": False, "image": image}
    try:
        subprocess.run([
            "docker", "build", "-f", str(ROOT / "dockerfiles/runtime.Dockerfile"),
            "--build-arg", f"PYTHON_BASE_IMAGE={profile.python_base_image}",
            "--build-arg", f"TORCH_INDEX_URL={profile.torch_index_url}",
            "--build-arg", f"COMMON_REQUIREMENTS_LOCK={profile.common_requirements_lock}",
            "--build-arg", f"REQUIREMENTS_LOCK={profile.requirements_lock}",
            "-t", image, str(ROOT),
        ], check=True, cwd=ROOT)
        image_id = subprocess.check_output([
            "docker", "image", "inspect", image, "--format", "{{.Id}}",
        ], text=True).strip()
        result["image_id"] = image_id
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
        for pattern in PATTERNS[args.family]:
            run_command += ["--pattern", pattern]
        code = subprocess.run(run_command, cwd=ROOT).returncode
        result["successful"] = code == 0
        return code
    except subprocess.CalledProcessError as error:
        result["error"] = str(error)
        return error.returncode or 1
    finally:
        result["duration_s"] = time.perf_counter() - started
        atomic_write_json(output / "runtime.json", result)


if __name__ == "__main__":
    raise SystemExit(main())

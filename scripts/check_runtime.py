"""构建指定锁定环境，离线运行真实小模型接口测试并拒绝跳过。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.artifacts import atomic_write_json  # noqa: E402 -- 脚本先设置仓库导入路径。
from acprof.host.dependency_images import prepare_environment_image  # noqa: E402 -- 脚本先设置仓库导入路径。
from acprof.runtime_profiles import DEFAULT_PROFILES, PROFILES, environment_id  # noqa: E402 -- 脚本先设置仓库导入路径。

PATTERNS = {
    "nlp": ("test_nlp_runtime.py",),
    "cv": ("test_cv_runtime.py",),
    "audio": ("test_audio_runtime_optional.py",),
    "diffusion": ("test_diffusion_runtime.py",),
    "structured": ("test_structured_runtime.py",),
    "timeseries": ("test_timeseries_runtime.py",),
    "multimodal": ("test_multimodal_runtime.py", "test_multimodal_generation_runtime.py", "test_audio_generation_runtime.py"),
}
RUNTIME_PATTERNS = {"onnxruntime": ("test_onnx_runtime_optional.py", "test_onnx_tasks_runtime.py",
                                    "test_request_completion.py", "test_onnx_server_runtime.py",
                                    "test_onnx_planning_runtime.py")}
ONNX_ENVIRONMENT_CHECK = """import importlib.util
for name in ('torch', 'transformers'):
    assert importlib.util.find_spec(name) is None, f'{name} must not be installed in the ONNX job'
for name in ('onnx', 'onnxruntime', 'numpy', 'flask', 'PIL', 'tokenizers'):
    assert importlib.util.find_spec(name) is not None, f'required ONNX dependency missing: {name}'
import onnx, onnxruntime, numpy, flask, PIL, tokenizers
print('ONNX dependencies imported; torch and transformers are absent')
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--family", choices=PATTERNS)
    selection.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--variant", choices=("cpu", "cu124", "cu128"), default="cpu")
    parser.add_argument("--build-only", action="store_true", help="仅构建并核验完整依赖清单，不宣称模型接口验证通过")
    parser.add_argument("--test-pattern", action="append", help="覆盖任务族默认接口测试，可重复指定扩展测试文件模式")
    parser.add_argument("--basic-e2e", action="store_true", help="ONNX CPU 表格、图像、文本服务的真实 basic 采集与审计回归")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="单个容器验证命令的超时秒数")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.build_only and args.test_pattern:
        parser.error("--build-only 不运行接口测试，不能与 --test-pattern 同用")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds 必须大于零")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("验证输出目录必须为空")
    name = args.profile or DEFAULT_PROFILES[(args.family, args.variant)]
    profile = PROFILES[name]
    runtime_type = profile.environment.runtime_type
    patterns = args.test_pattern or RUNTIME_PATTERNS.get(runtime_type) or PATTERNS.get(profile.family)
    if args.basic_e2e and (name != "onnxruntime-cpu" or args.build_only):
        parser.error("--basic-e2e 仅适用于 onnxruntime-cpu 接口验证")
    if profile.adapter != "family-default" and not args.build_only and not args.test_pattern:
        parser.error("自定义 adapter 请通过主流程独立 runtime validation 验证；此入口可用 --build-only 核验依赖")
    if not args.build_only and not patterns:
        parser.error("此 runtime/family 尚无默认测试，请显式提供 --test-pattern")
    started = time.perf_counter()
    result = {"schema_version": 1, "family": profile.family, "profile": profile.to_dict(),
              "validation_scope": "dependencies" if args.build_only else "offline_cpu_interfaces",
              "device": None if args.build_only else "cpu", "successful": False,
              "test_patterns": [] if args.build_only else list(patterns)}
    container_name = "acprof-runtime-check-" + uuid.uuid4().hex[:12]
    container_started = False
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
            "docker", "run", "--rm", "--name", container_name, "--network", "none", "--cpus", "2", "--memory", "4g",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{ROOT}:/workspace:ro", "-v", f"{output}:/evidence", "-w", "/workspace",
        ]
        for env in ("HOME=/tmp", "USER=acprof", "LOGNAME=acprof", "HF_HOME=/tmp/hf", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1"):
            command += ["-e", env]
        container_started = True
        subprocess.run([*command, image_id, "python", "-m", "pip", "check"], check=True, timeout=args.timeout_seconds)
        if runtime_type == "onnxruntime":
            subprocess.run([*command, image_id, "python", "-c", ONNX_ENVIRONMENT_CHECK],
                           check=True, timeout=args.timeout_seconds)
        run_command = [*command, image_id, "python", "scripts/run_tests.py",
                       "--require-no-skips", "--report", "/evidence/tests.json"]
        for pattern in result["test_patterns"]:
            run_command += ["--pattern", pattern]
        code = subprocess.run(run_command, cwd=ROOT, timeout=args.timeout_seconds).returncode
        if code == 0 and args.basic_e2e:
            from scripts.check_onnx_basic import run_basic_e2e
            from examples.onnxruntime.fixtures import BASIC_SCENARIOS
            result["basic_task_e2e"] = {}
            for scenario in BASIC_SCENARIOS:
                directory = "basic" if scenario == "tabular" else "basic-" + scenario
                result["basic_task_e2e"][scenario] = run_basic_e2e(
                    image_id, output / directory, timeout_seconds=args.timeout_seconds, scenario=scenario)
            result["basic_e2e"] = result["basic_task_e2e"]["tabular"]
            code = 0 if all(item["successful"] for item in result["basic_task_e2e"].values()) else 1
        result["successful"] = code == 0
        return code
    except subprocess.CalledProcessError as error:
        result["error"] = str(error)
        return error.returncode or 1
    except subprocess.TimeoutExpired as error:
        result["error"] = f"validation timeout: {error}"
        return 1
    except (RuntimeError, ValueError, OSError) as error:
        result["error"] = str(error)
        return 1
    finally:
        cleanup_error = ""
        if container_started:
            # A timed-out docker CLI does not guarantee the container has stopped.
            try:
                cleanup = subprocess.run(["docker", "rm", "-f", container_name], capture_output=True,
                                         text=True, timeout=30, check=False)
                if cleanup.returncode and "No such container" not in (cleanup.stderr or ""):
                    cleanup_error = cleanup.stderr or "docker container cleanup failed"
            except (OSError, subprocess.TimeoutExpired) as error:
                cleanup_error = str(error)
            result["cleanup"] = {"successful": not cleanup_error, "error": cleanup_error}
            if cleanup_error:
                result["successful"] = False
        result["duration_s"] = time.perf_counter() - started
        atomic_write_json(output / "runtime.json", result)
        if cleanup_error:
            return 1


if __name__ == "__main__":
    raise SystemExit(main())

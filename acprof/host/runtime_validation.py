"""在资源矩阵之前验证接口；验证容器退出后才开始正式采集。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from acprof.container.runtime_validate import RESULT_PREFIX


def validate_runtime(
    *, task_info: Any, image_info: Any, planned: Any, cpu_list: list[int],
    mem_list: list[int], gpu_list: list[str], output_dir: str,
    timeout_seconds: float = 300.0,
) -> dict:
    if not getattr(image_info, "runtime_environment", {}):
        raise ValueError("镜像缺少 runtime_environment；请使用当前版本重新构建运行环境")
    from acprof.host.docker_runtime import _inspect_container_state
    from acprof.host.env_utils import hf_offline_docker_env_args
    from acprof.artifacts import require_schema_version

    plan = json.loads(Path(planned.plan_file).read_text())
    require_schema_version(plan, 2, "input_scale_plan.json")
    entry = min(plan["entries"], key=lambda item: float(item["input_scale"]))
    encoded = json.dumps(entry["payload"], ensure_ascii=False).encode()
    report: dict[str, Any] = {
        "schema_version": 1, "status": "running", "image_id": image_info.tag,
        "build_fingerprint": image_info.runtime_environment["build_fingerprint"],
        "input_scale": entry["input_scale"], "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "scope": "isolated_minimum_scale_predict_and_postprocess_before_measurement",
        "devices": {},
        "profiler_validation": "separate_profiler_plans; inference_success_does_not_prove_profiler_support",
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    failure = None
    with tempfile.TemporaryDirectory(prefix="acprof-runtime-validation-") as temporary:
        payload = Path(temporary) / "payload.json"
        payload.write_bytes(encoded)
        for mode in dict.fromkeys(gpu_list):
            name = "acprof-validate-" + uuid.uuid4().hex[:16]
            command = [
                "docker", "run", "--name", name, "--network", "none",
                f"--cpus={max(cpu_list)}", f"--memory={max(mem_list)}g",
                "-v", f"{payload}:/validation-input.json:ro",
                "-e", f"MODEL_ID={task_info.model_id}",
                "-e", f"MODEL_REVISION={task_info.model_revision}",
                "-e", f"TASK_FAMILY={task_info.task_family}",
                "-e", f"TASK_TYPE={task_info.pipeline_tag}",
                "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
                "-e", f"USE_GPU={int(mode == 'on')}",
                "-e", f"TORCH_NUM_THREADS={max(cpu_list)}",
                *hf_offline_docker_env_args(),
            ]
            if mode == "on":
                command += ["--gpus", "all"]
            command += ["--entrypoint", "python", image_info.tag, "-m", "acprof.container.runtime_validate", "/validation-input.json"]
            print(f"[runtime-check] {mode}: 验证加载、推理和输出协议（独立容器）", flush=True)
            log = ""
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
                log = (result.stdout or "") + "\n" + (result.stderr or "")
                records = [line[len(RESULT_PREFIX):] for line in (result.stdout or "").splitlines() if line.startswith(RESULT_PREFIX)]
                state = _inspect_container_state(name) or {}
                if state.get("OOMKilled"):
                    device_result = {"status": "resource_limit", "error": "validation_container_oom", "mem_cap_gb": max(mem_list)}
                elif records:
                    device_result = json.loads(records[-1])
                    if not isinstance(device_result, dict) or device_result.get("status") not in {"ok", "error"}:
                        raise ValueError("invalid runtime validation response")
                    if result.returncode and device_result.get("status") == "ok":
                        device_result = {"status": "error", "error": f"container exit {result.returncode}"}
                else:
                    device_result = {"status": "error", "error": log[-4000:] or f"container exit {result.returncode}"}
            except subprocess.TimeoutExpired as exc:
                def decoded(value):
                    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")

                log = decoded(exc.stdout) + "\n" + decoded(exc.stderr)
                device_result = {"status": "error", "error": f"runtime_validation_timeout ({timeout_seconds:g}s)"}
            except (ValueError, OSError) as exc:
                device_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
            (root / f"runtime_validation_{mode}.log").write_text(log)
            report["devices"][mode] = device_result
            if device_result["status"] == "error":
                failure = f"{mode}: {device_result.get('error', 'runtime validation failed')}"
                break
            print(f"[runtime-check] {mode}: {device_result['status']}", flush=True)
    report["status"] = "error" if failure else (
        "ok" if all(item["status"] == "ok" for item in report["devices"].values()) else "resource_limited"
    )
    (root / "runtime_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if failure:
        raise RuntimeError(f"运行环境验证失败，未进入资源矩阵。{failure}\n完整日志：{root / 'runtime_validation.json'}")
    return report

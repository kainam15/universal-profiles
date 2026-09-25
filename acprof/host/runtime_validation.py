"""在资源矩阵之前验证接口；验证容器退出后才开始正式采集。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from acprof.container.runtime_validate import RESULT_PREFIX
from acprof.host.gpu_device import gpu_docker_args
from acprof.runtime_settings import runtime_docker_env_args


def validate_runtime(
    *, task_info: Any, image_info: Any, planned: Any, cpu_list: list[int],
    mem_list: list[int], gpu_list: list[str], output_dir: str,
    timeout_seconds: float = 300.0,
    mode: str = "full",
) -> dict:
    if mode not in {"basic", "full"} or mode == "basic" and gpu_list != ["off"]:
        raise ValueError("contract probe mode must be basic (CPU only) or full")
    if (not gpu_list or any(value not in {"off", "on"} for value in gpu_list) or not cpu_list or not mem_list
            or any(type(value) is not int or value <= 0 for value in [*cpu_list, *mem_list])
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("runtime validation requires devices and positive resource/timeout limits")
    if not getattr(image_info, "runtime_environment", {}):
        raise ValueError("镜像缺少 runtime_environment；请使用当前版本重新构建运行环境")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_info.tag):
        raise ValueError("runtime validation requires an immutable image ID")
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
        "scope": "isolated_import_and_signature_only" if mode == "basic" else
                 "isolated_minimum_scale_predict_and_postprocess_before_measurement",
        "devices": {},
        "mode": mode,
        "profiler_validation": "separate_profiler_plans; inference_success_does_not_prove_profiler_support",
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    failure = None
    with tempfile.TemporaryDirectory(prefix="acprof-runtime-validation-") as temporary:
        payload = Path(temporary) / "payload.json"
        payload.write_bytes(encoded)
        for device_mode in dict.fromkeys(gpu_list):
            name = "acprof-validate-" + uuid.uuid4().hex[:16]
            command = [
                "docker", "run", "--name", name, "--network", "none",
                "--read-only", "--cap-drop=ALL", "--security-opt", "no-new-privileges", "--pids-limit=256",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
                f"--cpus={max(cpu_list)}", f"--memory={max(mem_list)}g",
                "-v", f"{payload}:/validation-input.json:ro",
                "-e", f"MODEL_ID={task_info.model_id}",
                "-e", f"MODEL_REVISION={task_info.model_revision}",
                "-e", f"TASK_FAMILY={task_info.task_family}",
                "-e", f"TASK_TYPE={task_info.pipeline_tag}",
                "-e", f"RUNTIME_BACKEND={task_info.runtime_backend}",
                "-e", f"USE_GPU={int(device_mode == 'on')}",
                "-e", f"TORCH_NUM_THREADS={max(cpu_list)}",
                "-e", f"ACPROF_REQUEST_TIMEOUT_S={timeout_seconds:g}",
                *runtime_docker_env_args(),
                *hf_offline_docker_env_args(),
                "-e", "HF_MODULES_CACHE=/tmp/hf-modules", "-e", "XDG_CACHE_HOME=/tmp/cache",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", f"ACPROF_CONTRACT_PROBE_MODE={mode}",
            ]
            if device_mode == "on":
                command += gpu_docker_args()
            command += ["--entrypoint", "python", image_info.tag, "-m", "acprof.container.runtime_validate", "/validation-input.json"]
            print(f"[runtime-check] {device_mode}: {mode} 契约验证（独立容器）", flush=True)
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
                    if getattr(task_info, "model_resolution", {}).get("contract") and device_result.get("status") == "ok":
                        required = {"import", "signature"} if mode == "basic" else {
                            "load", "preprocess", "predict", "postprocess", "validate_output"}
                        observed = {item.get("stage") for item in device_result.get("stages", [])
                                    if isinstance(item, dict) and item.get("status") == "verified"}
                        if not required <= observed or mode == "full" and any(
                            device_result.get("validation", {}).get(layer, {}).get("status") != "verified"
                            for layer in ("protocol", "task")
                        ):
                            raise ValueError("incomplete contract validation response")
                else:
                    device_result = {"status": "error", "error": log[-4000:] or f"container exit {result.returncode}"}
            except subprocess.TimeoutExpired as exc:
                def decoded(value):
                    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")

                log = decoded(exc.stdout) + "\n" + decoded(exc.stderr)
                device_result = {"status": "error", "error": f"runtime_validation_timeout ({timeout_seconds:g}s)"}
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                device_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
            (root / f"runtime_validation_{device_mode}.log").write_text(log)
            report["devices"][device_mode] = device_result
            if device_result["status"] == "error":
                failure = f"{device_mode}: {device_result.get('error', 'runtime validation failed')}"
                break
            print(f"[runtime-check] {device_mode}: {device_result['status']}", flush=True)
    report["status"] = "error" if failure else (
        "ok" if all(item["status"] == "ok" for item in report["devices"].values()) else "resource_limited"
    )
    from acprof.artifacts import atomic_write_json
    atomic_write_json(root / "runtime_validation.json", report)
    if getattr(task_info, "model_resolution", {}):
        from acprof.model_contract import record_runtime_validation, write_model_resolution
        record_runtime_validation(task_info, report)
        write_model_resolution(task_info, root)
    if failure:
        raise RuntimeError(f"运行环境验证失败，未进入资源矩阵。{failure}\n完整日志：{root / 'runtime_validation.json'}")
    return report

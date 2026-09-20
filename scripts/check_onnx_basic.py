"""本地合成 ONNX 制品的 basic 服务回归；复用生产 client、cgroup 采样和结果审计。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acprof.analysis.audit import audit_result
from acprof.artifacts import atomic_write_json
from acprof.capabilities import apply_collection_result, apply_runtime_validation, measurement_report
from acprof.config import STATIC_META_SCHEMA_VERSION
from acprof.result_csv import expected_measurements, merge_result_csvs, read_result_csv
from examples.onnxruntime.fixtures import BASIC_SCENARIOS


def audit_basic_capabilities(capability) -> None:
    report = capability.to_dict()
    if report["execution"].get("cpu", {}).get("status") != "verified":
        raise ValueError("CPU execution must have verified protocol and task output evidence")
    if not capability.collection_complete or not report["requested_measurements_complete"]:
        raise ValueError("basic capability report lacks required collected evidence")
    for metric in ("latency", "throughput", "container_cpu", "container_memory"):
        if report["measurement"][metric]["status"] != "verified":
            raise ValueError(f"required capability is not verified: {metric}")


def _require_workload_facts(actual: dict, expected: dict, path: str = "workload") -> None:
    for key, value in expected.items():
        observed = actual.get(key)
        if isinstance(value, dict) and isinstance(observed, dict):
            _require_workload_facts(observed, value, f"{path}.{key}")
        elif observed != value:
            raise ValueError(f"actual workload differs at {path}.{key}: expected {value!r}, observed {observed!r}")


def audit_basic_rows(rows: list[dict], *, scenario: str = "tabular") -> None:
    """Protect required evidence and workload relationships, without timing thresholds."""
    specification = BASIC_SCENARIOS[scenario]
    if len(rows) != 3 or sum(row.get("warmup") == "1" for row in rows) != 1:
        raise ValueError("expected one warmup and two formal measurement rows")
    for row in rows:
        if row.get("status") != "ok" or row.get("error"):
            raise ValueError(f"basic collection failed: {row.get('error') or row.get('status')}")
        for field in ("latency_app_s", "throughput_samples_per_s", "container_cpu_util_avg_pct",
                      "container_mem_usage_avg_bytes", "resource_usage_iters"):
            value = float(row.get(field, "nan"))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"required finite basic metric missing: {field}")
        latency = float(row["latency_app_s"])
        if latency <= 0 or not math.isclose(float(row["throughput_samples_per_s"]) * latency,
                                            specification["batch_size"],
                                            rel_tol=0.02, abs_tol=0.001):
            raise ValueError("throughput must preserve batch_size / application latency semantics")
        if not 0 < float(row["container_mem_usage_avg_bytes"]) <= 1024 ** 3:
            raise ValueError("container memory must fit the actual 1 GiB limit")
        if float(row["resource_usage_iters"]) < 2:
            raise ValueError("CPU/memory collection needs at least two actual samples")
        contract = json.loads(row["workload_contract"])
        count = int(row["repeat_in_window"])
        if count < 1 or contract.get("request_count") != count:
            raise ValueError("actual workload request count does not match the measured window")
        variants = contract.get("variants", [])
        if not variants or sum(item["count"] for item in variants) != count:
            raise ValueError("actual workload variants do not cover every measured request")
        for item in variants:
            _require_workload_facts(item["contract"], specification["expected_workload"])
        for unavailable in ("latency_s", "cpu_energy_total_j", "cpu_instructions_per_request"):
            if math.isfinite(float(row.get(unavailable, "nan"))):
                raise ValueError(f"unrequested full metric must remain missing: {unavailable}")


def run_basic_e2e(image_id: str, output: Path, *, timeout_seconds: int = 180,
                  scenario: str = "tabular") -> dict:
    import requests

    specification = BASIC_SCENARIOS[scenario]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("basic E2E evidence directory must be empty")
    report = {"schema_version": 1, "successful": False, "scope": "synthetic_onnx_basic_container",
              "image_id": image_id, "profiling_mode": "basic", "scenario": scenario,
              "task": specification["task"], "family": specification["family"]}
    name = "acprof-onnx-basic-" + uuid.uuid4().hex[:12]
    prepare_name = name + "-prepare"
    deadline = time.monotonic() + timeout_seconds
    commands = []

    def run(command, *, log=None, check=True, environment=None):
        commands.append(command)
        remaining = max(0.1, deadline - time.monotonic())
        kwargs = dict(cwd=ROOT, env=environment, text=True, check=check, timeout=remaining)
        if log:
            with (output / log).open("w") as stream:
                return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
        return subprocess.run(command, capture_output=True, **kwargs)

    container_environment = {
        "HOME": "/tmp", "USER": "acprof", "LOGNAME": "acprof", "HF_HOME": "/tmp/hf",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "MODEL_ID": "acprof/synthetic-" + scenario, "MODEL_REVISION": "fixture-v1",
        "MODEL_LOCAL_PATH": "/evidence/model", "TASK_FAMILY": specification["family"],
        "TASK_TYPE": specification["task"], "RUNTIME_BACKEND": "onnxruntime", "USE_GPU": "0",
        "ACPROF_BASIC_SCENARIO": scenario,
        "ACPROF_RUNTIME_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    common = ["--cpus", "2", "--memory", "1g", "--memory-swap", "1g",
              "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
              "--security-opt", "no-new-privileges", "-v", f"{ROOT}:/workspace:ro", "-w", "/workspace"]
    for key, value in container_environment.items():
        common.extend(("-e", f"{key}={value}"))
    try:
        info = json.loads(run(["docker", "info", "--format", "{{json .}}"] ).stdout)
        if info.get("OSType") != "linux" or str(info.get("CgroupVersion")) != "2":
            raise RuntimeError("basic E2E requires local Linux Docker Engine and cgroup v2")
        report["docker"] = {key: info[key] for key in ("ServerVersion", "OSType", "CgroupVersion")}
        from scripts.check_runtime import ONNX_ENVIRONMENT_CHECK
        preparation = ONNX_ENVIRONMENT_CHECK + """
import os
from pathlib import Path
from examples.onnxruntime.fixtures import prepare_basic_scenario
prepare_basic_scenario(os.environ['ACPROF_BASIC_SCENARIO'], Path('/evidence'))
"""
        run(["docker", "run", "--rm", "--name", prepare_name, "--network", "none", *common,
             "-v", f"{output}:/evidence", image_id, "python", "-c", preparation], log="preparation.log")
        validation = json.loads((output / "output_validation.json").read_text())
        payload = json.loads((output / "payload.json").read_text())
        if not validation["reference"]["known_reference_passed"]:
            raise ValueError("known numerical output reference was not verified")
        # Separate validation process has exited before the service and measured client start.
        # Docker internal networks do not publish ports. Only loopback is exposed;
        # all artifacts are local and Hub offline mode remains enabled.
        run(["docker", "run", "-d", "--name", name, "--network", "bridge",
             "--publish", "127.0.0.1::8002", *common, "-v", f"{output}:/evidence:ro",
             image_id, "python", "-m", "acprof.container.server"])
        port = run(["docker", "port", name, "8002/tcp"]).stdout.strip().split(":")[-1]
        base_url = f"http://127.0.0.1:{int(port)}"
        session = requests.Session()
        session.trust_env = False
        ready_deadline = min(deadline, time.monotonic() + 40)
        while True:
            try:
                response = session.get(base_url + "/ready", timeout=1)
                response.raise_for_status()
                ready = response.json()
                if ready.get("status") != "ok":
                    raise ValueError(f"service is not ready: {ready}")
                break
            except requests.RequestException:
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError("ONNX service did not become ready within 40 seconds")
                state = run(["docker", "inspect", "--format", "{{.State.Running}}", name]).stdout.strip()
                if state != "true":
                    raise RuntimeError("ONNX service exited before readiness")
                time.sleep(0.1)
        atomic_write_json(output / "ready.json", ready)
        response = session.post(base_url + "/predict", json=payload, timeout=10)
        response.raise_for_status()
        protocol = response.json()
        if (protocol.get("output_shape") != specification["output_shape"]
                or protocol.get("n_results") != specification["n_results"]):
            raise ValueError(f"service output protocol mismatch: {protocol}")
        atomic_write_json(output / "response.json", protocol)
        environment = os.environ.copy()
        # Clear per-case overrides so this diagnostic never inherits a real run's output/profilers.
        environment.update({key: "" for key in (
            "COMPUTE_PROFILE_PLAN_FILE", "EXECUTION_PROFILE_PLAN_FILE", "IDLE_DIAG_PATH",
            "CLIENT_ERROR_PATH", "INPUT_SCALES", "SNIFF_GROUPS_PATH", "IDLE_DEBUG", "USE_MIPS")})
        environment.update(container_environment)
        environment.update({
            "BASE_URL": base_url, "PIPELINE_TAG": specification["task"], "CONTAINER_NAME": name,
            "MODEL_LOCAL_PATH": str(output / "model"), "PROFILING_MODE": "basic", "GPU_MODE": "off",
            "CPU_CORES": "2", "MEM_CAP_GB": "1", "BATCH_SIZE": str(specification["batch_size"]),
            "WARMUP": "1", "REPEAT": "2",
            "REPEAT_IN_WINDOW": "0", "REPEAT_WINDOW_SECONDS": "0.5", "AUTO_WARMUP_REQUESTS": "0",
            "SAMPLE_HZ": "20", "IDLE_SECONDS": "0", "IDLE_COOLDOWN_SECONDS": "0",
            "REQUEST_TIMEOUT_SECONDS": "10", "ACPROF_WECOM_WEBHOOK_URL": "",
            "INPUT_SCALE_PLAN_FILE": str(output / "input_scale_plan.json"), "CASE_NAME": name,
            "OUT_CSV": str(output / "result_case.csv"),
        })
        run([sys.executable, "-m", "acprof.host.client"], environment=environment, log="client.log")
        expected = expected_measurements([2], [1], ["off"], [specification["input_scale"]], 1, 2)
        merge_result_csvs([str(output / "result_case.csv")], str(output / "result_all.csv"), expected=expected)
        _, rows = read_result_csv(output / "result_all.csv", expected=expected)
        audit_basic_rows(rows, scenario=scenario)
        capability = measurement_report("basic", gpu_modes=["off"])
        apply_runtime_validation(capability, {"devices": {"off": validation["runtime_validation"]}})
        apply_collection_result(capability, rows)
        report["capability_report"] = capability.to_dict()
        audit_basic_capabilities(capability)
        # This records the diagnostic's measured plan; it does not invoke the hardware matrix.
        atomic_write_json(output / "static_meta.json", {
            "schema_version": STATIC_META_SCHEMA_VERSION, "validation_scope": "synthetic_onnx_basic_container",
            "model_id": container_environment["MODEL_ID"], "task_family": specification["family"],
            "pipeline_tag": specification["task"], "runtime_backend": "onnxruntime",
            "batch_size": specification["batch_size"],
            "profiling_mode": "basic", "input_scale_plan_sha256": hashlib.sha256(
                (output / "input_scale_plan.json").read_bytes()).hexdigest(),
            "compute_profile_tools": [], "execution_profile_tools": [],
            "capability_report": capability.to_dict(),
        })
        atomic_write_json(output / "run_state.json", {
            "schema_version": 1, "status": "complete", "run_id": name,
            "validation_scope": "synthetic_onnx_basic_container",
            "options": {"cpus": "2", "mems": "1", "gpus": "off", "warmup": 1, "repeat": 2,
                        "batch_size": specification["batch_size"]},
            "runtime": {"planned": {"scales": [specification["input_scale"]]}},
        })
        audit = audit_result(output)
        atomic_write_json(output / "audit.json", audit)
        if not audit["valid"] or audit["counts"]["formal_ok"] != 2 or audit["completion"] != "complete":
            raise ValueError(f"saved basic result failed audit: {audit['issues']}")
        report.update(successful=True, rows=len(rows), audit="audit.json",
                      output_validation="output_validation.json", required_metrics_verified=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    finally:
        cleanup_errors = []
        try:
            with (output / "service.log").open("w") as log:
                subprocess.run(["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as error:
            report["service_log_error"] = str(error)
        for container in (name, prepare_name):
            try:
                # A missing --rm preparation container is expected.
                removed = subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True, timeout=20)
                if removed.returncode and "No such container" not in removed.stderr:
                    cleanup_errors.append(removed.stderr.strip())
            except (OSError, subprocess.TimeoutExpired) as error:
                cleanup_errors.append(str(error))
        report["cleanup"] = {"successful": not cleanup_errors, "errors": cleanup_errors}
        report["successful"] = report["successful"] and not cleanup_errors
        atomic_write_json(output / "commands.json", {"commands": commands})
        atomic_write_json(output / "basic.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--scenario", choices=BASIC_SCENARIOS, default="tabular")
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return 0 if run_basic_e2e(args.image_id, args.output_dir, timeout_seconds=args.timeout_seconds,
                             scenario=args.scenario)["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

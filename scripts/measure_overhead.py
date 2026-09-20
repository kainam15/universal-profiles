"""以原实验的固定镜像和输入，诊断 RAPL/NVML/cgroup 监测线程对请求的影响。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.analysis.uncertainty import bootstrap_mean_interval
from acprof.artifacts import atomic_write_json
from acprof.runtime_settings import RUNTIME_ENV_NAMES


@contextmanager
def source_runtime_environment(recorded):
    """Use the source's thread/device settings, never accidental caller overrides."""
    names = (*RUNTIME_ENV_NAMES, "OMP_NUM_THREADS", "MKL_NUM_THREADS", "DEVICE_INDEX", "CUDA_VISIBLE_DEVICES")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            value = recorded.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def summarize_overhead(rows, *, seed=0):
    groups = defaultdict(dict)
    for row in rows:
        round_index, scenario = row["round"], row["scenario"]
        value = row["latency_app_s"]
        if not math.isfinite(value) or value <= 0 or round_index in groups[scenario]:
            raise ValueError("开销实验存在无效数值或重复轮次")
        groups[scenario][round_index] = value
    baseline = groups.get("none")
    if not baseline:
        raise ValueError("缺少未启用 monitors 的基线")
    result = []
    for scenario, values in sorted(groups.items()):
        if scenario == "none":
            continue
        if set(values) != set(baseline):
            raise ValueError("对照与基线轮次不匹配")
        changes = [(values[index] / baseline[index] - 1) * 100 for index in sorted(values)]
        differences = [values[index] - baseline[index] for index in sorted(values)]
        low, high = bootstrap_mean_interval(changes, seed=seed)
        result.append({"scenario": scenario, "paired_rounds": len(changes),
                       "paired_mean_change_pct": statistics.fmean(changes),
                       "paired_mean_change_s": statistics.fmean(differences),
                       "paired_change_stdev_s": statistics.stdev(differences) if len(differences) > 1 else None,
                       "baseline_mean_s": statistics.fmean(baseline.values()),
                       "baseline_stdev_s": statistics.stdev(baseline.values()) if len(baseline) > 1 else None,
                       "latency_mean_s": statistics.fmean(values.values()),
                       "latency_stdev_s": statistics.stdev(values.values()) if len(values) > 1 else None,
                       "ci_low_pct": low, "ci_high_pct": high})
    return result


def measure_window(url, payload, *, count, monitors, token, perf_monitor=None,
                   control_window=None, timeout=60):
    import requests

    started = []
    lifecycle = time.perf_counter()
    timings = []
    stopped = []
    cleanup_errors = []
    perf_started = False
    perf_result = None
    contracts = defaultdict(int)
    try:
        if control_window is not None:
            control_window()
        for monitor in monitors:
            monitor.start()
            started.append(monitor)
        if perf_monitor is not None:
            perf_monitor.start()
            perf_started = True
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        for index in range(count):
            before = time.perf_counter()
            response = requests.post(url + "/predict", json=payload,
                                     headers={"Connection": "close", "X-Req-Id": f"{token}:{index}"}, timeout=timeout)
            elapsed = time.perf_counter() - before
            response.raise_for_status()
            body = response.json()
            if response.status_code != 200 or not isinstance(body, dict) or body.get("error"):
                raise RuntimeError("/predict did not return a completed successful response")
            timings.append(elapsed)
            contract = body.get("workload_contract")
            if contract is not None:
                contracts[json.dumps(contract, sort_keys=True, separators=(",", ":"))] += 1
        cpu_time = time.process_time() - cpu_started
        wall_time = time.perf_counter() - wall_started
    finally:
        # 任一请求失败也停止所有线程，避免污染下一次实验。
        if perf_monitor is not None:
            try:
                if perf_started:
                    perf_result = asdict(perf_monitor.stop(len(timings), statistics.fmean(timings) if timings else math.nan))
            except Exception as error:
                cleanup_errors.append(str(error))
            finally:
                try:
                    perf_monitor.close()
                except Exception as error:
                    cleanup_errors.append(str(error))
        # Match the production client: perf, resource, GPU energy, CPU energy.
        priority = {"ResourceUsageMonitor": 0, "GPUEnergyMonitor": 1, "CPUEnergyMonitor": 2}
        for monitor in sorted(reversed(monitors), key=lambda item: priority.get(type(item).__name__, 3)):
            try:
                if monitor in started:
                    value = monitor.stop()
                    error = value[2] if type(monitor).__name__ == "GPUEnergyMonitor" else value[1]
                    stopped.append({"monitor": type(monitor).__name__, "samples": len(value[-1]), "error": error})
            except Exception as error:
                cleanup_errors.append(str(error))
            finally:
                try:
                    monitor.close()
                except Exception as error:
                    cleanup_errors.append(str(error))
    if cleanup_errors:
        raise RuntimeError(f"监测器停止失败：{cleanup_errors}")
    if any(item["error"] or item["samples"] < 2 for item in stopped):
        raise RuntimeError(f"监测器未取得有效窗口：{stopped}")
    if perf_result is not None and not (math.isfinite(perf_result["instructions_total"]) and perf_result["instructions_total"] > 0):
        raise RuntimeError("perf 未取得有效 instructions")
    return {"latency_app_s": statistics.fmean(timings), "request_count": count,
            "host_process_cpu_s": cpu_time, "request_loop_wall_s": wall_time,
            "lifecycle_wall_s": time.perf_counter() - lifecycle, "monitors": stopped,
            "perf": perf_result, "workload_contracts": [
                {"count": n, "contract": json.loads(contract)} for contract, n in contracts.items()]}


def validate_capture(command, pcap, *, token, count):
    parsed = subprocess.run(command, capture_output=True, text=True)
    if parsed.returncode:
        raise RuntimeError(f"packet parser failed: {parsed.stderr}")
    data = json.loads(parsed.stdout)
    requests = data.get("requests", {})
    if len(requests) != count or any(not key.startswith(token + ":") for key in requests):
        raise RuntimeError(f"packet coverage mismatch: expected={count}, actual={len(requests)}")
    atomic_write_json(pcap.with_suffix(".packets.json"), data)
    return len(requests)


def measure_profile_window(session, entry, *, scenario, rate, count, name, cpu, mem, gpu,
                           token, output, source, options):
    """Internal comparison; reuse collectors and the production idle lifecycle."""
    from acprof.monitors.energy_cpu import CPUEnergyMonitor
    from acprof.monitors.energy_nvml import GPUEnergyMonitor
    from acprof.monitors.resource_usage import ResourceUsageMonitor
    from acprof.monitors.perf_mips import PerfMIPSMonitor
    from acprof.host.packet_capture import _resolve_packet_latency_runtime

    idle = float(options["idle_seconds"])
    capture = capture_log = perf = None
    monitors = []
    gpu_monitor = cpu_monitor = resource = None
    control = None
    handed_off = False
    try:
        if scenario == "full":
            # Check existing capabilities without granting permissions.
            from acprof.host.packet_capture import _tcpdump_can_capture_without_sudo
            import shutil
            tcpdump = shutil.which("tcpdump")
            if not tcpdump or not _tcpdump_can_capture_without_sudo(tcpdump):
                raise RuntimeError("full 对照缺少现有 tcpdump capability；不会修改系统权限")
            pcap = output / f"{token}.pcap"
            runtime = _resolve_packet_latency_runtime(str(ROOT), str(pcap), options["sniff_iface"])
            if runtime is None:
                raise RuntimeError("full 对照缺少 tcpdump/tshark")
            capture_log = (output / f"{token}-tcpdump.log").open("w")
            capture = subprocess.Popen(runtime.tcpdump_cmd, stdout=capture_log, stderr=capture_log)
            time.sleep(0.2)
            if capture.poll() is not None:
                raise RuntimeError("tcpdump 在请求前退出")
            if gpu == "on":
                gpu_monitor = GPUEnergyMonitor(sample_hz=rate, device_index=int(os.environ.get("DEVICE_INDEX", "0")))
                monitors.append(gpu_monitor)
            cpu_monitor = CPUEnergyMonitor(sample_hz=rate, idle_seconds=idle, container_name=name)
            monitors.append(cpu_monitor)
            perf = PerfMIPSMonitor(name)
        if scenario in {"basic", "full"}:
            resource = ResourceUsageMonitor(sample_hz=rate, container_name=name, cpu_cores=cpu,
                                           mem_cap_gb=mem, use_gpu=gpu == "on",
                                           device_index=int(os.environ.get("DEVICE_INDEX", "0")))
            monitors.append(resource)
        time.sleep(float(options["idle_cooldown_seconds"]))
        if scenario == "full":
            # Importing the existing helper must not generate another input.
            previous_plan = os.environ.get("INPUT_SCALE_PLAN_FILE")
            try:
                os.environ["INPUT_SCALE_PLAN_FILE"] = str(source / "input_scale_plan.json")
                from acprof.host import client
            finally:
                if previous_plan is None:
                    os.environ.pop("INPUT_SCALE_PLAN_FILE", None)
                else:
                    os.environ["INPUT_SCALE_PLAN_FILE"] = previous_plan
            client.IDLE_SECONDS = idle
            client.IDLE_DEBUG = False
            control = lambda: client._run_matched_control_window(gpu_monitor, cpu_monitor, resource, perf)
        else:
            time.sleep(idle)
        handed_off = True
        result = measure_window(session.base_url, entry["payload"], count=count, monitors=monitors,
                                token=token, perf_monitor=perf, control_window=control,
                                timeout=float(options["request_timeout_seconds"]))
    finally:
        if capture is not None:
            # The production orchestrator drains libpcap before stopping tcpdump.
            # This wait and all packet parsing are outside the request timing.
            time.sleep(1.0)
            capture.terminate()
            try:
                capture.wait(timeout=5)
            except subprocess.TimeoutExpired:
                capture.kill()
                capture.wait()
            time.sleep(0.2)
        if capture_log is not None:
            capture_log.close()
        if not handed_off:
            for monitor in [*monitors, *([perf] if perf is not None else [])]:
                monitor.close()
    if scenario == "full":
        if capture.returncode != 0 or pcap.stat().st_size <= 24:
            raise RuntimeError("full 对照未取得有效 PCAP")
        result["pcap"] = str(pcap)
        result["packet_request_count"] = validate_capture(runtime.parse_cmd, pcap, token=token, count=count)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="已完成的新实验目录")
    parser.add_argument("--gpu", choices=("off", "on"), required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--sample-hz", default="5,20,100")
    parser.add_argument("--modes", help="内部对照场景，如 none,basic,full；此时 --sample-hz 必须为单值")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rates = [float(value) for value in args.sample_hz.split(",")]
    modes = args.modes.split(",") if args.modes else None
    if modes is not None and ("none" not in modes or len(modes) < 2 or len(set(modes)) != len(modes)
                              or set(modes) - {"none", "basic", "full"} or len(rates) != 1):
        parser.error("--modes 要求 none 与 basic/full，不允许重复；采样频率须为单值")
    if args.rounds < 3 or args.requests < 1 or len(set(rates)) != len(rates) or not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        parser.error("要求至少 3 轮、正请求数及不重复的正采样频率")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("诊断输出目录必须为空")
    from acprof.host.run_state import MeasurementLock, file_sha256, load_run_state
    from acprof.host.detect import TaskInfo
    from acprof.host.docker_runtime import ImageInfo, _start_container_session, _stop_container_session, require_image_identity
    from acprof.monitors.energy_cpu import CPUEnergyMonitor
    from acprof.monitors.energy_nvml import GPUEnergyMonitor
    from acprof.monitors.resource_usage import ResourceUsageMonitor
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.run_state import host_identity

    bootstrap_project_env(ROOT)

    source = args.source.resolve()
    state = load_run_state(source)
    if state.get("status") != "complete" or args.gpu not in state["options"]["gpus"].split(","):
        parser.error("需要已完成且包含所选设备的新实验")
    snapshot = state["runtime"]
    task, image = TaskInfo(**snapshot["task"]), ImageInfo(**snapshot["image"])
    plan_path = source / "input_scale_plan.json"
    expected_hash = state["artifacts"].get("input_scale_plan.json")
    if not expected_hash or file_sha256(plan_path) != expected_hash:
        parser.error("输入计划与原实验身份不一致")
    plan = json.loads(plan_path.read_text())
    entry = plan["entries"][0]
    cpu = max(map(int, state["options"]["cpus"].split(",")))
    mem = max(map(int, state["options"]["mems"].split(",")))
    name = "acprof-overhead-" + uuid4().hex[:12]
    recorded_environment = state["options"]["measurement_environment"]
    report = {"schema_version": 1, "kind": "collection_overhead_diagnostic" if modes else "monitor_overhead_diagnostic", "successful": False,
              "source_run_id": state["run_id"], "image_id": image.tag, "model_revision": task.model_revision,
              "gpu_mode": args.gpu, "cpu_cores": cpu, "mem_cap_gb": mem, "input_scale": entry["input_scale"],
              "input_scale_plan_sha256": expected_hash, "seed": args.seed, "rounds": [],
              "payload_sha256": hashlib.sha256(json.dumps(entry["payload"], sort_keys=True).encode()).hexdigest(),
              "measurement_environment": recorded_environment, "host_identity": host_identity(ROOT),
              "command": [sys.executable, *sys.argv],
              "scope": ("同一常驻模型、固定输入与线程的串行 /predict；none/basic/full 仅为内部采集器对照，"
                        "full 复用无请求对照与 PCAP/perf/RAPL/NVML/cgroup；不含 profiler、TUI、启动和离线合并成本，非正式画像"
                        if modes else "同一常驻模型的 HTTP 请求；比较监测线程，未开启 PCAP/perf/TUI，不是正式能耗实验")}
    try:
        with MeasurementLock(), source_runtime_environment(recorded_environment):
            require_image_identity(image.tag, image.runtime_environment)
            session = _start_container_session(task, cpu, mem, args.gpu, image, name, "[overhead]")
            try:
                measure_window(session.base_url, entry["payload"], count=5, monitors=[], token="warmup")
                rng = random.Random(args.seed)
                for round_index in range(args.rounds):
                    scenarios = list(modes) if modes else [0.0, *rates]
                    rng.shuffle(scenarios)
                    for rate in scenarios:
                        if modes:
                            scenario = rate
                            result = measure_profile_window(
                                session, entry, scenario=scenario, rate=rates[0], count=args.requests,
                                name=name, cpu=cpu, mem=mem, gpu=args.gpu,
                                token=f"overhead-{round_index}-{scenario}", output=output,
                                source=source, options=state["options"])
                            report["rounds"].append({"round": round_index, "scenario": scenario, **result})
                            atomic_write_json(output / "overhead.json", report)
                            print(f"[overhead] round={round_index + 1} {scenario}: {result['latency_app_s']:.6f}s/request", flush=True)
                            continue
                        # 监测器初始化与文件输出均不计入请求计时。
                        monitors = []
                        if rate:
                            monitors = [CPUEnergyMonitor(sample_hz=rate, container_name=name),
                                        ResourceUsageMonitor(sample_hz=rate, container_name=name, cpu_cores=cpu,
                                                             mem_cap_gb=mem, use_gpu=args.gpu == "on")]
                            if args.gpu == "on":
                                monitors.append(GPUEnergyMonitor(sample_hz=rate))
                        scenario = f"monitors-{rate:g}" if rate else "none"
                        result = measure_window(session.base_url, entry["payload"], count=args.requests,
                                                monitors=monitors, token=f"overhead-{round_index}-{scenario}")
                        report["rounds"].append({"round": round_index, "scenario": scenario, **result})
                        atomic_write_json(output / "overhead.json", report)
                        print(f"[overhead] round={round_index + 1} {scenario}: {result['latency_app_s']:.6f}s/request", flush=True)
                report["comparisons"] = summarize_overhead(report["rounds"], seed=args.seed)
                report["successful"] = True
            finally:
                _stop_container_session(name, "[overhead]")
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        atomic_write_json(output / "overhead.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

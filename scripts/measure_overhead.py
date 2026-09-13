"""以原实验的固定镜像和输入，诊断 RAPL/NVML/cgroup 监测线程对请求的影响。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.analysis.uncertainty import bootstrap_mean_interval
from acprof.artifacts import atomic_write_json


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
        low, high = bootstrap_mean_interval(changes, seed=seed)
        result.append({"scenario": scenario, "paired_rounds": len(changes),
                       "paired_mean_change_pct": statistics.fmean(changes),
                       "ci_low_pct": low, "ci_high_pct": high})
    return result


def measure_window(url, payload, *, count, monitors, token):
    import requests

    started = []
    lifecycle = time.perf_counter()
    timings = []
    stopped = []
    cleanup_errors = []
    try:
        for monitor in monitors:
            monitor.start()
            started.append(monitor)
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        for index in range(count):
            before = time.perf_counter()
            response = requests.post(url + "/predict", json=payload,
                                     headers={"Connection": "close", "X-Req-Id": f"{token}:{index}"}, timeout=60)
            timings.append(time.perf_counter() - before)
            response.raise_for_status()
        cpu_time = time.process_time() - cpu_started
        wall_time = time.perf_counter() - wall_started
    finally:
        # 任一请求失败也停止所有线程，避免污染下一次实验。
        for monitor in reversed(monitors):
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
    return {"latency_app_s": statistics.fmean(timings), "request_count": count,
            "host_process_cpu_s": cpu_time, "request_loop_wall_s": wall_time,
            "lifecycle_wall_s": time.perf_counter() - lifecycle, "monitors": stopped}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="已完成的新实验目录")
    parser.add_argument("--gpu", choices=("off", "on"), required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--sample-hz", default="5,20,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rates = [float(value) for value in args.sample_hz.split(",")]
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
    report = {"schema_version": 1, "kind": "monitor_overhead_diagnostic", "successful": False,
              "source_run_id": state["run_id"], "image_id": image.tag, "model_revision": task.model_revision,
              "gpu_mode": args.gpu, "cpu_cores": cpu, "mem_cap_gb": mem, "input_scale": entry["input_scale"],
              "input_scale_plan_sha256": expected_hash, "seed": args.seed, "rounds": [],
              "scope": "同一常驻模型的 HTTP 请求；比较监测线程，未开启 PCAP/perf/TUI，不是正式能耗实验"}
    try:
        with MeasurementLock():
            require_image_identity(image.tag, image.runtime_environment)
            session = _start_container_session(task, cpu, mem, args.gpu, image, name, "[overhead]")
            try:
                measure_window(session.base_url, entry["payload"], count=5, monitors=[], token="warmup")
                rng = random.Random(args.seed)
                for round_index in range(args.rounds):
                    scenarios = [0.0, *rates]
                    rng.shuffle(scenarios)
                    for rate in scenarios:
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

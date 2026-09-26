"""TUI 只读环境检查和结果摘要；不依赖 Textual。"""

from __future__ import annotations

import csv
import math
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from acprof.host.env_utils import load_project_env

from acprof.capabilities import Capability, measurement_requested
from acprof.host.preflight import probe_cpu_energy, probe_perf_instructions
from acprof.host.packet_capture import tcpdump_capability_available

from acprof.tui.commands import RunConfig, _csv_values

from acprof.tui.i18n import message


@dataclass(frozen=True)
class PreflightCheck:
    label: str
    status: str
    detail: str
    capability_status: str = ""


def _completed_command(
    command: Sequence[str],
    *,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _readable_rapl_paths(
    powercap_root: str | os.PathLike[str] = "/sys/class/powercap",
) -> list[str]:
    """Use the production package-domain reader, including real read checks."""
    from acprof.monitors.energy_cpu import _discover_rapl_domains
    return sorted(domain.energy_path for domain in _discover_rapl_domains(os.fspath(powercap_root)))


def quick_preflight(
    config: RunConfig,
    *,
    project_dir: str | os.PathLike[str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = _completed_command,
) -> list[PreflightCheck]:
    """Run read-only host checks; run.py remains the authoritative preflight."""
    checks: list[PreflightCheck] = []
    is_linux = platform.system() == "Linux"
    try:
        proc_version = Path("/proc/version").read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        proc_version = ""
    is_wsl = "microsoft" in proc_version.lower()
    checks.append(
        PreflightCheck(
            message('原生 Linux'),
            "ok" if is_linux and not is_wsl else "fail",
            platform.platform() if not is_wsl else message('检测到 WSL'),
        )
    )

    cgroup_v2 = Path("/sys/fs/cgroup/cgroup.controllers").is_file()
    checks.append(
        PreflightCheck(
            "cgroup v2",
            "ok" if cgroup_v2 else "fail",
            message('统一层级可用') if cgroup_v2 else message('未找到 cgroup.controllers'),
        )
    )

    docker_cli = shutil.which("docker")
    if not docker_cli:
        checks.append(PreflightCheck("Docker", "fail", message('未找到 docker CLI')))
    else:
        endpoint = "unknown"
        try:
            context = command_runner((docker_cli, "context", "show"), timeout=10.0)
            context_name = context.stdout.strip() if context.returncode == 0 else "default"
            inspected = command_runner(
                (
                    docker_cli,
                    "context",
                    "inspect",
                    context_name,
                    "--format",
                    '{{(index .Endpoints "docker").Host}}',
                ),
                timeout=10.0,
            )
            if inspected.returncode == 0:
                endpoint = inspected.stdout.strip()
            docker_host_override = ("" if os.environ.get("DOCKER_CONTEXT", "").strip() else
                                    os.environ.get("DOCKER_HOST", "").strip())
            if docker_host_override:
                endpoint = docker_host_override
            info = command_runner(
                (docker_cli, "info", "--format", "{{.Name}}|{{.OperatingSystem}}"),
                timeout=15.0,
            )
            native = endpoint in {
                "unix:///var/run/docker.sock",
                "unix:/var/run/docker.sock",
            }
            is_desktop = "docker desktop" in info.stdout.lower()
            status = (
                "ok"
                if info.returncode == 0 and native and not is_desktop
                else "fail"
            )
            detail = (
                f"{endpoint} · {info.stdout.strip()}"
                if info.returncode == 0
                else ((info.stderr or info.stdout).strip() or message('docker info 失败'))
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            status = "fail"
            detail = message('Docker 检查失败：{0}', exc)
        checks.append(PreflightCheck(message('本机 Docker'), status, detail))

    for tool in (("tcpdump", "tshark") if measurement_requested(config.profiling_mode, "packet_latency") else ()):
        path = shutil.which(tool)
        checks.append(
            PreflightCheck(tool, "ok" if path else "fail", path or message('未安装'))
        )
        if tool == 'tcpdump' and path:
            allowed = os.geteuid() == 0
            detail = 'CAP_NET_RAW'
            try:
                if not allowed:
                    getcap = shutil.which('getcap')
                    if getcap:
                        result = command_runner((getcap, str(Path(path).resolve())), timeout=5.0)
                        allowed = result.returncode == 0 and tcpdump_capability_available(result.stdout)
                if not allowed:
                    detail = message('缺少抓包权限；请在设置 → 连接与权限中配置 CAP_NET_RAW。')
            except (OSError, subprocess.TimeoutExpired):
                detail = message('无法检查抓包权限；请确认 getcap 可用。')
            checks.append(PreflightCheck('tcpdump permissions', 'ok' if allowed else 'fail', detail,
                                         'available' if allowed else 'permission_denied'))

    ip_cli = shutil.which("ip") if measurement_requested(config.profiling_mode, "packet_latency") else None
    if ip_cli:
        try:
            iface = command_runner(
                (ip_cli, "link", "show", config.sniff_iface),
                timeout=5.0,
            )
            checks.append(
                PreflightCheck(
                    message('抓包网卡'),
                    "ok" if iface.returncode == 0 else "fail",
                    config.sniff_iface,
                )
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(PreflightCheck(message('抓包网卡'), "fail", str(exc)))
    elif measurement_requested(config.profiling_mode, "packet_latency"):
        checks.append(PreflightCheck(message('抓包网卡'), "fail", message('未找到 ip 命令')))
    else:
        checks.append(PreflightCheck(message('抓包网卡'), "ok", "not_requested (basic)", "not_requested"))

    rapl = probe_cpu_energy() if measurement_requested(config.profiling_mode, "cpu_energy") else Capability("not_requested", "basic", "profiling_mode")
    checks.append(
        PreflightCheck(
            "CPU RAPL",
            "ok" if rapl.status.value in {"available", "not_requested"} else "fail",
            f"{rapl.status.value}: {rapl.detail}", rapl.status.value,
        )
    )

    perf = Capability("not_requested", "basic", "profiling_mode")
    if measurement_requested(config.profiling_mode, "cpu_instructions"):
        probe_environ = os.environ.copy()
        try:
            load_project_env(
                project_dir if project_dir is not None else Path.cwd(), environ=probe_environ,
            )
        except ValueError as exc:
            perf = Capability("error", str(exc), "environment")
        else:
            perf = probe_perf_instructions(env=probe_environ)
    if perf.status.value == "available":
        detail = message('普通用户 perf 可用，已读到 instructions 计数并通过跨用户 PID 附加检查')
        checks.append(PreflightCheck("perf instructions", "ok", detail, perf.status.value))
    else:
        checks.append(PreflightCheck("perf instructions", "ok" if perf.status.value == "not_requested" else "fail", f"{perf.status.value}: {perf.detail}", perf.status.value))

    if "on" in _csv_values(config.gpus.lower()):
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                gpu = command_runner(
                    (nvidia_smi, "--query-gpu=name", "--format=csv,noheader"),
                    timeout=10.0,
                )
                detail = gpu.stdout.strip() or gpu.stderr.strip() or message('GPU 查询失败')
                status = "ok" if gpu.returncode == 0 else "fail"
            except (OSError, subprocess.TimeoutExpired) as exc:
                status, detail = "fail", str(exc)
        else:
            status, detail = "fail", message('未找到 nvidia-smi')
        checks.append(PreflightCheck("NVIDIA GPU", status, detail))

    return checks


@dataclass(frozen=True)
class ResultSummary:
    rows: int
    ok_rows: int
    error_rows: int
    warmup_rows: int
    cases: int
    min_latency_s: float | None = None
    max_latency_s: float | None = None
    avg_latency_s: float | None = None


def summarize_result_csv(result_csv: str | Path) -> ResultSummary:
    """Read a completed result CSV once, outside timed collection windows."""
    path = Path(result_csv).expanduser()
    rows = ok_rows = error_rows = warmup_rows = 0
    cases: set[tuple[str, str, str]] = set()
    latencies: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            status = str(row.get("status") or "").strip().lower()
            is_warmup = str(row.get("warmup") or "0").strip() == "1"
            if status == "ok":
                ok_rows += 1
                if not is_warmup:
                    raw_lat = (
                        row.get("latency_app_s")
                        or row.get("latency_s")
                        or ""
                    ).strip()
                    try:
                        lat_val = float(raw_lat)
                        if math.isfinite(lat_val) and lat_val > 0:
                            latencies.append(lat_val)
                    except ValueError:
                        pass
            elif status == "error":
                error_rows += 1
            if is_warmup:
                warmup_rows += 1
            cases.add(
                (
                    str(row.get("cpu_cores") or ""),
                    str(row.get("mem_cap_gb") or ""),
                    str(row.get("gpu_mode") or ""),
                )
            )
    min_lat = min(latencies) if latencies else None
    max_lat = max(latencies) if latencies else None
    avg_lat = (sum(latencies) / len(latencies)) if latencies else None
    return ResultSummary(
        rows=rows,
        ok_rows=ok_rows,
        error_rows=error_rows,
        warmup_rows=warmup_rows,
        cases=len(cases),
        min_latency_s=min_lat,
        max_latency_s=max_lat,
        avg_latency_s=avg_lat,
    )

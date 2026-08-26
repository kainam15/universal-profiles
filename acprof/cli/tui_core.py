"""Pure helpers for the AC-Prof terminal user interface.

This module intentionally has no Textual dependency.  Keeping command
construction, validation, progress parsing, and quick host checks here makes
the TUI a thin controller around the existing scientific collection CLI.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from typing import Callable, Iterable, Sequence

from acprof.config import (
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
)


TASK_FAMILIES = ("nlp", "cv", "audio", "timeseries", "diffusion")
GPU_MODES = ("off", "on")
COMPUTE_PROFILE_TOOLS = ("none", "both", "torch", "ncu")
EXECUTION_PROFILE_TOOLS = ("none", "both", "massif", "nsys")
NOTIFY_MODES = ("auto", "none", "wecom")


class TuiConfigError(ValueError):
    """Raised when a TUI form cannot produce a safe run command."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(str(error) for error in errors if str(error))
        super().__init__("；".join(self.errors))


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _positive_int_csv(value: str, label: str) -> list[int]:
    raw_values = _csv_values(value)
    if not raw_values:
        raise TuiConfigError([f"{label}不能为空"])
    try:
        values = [int(item) for item in raw_values]
    except ValueError as exc:
        raise TuiConfigError([f"{label}必须是逗号分隔的整数"]) from exc
    if any(item <= 0 for item in values):
        raise TuiConfigError([f"{label}必须全部大于 0"])
    return values


def _positive_float_csv(value: str, label: str) -> list[float]:
    raw_values = _csv_values(value)
    if not raw_values:
        return []
    try:
        values = [float(item) for item in raw_values]
    except ValueError as exc:
        raise TuiConfigError([f"{label}必须是逗号分隔的数字"]) from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise TuiConfigError([f"{label}必须全部大于 0"])
    return values


def _number(value: str | int | float, label: str, *, integer: bool) -> int | float:
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        kind = "整数" if integer else "数字"
        raise TuiConfigError([f"{label}必须是{kind}"]) from exc


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return f"{value:.1f}"
    return str(value)


@dataclass(frozen=True)
class RunConfig:
    """Values exposed by the interactive run form."""

    model: str = ""
    task: str = ""
    task_family: str = ""
    backend: str = ""
    cpus: str = "1,2,4,8"
    mems: str = "2,4,8,16"
    gpus: str = "off,on"
    input_scales: str = ""
    workload_spec: str = ""
    output_dir: str = "results"
    batch_size: int = 1
    warmup: int = 2
    repeat: int = 5
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS
    sample_hz: float = 20.0
    idle_seconds: float = DEFAULT_IDLE_SECONDS
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS
    compute_profile_tool: str = "both"
    execution_profile_tool: str = "none"
    sniff_iface: str = "docker0"
    notify: str = "auto"
    prune_startup_oom: bool = True
    skip_build: bool = False
    idle_debug: bool = False
    allow_cgroup_v1: bool = False

    @classmethod
    def smoke(cls, model: str = "") -> "RunConfig":
        """Return a deliberately tiny, low-overhead validation run."""
        return cls(
            model=model,
            cpus="1",
            mems="4",
            gpus="off",
            input_scales="64",
            output_dir="results/smoke",
            warmup=0,
            repeat=1,
            repeat_in_window=1,
            compute_profile_tool="none",
            execution_profile_tool="none",
        )

    @classmethod
    def main_matrix(cls, model: str = "") -> "RunConfig":
        """Return the normal matrix with isolated profilers deferred."""
        return cls(
            model=model,
            compute_profile_tool="none",
            execution_profile_tool="none",
        )

    def validate(self, *, project_dir: Path | None = None) -> "RunConfig":
        """Normalize form values and reject invalid or misleading runs."""
        errors: list[str] = []
        model = self.model.strip()
        if not model:
            errors.append("模型 ID 不能为空")

        try:
            cpus = _positive_int_csv(self.cpus, "CPU 列表")
        except TuiConfigError as exc:
            errors.extend(exc.errors)
            cpus = []
        try:
            mems = _positive_int_csv(self.mems, "内存列表")
        except TuiConfigError as exc:
            errors.extend(exc.errors)
            mems = []

        gpus = _csv_values(self.gpus.lower())
        if not gpus:
            errors.append("GPU 模式不能为空")
        elif any(item not in GPU_MODES for item in gpus):
            errors.append("GPU 模式只能包含 off 或 on")

        if self.prune_startup_oom:
            if len(cpus) != len(set(cpus)):
                errors.append("启用启动 OOM 剪枝时 CPU 列表不能重复")
            if len(mems) != len(set(mems)):
                errors.append("启用启动 OOM 剪枝时内存列表不能重复")
            if len(gpus) != len(set(gpus)):
                errors.append("启用启动 OOM 剪枝时 GPU 模式不能重复")

        try:
            _positive_float_csv(self.input_scales, "输入规模")
        except TuiConfigError as exc:
            errors.extend(exc.errors)

        batch_size = _number(self.batch_size, "Batch size", integer=True)
        warmup = _number(self.warmup, "Warmup", integer=True)
        repeat = _number(self.repeat, "Repeat", integer=True)
        repeat_in_window = _number(
            self.repeat_in_window,
            "每窗口请求数",
            integer=True,
        )
        repeat_window_seconds = _number(
            self.repeat_window_seconds,
            "自动窗口秒数",
            integer=False,
        )
        sample_hz = _number(self.sample_hz, "采样频率", integer=False)
        idle_seconds = _number(self.idle_seconds, "Idle 秒数", integer=False)
        idle_cooldown_seconds = _number(
            self.idle_cooldown_seconds,
            "Idle cooldown 秒数",
            integer=False,
        )

        if batch_size <= 0:
            errors.append("Batch size 必须大于 0")
        if warmup < 0:
            errors.append("Warmup 不能小于 0")
        if repeat <= 0:
            errors.append("Repeat 必须大于 0")
        if repeat_in_window < 0:
            errors.append("每窗口请求数不能小于 0")
        if repeat_window_seconds <= 0.0:
            errors.append("自动窗口秒数必须大于 0")
        if not math.isfinite(float(repeat_window_seconds)):
            errors.append("自动窗口秒数必须是有限数字")
        if not math.isfinite(float(sample_hz)) or sample_hz <= 0.0:
            errors.append("采样频率必须大于 0")
        if not math.isfinite(float(idle_seconds)) or idle_seconds < 0.0:
            errors.append("Idle 秒数不能小于 0")
        if (
            not math.isfinite(float(idle_cooldown_seconds))
            or idle_cooldown_seconds < 0.0
        ):
            errors.append("Idle cooldown 秒数不能小于 0")

        task_family = self.task_family.strip().lower()
        if task_family and task_family not in TASK_FAMILIES:
            errors.append("任务族必须是 nlp/cv/audio/timeseries/diffusion")
        if self.compute_profile_tool not in COMPUTE_PROFILE_TOOLS:
            errors.append("无效的计算分析器")
        if self.execution_profile_tool not in EXECUTION_PROFILE_TOOLS:
            errors.append("无效的执行分析器")
        if self.notify not in NOTIFY_MODES:
            errors.append("无效的通知模式")
        if not self.output_dir.strip():
            errors.append("输出目录不能为空")
        if not self.sniff_iface.strip():
            errors.append("抓包网卡不能为空")

        workload_spec = self.workload_spec.strip()
        if workload_spec and project_dir is not None:
            workload_path = Path(workload_spec).expanduser()
            if not workload_path.is_absolute():
                workload_path = project_dir / workload_path
            if not workload_path.is_file():
                errors.append(f"Workload manifest 不存在：{workload_spec}")

        if errors:
            raise TuiConfigError(errors)

        return replace(
            self,
            model=model,
            task=self.task.strip(),
            task_family=task_family,
            backend=self.backend.strip(),
            cpus=",".join(str(value) for value in cpus),
            mems=",".join(str(value) for value in mems),
            gpus=",".join(gpus),
            input_scales=",".join(_csv_values(self.input_scales)),
            workload_spec=workload_spec,
            output_dir=self.output_dir.strip(),
            batch_size=int(batch_size),
            warmup=int(warmup),
            repeat=int(repeat),
            repeat_in_window=int(repeat_in_window),
            repeat_window_seconds=float(repeat_window_seconds),
            sample_hz=float(sample_hz),
            idle_seconds=float(idle_seconds),
            idle_cooldown_seconds=float(idle_cooldown_seconds),
            sniff_iface=self.sniff_iface.strip(),
        )

    def result_dir(self, project_dir: Path) -> Path:
        output_root = Path(self.output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = project_dir / output_root
        return output_root / self.model.replace("/", "--")

    def result_csv(self, project_dir: Path) -> Path:
        return self.result_dir(project_dir) / "result_all.csv"


def build_run_command(
    config: RunConfig,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    """Build the existing ``run.py`` command without duplicating its work."""
    config = config.validate(project_dir=project_dir)
    command = [
        str(python_executable),
        "-u",
        str(project_dir / "run.py"),
        "--model",
        config.model,
        "--cpus",
        config.cpus,
        "--mems",
        config.mems,
        "--gpus",
        config.gpus,
        "--batch-size",
        str(config.batch_size),
        "--warmup",
        str(config.warmup),
        "--repeat",
        str(config.repeat),
        "--repeat-in-window",
        str(config.repeat_in_window),
        "--repeat-window-seconds",
        _format_number(config.repeat_window_seconds),
        "--sample-hz",
        _format_number(config.sample_hz),
        "--idle-seconds",
        _format_number(config.idle_seconds),
        "--idle-cooldown-seconds",
        _format_number(config.idle_cooldown_seconds),
        "--compute-profile-tool",
        config.compute_profile_tool,
        "--execution-profile-tool",
        config.execution_profile_tool,
        "--sniff-iface",
        config.sniff_iface,
        "--output-dir",
        config.output_dir,
        "--notify",
        config.notify,
    ]
    for option, value in (
        ("--task", config.task),
        ("--task-family", config.task_family),
        ("--backend", config.backend),
        ("--input-scales", config.input_scales),
        ("--workload-spec", config.workload_spec),
    ):
        if value:
            command.extend((option, value))
    if not config.prune_startup_oom:
        command.append("--no-prune-startup-oom")
    if config.skip_build:
        command.append("--skip-build")
    if config.idle_debug:
        command.append("--idle-debug")
    if config.allow_cgroup_v1:
        command.append("--allow-cgroup-v1")
    return command


def build_plot_command(
    result_csv: str | Path,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    return [
        str(python_executable),
        "-u",
        str(project_dir / "plot.py"),
        str(Path(result_csv).expanduser()),
    ]


def build_profile_command(
    result_dir: str | Path,
    *,
    tools: str = "torch,ncu",
    dry_run: bool = True,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    normalized_tools = ",".join(_csv_values(tools))
    if not normalized_tools:
        raise TuiConfigError(["补采工具不能为空"])
    command = [
        str(python_executable),
        "-u",
        str(project_dir / "profile.py"),
        str(Path(result_dir).expanduser()),
        "--tools",
        normalized_tools,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def format_command(command: Sequence[str], *, project_dir: Path | None = None) -> str:
    """Return a shell-safe, readable command preview."""
    display = list(command)
    if project_dir is not None:
        project_dir = project_dir.resolve()
        for index, item in enumerate(display):
            try:
                item_path = Path(item).resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if item_path.parent == project_dir and item_path.name in {
                "run.py",
                "plot.py",
                "profile.py",
            }:
                display[index] = item_path.name
    return shlex.join(display)


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CASE_RE = re.compile(
    r"# Case\s+(?P<current>\d+)/(?P<total>\d+):\s+"
    r"CPU=(?P<cpu>\d+),\s+MEM=(?P<mem>\d+)GB,\s+GPU=(?P<gpu>\S+)"
)
MATRIX_RE = re.compile(r"Resource matrix:.*=\s*(?P<total>\d+)\s+cases")
FINAL_CSV_RE = re.compile(r"Merged results:\s+(?P<path>.+)$")
MERGE_CSV_RE = re.compile(r"\[merge\]\s+Final CSV:\s+(?P<path>.+?)\s+\(\d+\s+rows\)")


@dataclass(frozen=True)
class ProgressSnapshot:
    stage: str = "等待"
    detail: str = "尚未启动"
    current_case: int = 0
    completed_cases: int = 0
    total_cases: int = 0
    cpu: str = "-"
    mem: str = "-"
    gpu: str = "-"
    measurement_active: bool = False
    warnings: int = 0
    errors: int = 0
    final_csv: str = ""


class RunProgressTracker:
    """Translate stable run.py log markers into low-frequency UI state."""

    def __init__(self) -> None:
        self.snapshot = ProgressSnapshot()

    def feed(self, raw_line: str) -> ProgressSnapshot:
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        state = self.snapshot
        updates: dict[str, object] = {}

        if "[WARN]" in line or "[warning]" in line.lower():
            updates["warnings"] = state.warnings + 1
        if "[ERROR]" in line or line.startswith("Traceback"):
            updates["errors"] = state.errors + 1

        match = MATRIX_RE.search(line)
        if match:
            updates.update(
                stage="准备矩阵",
                detail="资源矩阵与输入规模已确定",
                total_cases=int(match.group("total")),
            )

        match = CASE_RE.search(line)
        if match:
            current = int(match.group("current"))
            updates.update(
                stage="启动容器",
                detail=f"正在准备 case {current}/{match.group('total')}",
                current_case=current,
                completed_cases=max(state.completed_cases, current - 1),
                total_cases=int(match.group("total")),
                cpu=match.group("cpu"),
                mem=match.group("mem"),
                gpu=match.group("gpu"),
                measurement_active=False,
            )
        elif line.startswith("[build]"):
            updates.update(stage="构建镜像", detail=line)
        elif line.startswith("[scale]") or line.startswith("[probe]"):
            updates.update(stage="规划输入", detail=line)
        elif line.startswith("[compute-profile]"):
            updates.update(stage="计算分析", detail=line)
        elif line.startswith("[execution-profile]"):
            updates.update(stage="执行分析", detail=line)
        elif "[case] Server ready" in line:
            updates.update(stage="服务就绪", detail=line)
        elif "[case] Running workload" in line:
            updates.update(
                stage="正式测量",
                detail="采集窗口进行中；TUI 已停止常规重绘",
                measurement_active=True,
            )
        elif "[case] Stopping container" in line:
            updates.update(
                stage="清理 case",
                detail=line,
                measurement_active=False,
            )
        elif "[case] Done." in line:
            updates.update(
                stage="case 完成",
                detail=line,
                completed_cases=max(state.completed_cases, state.current_case),
                measurement_active=False,
            )
        elif "[oom-prune] Skipping" in line:
            updates.update(
                stage="OOM 剪枝",
                detail=line,
                completed_cases=max(state.completed_cases, state.current_case),
                measurement_active=False,
            )
        elif line.startswith("[merge]"):
            updates.update(stage="合并结果", detail=line, measurement_active=False)
        elif "Profiling complete!" in line:
            updates.update(
                stage="已完成",
                detail="采集与合并已完成",
                completed_cases=state.total_cases or state.completed_cases,
                measurement_active=False,
            )

        final_match = FINAL_CSV_RE.search(line) or MERGE_CSV_RE.search(line)
        if final_match:
            updates["final_csv"] = final_match.group("path").strip()

        if updates:
            self.snapshot = replace(state, **updates)
        return self.snapshot


@dataclass(frozen=True)
class PreflightCheck:
    label: str
    status: str
    detail: str


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
    """Find top-level RAPL counters without following cyclic sysfs links."""
    paths: list[str] = []
    try:
        for entry in os.scandir(powercap_root):
            if entry.name.count(":") != 1:
                continue
            energy_path = os.path.join(entry.path, "energy_uj")
            if os.path.isfile(energy_path) and os.access(energy_path, os.R_OK):
                paths.append(energy_path)
    except OSError:
        pass
    return sorted(paths)


def quick_preflight(
    config: RunConfig,
    *,
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
            "原生 Linux",
            "ok" if is_linux and not is_wsl else "fail",
            platform.platform() if not is_wsl else "检测到 WSL",
        )
    )

    cgroup_v2 = Path("/sys/fs/cgroup/cgroup.controllers").is_file()
    checks.append(
        PreflightCheck(
            "cgroup v2",
            "ok" if cgroup_v2 else ("warn" if config.allow_cgroup_v1 else "fail"),
            "统一层级可用" if cgroup_v2 else "未找到 cgroup.controllers",
        )
    )

    docker_cli = shutil.which("docker")
    if not docker_cli:
        checks.append(PreflightCheck("Docker", "fail", "未找到 docker CLI"))
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
            docker_host_override = os.environ.get("DOCKER_HOST", "").strip()
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
                else (info.stderr or info.stdout or "docker info 失败").strip()
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            status = "fail"
            detail = f"Docker 检查失败：{exc}"
        checks.append(PreflightCheck("本机 Docker", status, detail))

    for tool in ("tcpdump", "tshark"):
        path = shutil.which(tool)
        checks.append(
            PreflightCheck(tool, "ok" if path else "fail", path or "未安装")
        )

    ip_cli = shutil.which("ip")
    if ip_cli:
        try:
            iface = command_runner(
                (ip_cli, "link", "show", config.sniff_iface),
                timeout=5.0,
            )
            checks.append(
                PreflightCheck(
                    "抓包网卡",
                    "ok" if iface.returncode == 0 else "fail",
                    config.sniff_iface,
                )
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(PreflightCheck("抓包网卡", "fail", str(exc)))
    else:
        checks.append(PreflightCheck("抓包网卡", "fail", "未找到 ip 命令"))

    # sysfs powercap entries contain cyclic ``device``/``subsystem`` symlinks;
    # never recurse through them.  The production monitor likewise inspects
    # only top-level package domains (exactly one colon in intel-rapl:N).
    rapl_paths = _readable_rapl_paths()
    checks.append(
        PreflightCheck(
            "CPU RAPL",
            "ok" if rapl_paths else "fail",
            rapl_paths[0] if rapl_paths else "没有可读 energy_uj",
        )
    )

    perf_cli = shutil.which("perf")
    if perf_cli:
        try:
            perf = command_runner(
                (perf_cli, "stat", "-e", "instructions", "--", "true"),
                timeout=10.0,
            )
            perf_detail = (perf.stderr or perf.stdout).strip().splitlines()
            checks.append(
                PreflightCheck(
                    "perf instructions",
                    "ok" if perf.returncode == 0 else "fail",
                    perf_detail[-1] if perf_detail else f"exit {perf.returncode}",
                )
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(PreflightCheck("perf instructions", "fail", str(exc)))
    else:
        checks.append(PreflightCheck("perf instructions", "fail", "未找到 perf"))

    if "on" in _csv_values(config.gpus.lower()):
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                gpu = command_runner(
                    (nvidia_smi, "--query-gpu=name", "--format=csv,noheader"),
                    timeout=10.0,
                )
                detail = gpu.stdout.strip() or gpu.stderr.strip() or "GPU 查询失败"
                status = "ok" if gpu.returncode == 0 else "fail"
            except (OSError, subprocess.TimeoutExpired) as exc:
                status, detail = "fail", str(exc)
        else:
            status, detail = "fail", "未找到 nvidia-smi"
        checks.append(PreflightCheck("NVIDIA GPU", status, detail))

    return checks


@dataclass(frozen=True)
class ResultSummary:
    rows: int
    ok_rows: int
    error_rows: int
    warmup_rows: int
    cases: int


def summarize_result_csv(result_csv: str | Path) -> ResultSummary:
    """Read a completed result CSV once, outside timed collection windows."""
    path = Path(result_csv).expanduser()
    rows = ok_rows = error_rows = warmup_rows = 0
    cases: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            status = str(row.get("status") or "").strip().lower()
            if status == "ok":
                ok_rows += 1
            elif status == "error":
                error_rows += 1
            if str(row.get("warmup") or "0").strip() == "1":
                warmup_rows += 1
            cases.add(
                (
                    str(row.get("cpu_cores") or ""),
                    str(row.get("mem_cap_gb") or ""),
                    str(row.get("gpu_mode") or ""),
                )
            )
    return ResultSummary(
        rows=rows,
        ok_rows=ok_rows,
        error_rows=error_rows,
        warmup_rows=warmup_rows,
        cases=len(cases),
    )


def parse_slash_command(value: str) -> tuple[str, list[str]]:
    """Parse a slash command using shell quoting rules, without executing it."""
    try:
        parts = shlex.split(value.strip())
    except ValueError as exc:
        raise TuiConfigError([f"命令格式错误：{exc}"]) from exc
    if not parts or not parts[0].startswith("/"):
        raise TuiConfigError(["快捷命令必须以 / 开头"])
    return parts[0][1:].lower(), parts[1:]

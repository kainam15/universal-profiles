"""TUI 实验配置校验及现有 CLI 命令构造；不依赖 Textual。"""

from __future__ import annotations

import math
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from acprof.config import (
    DEFAULT_COMPUTE_PROFILE_TOOL,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)

from acprof.tui.i18n import message


TASK_FAMILIES = ("nlp", "cv", "audio", "timeseries", "diffusion", "multimodal", "structured")
GPU_MODES = ("off", "on")
COMPUTE_PROFILE_TOOLS = ("none", "both", "torch", "ncu")
EXECUTION_PROFILE_TOOLS = ("none", "both", "massif", "nsys")
NOTIFY_MODES = ("auto", "none", "wecom")


class TuiConfigError(ValueError):
    """Raised when a TUI form cannot produce a safe run command."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(
            error if isinstance(error, str) else str(error)
            for error in errors if str(error)
        )
        super().__init__("；".join(self.errors))


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _positive_int_csv(value: str, label: str) -> list[int]:
    raw_values = _csv_values(value)
    if not raw_values:
        raise TuiConfigError([message('{0}不能为空', label)])
    try:
        values = [int(item) for item in raw_values]
    except ValueError as exc:
        raise TuiConfigError([message('{0}必须是逗号分隔的整数', label)]) from exc
    if any(item <= 0 for item in values):
        raise TuiConfigError([message('{0}必须全部大于 0', label)])
    return values


def _positive_float_csv(value: str, label: str) -> list[float]:
    raw_values = _csv_values(value)
    if not raw_values:
        return []
    try:
        values = [float(item) for item in raw_values]
    except ValueError as exc:
        raise TuiConfigError([message('{0}必须是逗号分隔的数字', label)]) from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise TuiConfigError([message('{0}必须全部大于 0', label)])
    return values


def _number(value: str | int | float, label: str, *, integer: bool) -> int | float:
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        kind = message('整数') if integer else message('数字')
        raise TuiConfigError([message('{0}必须是{1}', label, kind)]) from exc


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
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    sample_hz: float = 20.0
    idle_seconds: float = DEFAULT_IDLE_SECONDS
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS
    compute_profile_tool: str = DEFAULT_COMPUTE_PROFILE_TOOL
    execution_profile_tool: str = "none"
    sniff_iface: str = "docker0"
    notify: str = "auto"
    prune_startup_oom: bool = True
    skip_build: bool = False
    resume: bool = False
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
            errors.append(message('模型 ID 不能为空'))

        try:
            cpus = _positive_int_csv(self.cpus, message('CPU 列表'))
        except TuiConfigError as exc:
            errors.extend(exc.errors)
            cpus = []
        try:
            mems = _positive_int_csv(self.mems, message('内存列表'))
        except TuiConfigError as exc:
            errors.extend(exc.errors)
            mems = []

        gpus = _csv_values(self.gpus.lower())
        if not gpus:
            errors.append(message('GPU 模式不能为空'))
        elif any(item not in GPU_MODES for item in gpus):
            errors.append(message('GPU 模式只能包含 off 或 on'))

        if self.prune_startup_oom:
            if len(cpus) != len(set(cpus)):
                errors.append(message('启用启动 OOM 剪枝时 CPU 列表不能重复'))
            if len(mems) != len(set(mems)):
                errors.append(message('启用启动 OOM 剪枝时内存列表不能重复'))
            if len(gpus) != len(set(gpus)):
                errors.append(message('启用启动 OOM 剪枝时 GPU 模式不能重复'))

        try:
            _positive_float_csv(self.input_scales, message('输入规模'))
        except TuiConfigError as exc:
            errors.extend(exc.errors)

        batch_size = _number(self.batch_size, "Batch size", integer=True)
        warmup = _number(self.warmup, "Warmup", integer=True)
        repeat = _number(self.repeat, "Repeat", integer=True)
        repeat_in_window = _number(
            self.repeat_in_window,
            message('每窗口请求数'),
            integer=True,
        )
        repeat_window_seconds = _number(
            self.repeat_window_seconds,
            message('自动窗口秒数'),
            integer=False,
        )
        request_timeout_seconds = _number(
            self.request_timeout_seconds,
            message('单请求超时秒数'),
            integer=False,
        )
        sample_hz = _number(self.sample_hz, message('采样频率'), integer=False)
        idle_seconds = _number(self.idle_seconds, message('Idle 秒数'), integer=False)
        idle_cooldown_seconds = _number(
            self.idle_cooldown_seconds,
            message('Idle cooldown 秒数'),
            integer=False,
        )

        if batch_size <= 0:
            errors.append(message('Batch size 必须大于 0'))
        if warmup < 0:
            errors.append(message('Warmup 不能小于 0'))
        if repeat <= 0:
            errors.append(message('Repeat 必须大于 0'))
        if repeat_in_window < 0:
            errors.append(message('每窗口请求数不能小于 0'))
        if repeat_window_seconds <= 0.0:
            errors.append(message('自动窗口秒数必须大于 0'))
        if not math.isfinite(float(repeat_window_seconds)):
            errors.append(message('自动窗口秒数必须是有限数字'))
        if (
            not math.isfinite(float(request_timeout_seconds))
            or request_timeout_seconds <= 0.0
        ):
            errors.append(message('单请求超时秒数必须是大于 0 的有限数字'))
        if not math.isfinite(float(sample_hz)) or sample_hz <= 0.0:
            errors.append(message('采样频率必须大于 0'))
        if not math.isfinite(float(idle_seconds)) or idle_seconds < 0.0:
            errors.append(message('Idle 秒数不能小于 0'))
        if (
            not math.isfinite(float(idle_cooldown_seconds))
            or idle_cooldown_seconds < 0.0
        ):
            errors.append(message('Idle cooldown 秒数不能小于 0'))

        task_family = self.task_family.strip().lower()
        if task_family and task_family not in TASK_FAMILIES:
            errors.append(message('任务族必须是 nlp/cv/audio/timeseries/diffusion/multimodal/structured'))
        if self.compute_profile_tool not in COMPUTE_PROFILE_TOOLS:
            errors.append(message('无效的计算分析器'))
        if self.execution_profile_tool not in EXECUTION_PROFILE_TOOLS:
            errors.append(message('无效的执行分析器'))
        if self.notify not in NOTIFY_MODES:
            errors.append(message('无效的通知模式'))
        if not self.output_dir.strip():
            errors.append(message('输出目录不能为空'))
        if not self.sniff_iface.strip():
            errors.append(message('抓包网卡不能为空'))

        workload_spec = self.workload_spec.strip()
        if workload_spec and project_dir is not None:
            workload_path = Path(workload_spec).expanduser()
            if not workload_path.is_absolute():
                workload_path = project_dir / workload_path
            if not workload_path.is_file():
                errors.append(message('Workload manifest 不存在：{0}', workload_spec))

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
            request_timeout_seconds=float(request_timeout_seconds),
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
        "--request-timeout-seconds",
        _format_number(config.request_timeout_seconds),
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
    if config.resume:
        command.append("--resume")
    if config.idle_debug:
        command.append("--idle-debug")
    if config.allow_cgroup_v1:
        command.append("--allow-cgroup-v1")
    return command


def build_probe_command(
    config: RunConfig,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    """Build a one-request largest-scale diagnostic probe command."""
    config = config.validate(project_dir=project_dir)
    command = [
        str(python_executable),
        "-u",
        str(project_dir / "probe.py"),
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
        "--output-dir",
        config.output_dir,
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
    if config.skip_build:
        command.append("--skip-build")
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


def build_stats_command(
    result_csv: str | Path,
    output: str | Path,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    return [str(python_executable), "-u", str(project_dir / "stats.py"),
            str(Path(result_csv).expanduser()), "--output", str(output)]


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
        raise TuiConfigError([message('补采工具不能为空')])
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
            # Only entry-point names can be shortened. Resolving every flag,
            # numeric value and model ID needlessly touches the filesystem on
            # each form edit (and may be slow for paths on remote storage).
            if Path(item).name not in {"run.py", "probe.py", "plot.py", "profile.py", "stats.py"}:
                continue
            try:
                item_path = Path(item).resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if item_path.parent == project_dir and item_path.name in {
                "run.py",
                "probe.py",
                "plot.py",
                "profile.py",
                "stats.py",
            }:
                display[index] = item_path.name
    return shlex.join(display)


def parse_slash_command(value: str) -> tuple[str, list[str]]:
    """Parse a slash command using shell quoting rules, without executing it."""
    try:
        parts = shlex.split(value.strip())
    except ValueError as exc:
        raise TuiConfigError([message('命令格式错误：{0}', exc)]) from exc
    if not parts or not parts[0].startswith("/"):
        raise TuiConfigError([message('快捷命令必须以 / 开头')])
    return parts[0][1:].lower(), parts[1:]

"""Linux perf based retired-instruction and memory-behavior monitoring."""
from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional


MIPS_EXIT_CODE = 8
PERF_EVENT = "instructions"
PERF_OPTIONAL_EVENTS = (
    "cycles",
    "ref-cycles",
    "cache-references",
    "cache-misses",
    "dTLB-loads",
    "dTLB-load-misses",
)
PERF_EVENTS = (PERF_EVENT, *PERF_OPTIONAL_EVENTS)
PERF_TIMEOUT_MS = 86_400_000
PERF_PROBE_TIMEOUT_S = 5.0
PERF_STOP_TIMEOUT_S = 5.0
_RESOLVED_COMMAND_PREFIX: Optional[List[str]] = None


class MIPSProfilingError(RuntimeError):
    """Raised when required MIPS profiling cannot continue."""


@dataclass
class PerfStatParsed:
    instructions_total: int
    perf_elapsed_s: float
    cycles_total: float = float("nan")
    ref_cycles_total: float = float("nan")
    ipc: float = float("nan")
    running_pct: float = float("nan")
    cache_references_total: float = float("nan")
    cache_misses_total: float = float("nan")
    dtlb_loads_total: float = float("nan")
    dtlb_load_misses_total: float = float("nan")


@dataclass
class MIPSResult:
    instructions_total: float
    instructions_per_request: float
    perf_elapsed_s: float
    cpu_mips_app: float
    cycles_per_request: float = float("nan")
    ref_cycles_per_request: float = float("nan")
    ipc: float = float("nan")
    running_pct: float = float("nan")
    cache_references_per_request: float = float("nan")
    cache_misses_per_request: float = float("nan")
    cache_miss_rate_pct: float = float("nan")
    dtlb_loads_per_request: float = float("nan")
    dtlb_load_misses_per_request: float = float("nan")
    dtlb_load_miss_rate_pct: float = float("nan")


def _clean_numeric_text(value: object) -> str:
    return str(value).strip().replace(",", "")


def _to_float(value: object) -> float:
    try:
        cleaned = _clean_numeric_text(value)
        if not cleaned or cleaned.startswith("<"):
            return float("nan")
        return float(cleaned)
    except Exception:
        return float("nan")


def _parse_elapsed_s(text: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+seconds\s+time\s+elapsed", text)
    if match:
        return float(match.group(1))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "seconds time elapsed" not in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        for part in parts:
            value = _to_float(part)
            if value == value and value >= 0:
                return value
    return float("nan")


def _event_label_matches(label: str, event_name: str) -> bool:
    """Match generic and hybrid-PMU perf CSV event labels."""
    normalized = str(label or "").strip()
    if normalized == event_name or normalized.startswith(f"{event_name}:"):
        return True
    return (
        f"/{event_name}/" in normalized
        or f"/{event_name}:" in normalized
    )


def parse_perf_stat_output(
    text: str,
    *,
    fallback_elapsed_s: Optional[float] = None,
    require_elapsed: bool = True,
) -> PerfStatParsed:
    event_values = {event_name: float("nan") for event_name in PERF_EVENTS}
    event_scopes: dict[str, set[str]] = {name: set() for name in PERF_EVENTS}
    invalid_events: set[str] = set()
    running_percentages = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        event_label = parts[2]
        for event_name in PERF_EVENTS:
            if _event_label_matches(event_label, event_name):
                value = _to_float(parts[0])
                event_scopes[event_name].add(event_label.replace(event_name, "EVENT", 1))
                if event_name in ("instructions", "cycles", "ref-cycles"):
                    running_pct = _to_float(parts[4]) if len(parts) > 4 else float("nan")
                    if math.isfinite(running_pct) and 0 < running_pct <= 100:
                        running_percentages.append(running_pct)
                    if running_pct == 0:
                        value = float("nan")
                if math.isfinite(value) and value >= 0.0:
                    previous = event_values[event_name]
                    event_values[event_name] = (
                        previous + value
                        if math.isfinite(previous)
                        else value
                    )
                else:
                    invalid_events.add(event_name)
                break

    # A partial hybrid-PMU total must not be used as the denominator of IPC.
    for name in ("cycles", "ref-cycles"):
        if name in invalid_events or event_scopes[name] != event_scopes["instructions"]:
            event_values[name] = float("nan")

    elapsed_s = _parse_elapsed_s(text)
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        fallback_elapsed = (
            _to_float(fallback_elapsed_s)
            if fallback_elapsed_s is not None
            else float("nan")
        )
        if math.isfinite(fallback_elapsed) and fallback_elapsed > 0:
            elapsed_s = fallback_elapsed

    instructions = event_values[PERF_EVENT]
    if not math.isfinite(instructions) or instructions < 0:
        raise MIPSProfilingError(
            "perf did not report a valid retired-instructions count for event "
            f"{PERF_EVENT!r}."
        )
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        if require_elapsed:
            raise MIPSProfilingError("perf did not report a valid elapsed time.")
        elapsed_s = float("nan")
    return PerfStatParsed(
        instructions_total=int(instructions),
        perf_elapsed_s=elapsed_s,
        cycles_total=event_values["cycles"],
        ref_cycles_total=event_values["ref-cycles"],
        ipc=(
            instructions / event_values["cycles"]
            if event_values["cycles"] > 0
            and "instructions" not in invalid_events
            and event_scopes["instructions"] == event_scopes["cycles"]
            else float("nan")
        ),
        running_pct=min(running_percentages) if running_percentages else float("nan"),
        cache_references_total=event_values["cache-references"],
        cache_misses_total=event_values["cache-misses"],
        dtlb_loads_total=event_values["dTLB-loads"],
        dtlb_load_misses_total=event_values["dTLB-load-misses"],
    )


def read_perf_event_paranoid(path: str = "/proc/sys/kernel/perf_event_paranoid") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "unavailable"


def _perf_probe_command(prefix: List[str]) -> List[str]:
    return [
        *prefix,
        "stat",
        "--no-big-num",
        "-x",
        ",",
        "-e",
        PERF_EVENT,
        "--",
        "sleep",
        "0.01",
    ]


def _perf_attach_probe_command(prefix: List[str], pid: int) -> List[str]:
    return [
        *prefix,
        "stat",
        "--no-big-num",
        "-x",
        ",",
        "-e",
        PERF_EVENT,
        "-p",
        str(pid),
        "--",
        "sleep",
        "0.01",
    ]


def _run_perf_probe(
    prefix: List[str],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "check": False,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": PERF_PROBE_TIMEOUT_S,
    }
    # A preflight must never wait for input from the user's terminal.
    kwargs["input"] = ""
    if env is not None:
        kwargs["env"] = env
    return subprocess.run(_perf_probe_command(prefix), **kwargs)


def _run_perf_attach_probe(
    prefix: List[str],
    pid: int,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "check": False,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": PERF_PROBE_TIMEOUT_S,
    }
    kwargs["input"] = ""
    if env is not None:
        kwargs["env"] = env
    return subprocess.run(_perf_attach_probe_command(prefix, pid), **kwargs)


def _probe_succeeded(result: subprocess.CompletedProcess) -> bool:
    if result.returncode != 0:
        return False
    try:
        parse_perf_stat_output(
            (result.stderr or "") + "\n" + (result.stdout or ""),
            require_elapsed=False,
        )
    except Exception:
        return False
    return True


def _attach_probe_succeeded(result: subprocess.CompletedProcess) -> bool:
    return result.returncode == 0


def resolve_perf_command_prefix(
    *, env: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Probe real instruction counts without changing the caller's privileges."""
    probe_environ = os.environ if env is None else env
    perf_path = shutil.which("perf", path=probe_environ.get("PATH", os.defpath))
    if not perf_path:
        raise MIPSProfilingError("Linux perf command was not found.")

    try:
        result = _run_perf_probe(["perf"], env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MIPSProfilingError(f"{type(exc).__name__}: {exc}") from exc
    if _probe_succeeded(result):
        # Docker services run as root. A successful child-process probe does
        # not establish permission to attach across user IDs, even at paranoid=-1.
        # PID 1 is stable on the required native Linux host; no container is started.
        try:
            attached = _run_perf_attach_probe(["perf"], 1, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MIPSProfilingError(f"perf attach to host PID 1: {type(exc).__name__}: {exc}") from exc
        if not _attach_probe_succeeded(attached):
            detail = (attached.stderr or attached.stdout or "permission probe failed").strip()
            raise MIPSProfilingError(
                "perf cannot attach to host PID 1; container-process access is not available. "
                "Configure CAP_PERFMON in TUI Settings > Connections and permissions "
                f"or see docs/Getting_Started.md#最小权限安装.\n{detail}"
            )
        return ["perf"]
    detail = "\n".join(
        part.strip() for part in (result.stderr, result.stdout) if part and part.strip()
    ) or f"perf did not report valid instructions (exit={result.returncode})"
    raise MIPSProfilingError(f"perf: {detail}; see docs/Getting_Started.md#最小权限安装")


def get_perf_command_prefix() -> List[str]:
    global _RESOLVED_COMMAND_PREFIX
    if _RESOLVED_COMMAND_PREFIX is None:
        _RESOLVED_COMMAND_PREFIX = resolve_perf_command_prefix()
    return list(_RESOLVED_COMMAND_PREFIX)


def resolve_perf_command_prefix_for_pid(pid: int) -> List[str]:
    perf_path = shutil.which("perf")
    if not perf_path:
        raise MIPSProfilingError("Linux perf command was not found.")

    try:
        direct = _run_perf_attach_probe(["perf"], pid)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MIPSProfilingError(f"{type(exc).__name__}: {exc}") from exc
    if _attach_probe_succeeded(direct):
        return ["perf"]

    last_error = (direct.stderr or direct.stdout or "perf attach probe failed").strip()
    raise MIPSProfilingError(f"{last_error}; see docs/Getting_Started.md#最小权限安装")


def _friendly_mips_error(detail: str) -> str:
    perf_path = shutil.which("perf") or "not found"
    paranoid = read_perf_event_paranoid()
    return (
        "[mips][ERROR] MIPS profiling requires Linux perf access to hardware "
        f"event {PERF_EVENT!r}.\n\n"
        "Detected:\n"
        f"  perf={perf_path}\n"
        f"  perf_event_paranoid={paranoid}\n"
        f"  last_error={detail.strip() or 'unavailable'}\n\n"
        "Recovery steps:\n"
        "  1. Install perf if missing, for example: sudo apt-get install -y linux-tools-common linux-tools-generic\n"
        "  2. Have an administrator grant cap_perfmon=ep to the real perf executable, "
        "restricted to the profiling group. See docs/Getting_Started.md#最小权限安装.\n"
        "  3. Log in again after group membership changes, then rerun acprof doctor.\n\n"
        "After fixing permissions, rerun AC-Prof as your normal user. Avoid "
        "`sudo python run.py ...` because it can leave result files owned by root."
    )


def require_mips_prerequisites() -> None:
    try:
        resolve_perf_command_prefix()
    except Exception as exc:
        print(_friendly_mips_error(str(exc)), file=sys.stderr)
        raise SystemExit(1) from None


def _docker_container_pid(container_name: str) -> int:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise MIPSProfilingError(
            result.stderr.strip() or f"docker inspect failed for {container_name}"
        )

    try:
        pid = int(result.stdout.strip())
    except ValueError as exc:
        raise MIPSProfilingError(f"invalid container pid: {result.stdout.strip()!r}") from exc
    if pid <= 0:
        raise MIPSProfilingError(f"container is not running: {container_name}")
    return pid


def _per_request(total: float, repeat: int) -> float:
    return (
        float(total) / float(repeat)
        if math.isfinite(total) and total >= 0.0
        else float("nan")
    )


def _miss_rate_pct(misses: float, references: float) -> float:
    return (
        float(misses) / float(references) * 100.0
        if (
            math.isfinite(misses)
            and math.isfinite(references)
            and misses >= 0.0
            and references > 0.0
        )
        else float("nan")
    )


class PerfMIPSMonitor:
    def __init__(
        self,
        container_name: str,
        command_prefix: Optional[List[str]] = None,
    ) -> None:
        self.container_name = container_name
        self.command_prefix = command_prefix
        self._proc: Optional[subprocess.Popen] = None
        self._t_start: Optional[float] = None

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            raise MIPSProfilingError("MIPS monitor is already running")
        if not self.container_name:
            raise MIPSProfilingError("CONTAINER_NAME is required for MIPS profiling")

        pid = _docker_container_pid(self.container_name)
        prefix = list(self.command_prefix or resolve_perf_command_prefix_for_pid(pid))
        if len(prefix) != 1 or os.path.basename(prefix[0]) != "perf":
            raise MIPSProfilingError("Only a direct perf executable is supported; configure cap_perfmon first")
        cmd = [
            *prefix,
            "stat",
            "--no-big-num",
            "-x",
            ",",
            "-e",
            ",".join(PERF_EVENTS),
            "-p",
            str(pid),
            "--timeout",
            str(PERF_TIMEOUT_MS),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._t_start = time.perf_counter()
        except OSError as exc:
            raise MIPSProfilingError(f"failed to start perf: {exc}") from exc

    def stop(self, repeat_in_window: int, latency_app_s: float) -> MIPSResult:
        if self._proc is None:
            raise MIPSProfilingError("MIPS monitor was not started")

        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
        try:
            stdout, stderr = self._proc.communicate(timeout=PERF_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            self._proc.kill()
            self._proc.communicate()
            raise MIPSProfilingError("perf did not stop after workload window") from exc

        output = (stderr or "") + "\n" + (stdout or "")
        wall_elapsed_s = (
            time.perf_counter() - self._t_start
            if self._t_start is not None
            else None
        )
        try:
            parsed = parse_perf_stat_output(output, fallback_elapsed_s=wall_elapsed_s)
        except MIPSProfilingError as exc:
            detail = output.strip() or str(exc)
            raise MIPSProfilingError(_friendly_mips_error(detail)) from exc

        repeat = max(1, int(repeat_in_window))
        instructions_per_request = float(parsed.instructions_total) / float(repeat)
        cache_references_per_request = _per_request(
            parsed.cache_references_total,
            repeat,
        )
        cache_misses_per_request = _per_request(
            parsed.cache_misses_total,
            repeat,
        )
        dtlb_loads_per_request = _per_request(
            parsed.dtlb_loads_total,
            repeat,
        )
        dtlb_load_misses_per_request = _per_request(
            parsed.dtlb_load_misses_total,
            repeat,
        )
        latency = _to_float(latency_app_s)
        cpu_mips_app = (
            instructions_per_request / latency / 1_000_000.0
            if latency == latency and latency > 0
            else float("nan")
        )
        return MIPSResult(
            instructions_total=float(parsed.instructions_total),
            instructions_per_request=instructions_per_request,
            perf_elapsed_s=parsed.perf_elapsed_s,
            cpu_mips_app=cpu_mips_app,
            cycles_per_request=_per_request(parsed.cycles_total, repeat),
            ref_cycles_per_request=_per_request(parsed.ref_cycles_total, repeat),
            ipc=parsed.ipc,
            running_pct=parsed.running_pct,
            cache_references_per_request=cache_references_per_request,
            cache_misses_per_request=cache_misses_per_request,
            cache_miss_rate_pct=_miss_rate_pct(
                parsed.cache_misses_total,
                parsed.cache_references_total,
            ),
            dtlb_loads_per_request=dtlb_loads_per_request,
            dtlb_load_misses_per_request=dtlb_load_misses_per_request,
            dtlb_load_miss_rate_pct=_miss_rate_pct(
                parsed.dtlb_load_misses_total,
                parsed.dtlb_loads_total,
            ),
        )

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.communicate(timeout=PERF_STOP_TIMEOUT_S)
            except Exception:
                pass

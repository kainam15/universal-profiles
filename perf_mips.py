"""Linux perf based retired-instruction MIPS monitoring."""
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
from typing import List, Optional


MIPS_EXIT_CODE = 8
PERF_EVENT = "instructions"
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


@dataclass
class MIPSResult:
    instructions_total: float
    instructions_per_request: float
    perf_elapsed_s: float
    cpu_mips_app: float


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


def parse_perf_stat_output(
    text: str,
    *,
    fallback_elapsed_s: Optional[float] = None,
    require_elapsed: bool = True,
) -> PerfStatParsed:
    instructions = float("nan")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or PERF_EVENT not in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if not any(part == PERF_EVENT or part.endswith(f"/{PERF_EVENT}/") for part in parts):
            continue
        if parts:
            instructions = _to_float(parts[0])
            break

    elapsed_s = _parse_elapsed_s(text)
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        fallback_elapsed = (
            _to_float(fallback_elapsed_s)
            if fallback_elapsed_s is not None
            else float("nan")
        )
        if math.isfinite(fallback_elapsed) and fallback_elapsed > 0:
            elapsed_s = fallback_elapsed

    if not math.isfinite(instructions) or instructions < 0:
        raise MIPSProfilingError(
            "perf did not report a valid retired-instructions count for event "
            f"{PERF_EVENT!r}."
        )
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        if require_elapsed:
            raise MIPSProfilingError("perf did not report a valid elapsed time.")
        elapsed_s = float("nan")
    return PerfStatParsed(int(instructions), elapsed_s)


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


def _run_perf_probe(prefix: List[str], password: str = "") -> subprocess.CompletedProcess:
    kwargs = {
        "capture_output": True,
        "text": True,
        "check": False,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": PERF_PROBE_TIMEOUT_S,
    }
    if password:
        kwargs["input"] = f"{password}\n"
    return subprocess.run(_perf_probe_command(prefix), **kwargs)


def _run_perf_attach_probe(
    prefix: List[str],
    pid: int,
    password: str = "",
) -> subprocess.CompletedProcess:
    kwargs = {
        "capture_output": True,
        "text": True,
        "check": False,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": PERF_PROBE_TIMEOUT_S,
    }
    if password:
        kwargs["input"] = f"{password}\n"
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


def resolve_perf_command_prefix() -> List[str]:
    perf_path = shutil.which("perf")
    if not perf_path:
        raise MIPSProfilingError("Linux perf command was not found.")

    direct = _run_perf_probe(["perf"])
    if _probe_succeeded(direct):
        return ["perf"]

    sudo_noninteractive = _run_perf_probe(["sudo", "-n", "perf"])
    if _probe_succeeded(sudo_noninteractive):
        return ["sudo", "-n", "perf"]

    sudo_password = os.environ.get("ACPROF_SUDO_PASSWORD", "").strip()
    if sudo_password:
        sudo_with_password = _run_perf_probe(
            ["sudo", "-S", "-p", "", "perf"],
            password=sudo_password,
        )
        if _probe_succeeded(sudo_with_password):
            return ["sudo", "-S", "-p", "", "perf"]

    last_error = (
        sudo_noninteractive.stderr
        or sudo_noninteractive.stdout
        or direct.stderr
        or direct.stdout
        or "perf probe failed"
    ).strip()
    raise MIPSProfilingError(last_error)


def get_perf_command_prefix() -> List[str]:
    global _RESOLVED_COMMAND_PREFIX
    if _RESOLVED_COMMAND_PREFIX is None:
        _RESOLVED_COMMAND_PREFIX = resolve_perf_command_prefix()
    return list(_RESOLVED_COMMAND_PREFIX)


def resolve_perf_command_prefix_for_pid(pid: int) -> List[str]:
    perf_path = shutil.which("perf")
    if not perf_path:
        raise MIPSProfilingError("Linux perf command was not found.")

    direct = _run_perf_attach_probe(["perf"], pid)
    if _attach_probe_succeeded(direct):
        return ["perf"]

    sudo_noninteractive = _run_perf_attach_probe(["sudo", "-n", "perf"], pid)
    if _attach_probe_succeeded(sudo_noninteractive):
        return ["sudo", "-n", "perf"]

    sudo_with_password: Optional[subprocess.CompletedProcess] = None
    sudo_password = os.environ.get("ACPROF_SUDO_PASSWORD", "").strip()
    if sudo_password:
        sudo_with_password = _run_perf_attach_probe(
            ["sudo", "-S", "-p", "", "perf"],
            pid,
            password=sudo_password,
        )
        if _attach_probe_succeeded(sudo_with_password):
            return ["sudo", "-S", "-p", "", "perf"]

    last_error_result = sudo_with_password or sudo_noninteractive or direct
    last_error = (
        last_error_result.stderr
        or last_error_result.stdout
        or "perf attach probe failed"
    ).strip()
    raise MIPSProfilingError(last_error)


def _friendly_mips_error(detail: str) -> str:
    perf_path = shutil.which("perf") or "not found"
    paranoid = read_perf_event_paranoid()
    password_state = "set" if os.environ.get("ACPROF_SUDO_PASSWORD", "").strip() else "not set"
    return (
        "[mips][ERROR] MIPS profiling requires Linux perf access to hardware "
        f"event {PERF_EVENT!r}.\n\n"
        "Detected:\n"
        f"  perf={perf_path}\n"
        f"  perf_event_paranoid={paranoid}\n"
        f"  ACPROF_SUDO_PASSWORD={password_state}\n"
        f"  last_error={detail.strip() or 'unavailable'}\n\n"
        "Recovery steps:\n"
        "  1. Install perf if missing, for example: sudo apt-get install -y linux-tools-common linux-tools-generic\n"
        "  2. Temporary permission fix for this boot:\n"
        "     echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid\n"
        "  3. Or set ACPROF_SUDO_PASSWORD in .env.local so AC-Prof can run sudo -S perf.\n\n"
        "Note: profiling Docker container PIDs may still require sudo perf even when "
        "perf_event_paranoid=0, because those processes are often owned by root.\n\n"
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


def _prefix_uses_password(prefix: List[str]) -> bool:
    return len(prefix) >= 3 and prefix[0] == "sudo" and "-S" in prefix


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
        password = ""
        if _prefix_uses_password(prefix):
            password = os.environ.get("ACPROF_SUDO_PASSWORD", "").strip()
            if not password:
                raise MIPSProfilingError(
                    "sudo -S perf was selected but ACPROF_SUDO_PASSWORD is not set"
                )
        cmd = [
            *prefix,
            "stat",
            "--no-big-num",
            "-x",
            ",",
            "-e",
            PERF_EVENT,
            "-p",
            str(pid),
            "--timeout",
            str(PERF_TIMEOUT_MS),
        ]
        stdin = subprocess.PIPE if _prefix_uses_password(prefix) else subprocess.DEVNULL
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._t_start = time.perf_counter()
            if _prefix_uses_password(prefix):
                assert self._proc.stdin is not None
                self._proc.stdin.write(f"{password}\n")
                self._proc.stdin.flush()
                self._proc.stdin.close()
                self._proc.stdin = None
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
        )

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.communicate(timeout=PERF_STOP_TIMEOUT_S)
            except Exception:
                pass

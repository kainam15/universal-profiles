"""Linux、Docker、cgroup 和 CPU 能耗采集的主机预检。"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = str(Path(__file__).resolve().parents[2])
NATIVE_DOCKER_SOCKET = "/var/run/docker.sock"


def _docker_info_is_docker_desktop(info: str) -> bool:
    normalized = (info or "").lower()
    return "docker desktop" in normalized or "name=docker-desktop" in normalized


def _docker_context_is_docker_desktop(context_name: str) -> bool:
    normalized = (context_name or "").strip().lower()
    return normalized in {"desktop-linux", "docker-desktop"} or normalized.startswith("desktop-")


def _process_is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in platform.release().lower()
    except Exception:
        return False


def _exit_unsupported_host(reason: str) -> None:
    print(
        "[infra][ERROR] AC-Prof requires a native Linux host; "
        f"{reason}.\n\n"
        "WSL and Docker Desktop do not reliably expose all host-side data "
        "sources used by this project, including RAPL, perf PMU events, the "
        "Docker bridge, cgroups, and the NVIDIA runtime.\n\n"
        "Run AC-Prof directly from native Ubuntu as a normal user:\n"
        f"  cd {PROJECT_DIR}\n"
        "  source .venv/bin/activate\n"
        "  unset DOCKER_HOST DOCKER_CONTEXT\n"
        "  docker context use default\n"
        "  python run.py --model <model-id> ...\n",
        file=sys.stderr,
    )
    sys.exit(1)


def require_native_linux_host() -> None:
    """Exit before profiling when the process is not on native Linux."""
    try:
        system = platform.system()
    except Exception:
        system = ""

    if system != "Linux":
        _exit_unsupported_host(f"detected host OS {system or 'unknown'}")
    if _process_is_wsl():
        _exit_unsupported_host("WSL was detected")


def detect_cgroup_version(
    *,
    cgroup_root: str = "/sys/fs/cgroup",
    proc_self_cgroup_path: str = "/proc/self/cgroup",
) -> str:
    """Return the host cgroup hierarchy version used by this process."""
    if os.path.isfile(os.path.join(cgroup_root, "cgroup.controllers")):
        return "v2"

    try:
        with open(proc_self_cgroup_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return "unknown"

    for raw_line in lines:
        parts = raw_line.strip().split(":", 2)
        if len(parts) == 3 and parts[1].strip():
            return "v1"
    return "unknown"


def require_cgroup_prerequisites() -> str:
    """仅支持统一 cgroup v2，在任何采集前拒绝其它层级。"""
    version = detect_cgroup_version()
    if version != "v2":
        print(f"[cgroup][ERROR] AC-Prof requires unified cgroup v2; detected {version}. "
              "Enable cgroup v2 in the host boot/systemd configuration and reboot.", file=sys.stderr)
        sys.exit(1)
    return version


def require_result_cgroup_compatibility(
    output_dir: str,
    *,
    cgroup_version: str,
) -> None:
    """Refuse to append partial case rows collected under another hierarchy."""
    result_dir = Path(output_dir)
    if not result_dir.is_dir():
        return

    partial_results = sorted(
        path
        for path in result_dir.glob("result_case_*.csv")
        if path.is_file() and path.stat().st_size > 0
    )
    if not partial_results:
        return

    static_meta_path = result_dir / "static_meta.json"
    try:
        with static_meta_path.open("r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        existing_version = str(existing_meta.get("cgroup_version") or "unknown")
    except (OSError, ValueError, TypeError, AttributeError):
        existing_version = "unknown"

    if existing_version == cgroup_version:
        return

    examples = ", ".join(path.name for path in partial_results[:3])
    if len(partial_results) > 3:
        examples += ", ..."
    print(
        "[cgroup][ERROR] Refusing to append to partial result files with a "
        "different or unknown cgroup provenance.\n\n"
        f"Current cgroup_version:  {cgroup_version}\n"
        f"Existing cgroup_version: {existing_version}\n"
        f"Partial files: {examples}\n\n"
        "Use a different --output-dir for this run, or archive the existing "
        "partial result files before retrying. AC-Prof will not mix them "
        "automatically.",
        file=sys.stderr,
    )
    sys.exit(1)


def _docker_host_is_native_socket(docker_host: str) -> bool:
    normalized = (docker_host or "").strip()
    if not normalized:
        return False
    if normalized.startswith("unix://"):
        normalized = normalized[len("unix://"):]
    elif normalized.startswith("unix:"):
        normalized = normalized[len("unix:"):]
    else:
        return False
    return os.path.normpath(normalized) == NATIVE_DOCKER_SOCKET


def _exit_nonlocal_docker(docker_host: str) -> None:
    print(
        "[infra][ERROR] AC-Prof is not connected to the native Docker socket "
        f"{NATIVE_DOCKER_SOCKET}.\n\n"
        f"Detected Docker endpoint: {docker_host or 'unknown'}\n\n"
        "Packet capture, container cgroups, perf PID attachment, and host "
        "profilers must observe containers created by the local Ubuntu daemon.\n\n"
        "Switch back to the native daemon before running again:\n"
        "  unset DOCKER_HOST DOCKER_CONTEXT\n"
        "  docker context use default\n"
        f"  test \"$(docker context inspect default --format "
        f"'{{{{(index .Endpoints \"docker\").Host}}}}')\" = "
        f"\"unix://{NATIVE_DOCKER_SOCKET}\"\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _exit_docker_desktop() -> None:
    print(
        "[infra][ERROR] AC-Prof is currently connected to Docker Desktop, not "
        "the native Linux Docker daemon.\n\n"
        "Docker Desktop cannot reliably expose the host /opt profiler installs, "
        "docker0 traffic, or NVIDIA GPU runtime needed by this project.\n\n"
        "Switch to native Docker before running again, for example:\n"
        "  docker context use default\n"
        "or run one command with:\n"
        "  DOCKER_HOST=unix:///var/run/docker.sock python run.py ...\n",
        file=sys.stderr,
    )
    sys.exit(1)


def require_native_docker() -> None:
    """Require the local native-Linux Docker daemon used by host monitors."""
    try:
        context_result = subprocess.run(
            ["docker", "context", "show"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if (
            context_result.returncode == 0
            and _docker_context_is_docker_desktop(context_result.stdout)
        ):
            _exit_docker_desktop()

        docker_host = os.environ.get("DOCKER_HOST", "").strip()
        if not docker_host and context_result.returncode == 0:
            context_name = context_result.stdout.strip()
            endpoint_result = subprocess.run(
                [
                    "docker",
                    "context",
                    "inspect",
                    context_name,
                    "--format",
                    '{{(index .Endpoints "docker").Host}}',
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
            if endpoint_result.returncode == 0:
                docker_host = endpoint_result.stdout.strip()

        if docker_host and not _docker_host_is_native_socket(docker_host):
            _exit_nonlocal_docker(docker_host)

        result = subprocess.run(
            [
                "docker",
                "info",
                "--format",
                "Name={{.Name}}\n"
                "OperatingSystem={{.OperatingSystem}}\n"
                "DockerRootDir={{.DockerRootDir}}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except FileNotFoundError:
        print(
            "[infra][ERROR] Docker CLI was not found. Install Docker and run AC-Prof "
            "against the native Linux Docker daemon.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(
            "[infra][ERROR] `docker info` timed out. Check that the native Linux "
            "Docker daemon is running before starting AC-Prof.",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        print(
            "[infra][ERROR] Could not talk to Docker. Start the native Linux Docker "
            f"daemon and retry.\n\nDocker output:\n{detail}",
            file=sys.stderr,
        )
        sys.exit(1)

    if _docker_info_is_docker_desktop(result.stdout):
        _exit_docker_desktop()


def require_cpu_energy_prerequisites() -> None:
    """Exit early when CPU/vCPU energy profiling cannot be collected."""
    try:
        from acprof.monitors import energy_cpu

        cpu_power_source = energy_cpu.detect_cpu_power_source()
        vcpu_power_method = energy_cpu.detect_vcpu_power_method()
    except Exception as exc:
        print(
            "[cpu-energy][ERROR] CPU/vCPU energy profiling is required, but "
            f"AC-Prof could not run the CPU energy detector: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if cpu_power_source == "rapl" and vcpu_power_method == "rapl_cgroup_cpu_share":
        return

    print(
        "[cpu-energy][ERROR] CPU/vCPU energy profiling is required, but AC-Prof "
        "cannot read the Linux RAPL powercap counters needed for CPU package "
        "energy and estimated vCPU energy.\n\n"
        f"Detected: cpu_power_source={cpu_power_source}, "
        f"vcpu_power_method={vcpu_power_method}\n\n"
        "Common cause on Linux: /sys/class/powercap/intel-rapl:*/energy_uj "
        "exists but is only readable by root.\n\n"
        "Check current permissions:\n"
        "  ls -l /sys/class/powercap/intel-rapl:*/energy_uj\n\n"
        "Temporary fix for the current boot:\n"
        "  sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj\n\n"
        "Persistent fix with systemd-tmpfiles:\n"
        "  echo 'z /sys/class/powercap/intel-rapl:*/energy_uj 0444 root root -' | "
        "sudo tee /etc/tmpfiles.d/acprof-rapl.conf\n"
        "  sudo systemd-tmpfiles --create /etc/tmpfiles.d/acprof-rapl.conf\n\n"
        "After fixing permissions, rerun AC-Prof as your normal user. Avoid "
        "`sudo python run.py ...` because it can leave result files owned by root.",
        file=sys.stderr,
    )
    sys.exit(1)

"""主机抓包前置检查与运行命令。"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional

from acprof.config import SERVER_PORT
from acprof.host.docker_runtime import (
    _run,
)


@dataclass
class PacketLatencyRuntime:
    mode: str
    tcpdump_cmd: List[str]
    parse_cmd: List[str]


class PacketLatencyError(RuntimeError):
    """Raised when required packet-level latency cannot be collected."""


TCPDUMP_CAPTURE_CAPABILITY = "cap_net_raw,cap_net_admin=eip"
PACKET_LATENCY_RECOVERY_STEPS = (
    "Recovery steps:\n"
    "  1. Install packet tools: sudo apt-get install -y tcpdump tshark\n"
    "  2. Grant capture capability: sudo setcap "
    f"{TCPDUMP_CAPTURE_CAPABILITY} $(command -v tcpdump)\n"
    "  3. Verify capability: getcap $(command -v tcpdump)\n"
    "  4. Verify Docker bridge: ip link show docker0\n"
    "  5. If your bridge differs, pass --sniff-iface <iface>."
)


def _tcpdump_can_capture_without_sudo(tcpdump_path: str) -> bool:
    if os.geteuid() == 0:
        return True

    result = _run(["getcap", tcpdump_path], check=False)
    if result.returncode != 0:
        return False

    caps = result.stdout.lower()
    return "cap_net_raw" in caps and "cap_net_admin" in caps


def _packet_latency_error(reason: str, detail: str = "") -> PacketLatencyError:
    parts = [f"packet latency is required but unavailable: {reason}."]
    if detail:
        parts.append(f"Details: {detail}")
    parts.append(PACKET_LATENCY_RECOVERY_STEPS)
    return PacketLatencyError("\n".join(parts))


def _sniff_interface_exists(sniff_iface: str) -> bool:
    if not sniff_iface:
        return False

    sysfs_path = os.path.join("/sys/class/net", sniff_iface)
    if os.path.exists(sysfs_path):
        return True

    ip_cmd = shutil.which("ip")
    if not ip_cmd:
        return False

    result = _run([ip_cmd, "link", "show", sniff_iface], check=False)
    return result.returncode == 0


def require_packet_latency_prerequisites(project_dir: str, sniff_iface: str) -> None:
    """Fail early when native-Linux packet latency cannot be collected."""
    del project_dir  # Kept for compatibility with existing callers.

    missing_tools = [
        name for name in ("tcpdump", "tshark")
        if shutil.which(name) is None
    ]
    if missing_tools:
        raise _packet_latency_error(
            f"missing required command(s): {', '.join(missing_tools)}"
        )

    if not _sniff_interface_exists(sniff_iface):
        raise _packet_latency_error(
            f"network interface {sniff_iface!r} was not found"
        )

    tcpdump_path = shutil.which("tcpdump")
    if not tcpdump_path:
        raise _packet_latency_error("tcpdump was not found")

    if not _ensure_tcpdump_capture_capability(tcpdump_path):
        raise _packet_latency_error(
            f"tcpdump lacks capture capability: {tcpdump_path}"
        )


def _try_set_tcpdump_capture_capability(tcpdump_path: str) -> bool:
    setcap_cmd = ["setcap", TCPDUMP_CAPTURE_CAPABILITY, tcpdump_path]

    if os.geteuid() == 0:
        result = _run(setcap_cmd, check=False)
        return result.returncode == 0

    result = _run(["sudo", "-n", *setcap_cmd], check=False)
    if result.returncode == 0:
        return True

    sudo_password = os.environ.get("ACPROF_SUDO_PASSWORD", "").strip()
    if not sudo_password:
        print(
            "[sniff][WARN] tcpdump lacks capture capability and sudo needs a password. "
            "Set ACPROF_SUDO_PASSWORD in .env.local or run "
            f"`sudo setcap {TCPDUMP_CAPTURE_CAPABILITY} {tcpdump_path}` once.",
            file=sys.stderr,
        )
        return False

    result = _run(
        ["sudo", "-S", "-p", "", *setcap_cmd],
        check=False,
        input=f"{sudo_password}\n",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        print(
            "[sniff][WARN] Failed to grant tcpdump capture capability via sudo. "
            f"Packet latency may remain nan. Details: {detail}",
            file=sys.stderr,
        )
        return False

    return True


def _ensure_tcpdump_capture_capability(tcpdump_path: str) -> bool:
    if _tcpdump_can_capture_without_sudo(tcpdump_path):
        return True

    print(f"[sniff] tcpdump capture capability missing; trying to grant it on {tcpdump_path}")
    if not _try_set_tcpdump_capture_capability(tcpdump_path):
        return False

    if _tcpdump_can_capture_without_sudo(tcpdump_path):
        print("[sniff] tcpdump capture capability is ready")
        return True

    print(
        "[sniff][WARN] setcap finished but tcpdump capability is still unavailable. "
        "Packet latency may remain nan.",
        file=sys.stderr,
    )
    return False


def _resolve_packet_latency_runtime(
    project_dir: str,
    pcap_file: str,
    sniff_iface: str,
) -> Optional[PacketLatencyRuntime]:
    del project_dir  # Packet capture and parsing now always run on the local Linux host.
    local_tcpdump = shutil.which("tcpdump")
    local_tshark = shutil.which("tshark")
    if local_tcpdump and local_tshark:
        tcpdump_cmd = (
            [local_tcpdump]
            if _ensure_tcpdump_capture_capability(local_tcpdump)
            else ["sudo", "-n", "tcpdump"]
        )
        return PacketLatencyRuntime(
            mode="local",
            tcpdump_cmd=tcpdump_cmd + [
                "-i",
                sniff_iface,
                "-s",
                "0",
                "-B",
                "4096",
                "-w",
                pcap_file,
                "tcp",
                "port",
                str(SERVER_PORT),
            ],
            parse_cmd=[
                sys.executable,
                "-m",
                "acprof.packet.sniff_parse_pcap",
                pcap_file,
                str(SERVER_PORT),
            ],
        )

    return None

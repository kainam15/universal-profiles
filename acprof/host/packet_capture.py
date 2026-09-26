"""主机抓包前置检查与运行命令。"""
from __future__ import annotations

import os
import shutil
import re
from dataclasses import dataclass
from typing import List, Optional

from acprof.config import SERVER_PORT
from acprof.installation import module_command
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


TCPDUMP_CAPTURE_CAPABILITY = "cap_net_raw=ep"
PACKET_LATENCY_RECOVERY_STEPS = (
    "Recovery steps:\n"
    "  1. Install packet tools: sudo apt-get install -y tcpdump tshark\n"
    "  2. Ask an administrator for group-restricted capture access:\n"
    "     docs/Getting_Started.md#最小权限安装 (sudo setcap cap_net_raw=ep on the real executable)\n"
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

    return tcpdump_capability_available(result.stdout)


def tcpdump_capability_available(capabilities: str) -> bool:
    """Share capability interpretation with the read-only TUI checks."""
    caps = capabilities.lower()
    return any("cap_net_raw" in names.split(",") and "e" in flags and "p" in flags
               for names, flags in re.findall(r"(cap_[a-z_,]+)=([eip]+)", caps))


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


def require_packet_latency_prerequisites(sniff_iface: str) -> None:
    """Fail early when native-Linux packet latency cannot be collected."""
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

    if not _tcpdump_can_capture_without_sudo(tcpdump_path):
        raise _packet_latency_error(
            f"tcpdump lacks capture capability: {tcpdump_path}"
        )


def _resolve_packet_latency_runtime(
    project_dir: str,
    pcap_file: str,
    sniff_iface: str,
) -> Optional[PacketLatencyRuntime]:
    del project_dir  # Packet capture and parsing now always run on the local Linux host.
    local_tcpdump = shutil.which("tcpdump")
    local_tshark = shutil.which("tshark")
    if local_tcpdump and local_tshark:
        if not _tcpdump_can_capture_without_sudo(local_tcpdump):
            raise _packet_latency_error(f"tcpdump lacks capture capability: {local_tcpdump}")
        return PacketLatencyRuntime(
            mode="local",
            tcpdump_cmd=[local_tcpdump,
                "-p",
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
                *module_command("acprof.packet.sniff_parse_pcap"),
                pcap_file,
                str(SERVER_PORT),
            ],
        )

    return None

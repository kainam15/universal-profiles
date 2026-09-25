"""Lazy public command dispatch for installed and standalone AC-Prof."""
from __future__ import annotations

import argparse
from importlib import import_module
import os
import runpy
import sys

from acprof import __version__


COMMANDS = {
    "run": "run", "tui": "tui", "probe": "probe", "plot": "plot", "doctor": "doctor",
    "profile": "posthoc", "audit": "audit", "stats": "stats", "inspect": "inspect", "auto": "auto",
    "coverage": "coverage",
}
WORKERS = {"acprof.host.client", "acprof.packet.sniff_parse_pcap",
           "acprof.packet.merge_packet_latency"}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if getattr(sys, "frozen", False):
        # PyInstaller's libraries must not override Docker/perf/tshark system libraries.
        original = os.environ.pop("LD_LIBRARY_PATH_ORIG", None)
        if original is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = original
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(line_buffering=True, write_through=True)
    if arguments and arguments[0] == "_worker":
        if len(arguments) < 2 or arguments[1] not in WORKERS:
            raise SystemExit("Unknown AC-Prof worker")
        sys.argv = arguments[1:]
        runpy.run_module(arguments[1], run_name="__main__")
        return 0
    parser = argparse.ArgumentParser(prog="acprof", description="AC-Prof 推理服务分析工具")
    parser.add_argument("--version", action="version", version=f"AC-Prof {__version__}")
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS),
                        help="使用 acprof <command> --help 查看参数")
    # Let each existing parser own all of its flags (including --help).
    if not arguments or arguments[0] not in COMMANDS:
        parser.parse_args(arguments)
        parser.print_help()
        return 0
    command = arguments.pop(0)
    previous = sys.argv
    try:
        sys.argv = [f"acprof {command}", *arguments]
        result = import_module(f"acprof.cli.{COMMANDS[command]}").main()
        return result if isinstance(result, int) else 0
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())

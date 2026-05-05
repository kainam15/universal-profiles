#!/usr/bin/env python3
"""Compatibility wrapper for `python merge_packet_latency.py`."""

from acprof.packet.merge_packet_latency import *  # noqa: F401,F403
from acprof.packet.merge_packet_latency import main


if __name__ == "__main__":
    main()


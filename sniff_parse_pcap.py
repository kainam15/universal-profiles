#!/usr/bin/env python3
"""Compatibility wrapper for `python sniff_parse_pcap.py`."""

from acprof.packet.sniff_parse_pcap import *  # noqa: F401,F403
from acprof.packet.sniff_parse_pcap import main


if __name__ == "__main__":
    main()


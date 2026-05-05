#!/usr/bin/env python3
"""Compatibility wrapper for `python client.py`."""

from acprof.host.client import *  # noqa: F401,F403
from acprof.host.client import run_cli


if __name__ == "__main__":
    run_cli()


#!/usr/bin/env python3
"""Compatibility wrapper for `python plot.py`."""

from acprof.cli.plot import *  # noqa: F401,F403
from acprof.cli.plot import main


if __name__ == "__main__":
    main()


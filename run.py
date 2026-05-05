#!/usr/bin/env python3
"""Compatibility wrapper for `python run.py`."""

from acprof.cli.run import *  # noqa: F401,F403
from acprof.cli.run import main


if __name__ == "__main__":
    main()


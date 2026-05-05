#!/usr/bin/env python3
"""Compatibility wrapper for `python compute_profile_runner.py`."""

from acprof.container.compute_profile_runner import *  # noqa: F401,F403
from acprof.container.compute_profile_runner import main


if __name__ == "__main__":
    main()


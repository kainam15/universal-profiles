#!/usr/bin/env python3
"""Compatibility wrapper for `python download_model.py`."""

from acprof.container.download_model import *  # noqa: F401,F403
from acprof.container.download_model import main


if __name__ == "__main__":
    main()


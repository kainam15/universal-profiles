#!/usr/bin/env python3
"""Compatibility wrapper for `python server.py`."""

from acprof.container.server import *  # noqa: F401,F403
from acprof.container.server import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)


#!/usr/bin/env python3
"""Convenience entry point for profiler-only collection and CSV backfill.

When imported as ``profile`` this file proxies Python's standard-library
module, avoiding breakage for ``cProfile`` while keeping the requested
``python profile.py ...`` command at the repository root.
"""

if __name__ == "__main__":
    from acprof.cli.posthoc import main

    main()
else:
    import importlib.util
    import sysconfig
    from pathlib import Path

    _stdlib_path = Path(sysconfig.get_path("stdlib")) / "profile.py"
    _stdlib_spec = importlib.util.spec_from_file_location(
        "_acprof_stdlib_profile",
        _stdlib_path,
    )
    if _stdlib_spec is None or _stdlib_spec.loader is None:
        raise ImportError(f"cannot load standard-library profile module: {_stdlib_path}")
    _stdlib_module = importlib.util.module_from_spec(_stdlib_spec)
    _stdlib_spec.loader.exec_module(_stdlib_module)
    for _name in dir(_stdlib_module):
        if _name not in {"__file__", "__name__", "__package__", "__spec__"}:
            globals()[_name] = getattr(_stdlib_module, _name)


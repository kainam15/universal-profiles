"""Resolve, preflight and collect one exact model using the existing run engine."""
from __future__ import annotations

from pathlib import Path
import sys


def main(argv=None) -> int:
    from acprof.cli.run_args import build_parser
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.automation import AutomaticRun
    from acprof.host.run_state import MeasurementLock, ResultDirectoryLock
    parser = build_parser(automatic=True)
    args = parser.parse_args(argv)
    bootstrap_project_env(Path.cwd())
    automatic = None
    try:
        automatic = AutomaticRun(args)
        with MeasurementLock(), ResultDirectoryLock(automatic.root):
            task = automatic.prepare()
        print(f"[auto] {task.model_id}@{task.model_revision}; profiling_mode={args.profiling_mode}", flush=True)
        from acprof.cli import run
        run.main(args=args, prepared_task=task, preparation_artifacts=automatic.preparation_artifacts)
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            error = None
        else:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[auto][ERROR] {error}", file=sys.stderr)
        return automatic.finish(error) if automatic else 2
    except KeyboardInterrupt:
        if automatic:
            automatic.finish("interrupted")
        raise
    return automatic.finish()


if __name__ == "__main__":
    raise SystemExit(main())

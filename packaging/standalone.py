"""PyInstaller entry point; same dispatch as the wheel console script."""
from acprof.cli.main import main

raise SystemExit(main())

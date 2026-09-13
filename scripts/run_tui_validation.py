"""由硬件对照脚本驱动真实 TUI 子进程路径，使用独立设置文件。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.tui.app import AcprofTui, PendingLaunch
from acprof.tui.commands import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", type=Path)
    parser.add_argument("--ui", choices=("headless", "terminal"), default="headless")
    args = parser.parse_args()
    payload = json.loads(args.command.read_text())
    config = RunConfig(**payload["config"])

    class ValidationTui(AcprofTui):
        def on_mount(self):
            # Textual 按 MRO 分发 Mount，父类 handler 会自动执行。
            self.call_after_refresh(self._launch, PendingLaunch(tuple(payload["command"]), "run", config))

        def _process_finished(self, kind, returncode, snapshot, launch_error):
            super()._process_finished(kind, returncode, snapshot, launch_error)
            self.save_screenshot(str(args.command.parent / "tui-finished.svg"))
            self.exit(returncode if not launch_error else 1)

    app = ValidationTui(initial_config=config, settings_path=args.command.parent / "tui-settings.json")
    return app.run(headless=args.ui == "headless", size=(120, 30) if args.ui == "headless" else None) or 0


if __name__ == "__main__":
    raise SystemExit(main())

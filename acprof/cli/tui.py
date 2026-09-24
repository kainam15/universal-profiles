"""TUI 命令参数与启动。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Sequence

from acprof.tui.app import AcprofTui

from acprof.tui.commands import RunConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AC-Prof interactive terminal interface",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Pre-fill the Hugging Face model ID (overrides the last run model)",
    )
    parser.add_argument(
        "--preset",
        choices=("default", "smoke", "main"),
        default=None,
        help="Initial form preset (overrides saved experiment defaults)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Initial result directory (overrides the preset or saved directory)",
    )
    parser.add_argument(
        "--color-system",
        choices=("truecolor", "256", "auto"),
        default="truecolor",
        help="Terminal colors: RGB by default; use 256 for older terminals or auto for environment detection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        app = AcprofTui(color_system=args.color_system)
    except ValueError as exc:
        parser.error(str(exc))
    config = app.initial_config
    model = config.model if args.model is None else args.model
    if args.preset == "smoke":
        config = RunConfig.smoke(model)
    elif args.preset == "main":
        config = RunConfig.main_matrix(model)
    elif args.preset == "default":
        config = RunConfig(model=model)
    else:
        config = replace(config, model=model)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    app.initial_config = config
    app._initial_preset = app._infer_preset(config)
    app.run()


if __name__ == "__main__":
    main()

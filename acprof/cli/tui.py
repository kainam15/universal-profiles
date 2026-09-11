"""TUI CLI 参数解析和旧导入路径的兼容入口。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Sequence

from acprof.tui.app import (
    AcprofTui,
    PROJECT_DIR,
    PYTHON_EXECUTABLE,
    PendingLaunch,
)

from acprof.tui.commands import (
    RunConfig,
    TuiConfigError,
    build_plot_command,
    build_probe_command,
    build_profile_command,
    build_run_command,
    format_command,
    parse_slash_command,
)

from acprof.tui.diagnostics import (
    PreflightCheck,
    quick_preflight,
    summarize_result_csv,
)

from acprof.tui.i18n import (
    LANGUAGE_OPTIONS,
    error_message,
    join_messages,
    message,
    translate,
)

from acprof.tui.input import (
    BarCursorApp,
    BarCursorInput as Input,
)

from acprof.tui.log import (
    SelectableLog,
)

from acprof.tui.progress import (
    ProgressSnapshot,
    RunProgressTracker,
)

from acprof.tui.scrollbar import (
    SolidScrollBarRender,
)

from acprof.tui.settings import (
    UiPreferences,
    default_settings_path,
    load_settings,
    save_settings,
)

from acprof.tui.themes import (
    THEME_CATALOG,
    THEME_OPTIONS,
)

from acprof.tui.views import (
    COLLAPSED_SYMBOL,
    ConfirmActionScreen,
    EXPANDED_SYMBOL,
    LogPanel,
    StatusCheckbox,
)


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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    app = AcprofTui()
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
    app.initial_config = config
    app._initial_preset = app._infer_preset(config)
    app.run()


if __name__ == "__main__":
    main()

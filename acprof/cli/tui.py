#!/usr/bin/env python3
"""Interactive, low-refresh terminal controller for AC-Prof."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Sequence

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
    from textual.screen import ModalScreen
    from textual.theme import Theme
    from textual.widgets import (
        Button,
        Checkbox,
        Collapsible,
        ContentSwitcher,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        ProgressBar,
        RichLog,
        Select,
        Static,
        TabbedContent,
        TabPane,
        Tabs,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised before tests install deps
    if exc.name == "textual":
        raise SystemExit(
            "AC-Prof TUI 需要 Textual。请运行：\n"
            "  .venv/bin/python -m pip install -r requirements.txt"
        ) from None
    raise

from acprof.cli.tui_core import (
    PreflightCheck,
    ProgressSnapshot,
    RunConfig,
    RunProgressTracker,
    TuiConfigError,
    build_plot_command,
    build_probe_command,
    build_profile_command,
    build_run_command,
    format_command,
    parse_slash_command,
    quick_preflight,
    summarize_result_csv,
)

from acprof.cli.tui_settings import (
    TuiSettings, UiPreferences, default_settings_path, load_settings, save_settings,
)
from acprof.cli.tui_themes import THEME_CATALOG, THEME_OPTIONS


PROJECT_DIR = Path(__file__).resolve().parents[2]
# Keep the virtual-environment path. Resolving this symlink would turn
# ``.venv/bin/python`` into the system interpreter and lose the venv.
PYTHON_EXECUTABLE = Path(sys.executable).absolute()


@dataclass(frozen=True)
class PendingLaunch:
    command: tuple[str, ...]
    kind: str
    config: RunConfig | None = None


class StatusCheckbox(Checkbox):
    """Checkbox whose selected mark is unambiguous in dark terminals."""

    BUTTON_INNER = "✓"


class ConfirmActionScreen(ModalScreen[bool]):
    """Small confirmation screen for long-running or mutating actions."""

    BINDINGS = [("escape", "cancel", "取消")]

    CSS = """
    ConfirmActionScreen {
        align: center middle;
    }

    #confirm-dialog {
        width: 92%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }

    #confirm-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #confirm-message {
        height: auto;
        max-height: 18;
        overflow-y: auto;
        margin-bottom: 1;
    }

    #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }

    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, title: str, message: str, confirm_label: str = "确认"):
        super().__init__()
        self.dialog_title = title
        self.message = message
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(self.dialog_title, id="confirm-title", markup=False)
            yield Static(self.message, id="confirm-message", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button("取消", id="confirm-no")
                yield Button(
                    self.confirm_label,
                    id="confirm-yes",
                    variant="warning",
                )

    @on(Button.Pressed, "#confirm-yes")
    def confirm(self) -> None:
        self.dismiss(True)

    def on_mount(self) -> None:
        for button in self.query(Button):
            button.active_effect_duration = 0

    @on(Button.Pressed, "#confirm-no")
    def cancel(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class AcprofTui(App[None]):
    """Full-screen controller for AC-Prof collection and diagnostics."""

    TITLE = "AC-Prof"
    SUB_TITLE = "推理实验控制台"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        ("f5", "request_run", "开始采集"),
        ("f6", "quick_check", "环境检查"),
        ("f2", "show_settings", "设置"),
        ("ctrl+x", "request_stop", "终止任务"),
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+q", "request_quit", "退出"),
    ]

    CSS_PATH = "tui.tcss"

    def __init__(
        self, initial_config: RunConfig | None = None, *, settings_path: Path | None = None,
    ):
        super().__init__()
        self.animation_level = "none"
        self.settings_path = settings_path or default_settings_path(PROJECT_DIR)
        self._saved_settings, self._settings_warning = load_settings(
            self.settings_path, PROJECT_DIR,
        )
        self.ui_preferences = self._saved_settings.ui
        self.initial_config = initial_config or self._saved_settings.run_defaults or RunConfig()
        for palette in THEME_CATALOG:
            self.register_theme(Theme(**palette.theme_kwargs()))
        self.theme = self.ui_preferences.theme
        self._process: subprocess.Popen[str] | None = None
        self._process_kind = ""
        self._process_lock = threading.Lock()
        self._pending_launch: PendingLaunch | None = None
        self._active_run_config: RunConfig | None = None
        self._active_command: tuple[str, ...] = ()
        self._started_monotonic = 0.0
        self._stop_requested = False
        self._latest_snapshot = ProgressSnapshot()
        self._check_running = False
        self._form_ready = False
        self._applying_config = False
        self._preview_timer = None
        self._ignored_preset_event: str | None = None
        self._initial_preset = self._infer_preset(self.initial_config)
        self._elapsed_timer = None
        self._matrix_rows: dict[int, object] = {}

    def compose(self) -> ComposeResult:
        # A ticking clock would force periodic redraws during RAPL windows.
        yield Header(show_clock=False)
        with TabbedContent(initial="run-tab", id="main-tabs"):
            with TabPane("实验配置", id="run-tab"):
                with ContentSwitcher(initial="run-form", id="experiment-pages"):
                    with VerticalScroll(id="run-form", classes="pane-scroll"):
                        yield Static("配置实验", classes="section-title")
                        with Grid(classes="form-grid"):
                            yield Label("模型 ID")
                            yield Input(
                                value=self.initial_config.model,
                                placeholder="google-bert/bert-base-uncased",
                                id="model",
                                classes="config-control",
                            )
                            yield Label("输出目录")
                            yield Input(
                                value=self.initial_config.output_dir,
                                id="output-dir",
                                classes="config-control",
                            )

                            yield Label("运行预设")
                            yield Select(
                                (
                                    ("自定义", "custom"),
                                    ("最小 Smoke", "smoke"),
                                    ("主矩阵（分析器关闭）", "main"),
                                    ("完整默认", "default"),
                                ),
                                value=self._initial_preset,
                                allow_blank=False,
                                id="run-preset",
                            )
                            yield Label("", id="preset-hint-label")
                            yield Static("预设自动填充 · 下方可打开高级参数", id="preset-hint", markup=False)

                            yield Label("CPU 列表")
                            yield Input(
                                value=self.initial_config.cpus,
                                placeholder="1,2,4,8",
                                id="cpus",
                                classes="config-control",
                            )
                            yield Label("内存 GB")
                            yield Input(
                                value=self.initial_config.mems,
                                placeholder="2,4,8,16",
                                id="mems",
                                classes="config-control",
                            )

                            yield Label("GPU 模式")
                            yield Select(
                                self._gpu_options(),
                                value=self.initial_config.gpus,
                                allow_blank=False,
                                id="gpus",
                                classes="config-control",
                            )
                            yield Label("输入规模")
                            yield Input(
                                value=self.initial_config.input_scales,
                                placeholder="留空自动规划；如 64,128,256",
                                id="input-scales",
                                classes="config-control",
                            )

                        yield Static("", id="config-summary", markup=False)

                        with Collapsible(
                            title="完整命令（自动更新）",
                            collapsed=True,
                            id="command-details",
                        ):
                            yield Static("", id="command-preview", markup=False)
                            yield Static(
                                "测量窗口内暂停常规界面刷新，不读取正在写入的 CSV。",
                                id="science-note",
                                markup=False,
                            )

                    with VerticalScroll(id="advanced-form", classes="pane-scroll"):
                        yield Static("采集参数", classes="section-title")
                        with Grid(classes="form-grid"):
                            yield Label("Batch size")
                            yield Input(
                                value=str(self.initial_config.batch_size),
                                id="batch-size",
                                classes="config-control",
                            )
                            yield Label("Warmup / Repeat")
                            yield Input(
                                value=(
                                    f"{self.initial_config.warmup},"
                                    f"{self.initial_config.repeat}"
                                ),
                                placeholder="2,5",
                                id="warmup-repeat",
                                classes="config-control",
                            )

                            yield Label("窗口请求数")
                            yield Input(
                                value=str(self.initial_config.repeat_in_window),
                                placeholder="0 表示自动校准",
                                id="repeat-in-window",
                                classes="config-control",
                            )
                            yield Label("自动窗口秒数")
                            yield Input(
                                value=str(self.initial_config.repeat_window_seconds),
                                id="repeat-window-seconds",
                                classes="config-control",
                            )

                            yield Label("采样频率 Hz")
                            yield Input(
                                value=str(self.initial_config.sample_hz),
                                id="sample-hz",
                                classes="config-control",
                            )
                            yield Label("Idle 基线测量秒")
                            yield Input(
                                value=str(self.initial_config.idle_seconds),
                                id="idle-seconds",
                                classes="config-control",
                            )

                            yield Label("基线前冷却秒")
                            yield Input(
                                value=str(self.initial_config.idle_cooldown_seconds),
                                id="idle-cooldown-seconds",
                                classes="config-control",
                            )
                            yield Label("单请求超时秒")
                            yield Input(
                                value=str(self.initial_config.request_timeout_seconds),
                                id="request-timeout-seconds",
                                classes="config-control",
                            )

                            yield Label("计算分析器")
                            yield Select(
                                (
                                    ("关闭（先跑主矩阵）", "none"),
                                    ("Torch + NCU", "both"),
                                    ("仅 Torch", "torch"),
                                    ("仅 NCU", "ncu"),
                                ),
                                value=self.initial_config.compute_profile_tool,
                                allow_blank=False,
                                id="compute-profile-tool",
                                classes="config-control",
                            )
                            yield Label("执行分析器")
                            yield Select(
                                (
                                    ("关闭", "none"),
                                    ("Massif + Nsys", "both"),
                                    ("仅 Massif", "massif"),
                                    ("仅 Nsys", "nsys"),
                                ),
                                value=self.initial_config.execution_profile_tool,
                                allow_blank=False,
                                id="execution-profile-tool",
                                classes="config-control",
                            )

                            yield Label("抓包网卡")
                            yield Input(
                                value=self.initial_config.sniff_iface,
                                id="sniff-iface",
                                classes="config-control",
                            )
                            yield Label("通知")
                            yield Select(
                                (
                                    ("自动", "auto"),
                                    ("关闭", "none"),
                                    ("企业微信", "wecom"),
                                ),
                                value=self.initial_config.notify,
                                allow_blank=False,
                                id="notify",
                                classes="config-control",
                            )

                        yield Static("识别覆盖（通常留空）", classes="section-title")
                        with Grid(classes="form-grid"):
                            yield Label("Task")
                            yield Input(
                                value=self.initial_config.task,
                                placeholder="如 text-generation",
                                id="task",
                                classes="config-control",
                            )
                            yield Label("Task family")
                            yield Select(
                                (
                                    ("自动识别", ""),
                                    ("NLP", "nlp"),
                                    ("CV", "cv"),
                                    ("Audio", "audio"),
                                    ("Time series", "timeseries"),
                                    ("Diffusion", "diffusion"),
                                ),
                                value=self.initial_config.task_family,
                                allow_blank=False,
                                id="task-family",
                                classes="config-control",
                            )
                            yield Label("Backend")
                            yield Input(
                                value=self.initial_config.backend,
                                placeholder="留空自动识别",
                                id="backend",
                                classes="config-control",
                            )
                            yield Label("Workload manifest")
                            yield Input(
                                value=self.initial_config.workload_spec,
                                placeholder="音频 manifest，可留空",
                                id="workload-spec",
                                classes="config-control",
                            )

                        with Horizontal(classes="checkbox-row"):
                            yield StatusCheckbox(
                                "启动 OOM 剪枝",
                                value=self.initial_config.prune_startup_oom,
                                id="prune-startup-oom",
                                classes="config-control option-checkbox",
                            )
                            yield StatusCheckbox(
                                "复用现有镜像",
                                value=self.initial_config.skip_build,
                                id="skip-build",
                                classes="config-control option-checkbox",
                            )
                            yield StatusCheckbox(
                                "Idle 诊断",
                                value=self.initial_config.idle_debug,
                                id="idle-debug",
                                classes="config-control option-checkbox",
                            )
                            yield StatusCheckbox(
                                "允许 cgroup v1（仅诊断）",
                                value=self.initial_config.allow_cgroup_v1,
                                id="allow-cgroup-v1",
                                classes="config-control option-checkbox",
                            )


                        yield Static("下次启动使用的实验配置", classes="section-title")
                        yield Static(
                            "点击后记住当前实验表单。下次打开此项目自动填入，命令行指定的模型和预设优先。",
                            classes="page-hint", markup=False,
                        )
                        yield Static("", id="saved-run-summary", markup=False)
                        with Horizontal(classes="button-row"):
                            yield Button("记住实验配置", id="save-run-default")

                with Horizontal(id="run-actions", classes="action-bar"):
                    yield Button("高级参数", id="open-run-settings")
                    yield Static("", id="action-spacer")
                    yield Button("环境检查", id="quick-check")
                    yield Button("探测最大输入", id="probe-largest")
                    yield Button("开始采集", id="start-run", variant="primary")

            with TabPane("运行监控", id="monitor-tab"):
                with Vertical(classes="pane-scroll"):
                    with Grid(id="status-grid"):
                        yield Static("阶段", classes="status-label")
                        yield Static("等待", id="status-stage", markup=False)
                        yield Static("运行时间", classes="status-label")
                        yield Static("-", id="status-elapsed", markup=False)

                        yield Static("Case", classes="status-label")
                        yield Static("0/0", id="status-case", markup=False)
                        yield Static("资源", classes="status-label")
                        yield Static("CPU=-  MEM=-  GPU=-", id="status-resource", markup=False)

                        yield Static("警告 / 错误", classes="status-label")
                        yield Static("0 / 0", id="status-errors", markup=False)
                        yield Static("详情", classes="status-label")
                        yield Static("尚未启动", id="status-detail", markup=False)

                    yield ProgressBar(
                        total=1,
                        show_eta=False,
                        id="case-progress",
                    )
                    with Collapsible(
                        title="资源矩阵",
                        collapsed=True,
                        id="matrix-board",
                    ):
                        yield DataTable(
                            id="matrix-table",
                            show_cursor=False,
                        )
                    with Horizontal(classes="button-row"):
                        yield Button("终止当前任务", id="stop-run", variant="error", disabled=True)
                        yield Button("清空显示日志", id="clear-log")
                    yield RichLog(
                        id="run-log",
                        max_lines=self.ui_preferences.log_max_lines,
                        wrap=self.ui_preferences.log_wrap,
                        highlight=False,
                        markup=False,
                    )

            with TabPane("结果工具", id="results-tab"):
                with VerticalScroll(classes="pane-scroll"):
                    yield Static("已有结果", classes="section-title")
                    with Grid(classes="form-grid"):
                        yield Label("结果目录")
                        yield Input("", id="result-dir")
                        yield Label("结果 CSV")
                        yield Input("", id="result-csv")
                        yield Label("补采工具")
                        yield Input("torch,ncu", id="profile-tools")
                        yield Label("")
                        yield Static("")
                    with Horizontal(classes="button-row"):
                        yield Button("读取摘要", id="summarize-results")
                        yield Button("生成图表", id="plot-results", variant="primary")
                        yield Button("补采计划（dry-run）", id="profile-dry-run")
                        yield Button("执行补采", id="profile-run", variant="warning")
                    yield Static(
                        "选择或完成一次实验后，这里会显示结果摘要。",
                        id="result-summary",
                        markup=False,
                    )

            with TabPane("设置", id="settings-tab"):
                with VerticalScroll(classes="pane-scroll"):
                    yield Static("显示与日志", classes="section-title")
                    yield Static(
                        "修改立即生效，点击保存后下次启动沿用。",
                        classes="page-hint", markup=False,
                    )
                    with Grid(classes="form-grid"):
                        yield Label("界面主题")
                        yield Select(
                            THEME_OPTIONS,
                            value=self.ui_preferences.theme, allow_blank=False,
                            id="ui-theme", classes="ui-preference",
                        )
                        yield Label("保留日志行数")
                        yield Select(
                            ((str(n), n) for n in (500, 1000, 3000, 10000)),
                            value=self.ui_preferences.log_max_lines, allow_blank=False,
                            id="ui-log-lines", classes="ui-preference",
                        )
                    with Vertical(classes="settings-options"):
                        yield StatusCheckbox(
                            "新日志自动换行", value=self.ui_preferences.log_wrap,
                            id="ui-log-wrap", classes="ui-preference option-checkbox",
                        )
                        yield StatusCheckbox(
                            "显示底部快捷命令框",
                            value=self.ui_preferences.show_command_bar,
                            id="ui-command-bar", classes="ui-preference option-checkbox",
                        )
                    yield Static("", id="settings-location", classes="page-hint", markup=False)
                yield Static("", id="settings-status", markup=False)
                with Horizontal(id="settings-actions", classes="action-bar"):
                    yield Button("恢复界面默认", id="restore-ui-defaults")
                    yield Button("保存设置", id="save-ui-settings", variant="primary")

        with Vertical(id="bottom-panel"):
            with Horizontal(id="slash-command-bar"):
                yield Input(
                    placeholder="快捷命令：输入 /help 查看可用命令，按 Enter 执行",
                    id="slash-command",
                )
            yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        self._configure_interaction()
        self._apply_ui_preferences()
        self._update_saved_settings_summary()
        self._update_responsive_layout()
        self._form_ready = True
        self._refresh_command_preview(notify=False)
        table = self.query_one("#matrix-table", DataTable)
        table.add_column("Case", key="case")
        table.add_column("CPU", key="cpu")
        table.add_column("MEM (GB)", key="mem")
        table.add_column("GPU", key="gpu")
        table.add_column("状态", key="status")
        self.query_one("#model", Input).focus()
        if self._settings_warning:
            self.notify(self._settings_warning, title="设置读取提示", severity="warning", timeout=8)

    def on_resize(self) -> None:
        self._update_responsive_layout()

    def _update_responsive_layout(self) -> None:
        self.set_class(self.size.width < 110, "narrow")
        self.set_class(self.size.height < 35, "short")

    def _apply_ui_preferences(self) -> None:
        self.theme = self.ui_preferences.theme
        log = self.query_one("#run-log", RichLog)
        log.wrap = self.ui_preferences.log_wrap
        log.max_lines = self.ui_preferences.log_max_lines
        self.query_one("#bottom-panel").set_class(
            not self.ui_preferences.show_command_bar, "command-hidden",
        )

    def _update_saved_settings_summary(self) -> None:
        config = self._saved_settings.run_defaults
        summary = (
            f"已记住：{config.model or '模型待填写'} · CPU {config.cpus} · 内存 {config.mems} GB"
            if config else "尚未保存实验配置，启动时使用项目默认值。"
        )
        self.query_one("#saved-run-summary", Static).update(summary)
        self.query_one("#settings-location", Static).update(f"保存位置：{self.settings_path}")

    @on(Select.Changed, ".ui-preference")
    @on(Checkbox.Changed, ".ui-preference")
    def _ui_preference_changed(self) -> None:
        if not self._form_ready or self._is_busy():
            return
        preferences = UiPreferences(
            theme=self._select("ui-theme"),
            log_max_lines=int(self.query_one("#ui-log-lines", Select).value),
            log_wrap=self._checked("ui-log-wrap"),
            show_command_bar=self._checked("ui-command-bar"),
        )
        if preferences == self.ui_preferences:
            return
        self.ui_preferences = preferences
        self._apply_ui_preferences()
        self.query_one("#settings-status", Static).update("已应用 · 点击保存设置可在下次启动时沿用")

    @on(Button.Pressed, "#restore-ui-defaults")
    def restore_ui_defaults(self) -> None:
        if self._is_busy():
            return
        defaults = UiPreferences()
        self.query_one("#ui-theme", Select).value = defaults.theme
        self.query_one("#ui-log-lines", Select).value = defaults.log_max_lines
        self.query_one("#ui-log-wrap", Checkbox).value = defaults.log_wrap
        self.query_one("#ui-command-bar", Checkbox).value = defaults.show_command_bar
        self.ui_preferences = defaults
        self._apply_ui_preferences()
        self.query_one("#settings-status", Static).update("界面已恢复默认 · 点击保存设置可保留")

    def _save_settings(self, *, remember_run: bool) -> None:
        if self._is_busy():
            self.notify("任务完成后可保存设置", severity="warning")
            return
        try:
            config = (
                self._collect_config(allow_empty_model=True)
                if remember_run else self._saved_settings.run_defaults
            )
            settings = TuiSettings(ui=self.ui_preferences, run_defaults=config)
            save_settings(self.settings_path, settings, PROJECT_DIR)
        except (OSError, ValueError, TuiConfigError) as exc:
            self.notify(str(exc), title="设置未保存", severity="error")
            self.query_one("#settings-status", Static).update("保存失败 · 请检查配置或文件权限")
            return
        self._saved_settings = settings
        self._update_saved_settings_summary()
        message = "已保存界面设置和当前实验配置" if remember_run else "界面设置已保存"
        self.query_one("#settings-status", Static).update(message)
        self.notify(message, timeout=3)

    @on(Button.Pressed, "#save-ui-settings")
    def save_ui_settings(self) -> None:
        self._save_settings(remember_run=False)

    @on(Button.Pressed, "#save-run-default")
    def save_run_default(self) -> None:
        self._save_settings(remember_run=True)

    def action_show_settings(self) -> None:
        self._activate_tab("settings-tab")

    @on(Button.Pressed, "#open-run-settings")
    def open_run_settings(self) -> None:
        self._activate_tab("run-tab")
        pages = self.query_one("#experiment-pages", ContentSwitcher)
        show_advanced = pages.current != "advanced-form"
        pages.current = "advanced-form" if show_advanced else "run-form"
        self.query_one("#open-run-settings", Button).label = (
            "返回基本配置" if show_advanced else "高级参数"
        )

    @staticmethod
    def _infer_preset(config: RunConfig) -> str:
        model = config.model
        if config == RunConfig.smoke(model):
            return "smoke"
        if config == RunConfig.main_matrix(model):
            return "main"
        if config == RunConfig(model=model):
            return "default"
        return "custom"

    def _gpu_options(self) -> list[tuple[str, str]]:
        options = [("仅 CPU", "off"), ("仅 GPU", "on"), ("CPU + GPU", "off,on")]
        if self.initial_config.gpus not in {value for _, value in options}:
            options.append((f"自定义：{self.initial_config.gpus}", self.initial_config.gpus))
        return options

    @staticmethod
    def _matches_preset(config: RunConfig, preset: str) -> bool:
        if preset == "smoke":
            return config == RunConfig.smoke(config.model)
        if preset == "main":
            return config == RunConfig.main_matrix(config.model)
        if preset == "default":
            return config == RunConfig(model=config.model)
        return preset == "custom"

    def _input(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _select(self, widget_id: str) -> str:
        value = self.query_one(f"#{widget_id}", Select).value
        return "" if value is Select.NULL else str(value)

    def _checked(self, widget_id: str) -> bool:
        return bool(self.query_one(f"#{widget_id}", Checkbox).value)

    def _configure_interaction(self) -> None:
        # Textual ignores another click while a button's active effect lasts
        # (200 ms by default). Use focus/hover styling for immediate feedback.
        for button in self.query(Button):
            button.active_effect_duration = 0
        for field in self.query(Input):
            field.cursor_blink = False

    def _cancel_preview_timer(self) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer = None

    @on(Input.Changed, ".config-control")
    @on(Select.Changed, ".config-control")
    @on(Checkbox.Changed, ".config-control")
    def _configuration_changed(self) -> None:
        if not self._form_ready or self._applying_config or self._is_busy():
            return
        self._cancel_preview_timer()
        # Coalesce typing and preset field updates into one validation/render.
        self._preview_timer = self.set_timer(0.05, self._sync_form_state)

    def _sync_form_state(self) -> None:
        self._preview_timer = None
        if self._is_busy() or self._applying_config:
            return
        self._refresh_command_preview(notify=False, sync_preset=True)

    @on(Select.Changed, "#run-preset")
    def _run_preset_changed(self, event: Select.Changed) -> None:
        if not self._form_ready or self._applying_config or self._is_busy():
            return
        preset = str(event.value)
        if self._ignored_preset_event == preset:
            self._ignored_preset_event = None
            return
        if preset == "smoke":
            self.preset_smoke()
        elif preset == "main":
            self.preset_main()
        elif preset == "default":
            self.preset_default()

    @staticmethod
    def _pair(value: str, label: str) -> tuple[str, str]:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2 or not all(parts):
            raise TuiConfigError([f"{label}必须填写两个逗号分隔的值"])
        return parts[0], parts[1]

    def _collect_config(self, *, allow_empty_model: bool = False) -> RunConfig:
        warmup, repeat = self._pair(self._input("warmup-repeat"), "Warmup / Repeat")
        config = RunConfig(
            model=self._input("model"),
            task=self._input("task"),
            task_family=self._select("task-family"),
            backend=self._input("backend"),
            cpus=self._input("cpus"),
            mems=self._input("mems"),
            gpus=self._select("gpus"),
            input_scales=self._input("input-scales"),
            workload_spec=self._input("workload-spec"),
            output_dir=self._input("output-dir"),
            batch_size=self._input("batch-size"),  # normalized by RunConfig
            warmup=warmup,
            repeat=repeat,
            repeat_in_window=self._input("repeat-in-window"),
            repeat_window_seconds=self._input("repeat-window-seconds"),
            request_timeout_seconds=self._input("request-timeout-seconds"),
            sample_hz=self._input("sample-hz"),
            idle_seconds=self._input("idle-seconds"),
            idle_cooldown_seconds=self._input("idle-cooldown-seconds"),
            compute_profile_tool=self._select("compute-profile-tool"),
            execution_profile_tool=self._select("execution-profile-tool"),
            sniff_iface=self._input("sniff-iface"),
            notify=self._select("notify"),
            prune_startup_oom=self._checked("prune-startup-oom"),
            skip_build=self._checked("skip-build"),
            idle_debug=self._checked("idle-debug"),
            allow_cgroup_v1=self._checked("allow-cgroup-v1"),
        )
        if allow_empty_model and not config.model:
            validated = replace(config, model="settings/default-model").validate(
                project_dir=PROJECT_DIR,
            )
            return replace(validated, model="")
        return config.validate(project_dir=PROJECT_DIR)

    def _apply_config(self, config: RunConfig, *, preset: str = "custom") -> None:
        self._cancel_preview_timer()
        self._applying_config = True
        values = {
            "model": config.model,
            "task": config.task,
            "backend": config.backend,
            "cpus": config.cpus,
            "mems": config.mems,
            "input-scales": config.input_scales,
            "workload-spec": config.workload_spec,
            "output-dir": config.output_dir,
            "batch-size": str(config.batch_size),
            "warmup-repeat": f"{config.warmup},{config.repeat}",
            "repeat-in-window": str(config.repeat_in_window),
            "repeat-window-seconds": str(config.repeat_window_seconds),
            "request-timeout-seconds": str(config.request_timeout_seconds),
            "sample-hz": str(config.sample_hz),
            "idle-seconds": str(config.idle_seconds),
            "idle-cooldown-seconds": str(config.idle_cooldown_seconds),
            "sniff-iface": config.sniff_iface,
        }
        # Value watchers post Changed messages asynchronously. Suppressing
        # them here avoids dozens of queued debounce timers after a preset.
        try:
            with self.prevent(Input.Changed, Select.Changed, Checkbox.Changed):
                with self.batch_update():
                    for widget_id, value in values.items():
                        self.query_one(f"#{widget_id}", Input).value = value
                    for widget_id, value in {
                        "task-family": config.task_family,
                        "gpus": config.gpus,
                        "compute-profile-tool": config.compute_profile_tool,
                        "execution-profile-tool": config.execution_profile_tool,
                        "notify": config.notify,
                    }.items():
                        self.query_one(f"#{widget_id}", Select).value = value
                    for widget_id, value in {
                        "prune-startup-oom": config.prune_startup_oom,
                        "skip-build": config.skip_build,
                        "idle-debug": config.idle_debug,
                        "allow-cgroup-v1": config.allow_cgroup_v1,
                    }.items():
                        self.query_one(f"#{widget_id}", Checkbox).value = value
                    self.query_one("#run-preset", Select).value = preset
                    self._ignored_preset_event = None
        finally:
            self._applying_config = False
        if self._form_ready:
            self._refresh_command_preview(notify=False)

    def _show_config_error(self, exc: TuiConfigError) -> None:
        message = "\n".join(f"• {error}" for error in exc.errors)
        self.notify(message, title="配置有误", severity="error", timeout=8)

    def _refresh_command_preview(
        self, *, notify: bool = True, sync_preset: bool = False
    ) -> bool:
        config = None
        try:
            config = self._collect_config()
            command = build_run_command(
                config,
                project_dir=PROJECT_DIR,
                python_executable=PYTHON_EXECUTABLE,
            )
        except TuiConfigError as exc:
            self.query_one("#config-summary", Static).update(
                "配置待完善 · " + "；".join(exc.errors[:2])
            )
            self.query_one("#command-preview", Static).update(
                "配置尚未完成：" + "；".join(exc.errors)
            )
            if notify:
                self._show_config_error(exc)
            return False
        finally:
            if sync_preset:
                selected_preset = self._select("run-preset")
                if selected_preset != "custom" and (
                    config is None or not self._matches_preset(config, selected_preset)
                ):
                    self.query_one("#run-preset", Select).value = "custom"
        case_count = (
            len(config.cpus.split(","))
            * len(config.mems.split(","))
            * len(config.gpus.split(","))
        )
        scale_summary = (
            f"{len(config.input_scales.split(','))} 档"
            if config.input_scales
            else "自动规划"
        )
        profiler_summary = (
            "分析器关闭"
            if config.compute_profile_tool == "none"
            and config.execution_profile_tool == "none"
            else (
                f"计算={config.compute_profile_tool} · "
                f"执行={config.execution_profile_tool}"
            )
        )
        self.query_one("#config-summary", Static).update(
            f"{case_count} 个资源 case · 输入规模 {scale_summary} · "
            f"单请求超时 {config.request_timeout_seconds:g}s · "
            f"{profiler_summary} · 输出 {config.output_dir}"
        )
        self.query_one("#command-preview", Static).update(
            format_command(command, project_dir=PROJECT_DIR)
        )
        if notify:
            self.notify("命令预览已更新", timeout=2)
        return True

    def _is_busy(self) -> bool:
        with self._process_lock:
            return self._process is not None or bool(self._process_kind)

    def _set_busy(self, busy: bool) -> None:
        # Configuration changes during a run can queue preview redraws and
        # make the visible settings differ from the running subprocess.
        if busy:
            self._cancel_preview_timer()
        for widget in self.query(
            ".config-control, #run-preset, .ui-preference, "
            "#save-run-default, #restore-ui-defaults, #save-ui-settings"
        ):
            widget.disabled = busy
        for selector in (
            "#start-run",
            "#probe-largest",
            "#quick-check",
            "#plot-results",
            "#profile-dry-run",
            "#profile-run",
        ):
            self.query_one(selector, Button).disabled = busy
        self.query_one("#stop-run", Button).disabled = not busy

    def _activate_tab(self, tab_id: str) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Clear the outgoing field's focus before hiding its pane. Otherwise
        # Textual may restore that focus and activate the old pane again.
        tabs.query_one(Tabs).focus()
        tabs.active = tab_id

    def preset_smoke(self) -> None:
        self._apply_config(RunConfig.smoke(self._input("model")), preset="smoke")
        self.notify("已应用最小 Smoke 预设", timeout=3)

    def preset_main(self) -> None:
        self._apply_config(
            RunConfig.main_matrix(self._input("model")),
            preset="main",
        )
        self.notify("已应用主矩阵预设（分析器关闭）", timeout=3)

    def preset_default(self) -> None:
        self._apply_config(
            RunConfig(model=self._input("model")),
            preset="default",
        )
        self.notify("已恢复完整默认配置", timeout=3)

    @on(Button.Pressed, "#start-run")
    def start_run_button(self) -> None:
        self.action_request_run()

    @on(Button.Pressed, "#probe-largest")
    def probe_largest_button(self) -> None:
        self.action_request_probe()

    def action_request_probe(self) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        try:
            config = self._collect_config()
            command = build_probe_command(
                config,
                project_dir=PROJECT_DIR,
                python_executable=PYTHON_EXECUTABLE,
            )
        except TuiConfigError as exc:
            self._show_config_error(exc)
            return

        cpu = min(int(value) for value in config.cpus.split(","))
        memory_candidates = sorted(
            set(int(value) for value in config.mems.split(","))
        )
        gpu_modes = config.gpus.split(",")
        gpu = "off" if "off" in gpu_modes else "on"
        largest_scale = (
            max(float(value) for value in config.input_scales.split(","))
            if config.input_scales
            else None
        )
        scale_text = f"{largest_scale:g}" if largest_scale is not None else "自动规划后的最大值"
        memory_text = ",".join(
            f"{value}GB" for value in memory_candidates
        )
        preview = format_command(command, project_dir=PROJECT_DIR)
        self._pending_launch = PendingLaunch(tuple(command), "probe", config)
        self.push_screen(
            ConfirmActionScreen(
                "探测最低配置的最大输入？",
                f"资源：CPU={cpu}、GPU={gpu}\n"
                f"内存候选：{memory_text}（从小到大）\n"
                f"输入规模：{scale_text}\n\n"
                "每档使用全新容器并最多执行一次最大输入请求；OOM 时自动尝试下一档，"
                "第一个成功值就是最低可用内存。结果单独写入 "
                "probes/，不会写入或修改正式实验 CSV。最大输入请求不设超时，"
                "可用 /stop 手动终止。\n\n"
                + preview,
                "开始探测",
            ),
            self._confirmed_launch,
        )

    def action_request_run(self) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        try:
            config = self._collect_config()
            command = build_run_command(
                config,
                project_dir=PROJECT_DIR,
                python_executable=PYTHON_EXECUTABLE,
            )
        except TuiConfigError as exc:
            self._show_config_error(exc)
            return
        preview = format_command(command, project_dir=PROJECT_DIR)
        self.query_one("#command-preview", Static).update(preview)
        self._pending_launch = PendingLaunch(tuple(command), "run", config)
        self.push_screen(
            ConfirmActionScreen(
                "开始 AC-Prof 采集？",
                "将启动独立采集进程。正式测量窗口内 TUI 会停止常规日志刷新。\n\n"
                + preview,
                "开始采集",
            ),
            self._confirmed_launch,
        )

    def _confirmed_launch(self, confirmed: bool | None) -> None:
        pending = self._pending_launch
        self._pending_launch = None
        if not confirmed or pending is None:
            return
        self._launch(pending)

    def _launch(self, pending: PendingLaunch) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        self._active_run_config = pending.config if pending.kind == "run" else None
        self._active_command = pending.command
        self._process_kind = pending.kind
        self._started_monotonic = time.monotonic()
        self._stop_requested = False
        self._latest_snapshot = ProgressSnapshot(stage="启动中", detail="正在创建子进程")
        if pending.kind == "run" and pending.config is not None:
            result_dir = pending.config.result_dir(PROJECT_DIR)
            result_csv = pending.config.result_csv(PROJECT_DIR)
            self.query_one("#result-dir", Input).value = str(result_dir)
            self.query_one("#result-csv", Input).value = str(result_csv)
        self._set_busy(True)
        self._activate_tab("monitor-tab")
        # Move focus away from Input widgets so their caret-blink timers do
        # not cause redraws while a scientific measurement is active.
        self.query_one("#stop-run", Button).focus()
        # Start the elapsed-time ticker; it self-gates on measurement_active
        # to avoid any redraws during formal energy/latency windows.
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()
        self._elapsed_timer = self.set_interval(1.0, self._tick_elapsed)
        # Populate the resource matrix board for run tasks.
        if pending.kind == "run" and pending.config is not None:
            self._init_matrix_for_run(pending.config)
        else:
            self._clear_matrix()
        log = self.query_one("#run-log", RichLog)
        log.write(f"$ {format_command(pending.command, project_dir=PROJECT_DIR)}")
        log.write("[TUI] 子进程输出通过管道读取；tmux pane 捕获已对该子进程禁用。")
        self._render_snapshot(self._latest_snapshot)
        self._execute_command(list(pending.command), pending.kind)

    def _tick_elapsed(self) -> None:
        """Update elapsed time display; skipped during measurement windows."""
        if self._latest_snapshot.measurement_active:
            return  # Zero redraws during RAPL/latency measurement windows.
        if not self._started_monotonic or not self._is_busy():
            return
        elapsed = self._format_elapsed(time.monotonic() - self._started_monotonic)
        self.query_one("#status-elapsed", Static).update(elapsed)

    def _init_matrix_for_run(self, config: RunConfig) -> None:
        """Pre-populate the resource matrix board from the run configuration."""
        table = self.query_one("#matrix-table", DataTable)
        table.clear()
        self._matrix_rows.clear()
        # run.py iterates CPU → MEM → GPU (innermost).
        cpus = config.cpus.split(",")
        mems = config.mems.split(",")
        gpus = config.gpus.split(",")
        case_num = 0
        for cpu in cpus:
            for mem in mems:
                for gpu in gpus:
                    case_num += 1
                    key = table.add_row(
                        str(case_num), cpu.strip(), mem.strip(),
                        gpu.strip(), "⋯ 等待",
                    )
                    self._matrix_rows[case_num] = key
        self.query_one("#matrix-board", Collapsible).collapsed = False

    def _clear_matrix(self) -> None:
        """Clear the matrix board for non-run tasks."""
        self.query_one("#matrix-table", DataTable).clear()
        self._matrix_rows.clear()

    _STAGE_CSS_CLASS = {
        "等待": "stage-idle",
        "已完成": "stage-success",
        "探测完成": "stage-success",
        "找到最低可用内存": "stage-success",
        "case 完成": "stage-success",
        "失败": "stage-error",
        "探测失败": "stage-error",
        "已终止": "stage-error",
        "正式测量": "stage-measuring",
        "最大尺度探测": "stage-measuring",
    }
    _STAGE_CLASSES = frozenset({
        "stage-idle", "stage-running", "stage-measuring",
        "stage-success", "stage-error",
    })

    _MATRIX_STATUS = {
        "启动容器": "▶ 准备中",
        "构建镜像": "🔧 构建",
        "规划输入": "📐 规划",
        "服务就绪": "▶ 就绪",
        "正式测量": "⏱ 测量中",
        "清理 case": "⏳ 清理",
        "case 完成": "✓ 完成",
        "OOM 剪枝": "⊘ 剪枝",
        "失败": "✗ 失败",
    }

    @work(thread=True, group="process", exclusive=True, exit_on_error=False)
    def _execute_command(self, command: list[str], kind: str) -> None:
        tracker = RunProgressTracker() if kind in {"run", "probe"} else None
        suppressed_lines = 0
        deferred_important_lines: list[str] = []
        process: subprocess.Popen[str] | None = None
        launch_error = ""
        returncode = 1
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["MPLBACKEND"] = "Agg"
            child_env["ACPROF_TUI"] = "1"
            # run.py otherwise captures the entire full-screen pane, including
            # ANSI redraws, when the TUI itself is launched inside tmux.
            child_env.pop("TMUX", None)
            child_env.pop("TMUX_PANE", None)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=(os.name == "posix"),
            )
            with self._process_lock:
                self._process = process
            self.call_from_thread(self._process_started, process.pid, kind)
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                before = tracker.snapshot if tracker is not None else None
                snapshot = tracker.feed(line) if tracker is not None else None
                state_changed = snapshot != before if snapshot is not None else False
                important = (
                    "[ERROR]" in line
                    or "[WARN]" in line
                    or line.startswith("Traceback")
                )
                if (
                    before is not None
                    and before.measurement_active
                    and snapshot is not None
                    and snapshot.measurement_active
                ):
                    if important:
                        deferred_important_lines.append(line)
                    else:
                        suppressed_lines += 1
                    continue
                if (
                    before is not None
                    and before.measurement_active
                    and snapshot is not None
                    and not snapshot.measurement_active
                ):
                    if deferred_important_lines:
                        self.call_from_thread(
                            self._show_deferred_lines,
                            tuple(deferred_important_lines),
                        )
                        deferred_important_lines.clear()
                    if suppressed_lines:
                        self.call_from_thread(
                            self._show_suppressed_count,
                            suppressed_lines,
                        )
                        suppressed_lines = 0
                self.call_from_thread(
                    self._consume_process_line,
                    line,
                    snapshot,
                    state_changed,
                )
            returncode = process.wait()
            process.stdout.close()
        except Exception as exc:  # process errors must become visible in the UI
            launch_error = f"{type(exc).__name__}: {exc}"
            if process is not None and process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:  # pragma: no cover
                        process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            if suppressed_lines:
                self.call_from_thread(self._show_suppressed_count, suppressed_lines)
            if deferred_important_lines:
                self.call_from_thread(
                    self._show_deferred_lines,
                    tuple(deferred_important_lines),
                )
            final_snapshot = tracker.snapshot if tracker is not None else None
            with self._process_lock:
                if self._process is process:
                    self._process = None
            self.call_from_thread(
                self._process_finished,
                kind,
                returncode,
                final_snapshot,
                launch_error,
            )

    def _process_started(self, pid: int, kind: str) -> None:
        self.query_one("#run-log", RichLog).write(
            f"[TUI] {kind} 进程已启动，PID={pid}"
        )

    def _show_suppressed_count(self, count: int) -> None:
        self.query_one("#run-log", RichLog).write(
            f"[TUI] 为降低测量干扰，本窗口隐藏了 {count} 行常规输出。"
        )

    def _show_deferred_lines(self, lines: tuple[str, ...]) -> None:
        log = self.query_one("#run-log", RichLog)
        log.write("[TUI] 测量窗口结束，显示期间延迟刷新的重要消息：")
        for line in lines:
            log.write(line)

    def _write_log(self, line: str) -> None:
        self.query_one("#run-log", RichLog).write(line)

    def _consume_process_line(
        self,
        line: str,
        snapshot: ProgressSnapshot | None,
        state_changed: bool,
    ) -> None:
        if line:
            self.query_one("#run-log", RichLog).write(line)
        if snapshot is not None:
            was_measuring = self._latest_snapshot.measurement_active
            self._latest_snapshot = snapshot
            if self._elapsed_timer is not None:
                if snapshot.measurement_active:
                    self._elapsed_timer.pause()
                elif was_measuring:
                    self._elapsed_timer.resume()
            if state_changed:
                self._render_snapshot(snapshot)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _render_snapshot(self, snapshot: ProgressSnapshot) -> None:
        elapsed = (
            self._format_elapsed(time.monotonic() - self._started_monotonic)
            if self._started_monotonic
            else "-"
        )
        # Stage text with visual category coloring.
        stage_widget = self.query_one("#status-stage", Static)
        stage_widget.update(snapshot.stage)
        stage_widget.set_classes(
            self._STAGE_CSS_CLASS.get(snapshot.stage, "stage-running")
        )
        self.query_one("#status-elapsed", Static).update(elapsed)
        self.query_one("#status-case", Static).update(
            f"当前 {snapshot.current_case or '-'} · "
            f"已完成 {snapshot.completed_cases}/{snapshot.total_cases}"
        )
        self.query_one("#status-resource", Static).update(
            f"CPU={snapshot.cpu}  MEM={snapshot.mem}GB  GPU={snapshot.gpu}"
        )
        self.query_one("#status-errors", Static).update(
            f"{snapshot.warnings} / {snapshot.errors}"
        )
        self.query_one("#status-detail", Static).update(snapshot.detail)
        total = max(1, snapshot.total_cases)
        self.query_one("#case-progress", ProgressBar).update(
            total=total,
            progress=min(snapshot.completed_cases, total),
        )
        # Update the resource matrix board.
        self._update_matrix_status(snapshot)

    def _update_matrix_status(self, snapshot: ProgressSnapshot) -> None:
        """Update the matrix board row for the current case."""
        case_num = snapshot.current_case
        if case_num <= 0 or not self._matrix_rows:
            return
        table = self.query_one("#matrix-table", DataTable)
        row_key = self._matrix_rows.get(case_num)
        if row_key is None:
            return
        # Correct resource columns with actual values from the log.
        if snapshot.cpu != "-":
            table.update_cell(row_key, "cpu", snapshot.cpu)
            table.update_cell(row_key, "mem", snapshot.mem)
            table.update_cell(row_key, "gpu", snapshot.gpu)
        # Update status column.
        status = self._MATRIX_STATUS.get(snapshot.stage)
        if status:
            table.update_cell(row_key, "status", status)

    def _process_finished(
        self,
        kind: str,
        returncode: int,
        snapshot: ProgressSnapshot | None,
        launch_error: str,
    ) -> None:
        # Stop the elapsed-time ticker.
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()
            self._elapsed_timer = None
        # Show the final elapsed time.
        if self._started_monotonic:
            final_elapsed = self._format_elapsed(
                time.monotonic() - self._started_monotonic
            )
            self.query_one("#status-elapsed", Static).update(final_elapsed)
        self._set_busy(False)
        log = self.query_one("#run-log", RichLog)
        if launch_error:
            log.write(f"[TUI][ERROR] 无法运行命令：{launch_error}")
            self.notify(launch_error, title="任务启动失败", severity="error", timeout=8)
        elif returncode == 0:
            log.write(f"[TUI] {kind} 任务完成，退出码 0")
            self.notify("任务已完成", severity="information", timeout=5)
        elif self._stop_requested:
            log.write(f"[TUI] 任务已由用户终止，退出码 {returncode}")
            self.notify("任务已终止；部分 case 结果可能仍可续跑", severity="warning", timeout=7)
        else:
            log.write(f"[TUI][ERROR] {kind} 任务失败，退出码 {returncode}")
            self.notify(f"任务失败，退出码 {returncode}", severity="error", timeout=8)

        if kind == "run":
            if snapshot is not None:
                self._latest_snapshot = snapshot
            final_csv = snapshot.final_csv if snapshot is not None else ""
            if not final_csv and self._active_run_config is not None:
                final_csv = str(self._active_run_config.result_csv(PROJECT_DIR))
            if final_csv:
                final_path = Path(final_csv)
                if not final_path.is_absolute():
                    final_path = PROJECT_DIR / final_path
                self.query_one("#result-csv", Input).value = str(final_path)
                self.query_one("#result-dir", Input).value = str(final_path.parent)
                if final_path.is_file():
                    self._update_result_summary(str(final_path), notify=False)
            final_state = self._latest_snapshot
            if launch_error or (returncode != 0 and not self._stop_requested):
                final_state = replace(
                    final_state,
                    stage="失败",
                    detail=launch_error or f"采集进程退出码 {returncode}",
                    measurement_active=False,
                )
            elif self._stop_requested:
                final_state = replace(
                    final_state,
                    stage="已终止",
                    detail="用户请求终止；可使用相同输出目录续跑",
                    measurement_active=False,
                )
            self._latest_snapshot = final_state
            self._render_snapshot(final_state)
        elif kind == "probe":
            final_state = snapshot or self._latest_snapshot
            if launch_error or (returncode != 0 and not self._stop_requested):
                detail = launch_error or (
                    final_state.detail
                    if final_state.stage == "探测失败"
                    else f"探测进程退出码 {returncode}"
                )
                final_state = replace(
                    final_state,
                    stage="探测失败",
                    detail=detail,
                    measurement_active=False,
                )
            elif self._stop_requested:
                final_state = replace(
                    final_state,
                    stage="已终止",
                    detail="用户终止了最大尺度探测",
                    measurement_active=False,
                )
            elif final_state.stage != "探测完成":
                final_state = replace(
                    final_state,
                    stage="探测完成",
                    detail="最大尺度单次请求已完成",
                    measurement_active=False,
                )
            self._latest_snapshot = final_state
            self._render_snapshot(final_state)
        else:
            if launch_error or (returncode != 0 and not self._stop_requested):
                stage = "失败"
                detail = launch_error or f"{kind} 进程退出码 {returncode}"
            elif self._stop_requested:
                stage = "已终止"
                detail = f"用户终止了 {kind} 任务"
            else:
                stage = "已完成"
                detail = f"{kind} 任务已完成"
            self._latest_snapshot = ProgressSnapshot(stage=stage, detail=detail)
            self._render_snapshot(self._latest_snapshot)

        self._active_command = ()
        self._process_kind = ""
        self._stop_requested = False

    @on(Button.Pressed, "#stop-run")
    def stop_button(self) -> None:
        self.action_request_stop()

    def action_request_stop(self) -> None:
        if not self._is_busy():
            self.notify("当前没有运行中的任务", severity="warning")
            return
        self.push_screen(
            ConfirmActionScreen(
                "终止当前任务？",
                "将先向整个采集进程组发送 SIGINT，允许容器和监控器清理；"
                "超时后才会升级为 SIGTERM。已写入的 case 结果不会删除。",
                "终止任务",
            ),
            self._confirmed_stop,
        )

    def _confirmed_stop(self, confirmed: bool | None) -> None:
        if not confirmed:
            return
        self._stop_requested = True
        self._stop_process_gracefully()

    @work(thread=True, group="stop", exclusive=True, exit_on_error=False)
    def _stop_process_gracefully(self) -> None:
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        self.call_from_thread(
            self._write_log,
            "[TUI] 正在请求采集进程安全停止……",
        )
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGINT)
            else:  # pragma: no cover - formal collection is Linux only
                process.send_signal(signal.SIGINT)
            # run_single_case may need to stop tcpdump, monitors, and Docker;
            # allow that cleanup to finish before escalating the signal.
            process.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            return
        self.call_from_thread(
            self._write_log,
            "[TUI] SIGINT 超时，升级为 SIGTERM。",
        )
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover
                process.terminate()
        except ProcessLookupError:
            return

    @on(Button.Pressed, "#quick-check")
    def quick_check_button(self) -> None:
        self.action_quick_check()

    def action_quick_check(self) -> None:
        if self._is_busy() or self._check_running:
            self.notify("请等待当前任务完成", severity="warning")
            return
        # Host diagnostics do not require a model ID or a complete resource
        # matrix, so they remain usable on a freshly configured machine.
        config = RunConfig(
            model="preflight-only",
            gpus=self._select("gpus"),
            sniff_iface=self._input("sniff-iface"),
            allow_cgroup_v1=self._checked("allow-cgroup-v1"),
        )
        self._check_running = True
        self.query_one("#quick-check", Button).disabled = True
        self._activate_tab("monitor-tab")
        self.query_one("#run-log", RichLog).write("[TUI] 开始只读快速环境检查……")
        self._execute_quick_check(config)

    @work(thread=True, group="preflight", exclusive=True, exit_on_error=False)
    def _execute_quick_check(self, config: RunConfig) -> None:
        try:
            checks = quick_preflight(config)
            error = ""
        except Exception as exc:
            checks = []
            error = f"{type(exc).__name__}: {exc}"
        self.call_from_thread(self._show_quick_check, checks, error)

    def _show_quick_check(
        self,
        checks: Sequence[PreflightCheck],
        error: str,
    ) -> None:
        self._check_running = False
        if not self._is_busy():
            self.query_one("#quick-check", Button).disabled = False
        log = self.query_one("#run-log", RichLog)
        if error:
            log.write(f"[TUI][ERROR] 环境检查失败：{error}")
            self.notify(error, severity="error")
            return
        # Render a structured Rich table instead of plain text lines.
        from rich.table import Table as RichTable
        from rich.text import Text as RichText
        table = RichTable(
            title="环境检查结果",
            show_header=True,
            title_style="bold",
            border_style="dim",
        )
        table.add_column("检查项", style="bold", min_width=16)
        table.add_column("状态", justify="center", min_width=4)
        table.add_column("详情")
        status_style = {"ok": "green", "warn": "yellow", "fail": "red bold"}
        status_symbol = {"ok": "✓", "warn": "!", "fail": "✗"}
        for check in checks:
            style = status_style.get(check.status, "")
            symbol = status_symbol.get(check.status, "?")
            table.add_row(
                check.label,
                RichText(symbol, style=style),
                check.detail,
            )
        log.write(table)
        failures = sum(check.status == "fail" for check in checks)
        warnings = sum(check.status == "warn" for check in checks)
        log.write(
            f"[TUI] 快速检查完成：{len(checks) - failures - warnings} 通过，"
            f"{warnings} 警告，{failures} 失败。run.py 启动时仍会执行权威预检。"
        )
        severity = "error" if failures else ("warning" if warnings else "information")
        self.notify(
            f"环境检查：{failures} 失败，{warnings} 警告",
            severity=severity,
            timeout=6,
        )

    @on(Button.Pressed, "#clear-log")
    def clear_log_button(self) -> None:
        self.action_clear_log()

    def action_clear_log(self) -> None:
        self.query_one("#run-log", RichLog).clear()

    @on(Button.Pressed, "#summarize-results")
    def summarize_results_button(self) -> None:
        self._update_result_summary(self._input("result-csv"))

    def _update_result_summary(self, result_csv: str, *, notify: bool = True) -> None:
        if not result_csv:
            if notify:
                self.notify("请填写结果 CSV 路径", severity="warning")
            return
        try:
            summary = summarize_result_csv(result_csv)
        except (OSError, csv.Error, UnicodeError) as exc:
            self.query_one("#result-summary", Static).update(f"无法读取结果：{exc}")
            if notify:
                self.notify(str(exc), severity="error")
            return
        latency_info = ""
        if summary.avg_latency_s is not None:
            min_ms = summary.min_latency_s * 1000 if summary.min_latency_s is not None else 0
            max_ms = summary.max_latency_s * 1000 if summary.max_latency_s is not None else 0
            avg_ms = summary.avg_latency_s * 1000
            latency_info = (
                f"\n应用延迟（均值）：{avg_ms:.1f}ms "
                f"（范围 {min_ms:.1f}ms ~ {max_ms:.1f}ms）"
            )
        self.query_one("#result-summary", Static).update(
            "结果已读取\n"
            f"行数：{summary.rows}（成功 {summary.ok_rows} / 错误 {summary.error_rows}）\n"
            f"资源 case：{summary.cases}\n"
            f"Warmup 行：{summary.warmup_rows}（正常绘图会排除）"
            f"{latency_info}"
        )
        if notify:
            self.notify("结果摘要已更新", timeout=3)

    @on(Button.Pressed, "#plot-results")
    def plot_results_button(self) -> None:
        self._launch_plot()

    def _launch_plot(self, path: str | None = None) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        result_csv = path or self._input("result-csv")
        if not result_csv:
            self.notify("请填写结果 CSV 路径", severity="warning")
            return
        csv_path = Path(result_csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = PROJECT_DIR / csv_path
        if not csv_path.is_file():
            self.notify(f"结果 CSV 不存在：{csv_path}", severity="error")
            return
        command = build_plot_command(
            csv_path,
            project_dir=PROJECT_DIR,
            python_executable=PYTHON_EXECUTABLE,
        )
        self._launch(PendingLaunch(tuple(command), "plot"))

    @on(Button.Pressed, "#profile-dry-run")
    def profile_dry_run_button(self) -> None:
        self._launch_profile(dry_run=True)

    @on(Button.Pressed, "#profile-run")
    def profile_run_button(self) -> None:
        self._request_profile_run()

    def _profile_command(
        self,
        *,
        dry_run: bool,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> list[str] | None:
        directory = result_dir or self._input("result-dir")
        selected_tools = tools or self._input("profile-tools")
        if not directory:
            self.notify("请填写结果目录", severity="warning")
            return None
        result_path = Path(directory).expanduser()
        if not result_path.is_absolute():
            result_path = PROJECT_DIR / result_path
        if not result_path.is_dir():
            self.notify(f"结果目录不存在：{result_path}", severity="error")
            return None
        try:
            return build_profile_command(
                result_path,
                tools=selected_tools,
                dry_run=dry_run,
                project_dir=PROJECT_DIR,
                python_executable=PYTHON_EXECUTABLE,
            )
        except TuiConfigError as exc:
            self._show_config_error(exc)
            return None

    def _launch_profile(
        self,
        *,
        dry_run: bool,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        command = self._profile_command(
            dry_run=dry_run,
            result_dir=result_dir,
            tools=tools,
        )
        if command is not None:
            self._launch(PendingLaunch(tuple(command), "profile-dry-run" if dry_run else "profile"))

    def _request_profile_run(
        self,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        command = self._profile_command(
            dry_run=False,
            result_dir=result_dir,
            tools=tools,
        )
        if command is None:
            return
        self._pending_launch = PendingLaunch(tuple(command), "profile")
        self.push_screen(
            ConfirmActionScreen(
                "执行 profiler 补采？",
                "该操作会启动隔离 profiler，并在成功后原子回填现有结果。"
                "原文件会按项目规则备份。\n\n"
                + format_command(command, project_dir=PROJECT_DIR),
                "执行补采",
            ),
            self._confirmed_launch,
        )

    @on(Input.Submitted, "#slash-command")
    def slash_command_submitted(self, event: Input.Submitted) -> None:
        value = event.value
        event.input.value = ""
        try:
            command, args = parse_slash_command(value)
        except TuiConfigError as exc:
            self._show_config_error(exc)
            return

        if command == "run":
            self.action_request_run()
        elif command == "probe":
            self.action_request_probe()
        elif command == "check":
            self.action_quick_check()
        elif command in {"stop", "cancel"}:
            self.action_request_stop()
        elif command == "status":
            snapshot = self._latest_snapshot
            self.query_one("#run-log", RichLog).write(
                f"[TUI] status={snapshot.stage}; "
                f"case={snapshot.completed_cases}/{snapshot.total_cases}; "
                f"resource=CPU {snapshot.cpu}, MEM {snapshot.mem}GB, GPU {snapshot.gpu}; "
                f"warnings={snapshot.warnings}; errors={snapshot.errors}"
            )
            self._activate_tab("monitor-tab")
        elif command == "smoke":
            self.preset_smoke()
        elif command == "main":
            self.preset_main()
        elif command in {"matrix", "board"}:
            board = self.query_one("#matrix-board", Collapsible)
            board.collapsed = not board.collapsed
            self._activate_tab("monitor-tab")
        elif command in {"defaults", "default"}:
            self.preset_default()
        elif command == "preview":
            self._refresh_command_preview()
            self._activate_tab("run-tab")
        elif command == "plot":
            self._launch_plot(args[0] if args else None)
        elif command == "profile":
            self._launch_profile(
                dry_run=True,
                result_dir=args[0] if args else None,
                tools=args[1] if len(args) > 1 else None,
            )
        elif command in {"profile-run", "profile!"}:
            self._request_profile_run(
                result_dir=args[0] if args else None,
                tools=args[1] if len(args) > 1 else None,
            )
        elif command in {"results", "summary"}:
            path = args[0] if args else self._input("result-csv")
            if args:
                self.query_one("#result-csv", Input).value = path
            self._update_result_summary(path)
            self._activate_tab("results-tab")
        elif command == "clear":
            self.action_clear_log()
        elif command == "settings":
            self.action_show_settings()
        elif command == "help":
            self.query_one("#run-log", RichLog).write(
                "[TUI] /run 采集 · /probe 最大输入探测 · /check 环境检查 · "
                "/status 状态 · /stop 终止 · "
                "/smoke 最小预设 · /main 主矩阵 · /defaults 默认 · /preview 命令预览 · "
                "/matrix 切换矩阵看板 · /plot [csv] 绘图 · /profile [dir] [tools] 补采计划 · "
                "/profile-run [dir] [tools] 执行补采 · /results [csv] 摘要 · "
                "/settings 设置 · /clear 清日志 · /quit 退出"
            )
            self._activate_tab("monitor-tab")
        elif command in {"quit", "exit"}:
            self.action_request_quit()
        else:
            self.notify(f"未知快捷命令：/{command}", severity="error")

    def action_request_quit(self) -> None:
        if self._is_busy():
            self.notify("任务仍在运行，请先使用 /stop 安全终止", severity="warning", timeout=6)
            return
        self.exit()

    def on_unmount(self) -> None:
        """Best-effort guard against leaving collectors behind on normal exit."""
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover
                process.terminate()
        except ProcessLookupError:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AC-Prof interactive terminal interface",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Pre-fill the Hugging Face model ID",
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

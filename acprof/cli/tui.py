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
    from textual.widgets import (
        Button,
        Checkbox,
        Collapsible,
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
    build_profile_command,
    build_run_command,
    format_command,
    parse_slash_command,
    quick_preflight,
    summarize_result_csv,
)


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

    @on(Button.Pressed, "#confirm-no")
    def cancel(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class AcprofTui(App[None]):
    """Full-screen controller that delegates all experiments to run.py."""

    TITLE = "AC-Prof"
    SUB_TITLE = "低干扰实验控制台"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        ("f5", "request_run", "开始采集"),
        ("f6", "quick_check", "环境检查"),
        ("ctrl+x", "request_stop", "终止任务"),
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+q", "request_quit", "退出"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    HeaderIcon {
        display: none;
    }

    HeaderClockSpace {
        display: none;
    }

    #main-tabs {
        height: 1fr;
    }

    .pane-scroll {
        padding: 1 2;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0;
    }

    .form-grid {
        layout: grid;
        grid-size: 4;
        grid-columns: 16 1fr 18 1fr;
        grid-gutter: 1 1;
        height: auto;
    }

    .form-grid Label {
        height: 3;
        content-align: right middle;
        padding-right: 1;
    }

    .form-grid Input, .form-grid Select {
        width: 1fr;
    }

    .checkbox-row {
        height: auto;
        margin: 1 0;
    }

    .option-checkbox {
        margin-right: 2;
        color: $text-muted;
        background: $surface;
        border: tall $panel;
    }

    .option-checkbox > .toggle--button {
        color: $panel;
        background: $panel;
    }

    .option-checkbox.-on {
        color: $text;
        background: $success 45%;
        border: tall $success;
        text-style: bold;
    }

    .option-checkbox.-on > .toggle--button {
        color: $text;
        background: $success;
        text-style: bold;
    }

    .option-checkbox.-on > .toggle--label {
        color: $text;
        background: transparent;
        text-style: bold;
    }

    .option-checkbox:focus {
        border: tall $accent;
        background-tint: $accent 5%;
    }

    .option-checkbox:focus > .toggle--label {
        color: $text;
        background: transparent;
    }

    .button-row {
        height: auto;
        margin: 1 0;
    }

    .button-row Button {
        margin-right: 1;
    }

    .primary-actions {
        align-horizontal: right;
    }

    #config-summary {
        height: auto;
        border-left: thick $accent;
        padding: 0 1;
        margin: 1 0;
    }

    #preset-hint {
        height: 3;
        color: $text-muted;
        content-align: left middle;
    }

    #advanced-settings, #command-details {
        height: auto;
        margin-top: 1;
    }

    .advanced-body {
        height: auto;
        padding: 0 1;
    }

    #command-preview {
        height: auto;
        min-height: 3;
        max-height: 8;
        border: round $panel;
        padding: 0 1;
        overflow-y: auto;
    }

    #science-note {
        color: $text-muted;
        height: auto;
        margin: 1 0;
    }

    #status-grid {
        layout: grid;
        grid-size: 4;
        grid-columns: 14 1fr 14 1fr;
        grid-gutter: 0 1;
        height: auto;
        border: round $panel;
        padding: 1;
        margin-bottom: 1;
    }

    #status-grid .status-label {
        color: $text-muted;
        content-align: right middle;
        padding-right: 1;
    }

    #case-progress {
        margin-bottom: 1;
    }

    #run-log {
        height: 1fr;
        border: round $panel;
        padding: 0 1;
    }

    #result-summary {
        height: auto;
        min-height: 5;
        border: round $panel;
        padding: 1;
        margin-top: 1;
    }

    #slash-command {
        dock: bottom;
        height: 3;
        border-top: solid $panel;
    }
    """

    def __init__(self, initial_config: RunConfig | None = None):
        super().__init__()
        self.initial_config = initial_config or RunConfig()
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

    def compose(self) -> ComposeResult:
        # A ticking clock would force periodic redraws during RAPL windows.
        yield Header(show_clock=False)
        with TabbedContent(initial="run-tab", id="main-tabs"):
            with TabPane("实验配置", id="run-tab"):
                with VerticalScroll(classes="pane-scroll"):
                    yield Static("快速配置", classes="section-title")
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
                        yield Label("")
                        yield Static("预设会自动填充其余参数", id="preset-hint", markup=False)

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
                            (("仅 CPU", "off"), ("仅 GPU", "on"), ("CPU + GPU", "off,on")),
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
                        title="高级设置",
                        collapsed=True,
                        id="advanced-settings",
                    ):
                        with Vertical(classes="advanced-body"):
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
                                yield Label("")
                                yield Static("")

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

                    with Horizontal(classes="button-row primary-actions"):
                        yield Button("快速环境检查", id="quick-check", variant="primary")
                        yield Button("开始采集", id="start-run", variant="success")

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
                    with Horizontal(classes="button-row"):
                        yield Button("终止当前任务", id="stop-run", variant="error", disabled=True)
                        yield Button("清空显示日志", id="clear-log")
                    yield RichLog(
                        id="run-log",
                        max_lines=3000,
                        wrap=True,
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

        yield Input(
            placeholder="快捷命令：/run /check /status /stop /plot /profile /help",
            id="slash-command",
        )
        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        self._form_ready = True
        self._refresh_command_preview(notify=False)
        self.query_one("#model", Input).focus()

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

    @on(Input.Changed, ".config-control")
    @on(Select.Changed, ".config-control")
    @on(Checkbox.Changed, ".config-control")
    def _configuration_changed(self) -> None:
        if not self._form_ready or self._applying_config or self._is_busy():
            return
        if self._preview_timer is not None:
            self._preview_timer.stop()
        # Coalesce typing and preset field updates into one validation/render.
        self._preview_timer = self.set_timer(0.05, self._sync_form_state)

    def _sync_form_state(self) -> None:
        self._preview_timer = None
        try:
            config = self._collect_config()
        except TuiConfigError:
            config = None
        selected_preset = self._select("run-preset")
        if (
            selected_preset != "custom"
            and (config is None or not self._matches_preset(config, selected_preset))
        ):
            self.query_one("#run-preset", Select).value = "custom"
        self._refresh_command_preview(notify=False)

    @on(Select.Changed, "#run-preset")
    def _run_preset_changed(self, event: Select.Changed) -> None:
        if not self._form_ready or self._applying_config:
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

    def _collect_config(self) -> RunConfig:
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
        return config.validate(project_dir=PROJECT_DIR)

    def _apply_config(self, config: RunConfig, *, preset: str = "custom") -> None:
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
            "sample-hz": str(config.sample_hz),
            "idle-seconds": str(config.idle_seconds),
            "idle-cooldown-seconds": str(config.idle_cooldown_seconds),
            "sniff-iface": config.sniff_iface,
        }
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
        preset_widget = self.query_one("#run-preset", Select)
        if str(preset_widget.value) != preset:
            self._ignored_preset_event = preset
            preset_widget.value = preset
        self._applying_config = False
        if self._form_ready:
            self._refresh_command_preview(notify=False)

    def _show_config_error(self, exc: TuiConfigError) -> None:
        message = "\n".join(f"• {error}" for error in exc.errors)
        self.notify(message, title="配置有误", severity="error", timeout=8)

    def _refresh_command_preview(self, *, notify: bool = True) -> bool:
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
        for selector in (
            "#start-run",
            "#quick-check",
            "#plot-results",
            "#profile-dry-run",
            "#profile-run",
        ):
            self.query_one(selector, Button).disabled = busy
        self.query_one("#stop-run", Button).disabled = not busy

    def _activate_tab(self, tab_id: str) -> None:
        self.query_one("#main-tabs", TabbedContent).active = tab_id

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
        log = self.query_one("#run-log", RichLog)
        log.write(f"$ {format_command(pending.command, project_dir=PROJECT_DIR)}")
        log.write("[TUI] 子进程输出通过管道读取；tmux pane 捕获已对该子进程禁用。")
        self._render_snapshot(self._latest_snapshot)
        self._execute_command(list(pending.command), pending.kind)

    @work(thread=True, group="process", exclusive=True, exit_on_error=False)
    def _execute_command(self, command: list[str], kind: str) -> None:
        tracker = RunProgressTracker() if kind == "run" else None
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
            self._latest_snapshot = snapshot
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
        self.query_one("#status-stage", Static).update(snapshot.stage)
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

    def _process_finished(
        self,
        kind: str,
        returncode: int,
        snapshot: ProgressSnapshot | None,
        launch_error: str,
    ) -> None:
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
        symbols = {"ok": "✓", "warn": "!", "fail": "✗"}
        for check in checks:
            log.write(f"[{symbols.get(check.status, '?')}] {check.label}: {check.detail}")
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
        self.query_one("#result-summary", Static).update(
            "结果已读取\n"
            f"行数：{summary.rows}（成功 {summary.ok_rows} / 错误 {summary.error_rows}）\n"
            f"资源 case：{summary.cases}\n"
            f"Warmup 行：{summary.warmup_rows}（正常绘图会排除）"
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
        elif command in {"main", "matrix"}:
            self.preset_main()
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
        elif command == "help":
            self.query_one("#run-log", RichLog).write(
                "[TUI] /run 采集 · /check 环境检查 · /status 状态 · /stop 终止 · "
                "/smoke 最小预设 · /main 主矩阵 · /defaults 默认 · /preview 命令预览 · "
                "/plot [csv] 绘图 · /profile [dir] [tools] 补采计划 · "
                "/profile-run [dir] [tools] 执行补采 · /results [csv] 摘要 · "
                "/clear 清日志 · /quit 退出"
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
        default="",
        help="Pre-fill the Hugging Face model ID",
    )
    parser.add_argument(
        "--preset",
        choices=("default", "smoke", "main"),
        default="default",
        help="Initial form preset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.preset == "smoke":
        config = RunConfig.smoke(args.model)
    elif args.preset == "main":
        config = RunConfig.main_matrix(args.model)
    else:
        config = RunConfig(model=args.model)
    AcprofTui(config).run()


if __name__ == "__main__":
    main()

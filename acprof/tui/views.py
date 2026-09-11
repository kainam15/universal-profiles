"""保留既有控件树的 TUI 页面构建和独立控件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    ContentSwitcher,
    DataTable,
    Label,
    ProgressBar,
    Static,
    TabPane,
)

from acprof.tui.i18n import LANGUAGE_OPTIONS

from acprof.tui.input import BarCursorInput as Input

from acprof.tui.log import SelectableLog

from acprof.tui.themes import THEME_OPTIONS

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


# Keep the original triangles with one extra cell before the title text.
COLLAPSED_SYMBOL = "▶ "
EXPANDED_SYMBOL = "▼ "


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
        tr = self.app.tr
        with Vertical(id="confirm-dialog"):
            yield Static(tr(self.dialog_title), id="confirm-title", markup=False)
            yield Static(tr(self.message), id="confirm-message", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button(tr("取消"), id="confirm-no")
                yield Button(
                    tr(self.confirm_label),
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


class LogPanel(Vertical):
    """Keep the log controls available when the log fills the terminal."""

    ALLOW_MAXIMIZE = True


def compose_run_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("实验配置", id="run-tab"):
        with ContentSwitcher(initial="run-form", id="experiment-pages"):
            with VerticalScroll(id="run-form", classes="pane-scroll"):
                yield app._localized_widget(Static("配置实验", classes="section-title"))
                with Grid(classes="form-grid"):
                    yield app._localized_widget(Label("模型 ID"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.model,
                        placeholder="google-bert/bert-base-uncased",
                        id="model",
                        classes="config-control",
                        tooltip="确认启动采集或探测后自动记住，下次打开时填入。",
                    ))
                    yield app._localized_widget(Label("输出目录"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.output_dir,
                        id="output-dir",
                        classes="config-control",
                    ))

                    yield app._localized_widget(Label("运行预设"))
                    yield app._localized_select(
                        (
                            ("自定义", "custom"),
                            ("最小 Smoke", "smoke"),
                            ("主矩阵（分析器关闭）", "main"),
                            ("完整默认", "default"),
                        ),
                        value=app._initial_preset,
                        allow_blank=False,
                        id="run-preset",
                    )
                    yield app._localized_widget(Label("", id="preset-hint-label"))
                    yield app._localized_widget(Static("预设自动填充 · 下方可打开高级参数", id="preset-hint", markup=False))

                    yield app._localized_widget(Label("CPU 列表"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.cpus,
                        placeholder="1,2,4,8",
                        id="cpus",
                        classes="config-control",
                    ))
                    yield app._localized_widget(Label("内存 GB"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.mems,
                        placeholder="2,4,8,16",
                        id="mems",
                        classes="config-control",
                    ))

                    yield app._localized_widget(Label("GPU 模式"))
                    yield app._localized_select(
                        app._gpu_options(),
                        value=app.initial_config.gpus,
                        allow_blank=False,
                        id="gpus",
                        classes="config-control",
                    )
                    yield app._localized_widget(Label("输入规模"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.input_scales,
                        placeholder="留空自动规划；如 64,128,256",
                        id="input-scales",
                        classes="config-control",
                    ))

                yield app._localized_widget(Static("", id="config-summary", markup=False))

                with app._localized_widget(Collapsible(
                    title="完整命令（自动更新）",
                    collapsed=True,
                    collapsed_symbol=COLLAPSED_SYMBOL,
                    expanded_symbol=EXPANDED_SYMBOL,
                    id="command-details",
                )):
                    yield app._localized_widget(Static("", id="command-preview", markup=False))
                    yield app._localized_widget(Static(
                        "测量窗口内暂停常规界面刷新，不读取正在写入的 CSV。",
                        id="science-note",
                        markup=False,
                    ))

            with VerticalScroll(id="advanced-form", classes="pane-scroll"):
                yield app._localized_widget(Static("采集参数", classes="section-title"))
                with Grid(classes="form-grid"):
                    yield app._localized_widget(Label("Batch size"))
                    yield app._localized_widget(Input(
                        value=str(app.initial_config.batch_size),
                        id="batch-size",
                        classes="config-control",
                    ))
                    yield app._localized_widget(Label("Warmup / Repeat"))
                    yield app._localized_widget(Input(
                        value=(
                            f"{app.initial_config.warmup},"
                            f"{app.initial_config.repeat}"
                        ),
                        placeholder="2,5",
                        id="warmup-repeat",
                        classes="config-control",
                    ))

                    yield app._localized_widget(Label("窗口请求数"))
                    yield app._localized_widget(Input(
                        value=str(app.initial_config.repeat_in_window),
                        placeholder="0 表示自动校准",
                        id="repeat-in-window",
                        classes="config-control",
                    ))
                    yield app._localized_widget(Label("自动窗口秒数"))
                    yield app._localized_widget(Input(
                        value=str(app.initial_config.repeat_window_seconds),
                        id="repeat-window-seconds",
                        classes="config-control",
                    ))

                    yield app._localized_widget(Label("采样频率 Hz"))
                    yield app._localized_widget(Input(
                        value=str(app.initial_config.sample_hz),
                        id="sample-hz",
                        classes="config-control",
                    ))
                    yield app._localized_widget(Label("Idle 基线测量秒"))
                    yield app._localized_widget(Input(
                        value=str(app.initial_config.idle_seconds),
                        id="idle-seconds",
                        classes="config-control",
                    ))

                    yield app._localized_widget(Label("基线前冷却秒"))
                    yield app._localized_widget(Input(
                        value=str(app.initial_config.idle_cooldown_seconds),
                        id="idle-cooldown-seconds",
                        classes="config-control",
                    ))
                    yield app._localized_widget(Label("单请求超时秒"))
                    yield app._localized_widget(Input(
                        value=str(app.initial_config.request_timeout_seconds),
                        id="request-timeout-seconds",
                        classes="config-control",
                    ))

                    yield app._localized_widget(Label("计算分析器"))
                    yield app._localized_select(
                        (
                            ("关闭（先跑主矩阵）", "none"),
                            ("Torch + NCU", "both"),
                            ("仅 Torch", "torch"),
                            ("仅 NCU", "ncu"),
                        ),
                        value=app.initial_config.compute_profile_tool,
                        allow_blank=False,
                        id="compute-profile-tool",
                        classes="config-control",
                    )
                    yield app._localized_widget(Label("执行分析器"))
                    yield app._localized_select(
                        (
                            ("关闭", "none"),
                            ("Massif + Nsys", "both"),
                            ("仅 Massif", "massif"),
                            ("仅 Nsys", "nsys"),
                        ),
                        value=app.initial_config.execution_profile_tool,
                        allow_blank=False,
                        id="execution-profile-tool",
                        classes="config-control",
                    )

                    yield app._localized_widget(Label("抓包网卡"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.sniff_iface,
                        id="sniff-iface",
                        classes="config-control",
                    ))
                    yield app._localized_widget(Label("通知"))
                    yield app._localized_select(
                        (
                            ("自动", "auto"),
                            ("关闭", "none"),
                            ("企业微信", "wecom"),
                        ),
                        value=app.initial_config.notify,
                        allow_blank=False,
                        id="notify",
                        classes="config-control",
                    )

                yield app._localized_widget(Static("识别覆盖（通常留空）", classes="section-title"))
                with Grid(classes="form-grid"):
                    yield app._localized_widget(Label("Task"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.task,
                        placeholder="如 text-generation",
                        id="task",
                        classes="config-control",
                    ))
                    yield app._localized_widget(Label("Task family"))
                    yield app._localized_select(
                        (
                            ("自动识别", ""),
                            ("NLP", "nlp"),
                            ("CV", "cv"),
                            ("Audio", "audio"),
                            ("Time series", "timeseries"),
                            ("Diffusion", "diffusion"),
                            ("Multimodal", "multimodal"),
                            ("结构化数据／策略", "structured"),
                        ),
                        value=app.initial_config.task_family,
                        allow_blank=False,
                        id="task-family",
                        classes="config-control",
                    )
                    yield app._localized_widget(Label("Backend"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.backend,
                        placeholder="留空自动识别",
                        id="backend",
                        classes="config-control",
                    ))
                    yield app._localized_widget(Label("Workload manifest"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.workload_spec,
                        placeholder="输入素材 manifest，可留空",
                        id="workload-spec",
                        classes="config-control",
                    ))

                with Horizontal(classes="checkbox-row"):
                    yield app._localized_widget(StatusCheckbox(
                        "启动 OOM 剪枝",
                        value=app.initial_config.prune_startup_oom,
                        id="prune-startup-oom",
                        classes="config-control option-checkbox",
                    ))
                    yield app._localized_widget(StatusCheckbox(
                        "复用现有镜像",
                        value=app.initial_config.skip_build,
                        id="skip-build",
                        classes="config-control option-checkbox",
                        tooltip="优先复用本地模型镜像；未找到时提示并自动构建。",
                    ))
                    yield app._localized_widget(StatusCheckbox(
                        "Idle 诊断",
                        value=app.initial_config.idle_debug,
                        id="idle-debug",
                        classes="config-control option-checkbox",
                    ))
                    yield app._localized_widget(StatusCheckbox(
                        "允许 cgroup v1（仅诊断）",
                        value=app.initial_config.allow_cgroup_v1,
                        id="allow-cgroup-v1",
                        classes="config-control option-checkbox",
                    ))


                yield app._localized_widget(Static("下次启动使用的实验配置", classes="section-title"))
                yield app._localized_widget(Static(
                    "点击后记住当前实验表单。下次打开此项目自动填入，命令行指定的模型和预设优先。",
                    classes="page-hint", markup=False,
                ))
                yield app._localized_widget(Static("", id="saved-run-summary", markup=False))
                with Horizontal(classes="button-row"):
                    yield app._localized_widget(Button("记住实验配置", id="save-run-default"))

        with Horizontal(id="run-actions", classes="action-bar"):
            yield app._localized_widget(Button("高级参数", id="open-run-settings"))
            yield app._localized_widget(Static("", id="action-spacer"))
            yield app._localized_widget(Button("环境检查", id="quick-check"))
            yield app._localized_widget(Button("探测最大输入", id="probe-largest"))
            yield app._localized_widget(Button("开始采集", id="start-run", variant="primary"))


def compose_monitor_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("运行监控", id="monitor-tab"):
        with Vertical(classes="pane-scroll"):
            with Grid(id="status-grid"):
                yield app._localized_widget(Static("阶段", classes="status-label"))
                yield app._localized_widget(Static("等待", id="status-stage", markup=False))
                yield app._localized_widget(Static("运行时间", classes="status-label"))
                yield app._localized_widget(Static("-", id="status-elapsed", markup=False))

                yield app._localized_widget(Static("Case", classes="status-label"))
                yield app._localized_widget(Static("0/0", id="status-case", markup=False))
                yield app._localized_widget(Static("资源", classes="status-label"))
                yield app._localized_widget(Static("CPU=-  MEM=-  GPU=-", id="status-resource", markup=False))

                yield app._localized_widget(Static("警告 / 错误", classes="status-label"))
                yield app._localized_widget(Static("0 / 0", id="status-errors", markup=False))
                yield app._localized_widget(Static("详情", classes="status-label"))
                yield app._localized_widget(Static("尚未启动", id="status-detail", markup=False))

            yield ProgressBar(
                total=1,
                show_eta=False,
                id="case-progress",
            )
            with app._localized_widget(Collapsible(
                title="资源矩阵",
                collapsed=True,
                collapsed_symbol=COLLAPSED_SYMBOL,
                expanded_symbol=EXPANDED_SYMBOL,
                id="matrix-board",
            )):
                yield DataTable(
                    id="matrix-table",
                    show_cursor=False,
                )
            with LogPanel(id="log-panel"):
                with Horizontal(id="log-toolbar"):
                    yield app._localized_widget(Static("日志", id="log-title", markup=False))
                    yield app._localized_widget(Button("复制选区", id="copy-log", classes="log-tool"))
                    yield app._localized_widget(Button("回到最新", id="follow-log", classes="log-tool"))
                    yield app._localized_widget(Button("放大日志", id="expand-log", classes="log-tool"))
                    yield app._localized_widget(Button("返回监控", id="restore-log", classes="log-tool"))
                    yield app._localized_widget(Button("清空日志", id="clear-log", classes="log-tool"))
                    yield app._localized_widget(Button(
                        "终止任务", id="stop-run", classes="log-tool",
                        variant="error", disabled=True,
                    ))
                yield app._localized_widget(Static(
                    "拖动选择 · Ctrl+C 复制 · F8 放大 · 正在跟随最新",
                    id="log-hint", markup=False,
                ))
                yield SelectableLog(
                    id="run-log",
                    max_lines=app.ui_preferences.log_max_lines,
                    wrap=app.ui_preferences.log_wrap,
                )


def compose_results_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("结果工具", id="results-tab"):
        with VerticalScroll(classes="pane-scroll"):
            yield app._localized_widget(Static("已有结果", classes="section-title"))
            with Grid(classes="form-grid"):
                yield app._localized_widget(Label("结果目录"))
                yield app._localized_widget(Input(app._saved_settings.last_result_dir, id="result-dir"))
                yield app._localized_widget(Label("结果 CSV"))
                yield app._localized_widget(Input(app._saved_settings.last_result_csv, id="result-csv"))
                yield app._localized_widget(Label("补采工具"))
                with Grid(id="profile-tools"):
                    for tool, label, tooltip in (
                        ("torch", "torch · CPU / GPU", "逻辑 FLOP；点击或按空格切换。"),
                        ("ncu", "ncu · GPU", "GPU 实际执行 FLOP；点击或按空格切换。"),
                        ("nsys", "nsys · GPU", "CUDA API、kernel 和 memcpy 时间线；点击或按空格切换。"),
                        ("massif", "massif · CPU", "CPU-only 进程生命周期内存峰值；点击或按空格切换。"),
                    ):
                        yield app._localized_widget(StatusCheckbox(
                            label,
                            value=tool in {"torch", "ncu"},
                            name=tool,
                            id=f"profile-tool-{tool}",
                            classes="option-checkbox profile-tool",
                            tooltip=tooltip,
                        ))
            with Horizontal(classes="button-row"):
                yield app._localized_widget(Button("读取摘要", id="summarize-results"))
                yield app._localized_widget(Button("生成图表", id="plot-results", variant="primary"))
                yield app._localized_widget(Button("补采计划（dry-run）", id="profile-dry-run"))
                yield app._localized_widget(Button("执行补采", id="profile-run", variant="warning"))
            yield app._localized_widget(Static(
                "选择或完成一次实验后，这里会显示结果摘要。",
                id="result-summary",
                markup=False,
            ))


def compose_settings_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("设置", id="settings-tab"):
        with VerticalScroll(classes="pane-scroll"):
            yield app._localized_widget(Static("显示与日志", classes="section-title"))
            yield app._localized_widget(Static(
                "修改立即生效，点击保存后下次启动沿用。",
                classes="page-hint", markup=False,
            ))
            with Grid(classes="form-grid"):
                yield app._localized_widget(Label("界面语言"))
                yield app._localized_select(
                    LANGUAGE_OPTIONS,
                    value=app.ui_preferences.language, allow_blank=False,
                    id="ui-language", classes="ui-preference",
                )
                yield app._localized_widget(Label("界面主题"))
                yield app._localized_select(
                    THEME_OPTIONS,
                    value=app.ui_preferences.theme, allow_blank=False,
                    id="ui-theme", classes="ui-preference",
                )
                yield app._localized_widget(Label("保留日志行数"))
                yield app._localized_select(
                    ((str(n), n) for n in (500, 1000, 3000, 10000)),
                    value=app.ui_preferences.log_max_lines, allow_blank=False,
                    id="ui-log-lines", classes="ui-preference",
                )
            with Vertical(classes="settings-options"):
                yield app._localized_widget(StatusCheckbox(
                    "日志自动换行", value=app.ui_preferences.log_wrap,
                    id="ui-log-wrap", classes="ui-preference option-checkbox",
                ))
                yield app._localized_widget(StatusCheckbox(
                    "显示底部快捷命令框",
                    value=app.ui_preferences.show_command_bar,
                    id="ui-command-bar", classes="ui-preference option-checkbox",
                ))
            yield app._localized_widget(Static("", id="settings-location", classes="page-hint", markup=False))
        yield app._localized_widget(Static("", id="settings-status", markup=False))
        with Horizontal(id="settings-actions", classes="action-bar"):
            yield app._localized_widget(Button("恢复界面默认", id="restore-ui-defaults"))
            yield app._localized_widget(Button("保存设置", id="save-ui-settings", variant="primary"))

"""TUI 页面布局、公共控件与确认弹窗。"""

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
    Label,
    Rule,
    Static,
    TabPane,
)
from textual.widgets.button import ButtonVariant

from acprof.tui.i18n import LANGUAGE_OPTIONS

from acprof.tui.input import BarCursorInput as Input

from acprof.tui.log import SelectableLog
from acprof.tui.presentation import NOT_APPLICABLE, format_input_number
from acprof.tui.rendering import CjkCompositor
from acprof.tui.table import ResizableDataTable

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

    # 打开时不预选按钮；鼠标悬停与主动使用 Tab 聚焦仍保留高亮。
    AUTO_FOCUS = ""

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

    def __init__(
        self, title: str, message: str, confirm_label: str = "确认", *,
        variant: ButtonVariant = "primary",
    ):
        super().__init__()
        self._compositor = CjkCompositor()
        self.dialog_title = title
        self.message = message
        self.confirm_label = confirm_label
        self.confirm_variant = variant

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
                    variant=self.confirm_variant,
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


def compose_number_field(app: AcprofTui, label: str, widget_id: str, value: int | float | str,
                         unit: str, *, placeholder: str = "") -> ComposeResult:
    yield app._localized_widget(Label(label))
    with Horizontal(classes="number-field"):
        yield app._localized_widget(Input(
            value=format_input_number(value), id=widget_id, classes="config-control",
            placeholder=placeholder,
        ))
        yield app._localized_widget(Label(unit, classes="field-unit"))


def compose_run_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("实验配置", id="run-tab"):
        with Vertical(classes="page-header"):
            yield app._localized_widget(Static("配置实验", id="run-title", classes="page-title"))
            yield app._localized_widget(Static("", id="config-summary", classes="page-summary", markup=False))
        with ContentSwitcher(initial="run-form", id="experiment-pages"):
            with VerticalScroll(id="run-form", classes="pane-scroll"):
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

                    yield app._localized_widget(Label("模型契约"))
                    yield app._localized_widget(Button("解析与验证", id="inspect-model"))

                    yield app._localized_widget(Label("运行预设"))
                    yield app._localized_select(
                        (
                            ("自定义", "custom"),
                            ("基础 CPU Smoke", "smoke"),
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
                with Grid(classes="form-grid"):
                    for label, field, unit in (
                        ("Warmup", "warmup", "次"), ("Repeat", "repeat", "次"),
                        ("采样频率", "sample_hz", "Hz"), ("Idle 基线", "idle_seconds", "s"),
                        ("冷却时间", "idle_cooldown_seconds", "s"), ("请求超时", "request_timeout_seconds", "s"),
                        ("Batch size", "batch_size", ""), ("窗口请求数", "repeat_in_window", "次"),
                        ("自动窗口", "repeat_window_seconds", "s"),
                    ):
                        yield from compose_number_field(
                            app, label, field.replace("_", "-"), getattr(app.initial_config, field), unit,
                            placeholder="0 表示自动校准" if field == "repeat_in_window" else "",
                        )
                with Grid(classes="form-grid"):
                    yield app._localized_widget(Label("画像模式"))
                    yield app._localized_select(
                        (("完整（RAPL / perf / 抓包）", "full"), ("基础（延迟 / CPU / 内存）", "basic")),
                        value=app.initial_config.profiling_mode,
                        allow_blank=False, id="profiling-mode", classes="config-control",
                    )
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
                    yield app._localized_widget(Label("模型接口声明"))
                    yield app._localized_widget(Input(
                        value=app.initial_config.model_spec,
                        placeholder="模型接口 JSON，可留空",
                        id="model-spec",
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
                with Horizontal(classes="checkbox-row"):
                    yield app._localized_widget(StatusCheckbox(
                        "恢复未完成实验",
                        value=app.initial_config.resume,
                        id="resume-run",
                        classes="config-control option-checkbox",
                        tooltip="使用相同参数与输出目录，保留已完成 case，重新测量中断的 case。",
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
            with Horizontal(classes="action-secondary"):
                yield app._localized_widget(Button("高级参数", id="open-run-settings"))
                yield app._localized_widget(Button("环境检查", id="quick-check"))
                yield app._localized_widget(Button("探测最大输入", id="probe-largest"))
            with Horizontal(classes="action-primary"):
                yield app._localized_widget(Button("开始采集", id="start-run", variant="primary"))


def compose_monitor_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("运行监控", id="monitor-tab"):
        with Vertical(id="monitor-panel"):
            with Grid(id="status-grid", classes="page-header"):
                yield app._localized_widget(Static("阶段", classes="status-label"))
                yield app._localized_widget(Static("等待", id="status-stage", markup=False))
                yield app._localized_widget(Static("运行时间", classes="status-label"))
                yield app._localized_widget(Static(NOT_APPLICABLE, id="status-elapsed", markup=False))

                yield app._localized_widget(Static("Case", classes="status-label"))
                yield app._localized_widget(Static("0/0", id="status-case", markup=False))
                yield app._localized_widget(Static("资源", classes="status-label"))
                yield app._localized_widget(Static("CPU=—  MEM=—  GPU=—", id="status-resource", markup=False))

                yield app._localized_widget(Static("警告 / 错误", classes="status-label"))
                yield app._localized_widget(Static("0 / 0", id="status-errors", markup=False))
                yield app._localized_widget(Static("详情", classes="status-label"))
                yield app._localized_widget(Static("尚未启动", id="status-detail", markup=False))

            with LogPanel(id="log-panel"):
                with Horizontal(id="log-toolbar", classes="action-bar"):
                    with Horizontal(classes="action-secondary"):
                        yield app._localized_widget(Static("日志", id="log-title", markup=False))
                        yield app._localized_widget(Button("复制选区", id="copy-log", classes="log-tool"))
                        yield app._localized_widget(Button("回到最新", id="follow-log", classes="log-tool"))
                        yield app._localized_widget(Button("放大日志", id="expand-log", classes="log-tool"))
                        yield app._localized_widget(Button("返回监控", id="restore-log", classes="log-tool"))
                        yield app._localized_widget(Button("清空日志", id="clear-log", classes="log-tool"))
                    with Horizontal(classes="action-primary"):
                        yield app._localized_widget(Button(
                            "终止任务", id="stop-run", classes="log-tool",
                            variant="error", disabled=True,
                        ))
                yield SelectableLog(
                    id="run-log",
                    max_lines=app.ui_preferences.log_max_lines,
                    wrap=app.ui_preferences.log_wrap,
                )


def compose_plot_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("绘图工具", id="plot-tab"):
        with Vertical(classes="page-header"):
            yield app._localized_widget(Static("绘图工具", classes="page-title"))
            yield app._localized_widget(Static(
                "读取结果 CSV，查看摘要或生成图表。", classes="page-summary", markup=False,
            ))
        with VerticalScroll(id="plot-body", classes="pane-scroll"):
            with Grid(classes="form-grid tool-form-grid"):
                yield app._localized_widget(Label("结果 CSV"))
                yield app._localized_widget(Input(app._saved_settings.last_result_csv, id="result-csv"))
            yield app._localized_widget(Static(
                "选择或完成一次实验后，这里会显示结果摘要。",
                id="result-summary",
                markup=False,
            ))
        with Horizontal(id="plot-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield app._localized_widget(Button("读取摘要", id="summarize-results"))
            with Horizontal(classes="action-primary"):
                yield app._localized_widget(Button("生成图表", id="plot-results", variant="primary"))


def compose_reports_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("统计报告", id="reports-tab"):
        with Vertical(classes="page-header"):
            yield app._localized_widget(Static("统计报告", classes="page-title"))
            yield app._localized_widget(Static(
                "CSV / 目录：计算统计；JSON：查看报告。采集结束后操作。",
                id="report-status", classes="page-summary", markup=False,
            ))
        with Vertical(id="report-panel"):
            yield app._localized_widget(Input(
                app._saved_settings.last_result_csv, id="report-source", classes="report-control",
                placeholder="实验目录、结果 CSV 或报告 JSON 路径",
            ))
            yield Rule(classes="content-divider")
            yield app._localized_widget(ResizableDataTable(
                id="report-table", cursor_type="row", zebra_stripes=True, fixed_columns=1))
            yield app._localized_widget(Static(
                "表格可滚动；选择一行查看口径与数据来源。",
                id="report-detail", markup=False,
            ))
        with Horizontal(id="report-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield app._localized_widget(Button("当前结果", id="report-current", classes="report-control"))
                yield app._localized_widget(Button("查看报告", id="report-open", classes="report-control"))
            with Horizontal(classes="action-primary"):
                yield app._localized_widget(Button("计算统计", id="report-calculate", classes="report-control", variant="primary"))


def compose_profile_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("补采工具", id="profile-tab"):
        with Vertical(classes="page-header"):
            yield app._localized_widget(Static("已有结果补采", classes="page-title"))
            yield app._localized_widget(Static(
                "补采计划与执行日志会显示在“运行监控”页。",
                classes="page-summary", markup=False,
            ))
        with VerticalScroll(id="profile-body", classes="pane-scroll"):
            with Grid(classes="form-grid tool-form-grid"):
                yield app._localized_widget(Label("结果目录"))
                yield app._localized_widget(Input(app._saved_settings.last_result_dir, id="result-dir"))
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
        with Horizontal(id="profile-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield app._localized_widget(Button("补采计划（dry-run）", id="profile-dry-run"))
            with Horizontal(classes="action-primary"):
                yield app._localized_widget(Button("执行补采", id="profile-run", variant="warning"))


def compose_settings_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("设置", id="settings-tab"):
        with Vertical(classes="page-header"):
            yield app._localized_widget(Static("显示与日志", classes="page-title"))
            yield app._localized_widget(Static(
                "修改立即生效，点击保存后下次启动沿用。",
                id="settings-status", classes="page-summary", markup=False,
            ))
        with VerticalScroll(id="settings-body", classes="pane-scroll"):
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
        with Horizontal(id="settings-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield app._localized_widget(Button("恢复界面默认", id="restore-ui-defaults"))
            with Horizontal(classes="action-primary"):
                yield app._localized_widget(Button("保存设置", id="save-ui-settings", variant="primary"))


def compose_images_tab(app: AcprofTui) -> ComposeResult:
    from acprof.tui.images import (
        IMAGE_HINT, IMAGE_PLATFORMS, ImageDetailPanel, ImageDetailResizeHandle, ImageTable, ImageTree, ImageTreeHeader, ImageWorkspace,
    )

    with TabPane("镜像管理", id="images-tab"):
        with Vertical(id="image-header", classes="page-header"):
            yield app._localized_widget(Static("镜像管理", classes="page-title"))
            yield app._localized_widget(Static(IMAGE_HINT, id="image-status", classes="page-summary", markup=False))
        with Vertical(id="image-panel"):
            with Horizontal(id="image-filters"):
                yield app._localized_select(
                    (("AC-Prof 镜像", "acprof"), ("全部镜像", "all"), ("模型相关", "models"),
                     ("公共依赖", "runtime"), ("公共基础", "base"),
                     *((IMAGE_PLATFORMS[key], key) for key in ("cpu", "cu124", "cu128")),
                     ("无标签", "untagged")),
                    value="acprof", allow_blank=False, id="image-scope", classes="image-control",
                )
                yield app._localized_widget(Input(
                    placeholder="搜索模型、依赖、标签或镜像 ID", id="image-search", classes="image-control",
                ))
                for label, view in (("镜像树", "tree"), ("镜像列表", "list"), ("层共享", "layers")):
                    yield app._localized_widget(Button(
                        label, id="image-view-" + view, classes="image-control image-view-button",
                        variant="primary" if view == "tree" else "default",
                    ))
            with ImageWorkspace(id="image-workspace"):
                with ContentSwitcher(initial="image-tree-view", id="image-browser"):
                    with Vertical(id="image-tree-view"):
                        yield app._localized_widget(ImageTreeHeader(
                            id="image-tree-header", classes="image-control", show_cursor=False))
                        yield ImageTree("", id="image-tree", classes="image-control")
                    for table in (
                        ImageTable(id="image-table", classes="image-control", cursor_type="row",
                                   zebra_stripes=True, fixed_columns=1),
                        ResizableDataTable(id="image-layer-table", classes="image-control", cursor_type="row", zebra_stripes=True),
                    ):
                        yield app._localized_widget(table)
                yield app._localized_widget(ImageDetailResizeHandle(id="image-detail-resize", classes="image-control"))
                yield ImageDetailPanel(id="image-detail-scroll")
        with Horizontal(id="image-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                for label, widget_id in (("勾选/取消", "image-toggle"),
                                         ("选择同模型", "image-model"), ("清空选择", "image-clear")):
                    yield app._localized_widget(Button(label, id=widget_id, classes="image-control", disabled=True))
            with Horizontal(classes="action-primary"):
                yield app._localized_widget(Button(
                    "删除所选", id="image-delete", classes="image-control", variant="error", disabled=True,
                ))

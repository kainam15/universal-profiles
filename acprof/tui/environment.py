"""Project connections and explicit system-permission setup in the TUI."""
from __future__ import annotations

from pathlib import Path
import shlex
from uuid import uuid4

from textual import on, work
from textual.app import SuspendNotSupported
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Collapsible, Label, Static, TabbedContent, TabPane

from acprof.host.env_utils import configurable_env_values, save_project_env
from acprof.host.permissions import build_permission_plan, execute_permission_plan
from acprof.tui.commands import RunConfig
from acprof.tui.diagnostics import quick_preflight
from acprof.tui.i18n import message
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.rendering import CjkCompositor
from acprof.tui.views import ConfirmActionScreen


FIELDS = (
    ('HF_TOKEN', 'Hugging Face Token', True, 'hf_…'),
    ('HF_ENDPOINT', 'Hugging Face 镜像源', False, 'https://huggingface.co'),
    ('HF_FALLBACK_ENDPOINTS', '备用镜像源', False, 'https://huggingface.co'),
    ('HTTP_PROXY', 'HTTP_PROXY', True, 'http://127.0.0.1:7890'),
    ('HTTPS_PROXY', 'HTTPS_PROXY', True, 'http://127.0.0.1:7890'),
    ('ALL_PROXY', 'ALL_PROXY', True, 'socks5://127.0.0.1:7890'),
    ('NO_PROXY', 'NO_PROXY', False, 'localhost,127.0.0.1,::1'),
    ('ACPROF_WECOM_WEBHOOK_URL', '企业微信 Webhook', True, 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=…'),
)


def field_id(key: str) -> str:
    return 'env-' + key.lower().replace('_', '-')


class PermissionReviewScreen(ConfirmActionScreen):
    """Keep approval actions reachable while the complete command plan scrolls."""

    CSS = ConfirmActionScreen.CSS + """
    PermissionReviewScreen #confirm-dialog { height: 80%; }
    PermissionReviewScreen #confirm-message { height: 1fr; max-height: 100%; }
    """

    def compose(self):
        tr = self.app.tr
        with Vertical(id='confirm-dialog'):
            yield Static(tr(self.dialog_title), id='confirm-title', markup=False)
            with VerticalScroll(id='confirm-message'):
                yield Static(tr(self.message), markup=False)
            with Horizontal(id='confirm-buttons'):
                yield Button(tr('取消'), id='confirm-no')
                yield Button(tr(self.confirm_label), id='confirm-yes', variant='primary')


class EnvironmentSettingsScreen(ModalScreen[bool]):
    BINDINGS = [('escape', 'close', '关闭')]
    DEFAULT_CSS = """
    EnvironmentSettingsScreen { align: center middle; }
    #environment-dialog { width: 96%; max-width: 120; height: 94%; border: round $accent; background: $surface; }
    #environment-heading { height: auto; padding: 0 1; }
    #environment-title { height: 1; text-style: bold; }
    #environment-status { height: auto; max-height: 3; overflow-y: auto; }
    #environment-tabs { height: 1fr; }
    .environment-scroll { height: 1fr; padding: 0 1; }
    .environment-fields { grid-size: 2; grid-columns: 22 1fr; height: auto; }
    .environment-fields Label { height: 3; content-align: left middle; }
    .environment-hint { height: auto; color: $text-muted; margin-bottom: 1; }
    #permission-report { height: auto; padding: 1 0; }
    .permission-buttons { height: 3; }
    .permission-buttons Button { width: auto; margin-right: 1; }
    #environment-actions { padding: 0 1; }
    """

    def __init__(self, project_dir: Path, backup_dir: Path, *, sniff_iface: str = 'docker0'):
        super().__init__()
        self._compositor = CjkCompositor()
        self.project_dir = project_dir
        self.backup_dir = backup_dir
        self.sniff_iface = sniff_iface
        self.values = configurable_env_values(project_dir)
        path = project_dir / '.env.local'
        if path.is_symlink():
            raise ValueError('.env.local must be a regular file, not a symbolic link')
        self.original_text = path.read_text(encoding='utf-8') if path.exists() else ''
        self._working = False
        self.saved = False

    def _fields(self, fields):
        tr = self.app.tr
        with Grid(classes='environment-fields'):
            for key, label, secret, placeholder in fields:
                yield Label(tr(label))
                yield Input(value=self.values[key], password=secret, placeholder=placeholder,
                            id=field_id(key), classes='environment-input')

    def compose(self):
        tr = self.app.tr
        with Vertical(id='environment-dialog'):
            with Vertical(id='environment-heading'):
                yield Static(tr('连接与权限'), id='environment-title')
                yield Static(tr(message('保存到 {0}；保存后用于本次会话和后续启动。', self.project_dir / '.env.local')),
                             id='environment-status', markup=False)
            with TabbedContent(id='environment-tabs'):
                with TabPane(tr('连接配置'), id='connections-tab'):
                    with VerticalScroll(classes='environment-scroll'):
                        yield Static(tr('Token 留空可使用 hf login；镜像源留空使用官方 Hub。备用地址用逗号分隔。'),
                                     classes='environment-hint', markup=False)
                        yield from self._fields(FIELDS[:3])
                        with Collapsible(title=tr('代理设置'), collapsed=True):
                            yield Static(tr('留空关闭对应代理。NO_PROXY 决定哪些地址直连；本机推理通常需要 localhost、127.0.0.1。'),
                                         classes='environment-hint', markup=False)
                            yield from self._fields(FIELDS[3:7])
                        with Collapsible(title=tr('企业微信通知'), collapsed=True):
                            yield Static(tr('配置 Webhook 后采集默认发送通知；留空关闭。保存和检查不会发送消息。'),
                                         classes='environment-hint', markup=False)
                            yield from self._fields(FIELDS[7:])
                        yield Checkbox(tr('显示敏感字段'), id='show-environment-secrets')
                        yield Static(tr('保存时备份原文件并设置为仅当前用户可读写。重新启动时，显式导出的环境变量仍优先于文件。'),
                                     classes='environment-hint', markup=False)
                with TabPane(tr('采集权限'), id='permissions-tab'):
                    with VerticalScroll(classes='environment-scroll'):
                        yield Static(tr('full 采集需要 perf 与 tcpdump 权限。检查包括跨用户 PID 附加；实际容器仍在采集前单独验证。'),
                                     classes='environment-hint', markup=False)
                        yield Checkbox('perf · CAP_PERFMON', value=True, id='permission-perf')
                        yield Checkbox('tcpdump · CAP_NET_RAW', value=True, id='permission-tcpdump')
                        with Horizontal(classes='permission-buttons'):
                            yield Button(tr('检查环境'), id='check-environment-permissions')
                            yield Button(tr('配置所选权限'), id='configure-environment-permissions', variant='primary')
                        yield Static(tr('点击检查查看当前权限与主机条件。'), id='permission-report', markup=False)
                        yield Static(tr('配置时先展示变更，再由系统 sudo 提示输入管理员密码；完成后自动返回。密码不会进入配置文件。'),
                                     classes='environment-hint', markup=False)
            with Horizontal(id='environment-actions', classes='action-bar'):
                with Horizontal(classes='action-secondary'):
                    yield Button(tr('关闭'), id='close-environment-settings')
                with Horizontal(classes='action-primary'):
                    yield Button(tr('保存配置'), id='save-environment-settings', variant='primary')

    @on(Checkbox.Changed, '#show-environment-secrets')
    def show_secrets(self, event: Checkbox.Changed):
        for key, _, secret, _ in FIELDS:
            if secret:
                self.query_one('#' + field_id(key), Input).password = not event.value

    def _status(self, text, *, error=False):
        widget = self.query_one('#environment-status', Static)
        widget.update(self.app.tr(text))
        widget.set_class(error, 'stage-error')

    @on(Button.Pressed, '#save-environment-settings')
    def save(self):
        if self._working:
            return
        values = {key: self.query_one('#' + field_id(key), Input).value for key, *_ in FIELDS}
        try:
            path = save_project_env(self.project_dir, values, expected_text=self.original_text)
            self.original_text = path.read_text(encoding='utf-8')
        except (OSError, ValueError) as exc:
            self._status(message('配置保存失败：{0}', str(exc)), error=True)
            return
        self.saved = True
        self._status(message('已保存到 {0}；新启动的任务使用这些配置。', path))

    def _set_working(self, working):
        self._working = working
        for widget in self.query('Button, Input, Checkbox'):
            widget.disabled = working

    @on(Button.Pressed, '#check-environment-permissions')
    def check_permissions(self):
        if self._working:
            return
        self._set_working(True)
        self.query_one('#permission-report', Static).update(self.app.tr('正在检查环境……'))
        self._check_permissions()

    @work(thread=True, exclusive=True, group='environment-check', exit_on_error=False)
    def _check_permissions(self):
        try:
            checks = quick_preflight(RunConfig(profiling_mode='full', gpus='off', sniff_iface=self.sniff_iface),
                                     project_dir=self.project_dir)
            report = '\n\n'.join(f'[{item.status}] {self.app.tr(item.label)}: {self.app.tr(item.detail)}'
                                 for item in checks)
            failures = sum(item.status == 'fail' for item in checks)
        except Exception as exc:
            report = f'{type(exc).__name__}: environment check failed'
            failures = 1
        self.app.call_from_thread(self._checked_permissions, report, failures)

    def _checked_permissions(self, report, failures):
        self.query_one('#permission-report', Static).update(report)
        self._status(message('环境检查完成：{0} 项失败；详细结果见下方。', failures), error=bool(failures))
        self._set_working(False)

    @on(Button.Pressed, '#configure-environment-permissions')
    def configure_permissions(self):
        if self._working:
            return
        names = tuple(name for name in ('perf', 'tcpdump')
                      if self.query_one('#permission-' + name, Checkbox).value)
        try:
            plan = build_permission_plan(names)
        except (OSError, ValueError) as exc:
            self._status(message('无法配置权限：{0}', str(exc)), error=True)
            return
        detail = message(
            '为当前用户授权所选工具，限制普通用户执行范围；不修改 perf_event_paranoid。\n'
            '会保存原权限，并通过系统终端请求 sudo 授权。\n\n{0}',
            '\n'.join(shlex.join(command) for command in plan.commands),
        )
        self.app.push_screen(PermissionReviewScreen('配置采集权限', detail, '执行配置'),
                             lambda confirmed: self._confirmed_permissions(confirmed, plan))

    def _confirmed_permissions(self, confirmed, plan):
        if not confirmed or self._working:
            return
        self._set_working(True)
        self.call_after_refresh(self._install_permissions, plan)

    def _install_permissions(self, plan):
        error = ''
        backup = self.backup_dir / f'permissions-before-{uuid4().hex[:12]}.json'
        try:
            with self.app.suspend():
                # Catch inside the context so Textual always restores its terminal.
                try:
                    execute_permission_plan(plan, backup_path=backup)
                except (Exception, KeyboardInterrupt) as exc:
                    error = str(exc) or type(exc).__name__
        except SuspendNotSupported:
            error = self.app.tr('当前终端不支持系统授权，请在原生终端启动 TUI。')
        self._set_working(False)
        if error:
            self._status(message('权限配置未完成：{0}', error), error=True)
        else:
            self._status(message('权限配置已执行，原权限保存在 {0}；正在复查。', backup))
            self.check_permissions()

    @on(Button.Pressed, '#close-environment-settings')
    def action_close(self):
        if not self._working:
            self.dismiss(self.saved)

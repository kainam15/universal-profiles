"""Progressive model-contract review; executable probes use the app's process manager."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from textual import on, work
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Label, Static

from acprof.artifacts import atomic_write_json
from acprof.host.model_inspection import explain_resolution
from acprof.model_contract import write_model_resolution
from acprof.model_review import apply_review, review_questions
from acprof.model_spec import task_model_spec
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.rendering import CjkCompositor


class ModelResolutionScreen(ModalScreen):
    BINDINGS = [("escape", "close", "关闭")]
    DEFAULT_CSS = """
    ModelResolutionScreen { align: center middle; }
    #resolution-dialog { width: 94%; max-width: 120; height: 92%; border: round $accent; background: $surface; padding: 0 1; }
    #resolution-title { height: auto; text-style: bold; }
    #resolution-scroll { height: 1fr; }
    #resolution-body, #resolution-error, #resolution-details, .resolution-reason { height: auto; }
    #resolution-error { color: $error; }
    .resolution-answer { width: 100%; }
    .resolution-buttons { height: 3; }
    .resolution-buttons Button { min-width: 10; width: 1fr; margin-right: 1; }
    """

    def __init__(self, config, *, task=None):
        super().__init__()
        self._compositor = CjkCompositor()
        self.config = config
        self.task_info = task
        self.error = ""

    def compose(self):
        tr = self.app.tr
        contract = self.task_info.model_resolution.get("contract", {}) if self.task_info else {}
        questions = review_questions(self.task_info) if self.task_info else []
        ready = contract.get("status") == "resolved"
        editable = bool(questions) and not any(item.get("read_only") for item in questions)
        with Vertical(id="resolution-dialog"):
            yield Label(tr("模型契约解析"), id="resolution-title")
            with VerticalScroll(id="resolution-scroll"):
                yield Static(explain_resolution(self.task_info) if self.task_info else tr("正在读取模型证据……"),
                             id="resolution-body", markup=False)
                for index, item in enumerate(questions):
                    yield Label(item["path"])
                    yield Static(item["reason"], classes="resolution-reason", markup=False)
                    if not item.get("read_only"):
                        value = json.dumps(item["value"], ensure_ascii=False) if item["value"] is not None else ""
                        yield Input(value=value, placeholder=tr("填写此字段的 JSON 值"),
                                    id=f"resolution-answer-{index}", classes="resolution-answer")
                if ready:
                    yield Static(tr("没有待确认字段；静态解析不代表推理已验证。"), markup=False)
                elif questions and not editable:
                    yield Static(tr("源码或选择仍有歧义，请提供本地模型声明或 adapter。"), markup=False)
                if self.task_info:
                    with Collapsible(title=tr("已解析字段与证据"), collapsed=True):
                        yield Static(explain_resolution(self.task_info, explain=True), id="resolution-details", markup=False)
                yield Static(tr("Probe 使用 CPU 2 核、4 GiB、300 秒上限；可能构建镜像和下载模型。"), markup=False)
                yield Static(self.error, id="resolution-error", markup=False)
            with Horizontal(classes="resolution-buttons"):
                yield Button(tr("应用字段"), id="resolution-apply", disabled=not editable)
                yield Button("basic Probe", id="resolution-basic", disabled=not ready)
                yield Button("full Probe", id="resolution-full", disabled=not ready)
            with Horizontal(classes="resolution-buttons"):
                yield Button(tr("使用契约"), id="resolution-use", disabled=not ready, variant="primary")
                yield Button(tr("关闭"), id="resolution-close")

    def on_mount(self):
        if self.task_info is None:
            self.resolve()

    @work(thread=True, exclusive=True, exit_on_error=False)
    def resolve(self):
        from acprof.host.detect import detect_task
        from acprof.host.env_utils import bootstrap_project_env
        try:
            bootstrap_project_env(Path.cwd())
            task = detect_task(self.config.model, override_tag=self.config.task or None,
                               override_backend=self.config.backend or None,
                               model_spec_path=self.config.model_spec or None)
            error = ""
        except (Exception, SystemExit) as exc:
            task, error = None, str(exc)
        self.app.call_from_thread(self.resolved, task, error)

    def resolved(self, task, error):
        if not self.is_mounted:
            return
        self.task_info, self.error = task, error
        self.refresh(recompose=True)

    @on(Button.Pressed, "#resolution-apply")
    def apply_answers(self):
        try:
            answers = {item["path"]: json.loads(self.query_one(f"#resolution-answer-{index}", Input).value)
                       for index, item in enumerate(review_questions(self.task_info)) if not item.get("read_only")}
            self.query_one("#resolution-apply", Button).disabled = True
            self.review(answers)
        except (ValueError, TypeError) as exc:
            self.query_one("#resolution-error", Static).update(str(exc))

    @work(thread=True, exclusive=True, exit_on_error=False)
    def review(self, answers):
        try:
            task, error = apply_review(self.task_info, answers), ""
        except (ValueError, TypeError, OSError) as exc:
            task, error = self.task_info, str(exc)
        self.app.call_from_thread(self.resolved, task, error)

    @on(Button.Pressed, "#resolution-use, #resolution-basic, #resolution-full")
    def use_contract(self, event: Button.Pressed):
        if not self.task_info or self.task_info.model_resolution.get("contract", {}).get("status") != "resolved":
            return
        try:
            root = Path(self.config.output_dir).expanduser() / self.task_info.model_id.replace("/", "--") / "model-contracts" / uuid.uuid4().hex[:12]
            root = root.absolute()
            spec = root / "acprof_model.json"
            atomic_write_json(spec, task_model_spec(self.task_info))
            write_model_resolution(self.task_info, root)
        except (OSError, ValueError) as exc:
            self.query_one("#resolution-error", Static).update(str(exc))
            return
        self.dismiss({"task": self.task_info, "spec": spec, "output": root,
                      "selection": {"task": self.config.task, "backend": self.config.backend},
                      "probe": {"resolution-basic": "basic", "resolution-full": "full"}.get(event.button.id)})

    @on(Button.Pressed, "#resolution-close")
    def action_close(self):
        self.dismiss(None)

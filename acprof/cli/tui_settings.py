"""Local, project-scoped preferences for the terminal controller.

Saving is explicit: reading preferences never creates or repairs a file, and
experiment defaults are only persisted when supplied by the caller.  The
whitelisted dataclasses contain no environment variables or credentials.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, get_type_hints

from acprof.cli.tui_core import RunConfig
from acprof.cli.tui_themes import UI_THEMES


LOG_MAX_LINES = (500, 1000, 3000, 10000)
SETTINGS_VERSION = 1


@dataclass(frozen=True)
class UiPreferences:
    theme: str = "acprof-dark"
    log_wrap: bool = True
    log_max_lines: int = 3000
    show_command_bar: bool = True

    def validate(self) -> "UiPreferences":
        _validate_field_types(asdict(self), UiPreferences, "界面设置")
        if self.theme not in UI_THEMES:
            raise ValueError("无效的界面主题，请在设置页重新选择")
        if self.log_max_lines not in LOG_MAX_LINES:
            raise ValueError("日志保留行数只能是 500、1000、3000 或 10000")
        return self


@dataclass(frozen=True)
class TuiSettings:
    ui: UiPreferences = field(default_factory=UiPreferences)
    run_defaults: RunConfig | None = None

    def validate(self, *, project_dir: Path) -> "TuiSettings":
        if not isinstance(self.ui, UiPreferences):
            raise ValueError("界面设置必须是 UiPreferences")
        ui = self.ui.validate()
        config = self.run_defaults
        if config is not None:
            if not isinstance(config, RunConfig):
                raise ValueError("实验默认配置必须是 RunConfig")
            _validate_field_types(asdict(config), RunConfig, "实验默认配置")
            # A reusable default need not select a model, but must otherwise
            # meet the same requirements as the live experiment form.
            model = config.model.strip()
            config = replace(config, model=model or "settings/default-model")
            try:
                config = config.validate(project_dir=project_dir)
            except OverflowError as exc:
                raise ValueError("实验默认配置中的数字超出支持范围") from exc
            config = replace(config, model=model)
        return replace(self, ui=ui, run_defaults=config)


def default_settings_path(project_dir: Path) -> Path:
    """Return an XDG user-config path isolated by resolved project directory."""
    configured_root = os.environ.get("XDG_CONFIG_HOME", "")
    config_root = Path(configured_root).expanduser() if configured_root else None
    if config_root is None or not config_root.is_absolute():
        config_root = Path.home() / ".config"
    project = str(Path(project_dir).expanduser().resolve())
    project_key = hashlib.sha256(project.encode("utf-8")).hexdigest()[:16]
    return config_root / "acprof" / project_key / "tui.json"


def _validate_field_types(
    values: dict[str, Any], cls: type, label: str
) -> None:
    """Reject coercions, especially JSON booleans masquerading as integers."""
    allowed = {item.name for item in fields(cls)}
    if values.keys() - allowed:
        raise ValueError(f"{label}包含无法识别的字段")
    annotations = get_type_hints(cls)
    for name, value in values.items():
        expected = annotations[name]
        valid = (
            type(value) in (int, float)
            if expected is float
            else type(value) is expected
        )
        if not valid:
            raise ValueError(f"{label}字段 {name} 的类型不正确")


def _decode_settings(payload: Any, project_dir: Path) -> TuiSettings:
    if not isinstance(payload, dict):
        raise ValueError("设置文件的最外层必须是 JSON 对象")
    version = payload.get("version", SETTINGS_VERSION)
    if type(version) is not int or version != SETTINGS_VERSION:
        raise ValueError("不支持此设置文件版本")
    ui_values = payload.get("ui", {})
    if not isinstance(ui_values, dict):
        raise ValueError("界面设置必须是 JSON 对象")
    _validate_field_types(ui_values, UiPreferences, "界面设置")
    ui = UiPreferences(**ui_values)
    defaults_values = payload.get("run_defaults")
    defaults = None
    if defaults_values is not None:
        if not isinstance(defaults_values, dict):
            raise ValueError("实验默认配置必须是 JSON 对象")
        _validate_field_types(defaults_values, RunConfig, "实验默认配置")
        defaults = RunConfig(**defaults_values)
    # Ignore unknown top-level keys; only recognized fields can be saved again.
    return TuiSettings(ui=ui, run_defaults=defaults).validate(project_dir=project_dir)


def load_settings(path: Path, project_dir: Path) -> tuple[TuiSettings, str]:
    """Load valid preferences, or return defaults and a readable warning.

    Missing files are a normal first launch.  Invalid files are left intact so
    the user can inspect them or explicitly overwrite them from Settings.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = _decode_settings(payload, Path(project_dir))
    except FileNotFoundError:
        return TuiSettings(), ""
    except (OSError, ValueError, UnicodeError) as exc:
        return TuiSettings(), f"无法读取本地设置，已使用默认值：{exc}"
    return settings, ""


def save_settings(path: Path, settings: TuiSettings, project_dir: Path) -> None:
    """Validate and atomically replace a local preferences file.

    Raise ValueError for invalid settings and OSError for file-system failures.
    A failed validation or replacement leaves the previous file untouched.
    """
    if not isinstance(settings, TuiSettings):
        raise ValueError("设置必须是 TuiSettings")
    normalized = settings.validate(project_dir=Path(project_dir))
    payload = {
        "version": SETTINGS_VERSION,
        "ui": asdict(normalized.ui),
        "run_defaults": (
            asdict(normalized.run_defaults)
            if normalized.run_defaults is not None
            else None
        ),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)

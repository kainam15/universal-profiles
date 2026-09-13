"""Local, project-scoped preferences for the terminal controller.

Reading preferences never creates or repairs a file. UI preferences and
experiment defaults are saved explicitly; the latest confirmed run/probe model
and used result paths are remembered automatically. The whitelisted dataclasses
contain no environment variables or credentials.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, get_type_hints

from acprof.tui.commands import RunConfig
from acprof.tui.i18n import UI_LANGUAGES, error_message, message
from acprof.tui.themes import UI_THEMES


LOG_MAX_LINES = (500, 1000, 3000, 10000)
SETTINGS_VERSION = 4


class UnsupportedSettingsError(ValueError):
    """设置版本不受支持，调用方不得静默回退或覆盖文件。"""


@dataclass(frozen=True)
class UiPreferences:
    theme: str = "acprof-dark"
    log_wrap: bool = True
    log_max_lines: int = 3000
    show_command_bar: bool = True
    language: str = "zh"

    def validate(self) -> "UiPreferences":
        _validate_field_types(asdict(self), UiPreferences, message('界面设置'))
        if self.language not in UI_LANGUAGES:
            raise ValueError(message("无效的界面语言，请在设置页重新选择"))
        if self.theme not in UI_THEMES:
            raise ValueError(message('无效的界面主题，请在设置页重新选择'))
        if self.log_max_lines not in LOG_MAX_LINES:
            raise ValueError(message('日志保留行数只能是 500、1000、3000 或 10000'))
        return self


@dataclass(frozen=True)
class TuiSettings:
    ui: UiPreferences = field(default_factory=UiPreferences)
    run_defaults: RunConfig | None = None
    last_model: str = ""
    last_result_dir: str = ""
    last_result_csv: str = ""

    def validate(self, *, project_dir: Path) -> "TuiSettings":
        if not isinstance(self.ui, UiPreferences):
            raise ValueError(message('界面设置必须是 UiPreferences'))
        ui = self.ui.validate()
        if type(self.last_model) is not str:
            raise ValueError(message('上次运行的模型 ID 必须是字符串'))
        if type(self.last_result_dir) is not str or type(self.last_result_csv) is not str:
            raise ValueError(message('上次使用的结果路径必须是字符串'))
        config = self.run_defaults
        if config is not None:
            if not isinstance(config, RunConfig):
                raise ValueError(message('实验默认配置必须是 RunConfig'))
            _validate_field_types(asdict(config), RunConfig, message('实验默认配置'))
            # A reusable default need not select a model, but must otherwise
            # meet the same requirements as the live experiment form.
            model = config.model.strip()
            config = replace(config, model=model or "settings/default-model")
            try:
                config = config.validate(project_dir=project_dir)
            except OverflowError as exc:
                raise ValueError(message('实验默认配置中的数字超出支持范围')) from exc
            config = replace(config, model=model)
        return replace(
            self, ui=ui, run_defaults=config, last_model=self.last_model.strip(),
            last_result_dir=self.last_result_dir.strip(),
            last_result_csv=self.last_result_csv.strip(),
        )


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
        raise ValueError(message('{0}包含无法识别的字段', label))
    annotations = get_type_hints(cls)
    for name, value in values.items():
        expected = annotations[name]
        valid = (
            type(value) in (int, float)
            if expected is float
            else type(value) is expected
        )
        if not valid:
            raise ValueError(message('{0}字段 {1} 的类型不正确', label, name))


def _decode_settings(payload: Any, project_dir: Path) -> TuiSettings:
    if not isinstance(payload, dict):
        raise ValueError(message('设置文件的最外层必须是 JSON 对象'))
    version = payload.get("version")
    if type(version) is not int or version != SETTINGS_VERSION:
        raise UnsupportedSettingsError(
            f"不支持设置文件版本 {version!r}；当前要求 version={SETTINGS_VERSION}。"
            "请归档旧设置文件后重新配置。"
        )
    ui_values = payload.get("ui", {})
    if not isinstance(ui_values, dict):
        raise ValueError(message('界面设置必须是 JSON 对象'))
    _validate_field_types(ui_values, UiPreferences, message('界面设置'))
    ui = UiPreferences(**ui_values)
    defaults_values = payload.get("run_defaults")
    defaults = None
    if defaults_values is not None:
        if not isinstance(defaults_values, dict):
            raise ValueError(message('实验默认配置必须是 JSON 对象'))
        if "allow_cgroup_v1" in defaults_values:
            raise UnsupportedSettingsError(
                "设置包含已删除的 allow_cgroup_v1；请归档旧设置文件后重新配置。"
            )
        _validate_field_types(defaults_values, RunConfig, message('实验默认配置'))
        defaults = RunConfig(**defaults_values)
    # Ignore unknown top-level keys; only recognized fields can be saved again.
    return TuiSettings(
        ui=ui, run_defaults=defaults, last_model=payload.get("last_model", ""),
        last_result_dir=payload.get("last_result_dir", ""),
        last_result_csv=payload.get("last_result_csv", ""),
    ).validate(project_dir=project_dir)


def load_settings(path: Path, project_dir: Path) -> tuple[TuiSettings, str]:
    """Load valid preferences, or return defaults and a readable warning.

    Missing files are a normal first launch. Retired versions and fields raise
    without modifying the file; malformed current values retain a warning.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = _decode_settings(payload, Path(project_dir))
    except UnsupportedSettingsError:
        raise
    except FileNotFoundError:
        return TuiSettings(), ""
    except (OSError, ValueError, UnicodeError) as exc:
        return TuiSettings(), message('无法读取本地设置，已使用默认值：{0}', error_message(exc))
    return settings, ""


def save_settings(path: Path, settings: TuiSettings, project_dir: Path) -> None:
    """Validate and atomically replace a local preferences file.

    Raise ValueError for invalid settings and OSError for file-system failures.
    A failed validation or replacement leaves the previous file untouched.
    """
    if not isinstance(settings, TuiSettings):
        raise ValueError(message('设置必须是 TuiSettings'))
    normalized = settings.validate(project_dir=Path(project_dir))
    payload = {
        "version": SETTINGS_VERSION,
        "ui": asdict(normalized.ui),
        "last_model": normalized.last_model,
        "last_result_dir": normalized.last_result_dir,
        "last_result_csv": normalized.last_result_csv,
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

"""保留原 tui_settings 导入路径的兼容入口。"""

from acprof.tui.commands import RunConfig
from acprof.tui.i18n import UI_LANGUAGES, error_message, message
from acprof.tui.themes import UI_THEMES
from acprof.tui.settings import (
    LOG_MAX_LINES,
    SETTINGS_VERSION,
    TuiSettings,
    UiPreferences,
    _decode_settings,
    _validate_field_types,
    default_settings_path,
    load_settings,
    save_settings,
)

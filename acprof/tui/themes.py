"""Theme surfaces and stable action colors for the TUI."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ThemePalette:
    name: str
    label: str
    secondary: str
    background: str
    surface: str
    panel: str
    foreground: str
    dark: bool = True

    def theme_kwargs(self) -> dict[str, str | bool | dict[str, str]]:
        values = asdict(self)
        values.pop("label")
        # Themes change surfaces; action colors keep the same meaning.
        semantics = (
            dict(primary="#66b8c4", success="#81b89a", warning="#e0bd65", error="#e68f91")
            if self.dark else
            dict(primary="#176978", success="#347459", warning="#806000", error="#aa3e49")
        )
        return {
            **values, **semantics, "accent": semantics["primary"], "ansi": False,
            "variables": {"ui-disabled": "#858585" if self.dark else "#757575"},
        }


# Explicit RGB palettes keep the VS Code appearance across terminal themes.
# Keep IDs and surfaces stable for saved preferences.
THEME_CATALOG = (
    ThemePalette(
        name="acprof-dark", label="深海蓝 · 深色",
        secondary="#93acbd",
        background="#15232d", surface="#1c2e3b", panel="#283d4b",
        foreground="#e2ebef",
    ),
    ThemePalette(
        name="acprof-graphite", label="石墨灰 · 深色",
        secondary="#999fb0",
        background="#202126", surface="#292b32", panel="#383b45",
        foreground="#eceef3",
    ),
    ThemePalette(
        name="acprof-forest", label="松林绿 · 深色",
        secondary="#a7b8a4",
        background="#182824", surface="#233830", panel="#334a40",
        foreground="#e5eee5",
    ),
    ThemePalette(
        name="acprof-plum", label="暮紫 · 深色",
        secondary="#b4accb",
        background="#282236", surface="#352e45", panel="#473e59",
        foreground="#f0eaf7",
    ),
    ThemePalette(
        name="acprof-amber", label="琥珀 · 深色",
        secondary="#bfa58c",
        background="#2b2520", surface="#383028", panel="#4a4034",
        foreground="#f3eadd",
    ),
    ThemePalette(
        name="acprof-light", label="纸白 · 浅色",
        secondary="#536f82",
        background="#edf3f5", surface="#ffffff", panel="#dce7ec",
        foreground="#1e3543",
        dark=False,
    ),
    ThemePalette(
        name="acprof-sand", label="暖砂 · 浅色",
        secondary="#73644f",
        background="#f4eddf", surface="#fffaf0", panel="#e6dbc7",
        foreground="#3e352a",
        dark=False,
    ),
    ThemePalette(
        name="acprof-mist", label="雾蓝 · 浅色",
        secondary="#586e8a",
        background="#e8eef8", surface="#f5f8ff", panel="#d4dfef",
        foreground="#26354d",
        dark=False,
    ),
)

UI_THEMES = tuple(palette.name for palette in THEME_CATALOG)
THEME_OPTIONS = tuple((palette.label, palette.name) for palette in THEME_CATALOG)

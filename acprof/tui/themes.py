"""Shared palette catalog for TUI registration, selection and persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ThemePalette:
    name: str
    label: str
    primary: str
    secondary: str
    background: str
    surface: str
    panel: str
    foreground: str
    success: str
    warning: str
    error: str
    dark: bool = True

    def theme_kwargs(self) -> dict[str, str | bool]:
        values = asdict(self)
        values.pop("label")
        return {**values, "accent": self.primary}


# Keep the original IDs stable so saved preferences retain their appearance.
THEME_CATALOG = (
    ThemePalette(
        name="acprof-dark", label="深海蓝 · 深色",
        primary="#66b8c4", secondary="#93acbd",
        background="#15232d", surface="#1c2e3b", panel="#283d4b",
        foreground="#e2ebef", success="#81b89a", warning="#ddb77d", error="#e68f91",
    ),
    ThemePalette(
        name="acprof-graphite", label="石墨灰 · 深色",
        primary="#bdc9dc", secondary="#999fb0",
        background="#202126", surface="#292b32", panel="#383b45",
        foreground="#eceef3", success="#a8c6a3", warning="#d7be8c", error="#e99ca5",
    ),
    ThemePalette(
        name="acprof-forest", label="松林绿 · 深色",
        primary="#90c6ae", secondary="#a7b8a4",
        background="#182824", surface="#233830", panel="#334a40",
        foreground="#e5eee5", success="#a8d298", warning="#e0c086", error="#e6a19a",
    ),
    ThemePalette(
        name="acprof-plum", label="暮紫 · 深色",
        primary="#c8acf0", secondary="#b4accb",
        background="#282236", surface="#352e45", panel="#473e59",
        foreground="#f0eaf7", success="#a6d0b3", warning="#e4c496", error="#ed9db4",
    ),
    ThemePalette(
        name="acprof-amber", label="琥珀 · 深色",
        primary="#e1b775", secondary="#bfa58c",
        background="#2b2520", surface="#383028", panel="#4a4034",
        foreground="#f3eadd", success="#b1c99e", warning="#f2cc8a", error="#ec9e99",
    ),
    ThemePalette(
        name="acprof-light", label="纸白 · 浅色",
        primary="#176978", secondary="#536f82",
        background="#edf3f5", surface="#ffffff", panel="#dce7ec",
        foreground="#1e3543", success="#347459", warning="#936019", error="#aa3e49",
        dark=False,
    ),
    ThemePalette(
        name="acprof-sand", label="暖砂 · 浅色",
        primary="#80572e", secondary="#73644f",
        background="#f4eddf", surface="#fffaf0", panel="#e6dbc7",
        foreground="#3e352a", success="#476841", warning="#86571c", error="#a0423e",
        dark=False,
    ),
    ThemePalette(
        name="acprof-mist", label="雾蓝 · 浅色",
        primary="#395b99", secondary="#586e8a",
        background="#e8eef8", surface="#f5f8ff", panel="#d4dfef",
        foreground="#26354d", success="#376949", warning="#855b1f", error="#a53f58",
        dark=False,
    ),
)

UI_THEMES = tuple(palette.name for palette in THEME_CATALOG)
THEME_OPTIONS = tuple((palette.label, palette.name) for palette in THEME_CATALOG)

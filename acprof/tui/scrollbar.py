"""Scrollbar rendering that does not depend on terminal block glyphs."""

from __future__ import annotations

from rich.color import Color
from rich.segment import Segment, Segments
from rich.style import Style
from textual.scrollbar import ScrollBarRender


class SolidScrollBarRender(ScrollBarRender):
    """Paint track and thumb with explicit backgrounds in whole terminal cells.

    Textual's fractional block characters can leave dark bands with some fonts
    and terminal renderers. Keeping the mouse metadata on plain colored spaces
    preserves Textual's existing click and drag handling without those glyphs.
    Set ``renderer`` on this app's scrollbar instances to opt in.
    """

    @classmethod
    def render_bar(
        cls,
        size: int = 25,
        virtual_size: float = 50,
        window_size: float = 20,
        position: float = 0,
        thickness: int = 1,
        vertical: bool = True,
        back_color: Color = Color.parse("#555555"),
        bar_color: Color = Color.parse("bright_magenta"),
    ) -> Segments:
        size = max(0, int(size))
        thickness = max(0, int(thickness))
        if not size or not thickness:
            return Segments(())

        blank = " " * (thickness if vertical else 1)
        track = Segment(blank, Style(bgcolor=back_color))
        segments = [track] * size

        if 0 < window_size < virtual_size:
            # Retain at least one track cell whenever there is room, so even a
            # nearly full viewport has a visible direction in which to scroll.
            thumb_size = max(
                1, min(max(1, size - 1), int(size * window_size / virtual_size + 0.5))
            )
            position_ratio = max(0.0, min(1.0, position / (virtual_size - window_size)))
            start = min(
                size - thumb_size, int((size - thumb_size) * position_ratio + 0.5)
            )
            end = start + thumb_size
            upper = Segment(
                blank, Style(bgcolor=back_color, meta={"@mouse.down": "scroll_up"})
            )
            thumb = Segment(
                blank, Style(bgcolor=bar_color, meta={"@mouse.down": "grab"})
            )
            lower = Segment(
                blank, Style(bgcolor=back_color, meta={"@mouse.down": "scroll_down"})
            )
            segments = [upper] * start + [thumb] * thumb_size + [lower] * (size - end)

        if vertical:
            return Segments(segments, new_lines=True)
        return Segments((segments + [Segment.line()]) * thickness, new_lines=False)

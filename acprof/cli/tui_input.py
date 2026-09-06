"""Native, blinking insertion cursors for the full-screen TUI's inputs."""
from __future__ import annotations

from rich.control import Control
from rich.style import Style
from textual.app import App
from textual.driver import Driver
from textual.errors import NoWidget
from textual.geometry import Offset
from textual.timer import Timer
from textual.widgets import Input


class BarCursorInput(Input):
    """Keep Input's editing/selection rendering without its simulated block."""

    def _restart_blink(self) -> None:
        super()._restart_blink()
        app = self.app
        if isinstance(app, BarCursorApp) and app.focused is self:
            # Also reset on a click/key that leaves the insertion point unchanged.
            app._input_cursor_restart_pending = True
            app.call_after_refresh(app._sync_input_cursor)

    def get_component_rich_style(
        self, *names: str, partial: bool = False, default: Style | None = None,
    ) -> Style:
        if names == ("input--cursor",) and not self.app.is_inline:
            # An empty style also preserves placeholder and selection colors.
            return Style.null()
        return super().get_component_rich_style(
            *names, partial=partial, default=default,
        )


class BarCursorApp(App[None]):
    """Show the terminal caret only over the focused, visible input.

    Textual 8.x hides the real cursor and paints a cell background instead.
    Its frame hooks place the native caret after layout and scrolling. One
    active-input timer toggles terminal visibility without repainting widgets;
    blink timing therefore does not depend on the terminal's blink setting.
    """

    _input_cursor_visible = False
    _input_cursor_style_set = False
    _input_cursor_phase = True
    _input_cursor_restart_pending = False
    _input_cursor_key: tuple[BarCursorInput, str, tuple[int, int]] | None = None
    _input_cursor_timer: Timer | None = None
    _input_cursor_blink_enabled = True
    _input_cursor_suspended = False
    INPUT_CURSOR_BLINK_INTERVAL = 0.5

    def get_driver_class(self) -> type[Driver]:
        driver_class = super().get_driver_class()

        class CursorDriver(driver_class):
            def start_application_mode(driver_self) -> None:
                super().start_application_mode()
                self._input_cursor_suspended = False

            def stop_application_mode(driver_self) -> None:
                # Covers normal exit, exceptions, and terminal suspension.
                try:
                    self._reset_input_cursor()
                finally:
                    super().stop_application_mode()

        return CursorDriver

    def _input_cursor_offset(self) -> Offset | None:
        field = self.focused
        if (
            not self.app_focus
            or not isinstance(field, BarCursorInput)
            or field.is_disabled
        ):
            return None
        # Input adds an extra cell at the end for scroll-into-view. The bar
        # belongs immediately after the final character, including when empty.
        offset = field.cursor_screen_offset - Offset(int(field.cursor_at_end), 0)
        if not field.content_region.contains(*offset):
            return None
        try:
            widget, _ = self.get_widget_at(*offset)
        except NoWidget:
            return None
        # Parent scrolling, tabs, modals, and overlays may cover the input.
        return offset if widget is field else None

    def _begin_update(self) -> None:
        super()._begin_update()
        if self._input_cursor_visible and self._driver is not None:
            self._driver.write("\x1b[?25l")
            self._input_cursor_visible = False

    def _end_update(self) -> None:
        try:
            self._sync_input_cursor()
        finally:
            super()._end_update()

    def _stop_input_cursor_timer(self) -> None:
        if self._input_cursor_timer is not None:
            self._input_cursor_timer.stop()
            self._input_cursor_timer = None

    def set_input_cursor_blink_enabled(self, enabled: bool) -> None:
        """Keep a steady caret, with no timer, during measurement windows."""
        if enabled == self._input_cursor_blink_enabled:
            return
        self._input_cursor_blink_enabled = enabled
        self._stop_input_cursor_timer()
        self._input_cursor_phase = True
        self._sync_input_cursor()

    def _toggle_input_cursor(self) -> None:
        self._input_cursor_phase = not self._input_cursor_phase
        self._sync_input_cursor()

    def _sync_input_cursor(self) -> None:
        driver = self._driver
        if (
            driver is None or driver.is_headless or driver.is_inline
            or not self._running or self._closed or self._input_cursor_suspended
        ):
            self._stop_input_cursor_timer()
            return
        offset = self._input_cursor_offset()
        if offset is None:
            self._stop_input_cursor_timer()
            self._input_cursor_key = None
            if self._input_cursor_visible:
                driver.write("\x1b[?25l")
                self._input_cursor_visible = False
                driver.flush()
            return

        field = self.focused
        assert isinstance(field, BarCursorInput)
        key = (field, field.value, field.selection)
        if key != self._input_cursor_key or self._input_cursor_restart_pending:
            self._input_cursor_key = key
            self._input_cursor_restart_pending = False
            self._input_cursor_phase = True
            self._stop_input_cursor_timer()
        if self._input_cursor_blink_enabled:
            if self._input_cursor_timer is None:
                self._input_cursor_timer = self.set_interval(
                    self.INPUT_CURSOR_BLINK_INTERVAL,
                    self._toggle_input_cursor,
                    name="input cursor blink",
                )
        else:
            self._input_cursor_phase = True

        self.cursor_position = offset
        # Use a steady terminal shape to avoid two independent blink clocks.
        shape = "" if self._input_cursor_style_set else "\x1b[6 q"
        self._input_cursor_style_set = True
        if self._input_cursor_phase:
            driver.write(shape + Control.move_to(*offset).segment.text + "\x1b[?25h")
            self._input_cursor_visible = True
        elif self._input_cursor_visible:
            driver.write("\x1b[?25l")
            self._input_cursor_visible = False
        driver.flush()

    def _reset_input_cursor(self) -> None:
        self._input_cursor_suspended = True
        self._stop_input_cursor_timer()
        self._input_cursor_key = None
        self._input_cursor_restart_pending = False
        self._input_cursor_phase = True
        if self._input_cursor_style_set and self._driver is not None:
            self._driver.write("\x1b[?25l\x1b[0 q")
            self._driver.flush()
        self._input_cursor_visible = False
        self._input_cursor_style_set = False

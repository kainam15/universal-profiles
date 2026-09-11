"""Selectable, bounded plain-text logs for the low-refresh terminal controller."""

from __future__ import annotations

from collections.abc import Sequence
import re

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.scrollbar import ScrollDown, ScrollTo, ScrollUp
from textual.widgets import TextArea
from textual.widgets.text_area import Selection


class SelectableLog(TextArea):
    """A read-only log which keeps original text separate from visual wrapping.

    Each ``write`` appends a log entry. Reading or selecting pauses tailing until
    ``follow_tail`` is called, so incoming output cannot steal the viewport.
    Retention counts original lines, not terminal rows created by soft wrapping.
    """

    BINDINGS = [Binding("ctrl+a", "select_all", "全选日志", show=False)]

    class FollowChanged(Message):
        """The user paused following, or explicitly returned to the latest output."""

        def __init__(self, log: SelectableLog, following: bool) -> None:
            super().__init__()
            self.log = log
            self.following = following

        @property
        def control(self) -> SelectableLog:
            return self.log

    def __init__(
        self,
        *,
        max_lines: int = 5000,
        wrap: bool = True,
        id: str | None = None,
        classes: str | None = None,
        name: str | None = None,
    ) -> None:
        self._mutating_log = True
        self._has_entries = False
        self._following = True
        self._max_lines = self._validate_max_lines(max_lines)
        super().__init__(
            read_only=True,
            soft_wrap=wrap,
            show_cursor=False,
            show_line_numbers=False,
            highlight_cursor_line=False,
            max_checkpoints=0,
            id=id,
            classes=classes,
            name=name,
        )
        self.cursor_blink = False
        self.match_cursor_bracket = False
        self._mutating_log = False

    @staticmethod
    def _validate_max_lines(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_lines must be a positive integer")
        return value

    @property
    def max_lines(self) -> int:
        return self._max_lines

    @max_lines.setter
    def max_lines(self, value: int) -> None:
        self._max_lines = self._validate_max_lines(value)
        self._trim_and_restore()

    @property
    def wrap(self) -> bool:
        return self.soft_wrap

    @wrap.setter
    def wrap(self, value: bool) -> None:
        self.soft_wrap = bool(value)
        if self.following and self.is_mounted:
            self.call_after_refresh(self._scroll_tail_if_following)

    @property
    def lines(self) -> Sequence[str]:
        """Retained original lines; visual wrapping never inserts newlines here."""
        return self.document.lines if self._has_entries else ()

    @property
    def following(self) -> bool:
        return self._following

    def _set_following(self, following: bool) -> None:
        if self._following != following:
            self._following = following
            self.post_message(self.FollowChanged(self, following))

    def _watch_selection(
        self, previous_selection: Selection, selection: Selection
    ) -> None:
        super()._watch_selection(previous_selection, selection)
        if not self._mutating_log and not selection.is_empty:
            self._set_following(False)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if (
            not self._mutating_log
            and new_value < old_value
            and not self.is_vertical_scroll_end
        ):
            self._set_following(False)

    def _on_mouse_down(self, event: events.MouseDown) -> None:
        # TextArea's inherited handler still owns dragging.
        self._set_following(False)

    def _on_click(self, event: events.Click) -> None:
        if event.chain not in (2, 3):
            return
        row, column = self.get_target_document_location(event)
        line = self.document[row]
        if event.chain == 3:
            self.selection = Selection((row, 0), (row, len(line)))
        else:
            for word in re.finditer(r"\w+|[^\w\s]+|\s+", line):
                if word.start() <= column < word.end():
                    self.selection = Selection((row, word.start()), (row, word.end()))
                    break
        event.stop()
        event.prevent_default()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._set_following(False)

    def action_cursor_up(self, select: bool = False) -> None:
        # Pause on user intent, before a new entry can interrupt scrolling.
        self._set_following(False)
        self.scroll_up(animate=False, immediate=True)

    def action_cursor_page_up(self) -> None:
        self._set_following(False)
        self.scroll_to(
            y=self.scroll_y - self.scrollable_content_region.height,
            animate=False,
            immediate=True,
        )

    def action_cursor_line_start(self, select: bool = False) -> None:
        self._set_following(False)
        self.scroll_home(animate=False, immediate=True)

    def action_cursor_down(self, select: bool = False) -> None:
        self.scroll_down(animate=False, immediate=True)

    def action_cursor_page_down(self) -> None:
        self.scroll_to(
            y=self.scroll_y + self.scrollable_content_region.height,
            animate=False,
            immediate=True,
        )

    def action_cursor_line_end(self, select: bool = False) -> None:
        # End only navigates; explicitly returning to latest resumes following.
        self.scroll_end(animate=False, immediate=True)

    def _on_scroll_up(self, event: ScrollUp) -> None:
        self.action_cursor_page_up()
        event.stop()
        event.prevent_default()

    def _on_scroll_down(self, event: ScrollDown) -> None:
        self.action_cursor_page_down()
        event.stop()
        event.prevent_default()

    def _on_scroll_to(self, message: ScrollTo) -> None:
        if self._allow_scroll:
            if message.y is not None:
                self._set_following(False)
            self.scroll_to(message.x, message.y, animate=False, immediate=True)
            message.stop()
            message.prevent_default()

    def _on_resize(self) -> None:
        # The inherited resize handler rewraps first; tailing waits for layout.
        if self.following:
            self.call_after_refresh(self._scroll_tail_if_following)

    def _scroll_tail_if_following(self) -> None:
        if self.following and self.is_mounted:
            self.scroll_end(animate=False, immediate=True, x_axis=False)

    def follow_tail(self) -> None:
        """Clear the selection, show the latest output, and resume following."""
        self.selection = Selection.cursor(self.document.end)
        self._set_following(True)
        self._scroll_tail_if_following()
        if self.is_mounted:
            self.call_after_refresh(self._scroll_tail_if_following)

    def _trim_lines(self) -> tuple[int, int]:
        remove_count = max(0, self.document.line_count - self.max_lines)
        if not remove_count:
            return 0, 0
        removed_rows = self.wrapped_document.location_to_offset((remove_count, 0)).y
        self.delete((0, 0), (remove_count, 0), maintain_selection_offset=False)
        return remove_count, removed_rows

    def _restore_after_write(
        self,
        selection: Selection,
        scroll_x: float,
        scroll_y: float,
        removed_lines: int,
        removed_rows: int,
    ) -> None:
        def relocate(location: tuple[int, int]) -> tuple[int, int]:
            row, column = location
            return (0, 0) if row < removed_lines else (row - removed_lines, column)

        self.selection = Selection(relocate(selection.start), relocate(selection.end))
        # No undo history may retain output that the user asked us to discard.
        self.history.clear()
        if not self.following:
            self.scroll_to(
                x=scroll_x,
                y=max(0, scroll_y - removed_rows),
                animate=False,
                immediate=True,
                force=True,
            )

    def _trim_and_restore(self) -> None:
        selection = self.selection
        scroll_x, scroll_y = self.scroll_x, self.scroll_y
        self._mutating_log = True
        try:
            removed_lines, removed_rows = self._trim_lines()
            self._restore_after_write(
                selection, scroll_x, scroll_y, removed_lines, removed_rows
            )
        finally:
            self._mutating_log = False
        self._scroll_tail_if_following()

    def write(self, content: str) -> SelectableLog:
        """Append an entry without changing a reader's selection or scroll anchor."""
        selection = self.selection
        scroll_x, scroll_y = self.scroll_x, self.scroll_y
        # A caller-provided final newline terminates this entry. Preserve any
        # additional blank lines, while avoiding a duplicated line separator.
        entry = content.removesuffix("\n")
        self._mutating_log = True
        try:
            self.insert(
                ("\n" if self._has_entries else "") + entry,
                self.document.end,
                maintain_selection_offset=True,
            )
            self._has_entries = True
            removed_lines, removed_rows = self._trim_lines()
            self._restore_after_write(
                selection, scroll_x, scroll_y, removed_lines, removed_rows
            )
        finally:
            self._mutating_log = False
        self._scroll_tail_if_following()
        if self.following and self.is_mounted:
            self.call_after_refresh(self._scroll_tail_if_following)
        return self

    def clear(self) -> SelectableLog:
        """Discard retained output and resume following future entries."""
        self._mutating_log = True
        try:
            self.load_text("")
            self._has_entries = False
        finally:
            self._mutating_log = False
        self._set_following(True)
        self.scroll_home(animate=False, immediate=True)
        return self

    def copy_selection(self) -> bool:
        """Copy selected original text; return whether there was a selection."""
        if not self.selected_text:
            return False
        self.app.copy_to_clipboard(self.selected_text)
        return True

    def copy_all(self) -> bool:
        """Copy all retained original text, including real line separators."""
        if not self._has_entries:
            return False
        self.app.copy_to_clipboard(self.text)
        return True

"""Keep wide characters intact across occluded Textual widget boundaries.

Textual 8.2.8 splits foreground text at hidden widget edges (#6357). Group
adjacent cells owned by the same widget before cropping; real occlusion still
clips normally. This adapter is local to AC-Prof screens, without patching
Textual globally. Recheck it when upgrading Textual's private compositor API.

Reference: https://github.com/Textualize/textual/issues/6357
"""

from bisect import bisect_left
from typing import Callable

from textual._compositor import ChopsUpdate, Compositor
from textual.geometry import Region
from textual.screen import Screen
from textual.strip import Strip


class CjkCompositor(Compositor):
    """Compose visible spans before cutting double-width characters."""

    def _render_chops(
        self, crop: Region, is_rendered_line: Callable[[int], bool],
    ) -> list[dict[int, Strip]]:
        cuts = self.cuts
        rows: list[dict[int, Strip]] = [{} for _ in cuts]
        claimed: list[set[int]] = [set() for _ in cuts]
        for region, clip, strips in self._get_renders(crop):
            visible = region.intersection(clip)
            for y, strip in zip(visible.line_range, strips):
                if not is_rendered_line(y):
                    continue
                boundaries, row, occupied = cuts[y], rows[y], claimed[y]
                index = bisect_left(boundaries, visible.x)
                stop = bisect_left(boundaries, visible.right)
                while index < stop:
                    if boundaries[index] in occupied:
                        index += 1
                        continue
                    first = index
                    index += 1
                    while index < stop and boundaries[index] not in occupied:
                        index += 1
                    start, end = boundaries[first], boundaries[index]
                    row[start] = strip.crop(start - visible.x, end - visible.x)
                    occupied.update(boundaries[first:index])
        # Front-to-back rendering claims spans out of horizontal order.
        return [dict(sorted(row.items())) for row in rows]

    def render_partial_update(self) -> ChopsUpdate | None:
        update = super().render_partial_update()
        if update is None:
            return None
        ends = [
            [x + (strip.cell_length if strip is not None else 0) for x, strip in row.items()]
            for row in update.chops
        ]
        spans = []
        for y, left, right in update.spans:
            start, end = left, right
            for x, strip in update.chops[y].items():
                if strip is not None and x < right and x + strip.cell_length > left:
                    start = min(start, x)
                    end = max(end, x + strip.cell_length)
            # A dirty region may end halfway through a character. Repaint the
            # intersecting visible span instead of replacing that half by space.
            spans.append((y, start, end))
        return ChopsUpdate(update.chops, spans, ends)


class CjkScreen(Screen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compositor = CjkCompositor()

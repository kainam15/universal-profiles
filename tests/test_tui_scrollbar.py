import unittest

from rich.color import Color
from rich.console import Console
from rich.style import Style

from acprof.cli.tui_scrollbar import SolidScrollBarRender


class SolidScrollBarRenderTests(unittest.TestCase):
    def render_cells(self, **kwargs):
        rendered = SolidScrollBarRender.render_bar(**kwargs)
        return [segment for segment in Console().render(rendered) if segment.text != "\n"]

    def test_thumb_is_contiguous_and_keeps_click_and_drag_targets(self):
        for back_hex, thumb_hex in (("#243746", "#77c5d5"), ("#edf3fa", "#3b5c95")):
            with self.subTest(background=back_hex):
                back, thumb = Color.parse(back_hex), Color.parse(thumb_hex)
                cells = self.render_cells(
                    size=20, virtual_size=100, window_size=20,
                    position=40, back_color=back, bar_color=thumb,
                )
                self.assertEqual(len(cells), 20)
                actions = [cell.style.meta.get("@mouse.down") for cell in cells]
                self.assertEqual(actions, ["scroll_up"] * 8 + ["grab"] * 4 + ["scroll_down"] * 8)
                for cell, action in zip(cells, actions):
                    self.assertEqual(cell.text, " ")
                    self.assertFalse(cell.style.reverse)
                    self.assertEqual(cell.style.bgcolor, thumb if action == "grab" else back)

    def test_scroll_range_reaches_both_ends_and_clamps_overscroll(self):
        starts = []
        for position in (-100, 0, 0.125, 20, 40.5, 60, 80, 1000):
            with self.subTest(position=position):
                cells = self.render_cells(size=20, virtual_size=100, window_size=20, position=position)
                actions = [cell.style.meta.get("@mouse.down") for cell in cells]
                grabbed = [index for index, action in enumerate(actions) if action == "grab"]
                self.assertEqual(len(cells), 20)
                self.assertEqual(grabbed, list(range(grabbed[0], grabbed[-1] + 1)))
                starts.append(grabbed[0])
                if position <= 0:
                    self.assertEqual(grabbed[0], 0)
                    self.assertNotIn("scroll_up", actions)
                if position >= 80:
                    self.assertEqual(grabbed[-1], 19)
                    self.assertNotIn("scroll_down", actions)
        self.assertEqual(starts, sorted(starts))

    def test_small_tracks_keep_a_draggable_thumb_and_scroll_direction(self):
        for size in (1, 2, 10):
            for virtual, window in ((100000, 1), (100, 99)):
                for position in (0, virtual - window):
                    with self.subTest(size=size, virtual=virtual, window=window, position=position):
                        cells = self.render_cells(size=size, virtual_size=virtual, window_size=window, position=position)
                        actions = [cell.style.meta.get("@mouse.down") for cell in cells]
                        self.assertIn("grab", actions)
                        self.assertEqual(len(cells), size)
                        if size > 1:
                            self.assertIn("scroll_down" if position == 0 else "scroll_up", actions)

    def test_horizontal_and_vertical_thickness_and_renderer_integration(self):
        console = Console(width=12, height=12)
        for vertical in (True, False):
            with self.subTest(vertical=vertical):
                width, height = (3, 12) if vertical else (12, 3)
                renderer = SolidScrollBarRender(
                    virtual_size=60, window_size=20, position=20,
                    thickness=3, vertical=vertical, style=Style(color="#ffbb66", bgcolor="#243746"),
                )
                lines = console.render_lines(renderer, console.options.update(width=width, height=height))
                self.assertEqual(len(lines), height)
                self.assertTrue(all(sum(segment.cell_length for segment in line) == width for line in lines))
                colors = [[segment.style.bgcolor for segment in line for _ in segment.text] for line in lines]
                if vertical:
                    self.assertTrue(all(len(set(row)) == 1 for row in colors))
                    self.assertEqual(sum(row[0] == Color.parse("#ffbb66") for row in colors), 4)
                else:
                    self.assertTrue(all(row == colors[0] for row in colors))
                    self.assertEqual(colors[0].count(Color.parse("#ffbb66")), 4)

    def test_non_scrollable_or_empty_track_has_no_interactive_thumb(self):
        for virtual, window in ((100, 100), (100, 101), (0, 0), (100, 0)):
            with self.subTest(virtual=virtual, window=window):
                cells = self.render_cells(size=12, virtual_size=virtual, window_size=window, position=200)
                self.assertEqual(len(cells), 12)
                self.assertTrue(all(not cell.style.meta for cell in cells))
                self.assertTrue(all(cell.style.bgcolor == Color.parse("#555555") for cell in cells))
        self.assertEqual(self.render_cells(size=0), [])
        self.assertEqual(self.render_cells(size=-1), [])
        self.assertEqual(self.render_cells(thickness=0), [])


if __name__ == "__main__":
    unittest.main()

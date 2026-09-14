import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from acprof.cli.tui import main
from acprof.tui.app import AcprofTui


class TuiColorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings_path = Path(temporary.name) / "tui.json"

    def render_surface(self, app):
        # Inspect emitted terminal codes, not SVG export (which always uses RGB).
        surface = app.get_css_variables()["surface"]
        return Strip([Segment(" ", Style(bgcolor=surface))]).render(app.console)

    def test_missing_colorterm_keeps_the_same_rgb_surface_as_vscode(self):
        for colorterm in (None, "", "truecolor", "24bit"):
            with self.subTest(colorterm=colorterm), patch.dict(
                os.environ, {"TERM": "xterm-256color"}, clear=True,
            ):
                if colorterm is not None:
                    os.environ["COLORTERM"] = colorterm
                original_environment = dict(os.environ)
                app = AcprofTui(settings_path=self.settings_path)
                self.assertIn("\x1b[48;2;28;46;59m", self.render_surface(app))
                self.assertEqual(dict(os.environ), original_environment)

    def test_256_color_mode_uses_extended_palette_even_in_truecolor_terminal(self):
        with patch.dict(os.environ, {"TERM": "xterm-256color", "COLORTERM": "truecolor"}, clear=True):
            app = AcprofTui(settings_path=self.settings_path, color_system="256")
            self.assertIn("\x1b[48;5;17m", self.render_surface(app))
            self.assertNotIn("48;2;", self.render_surface(app))

    def test_auto_mode_retains_terminal_capability_detection(self):
        for colorterm, expected in (("", "\x1b[48;5;17m"), ("truecolor", "\x1b[48;2;28;46;59m")):
            with self.subTest(colorterm=colorterm), patch.dict(
                os.environ, {"TERM": "xterm-256color", "COLORTERM": colorterm}, clear=True,
            ):
                app = AcprofTui(settings_path=self.settings_path, color_system="auto")
                self.assertIn(expected, self.render_surface(app))

    def test_cli_color_mode_applies_before_rendering(self):
        for mode, expected in (("truecolor", "\x1b[48;2;28;46;59m"), ("256", "\x1b[48;5;17m")):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"TERM": "xterm-256color"}, clear=True,
            ), patch("acprof.tui.app.default_settings_path", return_value=self.settings_path), patch.object(
                AcprofTui, "run", autospec=True,
            ) as run:
                main(["--color-system", mode])
                run.assert_called_once()
                self.assertIn(expected, self.render_surface(run.call_args.args[0]))


class TuiMonochromeTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_color_still_converts_the_rendered_frame_to_grayscale(self):
        for no_color in (False, True):
            environment = {"TERM": "xterm-256color"}
            if no_color:
                environment["NO_COLOR"] = "1"
            with self.subTest(no_color=no_color), tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, environment, clear=True,
            ):
                app = AcprofTui(settings_path=Path(directory) / "tui.json")
                async with app.run_test(size=(105, 27)) as pilot:
                    await pilot.pause()
                    colors = {
                        color.get_truecolor()
                        for strip in app.screen._compositor.render_strips()
                        for segment in strip if segment.style
                        for color in (segment.style.color, segment.style.bgcolor) if color
                    }
                    self.assertTrue(colors)
                    self.assertEqual(all(len(set(rgb)) == 1 for rgb in colors), no_color)


if __name__ == "__main__":
    unittest.main()

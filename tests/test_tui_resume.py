from pathlib import Path
import tempfile
import unittest

from textual.widgets import Checkbox, ContentSwitcher

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig, build_run_command


class TuiResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_checkbox_reaches_run_command_at_supported_sizes(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                app = AcprofTui(RunConfig.smoke("org/model"), settings_path=Path(directory) / "tui.json")
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
                    await pilot.pause()
                    checkbox = app.query_one("#resume-run", Checkbox)
                    checkbox.scroll_visible(animate=False, immediate=True)
                    await pilot.pause()
                    self.assertFalse(checkbox.value)
                    self.assertTrue(await pilot.click(checkbox))
                    await pilot.pause()
                    config = app._collect_config()
                    command = build_run_command(config, project_dir=Path.cwd())
                    self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()

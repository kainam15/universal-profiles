"""The model interface declaration must survive form input, commands and settings."""
from pathlib import Path
import tempfile
import unittest

from textual.widgets import ContentSwitcher, Input, Select

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig, TuiConfigError, build_probe_command, build_run_command
from acprof.tui.settings import TuiSettings, load_settings, save_settings


class TuiModelSpecTests(unittest.IsolatedAsyncioTestCase):
    async def test_declaration_input_reaches_run_probe_and_saved_defaults(self):
        root = Path(__file__).resolve().parents[1]
        path = "examples/multimodal/ultravox.model.json"
        for size in ((80, 24), (120, 30), (150, 45)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                settings_path = Path(directory) / "tui.json"
                app = AcprofTui(RunConfig.smoke("example/custom"), settings_path=settings_path)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
                    field = app.query_one("#model-spec", Input)
                    field.scroll_visible(animate=False, immediate=True)
                    await pilot.pause()
                    self.assertTrue(await pilot.click(field))
                    await pilot.press(*path)
                    self.assertEqual(field.value, path)
                    app.query_one("#ui-language", Select).value = "en"
                    await pilot.pause()
                    self.assertEqual(app.query_one("#model-spec", Input).value, path)
                    config = app._collect_config()
                    for builder in (build_run_command, build_probe_command):
                        command = builder(config, project_dir=root)
                        self.assertEqual(command[command.index("--model-spec") + 1], path)
                    save_settings(settings_path, TuiSettings(run_defaults=config), root)
                    restored, warning = load_settings(settings_path, root)
                    self.assertEqual(warning, "")
                    self.assertEqual(restored.run_defaults.model_spec, path)
                    app._apply_config(restored.run_defaults)
                    self.assertEqual(field.value, path)

    def test_missing_declaration_is_reported_before_command_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(model="example/custom", model_spec="missing.json")
            with self.assertRaisesRegex(TuiConfigError, "missing.json"):
                build_run_command(config, project_dir=Path(directory))


if __name__ == "__main__":
    unittest.main()

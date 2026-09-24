"""First-run presets must reach the real command without changing saved defaults."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import Select

from acprof.cli.tui import main
from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig, build_run_command
from acprof.tui.settings import TuiSettings, save_settings


class TuiOnboardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_selecting_smoke_builds_basic_cpu_command_without_notifications(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = AcprofTui(RunConfig(model="google-bert/bert-base-uncased"),
                            settings_path=Path(temporary) / "settings.json")
            async with app.run_test(size=(120, 30)) as pilot:
                await pilot.pause()
                preset = app.query_one("#run-preset", Select)
                preset.focus()
                await pilot.press("enter", "home", "down", "enter")
                await pilot.pause()
                self.assertEqual(preset.value, "smoke")
                command = build_run_command(app._collect_config(), project_dir=Path.cwd())
                for option, value in (("--profiling-mode", "basic"), ("--gpus", "off"),
                                      ("--notify", "none"), ("--repeat-in-window", "1")):
                    self.assertEqual(command[command.index(option) + 1], value)

    def test_cli_first_run_overrides_saved_full_matrix_without_rewriting_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            save_settings(path, TuiSettings(run_defaults=RunConfig(model="saved/model")), Path.cwd())
            original = path.read_bytes()
            with patch("acprof.tui.app.default_settings_path", return_value=path), patch(
                "textual.app.App.run", autospec=True,
            ) as run:
                main(["--model", "google-bert/bert-base-uncased", "--preset", "smoke",
                      "--output-dir", "results/first run"])
            app = run.call_args.args[0]
            command = build_run_command(app.initial_config, project_dir=Path.cwd())
            self.assertEqual(command[command.index("--profiling-mode") + 1], "basic")
            self.assertEqual(command[command.index("--output-dir") + 1], "results/first run")
            self.assertEqual(path.read_bytes(), original)

    def test_starting_without_overrides_preserves_saved_full_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            config = RunConfig(model="saved/model", cpus="2,4", output_dir="results/saved")
            save_settings(path, TuiSettings(run_defaults=config), Path.cwd())
            with patch("acprof.tui.app.default_settings_path", return_value=path), patch(
                "textual.app.App.run", autospec=True,
            ) as run:
                main([])
            self.assertEqual(run.call_args.args[0].initial_config, config)

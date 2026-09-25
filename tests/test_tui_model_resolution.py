"""Resolution UI asks for gaps only and hands probes to the managed subprocess."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import Button, Select, Static

from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig
import test_model_contract as fixture


class TuiModelResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_uses_managed_process_and_preserves_reviewed_revision(self):
        task = fixture.ModelContractTests().discover()
        with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
            "acprof.host.env_utils.bootstrap_project_env",
        ):
            app = AcprofTui(RunConfig(model=task.model_id, output_dir=directory), settings_path=Path(directory, "settings.json"))
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#ui-language", Select).value = "en"
                button = app.query_one("#inspect-model", Button)
                button.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                await pilot.click(button)
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(len(list(app.screen.query(".resolution-answer"))), 0)
                self.assertEqual(str(app.screen.query_one("#resolution-use", Button).label), "Use contract")
                await pilot.resize_terminal(120, 30)
                with patch.object(app, "_launch") as launch:
                    self.assertTrue(await pilot.click("#resolution-basic"))
                    await pilot.pause()
                command = launch.call_args.args[0].command
                self.assertEqual(command[command.index("--probe") + 1], "basic")
                self.assertEqual(command[command.index("--expected-revision") + 1], task.model_revision)
                self.assertNotIn("--gpus", command)

    async def test_only_unknown_input_is_editable_and_export_reaches_run_form(self):
        source = fixture.SOURCE.replace('audio = inputs.get("audio", None)', 'audio = inputs.get("audio", None)\n        speaker = inputs["speaker"]')
        task = fixture.ModelContractTests().discover(source)
        for size in ((80, 24), (120, 30), (150, 45)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory, patch(
                "acprof.host.detect.detect_task", return_value=task,
            ), patch("acprof.host.env_utils.bootstrap_project_env"):
                app = AcprofTui(RunConfig(model=task.model_id, output_dir=directory), settings_path=Path(directory, "settings.json"))
                async with app.run_test(size=size) as pilot:
                    button = app.query_one("#inspect-model", Button)
                    button.scroll_visible(animate=False, immediate=True)
                    await pilot.pause()
                    self.assertTrue(await pilot.click(button))
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    from acprof.tui.input import BarCursorInput
                    fields = list(app.screen.query(".resolution-answer"))
                    self.assertEqual(len(fields), 1)
                    field = fields[0]
                    self.assertIsInstance(field, BarCursorInput)
                    self.assertIn("speaker", str(app.screen.query_one("#resolution-body", Static).render()))
                    field.scroll_visible(animate=False, immediate=True)
                    await pilot.pause()
                    await pilot.click(field)
                    await pilot.press(*'{"literal":"narrator"}')
                    self.assertTrue(await pilot.click("#resolution-apply"))
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    self.assertEqual(len(list(app.screen.query(".resolution-answer"))), 0)
                    self.assertTrue(await pilot.click("#resolution-use"))
                    await pilot.pause()
                    path = app.query_one("#model-spec", BarCursorInput).value
                    spec = json.loads(Path(path).read_text())
                    self.assertEqual(spec["multimodal"]["inputs"]["speaker"], {"literal": "narrator"})
                    self.assertFalse(app._is_busy())

    async def test_measurement_disables_resolution(self):
        from acprof.tui.progress import ProgressSnapshot
        with tempfile.TemporaryDirectory() as directory:
            app = AcprofTui(RunConfig(model="fixture/model"), settings_path=Path(directory, "settings.json"))
            async with app.run_test(size=(80, 24)) as pilot:
                app._latest_snapshot = ProgressSnapshot(measurement_active=True)
                app._set_busy(True)
                await pilot.pause()
                self.assertTrue(app.query_one("#inspect-model", Button).disabled)

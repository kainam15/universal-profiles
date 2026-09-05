from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acprof.cli.tui_core import RunConfig
from acprof.cli.tui_settings import (
    TuiSettings,
    UiPreferences,
    default_settings_path,
    load_settings,
    save_settings,
)


class TuiSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.path = self.project / "preferences" / "tui.json"

    def write_payload(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_settings_path_uses_xdg_and_resolved_project_identity(self):
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.project / "xdg")}):
            first = default_settings_path(self.project)
            equivalent = default_settings_path(self.project / "child" / "..")
            other = default_settings_path(self.project / "another-project")
        self.assertEqual(first, equivalent)
        self.assertNotEqual(first, other)
        self.assertEqual(first.parent.parent, self.project / "xdg" / "acprof")
        self.assertEqual(first.name, "tui.json")
        self.assertFalse(first.exists())

    def test_empty_or_relative_xdg_uses_home_config(self):
        for configured in ("", "relative/path"):
            with self.subTest(configured=configured):
                with patch.dict(os.environ, {"XDG_CONFIG_HOME": configured}):
                    with patch("pathlib.Path.home", return_value=self.project):
                        path = default_settings_path(self.project)
                self.assertEqual(path.parent.parent, self.project / ".config" / "acprof")

    def test_first_launch_is_read_only_and_has_no_experiment_defaults(self):
        settings, warning = load_settings(self.path, self.project)
        self.assertEqual(settings, TuiSettings())
        self.assertIsNone(settings.run_defaults)
        self.assertEqual(warning, "")
        self.assertFalse(self.path.parent.exists())

    def test_roundtrip_ui_and_explicit_experiment_defaults(self):
        settings = TuiSettings(
            ui=UiPreferences(
                theme="acprof-light", log_wrap=False,
                log_max_lines=1000, show_command_bar=False,
            ),
            run_defaults=RunConfig.smoke("  demo/model  "),
        )
        save_settings(self.path, settings, self.project)
        restored, warning = load_settings(self.path, self.project)
        self.assertEqual(warning, "")
        self.assertEqual(restored.ui, settings.ui)
        self.assertEqual(restored.run_defaults, replace(settings.run_defaults, model="demo/model"))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_empty_model_is_valid_for_defaults_but_other_validation_remains(self):
        settings = TuiSettings(run_defaults=RunConfig(model="  ", cpus="1, 2", gpus="OFF"))
        save_settings(self.path, settings, self.project)
        restored, warning = load_settings(self.path, self.project)
        self.assertEqual(warning, "")
        self.assertEqual(restored.run_defaults.model, "")
        self.assertEqual(restored.run_defaults.cpus, "1,2")
        self.assertEqual(restored.run_defaults.gpus, "off")
        previous = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "CPU 列表"):
            save_settings(self.path, TuiSettings(run_defaults=RunConfig(cpus="0")), self.project)
        self.assertEqual(self.path.read_bytes(), previous)

    def test_malformed_file_warns_without_overwriting_source(self):
        self.path.parent.mkdir()
        for raw in (b"{broken", b"\xff\xfe"):
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                settings, warning = load_settings(self.path, self.project)
                self.assertEqual(settings, TuiSettings())
                self.assertIn("已使用默认值", warning)
                self.assertEqual(self.path.read_bytes(), raw)

    def test_bad_types_and_values_fall_back_without_coercion(self):
        bad_payloads = (
            [],
            {"version": True},
            {"version": 2},
            {"ui": []},
            {"ui": {"theme": "unknown"}},
            {"ui": {"log_wrap": "false"}},
            {"ui": {"show_command_bar": 0}},
            {"ui": {"log_max_lines": True}},
            {"ui": {"log_max_lines": 3000.0}},
            {"ui": {"log_max_lines": 5}},
            {"run_defaults": []},
            {"run_defaults": {"model": 5}},
            {"run_defaults": {"skip_build": "false"}},
            {"run_defaults": {"repeat": True}},
            {"run_defaults": {"repeat": 1.5}},
            {"run_defaults": {"sample_hz": True}},
            {"run_defaults": {"sample_hz": float("inf")}},
            {"run_defaults": {"sample_hz": 10 ** 400}},
            {"run_defaults": {"request_timeout_seconds": 0}},
            {"run_defaults": {"workload_spec": "missing-manifest.json"}},
            {"run_defaults": {"token": "do-not-store"}},
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                self.write_payload(payload)
                restored, warning = load_settings(self.path, self.project)
                self.assertEqual(restored, TuiSettings())
                self.assertTrue(warning)

    def test_unknown_top_level_fields_are_discarded_and_not_saved(self):
        self.write_payload({"ui": {"theme": "acprof-light"}, "token": "do-not-store"})
        settings, warning = load_settings(self.path, self.project)
        self.assertEqual(warning, "")
        self.assertEqual(settings.ui.theme, "acprof-light")
        save_settings(self.path, settings, self.project)
        self.assertNotIn("token", self.path.read_text(encoding="utf-8"))
        self.assertNotIn("do-not-store", self.path.read_text(encoding="utf-8"))

    def test_save_rejects_invalid_dataclasses_before_creating_file(self):
        invalid = (
            TuiSettings(ui=UiPreferences(log_wrap=1)),
            TuiSettings(ui=UiPreferences(log_max_lines=3)),
            TuiSettings(run_defaults=RunConfig(skip_build="false")),
            TuiSettings(run_defaults=RunConfig(repeat=True)),
            TuiSettings(run_defaults=RunConfig(sample_hz=10 ** 400)),
        )
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    save_settings(self.path, settings, self.project)
                self.assertFalse(self.path.parent.exists())

    def test_failed_atomic_replace_preserves_old_file_and_removes_temp(self):
        save_settings(self.path, TuiSettings(), self.project)
        previous = self.path.read_bytes()
        with patch("acprof.cli.tui_settings.os.replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                save_settings(
                    self.path, TuiSettings(ui=UiPreferences(theme="acprof-light")), self.project
                )
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_read_io_error_returns_default_and_warning(self):
        self.path.mkdir(parents=True)
        settings, warning = load_settings(self.path, self.project)
        self.assertEqual(settings, TuiSettings())
        self.assertIn("已使用默认值", warning)


if __name__ == "__main__":
    unittest.main()

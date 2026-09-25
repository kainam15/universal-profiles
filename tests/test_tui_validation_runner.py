"""验证辅助入口能挂载 TUI、执行子进程并退出，不启动真实采集。"""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from acprof.tui.commands import RunConfig


ROOT = Path(__file__).resolve().parents[1]


class TuiValidationRunnerTests(unittest.TestCase):
    def run_validation(self, child, *, returncode=0, expected="validation child finished", runner=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = replace(RunConfig.smoke("test/model"), output_dir=str(output / "results"), notify="none")
            command = output / "command.json"
            command.write_text(json.dumps({"config": asdict(config), "command": child}))
            result = subprocess.run([
                sys.executable, *(["-c", runner] if runner else []),
                str(ROOT / "scripts/run_tui_validation.py"), str(command),
            ], cwd=ROOT, env={**os.environ, "ACPROF_WECOM_WEBHOOK_URL": ""},
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, returncode, result.stderr)
            self.assertNotIn("DuplicateKey", result.stderr)
            self.assertTrue((output / "tui-finished.svg").is_file(), result.stderr)
            rendered = " ".join(ET.parse(output / "tui-finished.svg").getroot().itertext()).replace("\u00a0", " ")
            self.assertIn(expected, rendered)

    def test_headless_runner_mounts_once_and_records_completed_screen(self):
        self.run_validation([sys.executable, "-c", "print('validation child finished')"])

    def test_completion_before_monitor_refresh_keeps_output_and_exit_code(self):
        # Complete in the launch callback, before Textual can render the monitor.
        runner = '''import runpy, sys
from unittest.mock import patch
from acprof.tui.app import AcprofTui

def finish_immediately(self, command, kind):
    self._consume_process_line("validation child finished", None, False)
    self._process_finished(kind, int(command[-1]), None, "")

sys.argv = sys.argv[1:]
with patch.object(AcprofTui, "_execute_command", finish_immediately):
    runpy.run_path(sys.argv[0], run_name="__main__")
'''
        for returncode in (0, 7):
            with self.subTest(returncode=returncode):
                self.run_validation(["unused", str(returncode)], returncode=returncode, runner=runner)

    def test_finished_snapshot_selects_monitor_after_pending_focus(self):
        runner = '''import runpy, sys
from unittest.mock import patch
from acprof.tui.app import AcprofTui

def finish_after_focus(self, command, kind):
    self._consume_process_line("validation child finished", None, False)

    def pending_focus():
        self._activate_tab("run-tab")
        self.call_after_refresh(self._process_finished, kind, 0, None, "")

    self.call_after_refresh(pending_focus)

sys.argv = sys.argv[1:]
with patch.object(AcprofTui, "_execute_command", finish_after_focus):
    runpy.run_path(sys.argv[0], run_name="__main__")
'''
        self.run_validation(["unused"], runner=runner)

    def test_launch_error_is_rendered_before_nonzero_exit(self):
        self.run_validation([str(ROOT / "missing-validation-child")], returncode=1,
                            expected="FileNotFoundError")

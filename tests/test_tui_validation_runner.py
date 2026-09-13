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
    def test_headless_runner_mounts_once_and_records_completed_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = replace(RunConfig.smoke("test/model"), output_dir=str(output / "results"), notify="none")
            command = output / "command.json"
            command.write_text(json.dumps({"config": asdict(config), "command": [
                sys.executable, "-c", "print('validation child finished')",
            ]}))
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/run_tui_validation.py"), str(command),
            ], cwd=ROOT, env={**os.environ, "ACPROF_WECOM_WEBHOOK_URL": ""},
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("DuplicateKey", result.stderr)
            self.assertTrue((output / "tui-finished.svg").is_file(), result.stderr)
            rendered = " ".join(ET.parse(output / "tui-finished.svg").getroot().itertext()).replace("\u00a0", " ")
            self.assertIn("validation child finished", rendered)

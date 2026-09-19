"""离线接口验证可指定扩展测试，避免把任务族等同于运行时。"""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.check_runtime import main


class RuntimeCheckSelectionTests(unittest.TestCase):
    def test_explicit_patterns_override_family_default_and_are_recorded(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as run:
            code = main(['--profile', 'onnxruntime-cpu', '--test-pattern', 'test_custom_runtime.py',
                         '--test-pattern', 'test_custom_validation.py', '--output-dir', temporary])
            self.assertEqual(code, 0)
            command = run.call_args.args[0]
            self.assertIn('test_custom_runtime.py', command)
            self.assertIn('test_custom_validation.py', command)
            self.assertNotIn('test_structured_runtime.py', command)
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            self.assertEqual(result['test_patterns'], ['test_custom_runtime.py', 'test_custom_validation.py'])

    def test_build_only_cannot_claim_test_pattern_validation(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image',
        ) as build, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--profile', 'onnxruntime-cpu', '--build-only', '--test-pattern', 'test_custom.py',
                  '--output-dir', temporary])
        self.assertEqual(error.exception.code, 2)
        build.assert_not_called()


if __name__ == '__main__':
    unittest.main()

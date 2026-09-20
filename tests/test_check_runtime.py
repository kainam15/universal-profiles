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

from scripts.check_runtime import ONNX_ENVIRONMENT_CHECK, main


class RuntimeCheckSelectionTests(unittest.TestCase):
    def test_onnx_environment_guard_rejects_installed_torch_or_transformers(self):
        for forbidden in ('torch', 'transformers'):
            with self.subTest(package=forbidden), patch('importlib.util.find_spec',
                    side_effect=lambda name: object() if name == forbidden else None):
                with self.assertRaisesRegex(AssertionError, forbidden + ' must not be installed'):
                    exec(ONNX_ENVIRONMENT_CHECK, {})

    def test_onnx_environment_guard_rejects_missing_dependencies(self):
        with patch('importlib.util.find_spec', return_value=None):
            with self.assertRaisesRegex(AssertionError, 'required ONNX dependency missing: onnx'):
                exec(ONNX_ENVIRONMENT_CHECK, {})

    def test_timeout_fails_and_removes_named_container(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', side_effect=[
            subprocess.TimeoutExpired(['docker', 'run'], 1), subprocess.CompletedProcess([], 0),
        ]) as run:
            self.assertEqual(main(['--profile', 'onnxruntime-cpu', '--timeout-seconds', '1',
                                   '--output-dir', temporary]), 1)
            first = run.call_args_list[0].args[0]
            name = first[first.index('--name') + 1]
            self.assertEqual(run.call_args_list[-1].args[0], ['docker', 'rm', '-f', name])
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            self.assertFalse(result['successful'])
            self.assertIn('timeout', result['error'])

    def test_failed_required_tests_block_e2e_and_success(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})

        def execute(command, **kwargs):
            return subprocess.CompletedProcess(command, 1 if 'scripts/run_tests.py' in command else 0)

        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', side_effect=execute), patch(
            'scripts.check_onnx_basic.run_basic_e2e',
        ) as e2e:
            self.assertEqual(main(['--profile', 'onnxruntime-cpu', '--basic-e2e',
                                   '--output-dir', temporary]), 1)
            e2e.assert_not_called()
            self.assertFalse(json.loads((Path(temporary) / 'runtime.json').read_text())['successful'])

    def test_cleanup_timeout_preserves_failure_report(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', side_effect=[
            subprocess.TimeoutExpired(['docker', 'run'], 1),
            subprocess.TimeoutExpired(['docker', 'rm'], 30),
        ]):
            self.assertEqual(main(['--profile', 'onnxruntime-cpu', '--output-dir', temporary]), 1)
            result = json.loads((Path(temporary) / 'runtime.json').read_text())
            self.assertFalse(result['successful'])
            self.assertIn('validation timeout', result['error'])
            self.assertFalse(result['cleanup']['successful'])

    def test_basic_e2e_requires_all_three_task_families_to_succeed(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', return_value=subprocess.CompletedProcess([], 0)), patch(
            'scripts.check_onnx_basic.run_basic_e2e',
            side_effect=lambda *args, scenario='tabular', **kwargs: {'successful': scenario != 'image'},
        ) as e2e:
            self.assertEqual(main(['--profile', 'onnxruntime-cpu', '--basic-e2e',
                                   '--output-dir', temporary]), 1)
            self.assertEqual([call.kwargs['scenario'] for call in e2e.call_args_list],
                             ['tabular', 'image', 'text'])

    def test_onnx_profile_defaults_to_required_onnx_tests_and_forbids_torch(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        for profile in ('onnxruntime-cpu', 'onnxruntime-cv-cpu', 'onnxruntime-nlp-cpu'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary, patch(
                'scripts.check_runtime.prepare_environment_image', return_value=image,
            ), patch('scripts.check_runtime.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as run:
                code = main(['--profile', profile, '--output-dir', temporary])
                self.assertEqual(code, 0)
                commands = [call.args[0] for call in run.call_args_list]
                tests = next(command for command in commands if 'scripts/run_tests.py' in command)
                for pattern in ('test_onnx_runtime_optional.py', 'test_onnx_tasks_runtime.py',
                                'test_request_completion.py', 'test_onnx_server_runtime.py'):
                    self.assertIn(pattern, tests)
                self.assertIn('--require-no-skips', tests)
                guard = next(command for command in commands if '-c' in command)
                self.assertIn("'torch'", guard[-1])
                self.assertIn("'transformers'", guard[-1])
                self.assertIn("'onnxruntime'", guard[-1])

    def test_each_onnx_profile_build_only_records_dependencies_without_claiming_tests(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        for profile in ('onnxruntime-cpu', 'onnxruntime-cv-cpu', 'onnxruntime-nlp-cpu'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary, patch(
                'scripts.check_runtime.prepare_environment_image', return_value=image,
            ), patch('scripts.check_runtime.subprocess.run') as run:
                self.assertEqual(main(['--profile', profile, '--build-only', '--output-dir', temporary]), 0)
                run.assert_not_called()
                result = json.loads((Path(temporary) / 'runtime.json').read_text())
                self.assertTrue(result['successful'])
                self.assertEqual(result['validation_scope'], 'dependencies')
                self.assertEqual(result['test_patterns'], [])
                self.assertIsNone(result['device'])

    def test_explicit_patterns_override_family_default_and_are_recorded(self):
        image = SimpleNamespace(image_id='sha256:' + 'a' * 64, name='locked-env',
                                platform_image_id='sha256:' + 'b' * 64, manifest={})
        with tempfile.TemporaryDirectory() as temporary, patch(
            'scripts.check_runtime.prepare_environment_image', return_value=image,
        ), patch('scripts.check_runtime.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as run:
            code = main(['--profile', 'onnxruntime-cpu', '--test-pattern', 'test_custom_runtime.py',
                         '--test-pattern', 'test_custom_validation.py', '--output-dir', temporary])
            self.assertEqual(code, 0)
            command = next(call.args[0] for call in run.call_args_list
                           if 'scripts/run_tests.py' in call.args[0])
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

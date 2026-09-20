"""Runtime thread requests are independent from Docker resource limits."""
import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class RuntimeSettingsTests(unittest.TestCase):
    def settings(self):
        return importlib.import_module('acprof.runtime_settings')

    def test_default_preserves_torch_and_onnx_thread_conventions(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = self.settings()
            self.assertEqual(settings.runtime_threads(), 0)
            params = settings.onnx_runtime_parameters()
            self.assertEqual(params['effective']['threads'], 1)
            self.assertEqual(params['effective']['providers'], ['CPUExecutionProvider'])
            self.assertIsNone(params['requested']['intra_op_threads'])

    def test_backend_specific_then_generic_then_legacy_priority(self):
        for environment, expected in [({'TORCH_NUM_THREADS': '3'}, 3),
                                      ({'TORCH_NUM_THREADS': '3', 'ACPROF_RUNTIME_THREADS': '2'}, 2),
                                      ({'TORCH_NUM_THREADS': '3', 'ACPROF_RUNTIME_THREADS': '2',
                                        'ACPROF_ONNX_INTRA_OP_THREADS': '1'}, 1)]:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(self.settings().onnx_runtime_parameters()['effective']['threads'], expected)

    def test_invalid_explicit_setting_is_not_silently_clamped(self):
        for key, value in [('ACPROF_RUNTIME_THREADS', '-1'), ('ACPROF_ONNX_INTRA_OP_THREADS', '0'),
                           ('ACPROF_ONNX_INTER_OP_THREADS', '1.5'), ('ACPROF_REQUEST_TIMEOUT_S', 'nan')]:
            with self.subTest(key=key), patch.dict(os.environ, {key: value}, clear=True):
                settings = self.settings()
                with self.assertRaisesRegex(ValueError, key):
                    settings.request_timeout_s() if 'TIMEOUT' in key else settings.onnx_runtime_parameters()

    def test_cpu_only_provider_never_silently_falls_back(self):
        with patch.dict(os.environ, {'ACPROF_ONNX_PROVIDERS': 'CUDAExecutionProvider,CPUExecutionProvider'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'provider|Provider'):
                self.settings().onnx_runtime_parameters()

    def test_only_explicit_runtime_environment_is_forwarded(self):
        with patch.dict(os.environ, {'ACPROF_RUNTIME_THREADS': '2', 'UNRELATED': 'ignore'}, clear=True):
            settings = self.settings()
            self.assertEqual(settings.runtime_environment(), {'ACPROF_RUNTIME_THREADS': '2'})
            self.assertEqual(settings.runtime_docker_env_args(), ['-e', 'ACPROF_RUNTIME_THREADS=2'])

    def test_restore_identity_keeps_absent_defaults_and_records_explicit_request(self):
        from acprof.host.run_state import run_options
        args = SimpleNamespace(cpus='1,2', mems='2', gpus='off')
        with patch.dict(os.environ, {}, clear=True):
            original = run_options(args)['measurement_environment']
        self.assertNotIn('ACPROF_RUNTIME_THREADS', original)
        with patch.dict(os.environ, {'ACPROF_RUNTIME_THREADS': '3'}, clear=True):
            requested = run_options(args)['measurement_environment']
        self.assertEqual(requested, {**original, 'ACPROF_RUNTIME_THREADS': '3'})

    def test_execution_probe_preserves_explicit_legacy_request_not_quota_default(self):
        from acprof.host.execution_profile import _without_compute_thread_env
        command = ['docker', 'run', '--cpus=2', '-e', 'TORCH_NUM_THREADS=2', 'image']
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_without_compute_thread_env(command), ['docker', 'run', '--cpus=2', 'image'])
        with patch.dict(os.environ, {'TORCH_NUM_THREADS': '3'}, clear=True):
            self.assertEqual(_without_compute_thread_env(command),
                             ['docker', 'run', '--cpus=2', '-e', 'TORCH_NUM_THREADS=3', 'image'])

    def test_unlimited_probe_timeout_is_preserved_by_completion_hook(self):
        from acprof.container.execution import complete_prediction
        captured = []
        runtime = SimpleNamespace(wait_for_completion=lambda context, output, **kwargs:
                                  captured.append(kwargs['timeout_s']) or output)
        with patch.dict(os.environ, {'ACPROF_REQUEST_TIMEOUT_S': 'none'}, clear=True):
            self.assertEqual(complete_prediction(runtime, {}, 'ready'), 'ready')
        self.assertEqual(captured, [None])

    def test_profiler_keeps_unlimited_completion_unless_explicitly_requested(self):
        from acprof.host.profiler_common import _base_docker_cmd
        from acprof.container.execution import complete_prediction
        task = SimpleNamespace(model_id='fixture', model_revision='main', task_family='structured',
                               pipeline_tag='tabular-regression', runtime_backend='onnxruntime',
                               runtime_profile_id='onnxruntime-cpu')
        for override, expected in ((None, None), ('600', 600.0)):
            environment = {} if override is None else {'ACPROF_REQUEST_TIMEOUT_S': override}
            with self.subTest(override=override), patch.dict(os.environ, environment, clear=True):
                command = _base_docker_cmd(task_info=task, image_tag='fixture', cpu=2, mem=2,
                                          use_gpu=False, payload_file='/tmp/plan.json',
                                          profile_root='/tmp/profiles', tool_mount_roots=())
            forwarded = dict(command[i + 1].split('=', 1) for i, part in enumerate(command)
                             if part == '-e')
            captured = []
            runtime = SimpleNamespace(wait_for_completion=lambda ctx, output, **kwargs:
                                      captured.append(kwargs['timeout_s']) or output)
            with patch.dict(os.environ, forwarded, clear=True):
                self.assertEqual(complete_prediction(runtime, {}, 'ready'), 'ready')
            self.assertEqual(captured, [expected])

    def test_service_inherits_requested_deadline_with_explicit_override_priority(self):
        from acprof.host.docker_runtime import ImageInfo, _start_container_session
        task = SimpleNamespace(model_id='fixture', model_revision='main', task_family='structured',
                               pipeline_tag='tabular-regression', runtime_backend='onnxruntime')
        for timeout, override, expected in ((600, None, '600'), (None, None, 'none'), (600, '120', '120')):
            commands = []
            environment = {} if override is None else {'ACPROF_REQUEST_TIMEOUT_S': override}
            with self.subTest(timeout=timeout, override=override), patch.dict(os.environ, environment, clear=True), patch(
                'acprof.host.docker_runtime._run', side_effect=lambda command, **kwargs:
                commands.append(command) or SimpleNamespace(returncode=0, stdout='', stderr=''),
            ), patch('requests.get', return_value=SimpleNamespace(status_code=200, json=lambda: {'status': 'ok'})):
                _start_container_session(task, 1, 1, 'off', ImageInfo(tag='fixture'), 'test-deadline', '[test]',
                                         request_timeout_seconds=timeout)
            command = next(command for command in commands if command[:3] == ['docker', 'run', '-d'])
            values = [item for item in command if item.startswith('ACPROF_REQUEST_TIMEOUT_S=')]
            self.assertEqual(values[-1], 'ACPROF_REQUEST_TIMEOUT_S=' + expected)


if __name__ == '__main__':
    unittest.main()

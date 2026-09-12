import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import ImageInfo
from acprof.host.runtime_validation import validate_runtime
from acprof.host.profiler_common import _base_docker_cmd


class RuntimeValidationTests(unittest.TestCase):
    def task(self):
        return TaskInfo('Example/model', 'audio-text-to-text', 'multimodal',
                        'transformers_model', 'transformers', 'a' * 40, 'unit',
                        runtime_profile_id='moss-transformers560')

    def fixture(self, root):
        plan = root / 'input_scale_plan.json'
        plan.write_text(json.dumps({'entries': [
            {'input_scale': 10, 'payload': {'text': 'large'}},
            {'input_scale': 1, 'payload': {'text': 'small'}},
        ]}))
        return dict(task_info=self.task(), image_info=ImageInfo(
            tag='sha256:' + 'b' * 64, runtime_environment={'build_fingerprint': 'build'},
        ), planned=SimpleNamespace(plan_file=str(plan)), cpu_list=[1, 4], mem_list=[2, 8],
                    gpu_list=['off', 'on'], output_dir=str(root), timeout_seconds=30)

    def test_both_devices_use_smallest_planned_payload_and_separate_containers(self):
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[:2] == ['docker', 'run']:
                self.assertIn('--network', command)
                self.assertIn('--memory=8g', command)
                mount = command[command.index('-v') + 1].split(':')[0]
                self.assertEqual(json.loads(Path(mount).read_text()), {'text': 'small'})
                return subprocess.CompletedProcess(command, 0, stdout='ACPROF_RUNTIME_VALIDATION={"status":"ok"}\n', stderr='')
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.subprocess.run', side_effect=run,
        ), patch('acprof.host.docker_runtime._inspect_container_state', return_value={}):
            root = Path(temporary)
            report = validate_runtime(**self.fixture(root))
            saved = json.loads((root / 'runtime_validation.json').read_text())
            self.assertEqual(saved, report)
            self.assertEqual(report['status'], 'ok')
            self.assertEqual(set(report['devices']), {'off', 'on'})
            self.assertEqual(report['input_scale'], 1)
            self.assertEqual(list(root.glob('*.csv')), [])
        runs = [c for c in commands if c[:2] == ['docker', 'run']]
        removals = [c for c in commands if c[:3] == ['docker', 'rm', '-f']]
        self.assertEqual(len(runs), 2)
        self.assertNotEqual(runs[0][3], runs[1][3])
        self.assertEqual({c[3] for c in runs}, {c[3] for c in removals})
        self.assertNotIn('--gpus', runs[0])
        self.assertIn('--gpus', runs[1])

    def test_load_failure_keeps_stderr_and_stops_before_next_device(self):
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 1, stdout='', stderr='ImportError: wrong Transformers version')

        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.subprocess.run', side_effect=run,
        ), patch('acprof.host.docker_runtime._inspect_container_state', return_value={}):
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, 'wrong Transformers'):
                validate_runtime(**self.fixture(root))
            self.assertIn('wrong Transformers', (root / 'runtime_validation_off.log').read_text())
            self.assertEqual(json.loads((root / 'runtime_validation.json').read_text())['status'], 'error')
            self.assertEqual(list(root.glob('*.csv')), [])
        self.assertEqual(sum(c[:2] == ['docker', 'run'] for c in commands), 1)

    def test_timeout_removes_owned_container(self):
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[:2] == ['docker', 'run']:
                raise subprocess.TimeoutExpired(command, 30, stderr=b'loading processor')
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.subprocess.run', side_effect=run,
        ):
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                validate_runtime(**self.fixture(Path(temporary)))
            self.assertIn('loading processor', (Path(temporary) / 'runtime_validation_off.log').read_text())
        self.assertEqual(commands[-1][:3], ['docker', 'rm', '-f'])

    def test_invalid_validation_response_keeps_report_and_cleans_container(self):
        result = subprocess.CompletedProcess([], 0, stdout='ACPROF_RUNTIME_VALIDATION=[]\n', stderr='')
        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.subprocess.run', return_value=result,
        ) as run, patch('acprof.host.docker_runtime._inspect_container_state', return_value={}):
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, 'validation response'):
                validate_runtime(**self.fixture(root))
            self.assertEqual(json.loads((root / 'runtime_validation.json').read_text())['status'], 'error')
        self.assertEqual(run.call_args.args[0][:3], ['docker', 'rm', '-f'])

    def test_cgroup_oom_is_resource_limit_not_dependency_failure(self):
        result = subprocess.CompletedProcess([], 137, stdout='', stderr='Killed')
        with tempfile.TemporaryDirectory() as temporary, patch(
            'acprof.host.runtime_validation.subprocess.run', return_value=result,
        ), patch('acprof.host.docker_runtime._inspect_container_state', return_value={'OOMKilled': True}):
            report = validate_runtime(**self.fixture(Path(temporary)))
        self.assertEqual(report['status'], 'resource_limited')
        self.assertEqual(set(report['devices']), {'off', 'on'})

    def test_managed_profilers_keep_image_adapter_code(self):
        command = _base_docker_cmd(
            task_info=self.task(), image_tag='sha256:' + 'b' * 64,
            cpu=1, mem=8, use_gpu=True, payload_file='/tmp/payload.json',
            profile_root='/tmp/profiles', tool_mount_roots=[],
        )
        self.assertFalse(any('/app/acprof' in item for item in command))
        self.assertIn('--gpus', command)

    def test_cli_stops_before_matrix_when_runtime_validation_fails(self):
        import sys
        from acprof.cli import run
        from acprof.host.input_plan import PlannedInputScales

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', [
                'run.py', '--model', self.task().model_id, '--cpus', '1', '--mems', '8',
                '--gpus', 'off,on', '--input-scales', '1', '--notify', 'none', '--output-dir', temporary,
            ]))
            for name in (
                'bootstrap_project_env', 'require_native_linux_host', 'require_native_docker',
                'require_packet_latency_prerequisites', 'require_cpu_energy_prerequisites',
                'require_mips_prerequisites',
            ):
                stack.enter_context(patch('acprof.cli.run.' + name))
            stack.enter_context(patch('acprof.cli.run.require_cgroup_prerequisites', return_value='v2'))
            stack.enter_context(patch('acprof.host.detect.detect_task', return_value=self.task()))
            stack.enter_context(patch('acprof.host.docker_runtime.prepare_image', return_value=ImageInfo(
                tag='sha256:' + 'b' * 64, runtime_environment={'build_fingerprint': 'build'},
            )))
            stack.enter_context(patch('acprof.host.static_metadata.collect_static_meta', return_value=SimpleNamespace()))
            stack.enter_context(patch('acprof.host.static_metadata.enrich_static_meta_from_input_plan', return_value=SimpleNamespace()))
            stack.enter_context(patch('acprof.host.static_metadata.write_static_meta_json'))
            stack.enter_context(patch('acprof.host.input_plan.plan_input_scales', return_value=PlannedInputScales(
                scales=[1.0], source='manual', plan_file='',
            )))
            validation = stack.enter_context(patch(
                'acprof.host.runtime_validation.validate_runtime', side_effect=RuntimeError('adapter load failed'),
            ))
            matrix = stack.enter_context(patch('acprof.host.orchestrator.run_matrix'))
            with self.assertRaises(SystemExit) as exited:
                run.main()
            self.assertEqual(exited.exception.code, 1)
            validation.assert_called_once()
            matrix.assert_not_called()
            self.assertFalse(list(Path(temporary).rglob('*.csv')))


if __name__ == '__main__':
    unittest.main()

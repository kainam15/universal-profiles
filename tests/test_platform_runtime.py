"""平台可以不含 Torch，运行时声明仍受完整制品锁约束。"""
import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acprof import runtime_profiles as profiles
from acprof.dependency_locks import content_digest, python_lock_text, read_python_lock
from acprof.host.detect import TaskInfo
from runtime_fixture import ROOT, copy_dependency_tree


class PlatformRuntimeTests(unittest.TestCase):
    def platform(self, root):
        records = [record for record in read_python_lock(ROOT / 'dockerfiles/locks/platform-cpu.txt')
                   if record['name'] in {'pip', 'wheel', 'packaging', 'setuptools'}]
        (root / 'python-only.txt').write_text(python_lock_text(records))
        return profiles.PlatformSpec('python-only', requirements_lock='python-only.txt')

    def test_platform_does_not_require_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_dependency_tree(root)
            platform = self.platform(root)
            identity = profiles.platform_identity(platform, root)
            self.assertNotIn('torch_index_url', identity)
            self.assertNotIn('torch', {record['name'] for record in identity['packages']})

    def test_existing_torch_platform_and_environment_identity_is_stable(self):
        self.assertEqual(content_digest(profiles.platform_identity(profiles.PLATFORMS['cpu'], ROOT)),
                         '160a6790e36ae6bdf22474eb063af40a0f085203e7c5c7742a9fa00e0c669de2')
        self.assertEqual(profiles.environment_id(profiles.ENVIRONMENTS['nlp-cpu'], ROOT),
                         '3b87fe3c394edc75f049363f28da2a6ffba01a4f1b5f313e228d1f2afb7889ac')

    def test_runtime_version_is_validated_without_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_dependency_tree(root)
            platform = self.platform(root)
            runtime = profiles.RuntimeSpec('example-runtime', '0.1', package='example-runtime')
            environment = profiles.DependencyEnvironment('example', platform, platform.requirements_lock,
                                                        runtime=runtime)
            with self.assertRaisesRegex(ValueError, 'example-runtime'):
                profiles.environment_identity(environment, root)

    def test_non_torch_environment_identity_ignores_lock_order_and_comments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_dependency_tree(root)
            platform = self.platform(root)
            runtime = profiles.RuntimeSpec('packaging-runtime', '26.3', package='packaging')
            environment = profiles.DependencyEnvironment('example', platform, platform.requirements_lock,
                                                        runtime=runtime)
            first = profiles.environment_id(environment, root)
            path = root / platform.requirements_lock
            lines = [line for line in path.read_text().splitlines() if not line.startswith('#')]
            path.write_text('# unchanged artifacts\n' + '\n'.join(reversed(lines)) + '\n')
            self.assertEqual(first, profiles.environment_id(environment, root))
            changed = dataclasses.replace(environment, runtime=dataclasses.replace(runtime, version='26.4'))
            with self.assertRaisesRegex(ValueError, 'packaging==26.4'):
                profiles.environment_identity(changed, root)

    def test_same_locked_environment_identity_is_independent_of_runtime_role(self):
        environment = profiles.ENVIRONMENTS['nlp-cpu']
        declared = dataclasses.replace(environment, runtime=profiles.RuntimeSpec('torch', '2.11.0+cpu'))
        self.assertEqual(profiles.environment_id(environment, ROOT), profiles.environment_id(declared, ROOT))

    def test_partial_legacy_torch_declaration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Torch'):
            profiles.PlatformSpec('invalid', torch_version='2.11.0+cpu', requirements_lock='unused')

    def test_onnx_environment_has_no_torch_or_cuda_dependencies(self):
        identity = profiles.environment_identity(profiles.ENVIRONMENTS['onnxruntime-cpu'], ROOT)
        names = {record['name'] for record in identity['packages']}
        self.assertIn('onnxruntime', names)
        self.assertNotIn('torch', names)
        self.assertFalse(any(name.startswith('nvidia-') for name in names))

    def test_non_torch_profile_does_not_probe_torch_driver_or_read_overrides(self):
        from acprof.host.runtime_images import configure_runtime_profile
        platform = profiles.PlatformSpec('python-only', requirements_lock='unused')
        task = TaskInfo(model_id='example/model', model_revision='a' * 40,
                        pipeline_tag='tabular-regression', task_family='structured',
                        runtime_backend='example', library_name='example', detection_method='test')
        for runtime_type in ('example', 'torch'):
            environment = profiles.DependencyEnvironment('example', platform, 'unused',
                                                         runtime=profiles.RuntimeSpec(runtime_type, '1.0'))
            profile = profiles.RuntimeProfile('example', 'structured', environment)
            with self.subTest(runtime_type=runtime_type), patch(
                'acprof.host.runtime_images.select_runtime_profile', return_value=profile,
            ), patch('acprof.host.docker_runtime._select_nlp_torch_index_url',
                     side_effect=AssertionError('generic platform must not select a legacy Torch index')):
                self.assertIs(configure_runtime_profile(task), profile)


if __name__ == '__main__':
    unittest.main()

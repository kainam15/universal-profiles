"""依赖和平台选择回归：构建身份必须覆盖锁与基础镜像。"""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from acprof.host.detect import TaskInfo
from acprof.host.dependency_images import runtime_fingerprint
from acprof.runtime_profiles import select_runtime_profile
from runtime_fixture import ROOT, copy_dependency_tree


class DependencyLockTests(unittest.TestCase):
    def task(self, family='nlp'):
        return TaskInfo(model_id='example/model', model_revision='a' * 40, pipeline_tag='fill-mask',
                        task_family=family, runtime_backend='transformers_pipeline',
                        library_name='transformers', detection_method='test')

    def test_lock_comments_do_not_change_dependency_cache_identity(self):
        environment = select_runtime_profile(self.task()).environment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            first = runtime_fingerprint(environment, root)
            lock = root / environment.requirements_lock
            lock.write_text('# Same dependency environment, reviewed again.\n' + lock.read_text())
            self.assertEqual(first, runtime_fingerprint(environment, root))

    def test_every_family_defaults_to_complete_lock_and_digest(self):
        for family in ('nlp', 'cv', 'audio', 'diffusion', 'structured', 'timeseries', 'multimodal'):
            with self.subTest(family=family):
                environment = select_runtime_profile(self.task(family)).environment
                self.assertIn('@sha256:', environment.platform.python_base_image)
                locked = (ROOT / environment.requirements_lock).read_text()
                self.assertIn('# torch==', locked)
                self.assertIn('# flask==3.0.2', locked)

    def test_driver_selection_preserves_cuda124_with_pinned_wheel(self):
        from acprof.host.runtime_images import configure_runtime_profile
        task = self.task()
        with patch('acprof.host.docker_runtime._select_nlp_torch_index_url',
                   return_value='https://download.pytorch.org/whl/cu124'):
            profile = configure_runtime_profile(task)
        self.assertEqual(profile.profile_id, 'nlp-cu124')
        self.assertEqual(select_runtime_profile(task), profile)
        self.assertIn('torch==2.6.0+cu124', (ROOT / profile.environment.requirements_lock).read_text())

    def test_shared_lock_conflict_is_rejected_before_image_reuse(self):
        environment = select_runtime_profile(self.task()).environment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            common = root / environment.platform.requirements_lock
            common.write_text(common.read_text().replace('filelock-3.32.6-', 'filelock-3.32.7-'))
            with self.assertRaisesRegex(ValueError, '父层'):
                runtime_fingerprint(environment, root)


if __name__ == '__main__':
    unittest.main()

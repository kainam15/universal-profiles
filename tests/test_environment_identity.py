"""依赖内容身份、平台边界和完整锁的行为回归。"""
import dataclasses
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from acprof.dependency_locks import read_python_lock, require_exact_packages
from acprof.host.dependency_images import runtime_fingerprint
from acprof.runtime_profiles import ENVIRONMENTS, PROFILES, environment_id

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(ROOT / 'dockerfiles', self.root / 'dockerfiles')
        (self.root / 'acprof').mkdir()
        shutil.copyfile(ROOT / 'acprof/dependency_locks.py', self.root / 'acprof/dependency_locks.py')
        self.env = ENVIRONMENTS['audio-cpu']

    def test_profiles_share_exact_environments_without_merging_near_matches(self):
        self.assertEqual(len(PROFILES), 25)
        self.assertEqual(len({environment_id(p.environment, ROOT) for p in PROFILES.values()}), 21)
        for profile in ('onnxruntime-cpu', 'onnxruntime-cv-cpu', 'onnxruntime-nlp-cpu'):
            self.assertIs(PROFILES[profile].environment, ENVIRONMENTS['onnxruntime-cpu'])
        for variant in ('cpu', 'cu124'):
            self.assertIs(PROFILES['audio-' + variant].environment,
                          PROFILES['multimodal-transformers4576-' + variant].environment)
        self.assertNotEqual(environment_id(ENVIRONMENTS['audio-cu128'], ROOT),
                            environment_id(ENVIRONMENTS['multimodal-transformers4576'], ROOT))

    def test_comments_order_and_filename_do_not_change_identity(self):
        original = environment_id(self.env, self.root)
        path = self.root / self.env.requirements_lock
        lines = [line for line in path.read_text().splitlines() if line and not line.startswith('#')]
        replacement = path.with_name('renamed.txt')
        replacement.write_text('# review only\n\n' + '\n'.join(reversed(lines)) + '\n')
        changed = dataclasses.replace(self.env, environment_key='renamed',
                                      requirements_lock=str(replacement.relative_to(self.root)))
        self.assertEqual(original, environment_id(changed, self.root))
        self.assertEqual(runtime_fingerprint(self.env, self.root), runtime_fingerprint(changed, self.root))

    def test_artifact_source_hash_or_version_changes_environment_identity(self):
        original = environment_id(self.env, self.root)
        path = self.root / self.env.requirements_lock
        source = path.read_text()
        entry = next(line for line in source.splitlines() if line.startswith('librosa @'))
        for changed in (entry.replace('librosa-0.11.0-', 'librosa-0.11.1-'),
                        entry.replace('files.pythonhosted.org', 'example.org'),
                        entry.rsplit(':', 1)[0] + ':' + '1' * 64):
            with self.subTest(changed=changed):
                path.write_text(source.replace(entry, changed))
                self.assertNotEqual(original, environment_id(self.env, self.root))
        path.write_text(source)

    def test_system_package_change_changes_environment_identity(self):
        original = environment_id(self.env, self.root)
        path = self.root / self.env.platform.system_lock
        lock = json.loads(path.read_text())
        # 基础镜像继承包也属于平台内容，不能只比较显式 apt 安装列表。
        inherited = next(key for key in lock['packages']
                         if key not in {f"{p['name']}:{p['architecture']}" for p in lock['artifacts']})
        lock['packages'][inherited] += '.changed'
        path.write_text(json.dumps(lock))
        self.assertNotEqual(original, environment_id(self.env, self.root))

    def test_build_recipe_changes_cache_but_not_environment_identity(self):
        original = environment_id(self.env, self.root)
        build = runtime_fingerprint(self.env, self.root)
        path = self.root / 'dockerfiles/runtime.Dockerfile'
        path.write_text(path.read_text() + '\n# build change\n')
        self.assertEqual(original, environment_id(self.env, self.root))
        self.assertNotEqual(build, runtime_fingerprint(self.env, self.root))

    def test_parent_dependency_cannot_be_removed_or_replaced(self):
        path = self.root / self.env.requirements_lock
        source = path.read_text()
        entry = next(line for line in source.splitlines() if line.startswith('torch @'))
        for replacement in ('', entry.replace('torch-2.11.0', 'torch-2.11.1')):
            path.write_text(source.replace(entry, replacement))
            with self.assertRaisesRegex(ValueError, '父层'):
                environment_id(self.env, self.root)

    def test_unhashed_or_unpinned_input_is_rejected(self):
        path = self.root / 'bad-lock.txt'
        for content in ('torch==2.11.0\n', '-r another.txt\n', '', 'torch>=2\n'):
            path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_python_lock(path)

    def test_missing_extra_or_changed_packages_are_rejected(self):
        for actual in ({}, {'torch': '2.11', 'hidden': '1'}, {'torch': '2.12'}):
            with self.subTest(actual=actual), self.assertRaisesRegex(ValueError, 'package set'):
                require_exact_packages({'torch': '2.11'}, actual)


if __name__ == '__main__':
    unittest.main()

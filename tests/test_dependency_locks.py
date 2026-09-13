"""依赖和环境选择回归：构建身份必须覆盖锁与基础镜像。"""
import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acprof.host.detect import TaskInfo
from acprof.host.runtime_images import runtime_fingerprint
from acprof.runtime_profiles import select_runtime_profile

ROOT = Path(__file__).resolve().parents[1]


class DependencyLockTests(unittest.TestCase):
    def task(self, family="nlp"):
        return TaskInfo(model_id="example/model", model_revision="a" * 40, pipeline_tag="fill-mask", task_family=family,
                        runtime_backend="transformers_pipeline", library_name="transformers",
                        detection_method="test")

    def test_every_family_defaults_to_complete_lock_and_digest(self):
        for family in ("nlp", "cv", "audio", "diffusion", "structured", "timeseries", "multimodal"):
            with self.subTest(family=family):
                profile = select_runtime_profile(self.task(family))
                self.assertTrue(profile.requirements_lock)
                self.assertIn("@sha256:", profile.python_base_image)
                locked = (ROOT / profile.requirements_lock).read_text()
                self.assertIn("torch==", locked)
                self.assertIn("flask==3.0.2", locked)

    def test_driver_selection_preserves_cuda124_with_pinned_wheel(self):
        from acprof.host.runtime_images import configure_runtime_profile
        task = self.task()
        with patch("acprof.host.docker_runtime._select_nlp_torch_index_url", return_value="https://download.pytorch.org/whl/cu124"):
            profile = configure_runtime_profile(task)
        self.assertEqual(profile.profile_id, "nlp-cu124")
        self.assertEqual(select_runtime_profile(task), profile)
        self.assertIn("torch==2.6.0+cu124", (ROOT / profile.requirements_lock).read_text())

    def test_shared_lock_changes_invalidate_runtime_image(self):
        profile = select_runtime_profile(self.task())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dockerfiles/locks").mkdir(parents=True)
            (root / profile.requirements_lock).write_text("torch==2.11.0+cu128\n")
            (root / "dockerfiles/runtime.Dockerfile").write_text("FROM python\n")
            common = root / "dockerfiles/locks/common-cu128.txt"
            common.write_text("numpy==2.2.6\n")
            first = runtime_fingerprint(profile, root)
            common.write_text("numpy==2.2.7\n")
            self.assertNotEqual(first, runtime_fingerprint(profile, root))


if __name__ == "__main__":
    unittest.main()

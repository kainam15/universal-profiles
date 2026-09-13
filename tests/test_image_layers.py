import dataclasses
import tempfile
import unittest
from pathlib import Path

from acprof.host.detect import TaskInfo
from acprof.host.runtime_images import build_fingerprint, model_fingerprint, runtime_fingerprint
from acprof.runtime_profiles import select_runtime_profile


class ImageLayerIdentityTests(unittest.TestCase):
    def test_code_changes_reuse_runtime_and_model_but_refresh_final_image(self):
        task = TaskInfo(model_id="example/bert", model_revision="a" * 40, pipeline_tag="fill-mask",
                        task_family="nlp", runtime_backend="transformers_pipeline", library_name="transformers",
                        detection_method="test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acprof/container").mkdir(parents=True)
            (root / "dockerfiles/locks").mkdir(parents=True)
            for relative, content in {
                "acprof/container/download_model.py": "downloader = 1\n",
                "acprof/container/model_files.py": "selector = 1\n",
                "acprof/handler.py": "handler = 1\n",
                "dockerfiles/base.Dockerfile": "FROM python:3.10-slim\n",
                "dockerfiles/nlp.Dockerfile": "FROM base AS runtime\nRUN install-deps\n\nFROM runtime AS model\nCOPY models /models\n",
                "dockerfiles/runtime-model.Dockerfile": "FROM runtime\nCOPY downloader /opt\n",
                "dockerfiles/runtime.Dockerfile": "FROM python\nRUN install-locked-deps\n",
                "dockerfiles/locks/nlp-cu128.txt": "torch==2.11.0+cu128\n",
                "dockerfiles/locks/common-cu128.txt": "torch==2.11.0+cu128\n",
            }.items():
                (root / relative).write_text(content)
            profile = select_runtime_profile(task)
            runtime = runtime_fingerprint(profile, root)
            model = model_fingerprint(task, "sha256:" + "b" * 64, root)
            final = build_fingerprint(task, root)
            (root / "acprof/handler.py").write_text("handler = 2\n")
            self.assertEqual(runtime, runtime_fingerprint(profile, root))
            self.assertEqual(model, model_fingerprint(task, "sha256:" + "b" * 64, root))
            self.assertNotEqual(final, build_fingerprint(task, root))
            # 筛选规则变更只影响模型和最终层，不重新安装依赖。
            (root / "acprof/container/model_files.py").write_text("selector = 2\n")
            self.assertEqual(runtime, runtime_fingerprint(profile, root))
            changed_model = model_fingerprint(task, "sha256:" + "b" * 64, root)
            self.assertNotEqual(model, changed_model)
            # 新 commit、下载策略或真实运行环境都不得复用旧模型层。
            for changed in (dataclasses.replace(task, model_revision="c" * 40),
                            dataclasses.replace(task, model_download_policy="full")):
                self.assertNotEqual(changed_model, model_fingerprint(changed, "sha256:" + "b" * 64, root))
            self.assertNotEqual(changed_model, model_fingerprint(task, "sha256:" + "c" * 64, root))
            self.assertNotEqual(runtime, runtime_fingerprint(profile, root, {"TORCH_INDEX_URL": "cpu"}))


if __name__ == "__main__":
    unittest.main()

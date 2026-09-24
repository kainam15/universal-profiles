import dataclasses
import tempfile
import json
import unittest
from pathlib import Path

from acprof.host.detect import TaskInfo
from acprof.host.runtime_images import request_fingerprint, model_fingerprint
from acprof.host.dependency_images import runtime_fingerprint
from runtime_fixture import copy_dependency_tree
from acprof.runtime_profiles import select_runtime_profile


class ImageLayerIdentityTests(unittest.TestCase):
    def test_code_changes_reuse_runtime_and_model_but_refresh_final_image(self):
        task = TaskInfo(model_id="example/bert", model_revision="a" * 40, pipeline_tag="fill-mask",
                        task_family="nlp", runtime_backend="transformers_pipeline", library_name="transformers",
                        detection_method="test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_dependency_tree(root)
            (root / "acprof/container").mkdir()
            for relative, content in {
                "acprof/container/download_model.py": "downloader = 1\n",
                "acprof/container/model_files.py": "selector = 1\n",
                "acprof/model_spec.py": "spec = 1\n",
                "acprof/handler.py": "handler = 1\n",
                "dockerfiles/runtime-model.Dockerfile": "FROM runtime\nCOPY downloader /opt\n",
                "dockerfiles/runtime.Dockerfile": "FROM python\nRUN install-locked-deps\n",
            }.items():
                (root / relative).write_text(content)
            profile = select_runtime_profile(task)
            runtime = runtime_fingerprint(profile.environment, root)
            model = model_fingerprint(task, "sha256:" + "b" * 64, root)
            final = request_fingerprint(task, root)
            (root / "acprof/handler.py").write_text("handler = 2\n")
            self.assertEqual(runtime, runtime_fingerprint(profile.environment, root))
            self.assertEqual(model, model_fingerprint(task, "sha256:" + "b" * 64, root))
            self.assertNotEqual(final, request_fingerprint(task, root))
            # 筛选规则与模型声明解析变更影响模型和最终层，不重新安装依赖。
            changed_model = model
            for relative, content in (
                ("acprof/container/model_files.py", "selector = 2\n"),
                ("acprof/model_spec.py", "spec = 2\n"),
            ):
                with self.subTest(source=relative):
                    previous_model = changed_model
                    previous_final = request_fingerprint(task, root)
                    (root / relative).write_text(content)
                    self.assertEqual(runtime, runtime_fingerprint(profile.environment, root))
                    changed_model = model_fingerprint(task, "sha256:" + "b" * 64, root)
                    self.assertNotEqual(previous_model, changed_model)
                    self.assertNotEqual(previous_final, request_fingerprint(task, root))
            # 新 commit、下载策略或真实运行环境都不得复用旧模型层。
            for changed in (dataclasses.replace(task, model_revision="c" * 40),
                            dataclasses.replace(task, model_download_policy="full")):
                self.assertNotEqual(changed_model, model_fingerprint(changed, "sha256:" + "b" * 64, root))
            self.assertNotEqual(changed_model, model_fingerprint(task, "sha256:" + "c" * 64, root))
            changed_platform = dataclasses.replace(profile.environment.platform, python_base_image="python:3.10-slim@sha256:" + "c" * 64)
            lock = root / changed_platform.system_lock
            data = json.loads(lock.read_text())
            data["base_image"] = changed_platform.python_base_image
            lock.write_text(json.dumps(data))
            self.assertNotEqual(runtime, runtime_fingerprint(dataclasses.replace(profile.environment, platform=changed_platform), root))


if __name__ == "__main__":
    unittest.main()

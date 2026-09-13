import os
import dataclasses
import hashlib
import json
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.host import docker_runtime, runtime_images
from acprof.host.detect import TaskInfo
from acprof.runtime_profiles import RuntimeProfile
from acprof.dependency_locks import content_digest, package_versions, read_python_lock, system_lock_identity
from runtime_fixture import copy_dependency_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "nlp": ("fill-mask", "transformers_pipeline"),
    "audio": ("automatic-speech-recognition", "transformers_pipeline"),
    "cv": ("image-classification", "transformers_pipeline"),
    "diffusion": ("text-to-image", "diffusers"),
    "multimodal": ("image-text-to-text", "transformers_model"),
    "structured": ("tabular-regression", "sklearn"),
    "timeseries": ("time-series-forecasting", "chronos"),
}


class RuntimeImageBuildTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskInfo(
            model_id="example/model", pipeline_tag="fill-mask", task_family="nlp",
            runtime_backend="transformers_pipeline", library_name="transformers",
            model_revision="0123456789abcdef0123456789abcdef01234567",
            detection_method="test",
        )
        self.commands = []
        self.images = {}
        self.manifests = {}
        self.contexts = {}
        self.failed_dockerfile = None
        for mocked in (
            patch.dict(os.environ, {}, clear=True),
            patch.object(docker_runtime, "_run", side_effect=self.fake_run),
            patch.object(runtime_images, "inspect_identity", side_effect=self.images.get),
            patch.object(runtime_images, "verified_image", side_effect=self.verified_image),
            patch.object(docker_runtime, "_select_nlp_torch_index_url",
                         return_value="https://download.pytorch.org/whl/cu124"),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    def fake_run(self, command, **kwargs):
        self.commands.append(command)
        if command[:2] == ["docker", "build"]:
            recipe = Path(command[command.index("-f") + 1]).name
            if recipe == self.failed_dockerfile:
                return subprocess.CompletedProcess(command, 1, "", "build failed")
            arguments = dict(command[index + 1].split("=", 1)
                             for index, value in enumerate(command) if value == "--build-arg")
            identifier = "sha256:" + f"{len(self.commands):064x}"
            Path(command[command.index("--iidfile") + 1]).write_text(identifier)
            labels = {"org.acprof.model-files-key": arguments.get("MODEL_FILES_KEY", ""),
                      "org.acprof.platform-build-fingerprint": arguments.get("PLATFORM_BUILD_FINGERPRINT", ""),
                      "org.acprof.environment-build-fingerprint": arguments.get("ENVIRONMENT_BUILD_FINGERPRINT", "")}
            self.images[identifier] = {"image_id": identifier, "labels": labels}
            if recipe in {"platform.Dockerfile", "runtime.Dockerfile"}:
                context = Path(command[-1])
                expected = json.loads((context / "expectation.json").read_text())
                labels.update({"org.acprof.image-kind": "platform" if recipe == "platform.Dockerfile" else "environment",
                               "org.acprof.platform": expected["platform_id"],
                               "org.acprof.platform-build-fingerprint": expected["platform_build_fingerprint"]})
                if recipe == "runtime.Dockerfile":
                    labels["org.acprof.environment"] = expected["environment_id"]
                lock = read_python_lock(context / "requirements.lock")
                system = json.loads((context / "system.lock").read_text())
                self.manifests[identifier] = {
                    **expected, "schema_version": 1, "packages": package_versions(lock),
                    "system_packages": system["packages"],
                    "system_lock_sha256": content_digest(system_lock_identity(system)),
                    "dependency_lock_sha256": hashlib.sha256((context / "requirements.lock").read_bytes()).hexdigest(),
                }
                self.contexts[recipe] = {p.name: p.read_text() for p in context.iterdir() if p.is_file()}
        elif command[:2] == ["docker", "tag"]:
            self.images[command[3]] = self.images[command[2]]
        elif command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(self.manifests[command[-2]]), "")
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    def verified_image(self, task, name, fingerprint, project_dir=None):
        # 服务清单由 test_image_reuse 覆盖；依赖清单走实际核验函数。
        return docker_runtime.ImageInfo(tag=self.images[name]["image_id"], name=name)

    def build_commands(self):
        return [command for command in self.commands if command[:2] == ["docker", "build"]]

    def test_unlocked_profiles_fail_before_any_docker_command(self):
        for family in FAMILIES:
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "lock"):
                RuntimeProfile("test-" + family, family)
        self.assertEqual(self.commands, [])

    def test_locked_runtime_passes_fixed_model_revision_to_model_layer(self):
        docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        builds = self.build_commands()
        self.assertEqual([Path(cmd[cmd.index("-f") + 1]).name for cmd in builds], [
            "platform.Dockerfile", "runtime.Dockerfile", "runtime-model.Dockerfile", "runtime-final.Dockerfile",
        ])
        self.assertIn("# torch==2.11.0+cu128", self.contexts["runtime.Dockerfile"]["requirements.lock"])
        self.assertNotIn("acprof", self.contexts["runtime.Dockerfile"])
        self.assertIn("MODEL_REVISION=0123456789abcdef0123456789abcdef01234567", builds[2])

    def test_equal_locks_share_one_runtime_build_across_task_families(self):
        audio = dataclasses.replace(self.task, task_family="audio",
                                    pipeline_tag="automatic-speech-recognition",
                                    runtime_profile_id="audio-cpu")
        multimodal = dataclasses.replace(self.task, task_family="multimodal",
                                         pipeline_tag="image-text-to-text",
                                         runtime_backend="transformers_model",
                                         runtime_profile_id="multimodal-transformers4576-cpu")
        docker_runtime.build_image(audio, str(PROJECT_ROOT))
        docker_runtime.build_image(multimodal, str(PROJECT_ROOT))
        runtime_builds = [cmd for cmd in self.build_commands()
                          if Path(cmd[cmd.index("-f") + 1]).name == "runtime.Dockerfile"]
        self.assertEqual(len(runtime_builds), 1)

    def test_model_version_dots_survive_weights_and_service_image_names(self):
        self.task.model_id = "Qwen/Qwen2.5-0.5B"
        result = docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        builds = self.build_commands()
        weights_name = next(cmd[3] for cmd in self.commands if cmd[:2] == ["docker", "tag"] and cmd[3].startswith("acprof-weights-"))
        service_name = result.name
        self.assertRegex(weights_name, r"^acprof-weights-nlp-qwen--qwen2\.5-0\.5b:[0-9a-f]{20}$")
        self.assertRegex(service_name, r"^acprof-nlp-qwen--qwen2\.5-0\.5b:[0-9a-f]{20}$")
        self.assertEqual(result.name, service_name)
        self.assertIn("MODEL_ID=Qwen/Qwen2.5-0.5B", builds[2])
        self.assertEqual(self.task.model_id, "Qwen/Qwen2.5-0.5B")

    def test_mutable_model_revision_is_rejected_before_docker(self):
        self.task.model_revision = "main"
        with self.assertRaisesRegex(RuntimeError, "固定 model revision"):
            docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        self.assertEqual(self.commands, [])

    def test_hf_token_is_passed_only_as_buildkit_secret(self):
        with patch.dict(os.environ, {"HF_TOKEN": "test-secret-value"}):
            docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        for command in self.build_commands():
            recipe = Path(command[command.index("-f") + 1]).name
            self.assertEqual("--secret" in command, recipe == "runtime-model.Dockerfile")
            if recipe == "runtime-model.Dockerfile":
                self.assertIn("id=hf_token,env=HF_TOKEN", command)
        self.assertFalse(any("test-secret-value" in arg or arg.startswith("HF_TOKEN=")
                             for cmd in self.commands for arg in cmd))

    def test_wrong_dependency_cache_label_stops_before_building_more_layers(self):
        docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        name = next(name for name in self.images if name.startswith("acprof-runtime-env:"))
        self.images[name]["labels"]["org.acprof.environment-build-fingerprint"] = "wrong"
        count = len(self.build_commands())
        with self.assertRaisesRegex(RuntimeError, "缓存标签"):
            docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        self.assertEqual(len(self.build_commands()), count)

    def test_environment_id_label_must_match_even_with_valid_build_fingerprint(self):
        docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        name = next(name for name in self.images if name.startswith("acprof-runtime-env:"))
        self.images[name]["labels"]["org.acprof.environment"] = "wrong"
        count = len(self.build_commands())
        with self.assertRaisesRegex(RuntimeError, "缓存标签"):
            docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        self.assertEqual(len(self.build_commands()), count)

    def test_extra_installed_package_in_cached_environment_is_rejected(self):
        docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        name = next(name for name in self.images if name.startswith("acprof-runtime-env:"))
        self.manifests[self.images[name]["image_id"]]["packages"]["undeclared-package"] = "1.0"
        with self.assertRaisesRegex(ValueError, "extra=.*undeclared-package"):
            docker_runtime.build_image(self.task, str(PROJECT_ROOT))

    def test_retagged_parent_during_build_does_not_publish_environment_cache(self):
        from acprof.host.dependency_images import prepare_environment_image
        from acprof.runtime_profiles import ENVIRONMENTS
        def changed(command, **kwargs):
            result = self.fake_run(command, **kwargs)
            if command[:2] == ["docker", "build"] and Path(command[command.index("-f") + 1]).name == "runtime.Dockerfile":
                source = next(arg.split("=", 1)[1] for arg in command if arg.startswith("PLATFORM_IMAGE="))
                self.images[source] = {"image_id": "sha256:" + "e" * 64, "labels": {}}
            return result
        with patch.object(docker_runtime, "_run", side_effect=changed), self.assertRaisesRegex(RuntimeError, "父镜像引用发生变化"):
            prepare_environment_image(ENVIRONMENTS["audio-cpu"], PROJECT_ROOT)
        self.assertFalse(any(name.startswith("acprof-runtime-env:") for name in self.images))

    def test_changed_build_input_does_not_publish_dependency_cache(self):
        from acprof.host.dependency_images import prepare_environment_image
        from acprof.runtime_profiles import ENVIRONMENTS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            def changed(command, **kwargs):
                result = self.fake_run(command, **kwargs)
                if command[:2] == ["docker", "build"]:
                    recipe = root / "dockerfiles/platform.Dockerfile"
                    recipe.write_text(recipe.read_text() + "\n# changed during build\n")
                return result
            with patch.object(docker_runtime, "_run", side_effect=changed), self.assertRaisesRegex(RuntimeError, "发生变化"):
                prepare_environment_image(ENVIRONMENTS["audio-cpu"], root)
            self.assertFalse(any(name.startswith("acprof-platform-") for name in self.images))

    def test_input_change_between_declaration_read_and_fingerprinting_is_rejected(self):
        from acprof.host import dependency_images
        from acprof.runtime_profiles import ENVIRONMENTS, environment_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            changed = False
            def read_then_change(environment, project_dir):
                nonlocal changed
                identity = environment_identity(environment, project_dir)
                if not changed:
                    changed = True
                    path = root / environment.requirements_lock
                    path.write_text(path.read_text().replace("numpy-2.2.6-", "numpy-2.2.7-"))
                return identity
            with patch.object(dependency_images, "environment_identity", side_effect=read_then_change), self.assertRaisesRegex(RuntimeError, "输入发生变化"):
                dependency_images.prepare_environment_image(ENVIRONMENTS["audio-cpu"], root)
            self.assertFalse(any(name.startswith("acprof-runtime-env:") for name in self.images))

    def test_build_failure_stops_before_later_layers(self):
        for recipe in ("platform.Dockerfile", "runtime.Dockerfile", "runtime-model.Dockerfile", "runtime-final.Dockerfile"):
            with self.subTest(recipe=recipe):
                self.commands.clear()
                self.images.clear()
                self.failed_dockerfile = recipe
                with self.assertRaisesRegex(RuntimeError, "Docker 构建失败"):
                    docker_runtime.build_image(self.task, str(PROJECT_ROOT))
                last_build = self.build_commands()[-1]
                self.assertEqual(Path(last_build[last_build.index("-f") + 1]).name, recipe)


if __name__ == "__main__":
    unittest.main()

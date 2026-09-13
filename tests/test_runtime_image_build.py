import dataclasses
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.host import docker_runtime, runtime_images
from acprof.host.detect import TaskInfo
from acprof.runtime_profiles import RuntimeProfile


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
            name = command[command.index("-t") + 1]
            arguments = dict(command[index + 1].split("=", 1)
                             for index, value in enumerate(command) if value == "--build-arg")
            self.images[name] = {
                "image_id": "sha256:" + f"{len(self.commands):064x}",
                "labels": {"org.acprof.model-files-key": arguments.get("MODEL_FILES_KEY", "")},
            }
        else:
            self.assertEqual(command[:2], ["docker", "tag"])
        return subprocess.CompletedProcess(command, 0, "", "")

    def verified_image(self, task, name, fingerprint):
        # 镜像清单核验由 test_image_reuse 覆盖；此处观察构建参数与顺序。
        return docker_runtime.ImageInfo(tag=self.images[name]["image_id"], name=name)

    def build_commands(self):
        return [command for command in self.commands if command[:2] == ["docker", "build"]]

    def test_unlocked_families_pass_explicit_base_and_selected_torch(self):
        for family, (task_type, backend) in FAMILIES.items():
            with self.subTest(family=family):
                self.commands.clear()
                self.images.clear()
                task = dataclasses.replace(self.task, task_family=family,
                                           pipeline_tag=task_type, runtime_backend=backend)
                profile = RuntimeProfile("test-" + family, family)
                with patch.object(runtime_images, "select_runtime_profile", return_value=profile):
                    image = docker_runtime.build_image(task, str(PROJECT_ROOT))
                builds = self.build_commands()
                self.assertEqual([Path(cmd[cmd.index("-f") + 1]).name for cmd in builds], [
                    "base.Dockerfile", f"{family}.Dockerfile",
                    "runtime-model.Dockerfile", "runtime-final.Dockerfile",
                ])
                base = builds[0][builds[0].index("-t") + 1]
                self.assertRegex(base, r"^acprof-base:[0-9a-f]{20}$")
                self.assertIn("BASE_IMAGE=" + base, builds[1])
                self.assertEqual(builds[1][builds[1].index("--target") + 1], "runtime")
                if family in {"nlp", "diffusion", "multimodal", "structured"}:
                    self.assertIn("TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124", builds[1])
                    self.assertIn("TORCH_PACKAGE_SPEC=torch>=2.6,<2.7", builds[1])
                self.assertFalse(any("acprof-base:latest" in arg for cmd in builds for arg in cmd))
                self.assertTrue(image.tag.startswith("sha256:"))

    def test_locked_runtime_passes_fixed_model_revision_to_model_layer(self):
        docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        builds = self.build_commands()
        self.assertEqual([Path(cmd[cmd.index("-f") + 1]).name for cmd in builds], [
            "runtime.Dockerfile", "runtime-model.Dockerfile", "runtime-final.Dockerfile",
        ])
        self.assertIn("REQUIREMENTS_LOCK=dockerfiles/locks/nlp-cu128.txt", builds[0])
        self.assertIn("MODEL_REVISION=0123456789abcdef0123456789abcdef01234567", builds[1])

    def test_mutable_model_revision_is_rejected_before_docker(self):
        self.task.model_revision = "main"
        with self.assertRaisesRegex(RuntimeError, "固定 model revision"):
            docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        self.assertEqual(self.commands, [])

    def test_hf_token_is_passed_only_as_buildkit_secret(self):
        with patch.dict(os.environ, {"HF_TOKEN": "test-secret-value"}):
            docker_runtime.build_image(self.task, str(PROJECT_ROOT))
        for command in self.build_commands():
            self.assertIn("--secret", command)
            self.assertIn("id=hf_token,env=HF_TOKEN", command)
        self.assertFalse(any("test-secret-value" in arg or arg.startswith("HF_TOKEN=")
                             for cmd in self.commands for arg in cmd))

    def test_build_failure_stops_before_later_layers(self):
        for recipe in ("base.Dockerfile", "nlp.Dockerfile", "runtime-model.Dockerfile",
                       "runtime-final.Dockerfile"):
            with self.subTest(recipe=recipe):
                self.commands.clear()
                self.images.clear()
                self.failed_dockerfile = recipe
                with patch.object(runtime_images, "select_runtime_profile",
                                  return_value=RuntimeProfile("test-nlp", "nlp")), \
                        self.assertRaisesRegex(RuntimeError, "Docker 构建失败"):
                    docker_runtime.build_image(self.task, str(PROJECT_ROOT))
                last_build = self.build_commands()[-1]
                self.assertEqual(Path(last_build[last_build.index("-f") + 1]).name, recipe)


@unittest.skipUnless(os.environ.get("ACPROF_TEST_DOCKER_BUILD") == "1",
                     "requires ACPROF_TEST_DOCKER_BUILD=1 and Docker Buildx")
class DockerfileBaseImageTests(unittest.TestCase):
    def outline(self, family, *arguments):
        # outline 只解析 Dockerfile，不执行 RUN、下载模型或生成镜像。
        return subprocess.run([
            "docker", "build", "--call=outline", "--target", "runtime",
            "-f", str(PROJECT_ROOT / "dockerfiles" / f"{family}.Dockerfile"),
            *arguments, str(PROJECT_ROOT),
        ], capture_output=True, text=True, timeout=60, check=False)

    def test_missing_or_empty_base_image_is_rejected(self):
        for family in FAMILIES:
            for arguments in ((), ("--build-arg", "BASE_IMAGE=")):
                with self.subTest(family=family, arguments=arguments):
                    result = self.outline(family, *arguments)
                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertRegex(output, r"base name.*should not be blank")

    def test_explicit_base_image_is_accepted(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                result = self.outline(family, "--build-arg", "BASE_IMAGE=scratch")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertRegex(result.stdout, r"BASE_IMAGE\s+scratch\b")


if __name__ == "__main__":
    unittest.main()

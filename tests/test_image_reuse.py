import io
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from acprof.host import docker_runtime
from acprof.host import orchestrator
from acprof.host.detect import TaskInfo


class PrepareImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TaskInfo(
            model_id="Org/Model.v1",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="revision-1",
            detection_method="unit",
        )
        self.tag = "acprof-nlp-org--model_v1:latest"

    def test_existing_image_is_reused_without_building(self) -> None:
        query = subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr="")
        stdout = io.StringIO()
        with patch.object(docker_runtime, "_run", return_value=query) as docker, patch.object(
            docker_runtime, "build_image",
        ) as build, redirect_stdout(stdout):
            image = docker_runtime.prepare_image(self.task, "/project", reuse_existing=True)

        self.assertEqual(image.tag, self.tag)
        build.assert_not_called()
        docker.assert_called_once_with(
            ["docker", "image", "ls", "--quiet", "--filter", f"reference={self.tag}"],
            check=False,
        )
        self.assertIn("跳过构建并复用", stdout.getvalue())

    def test_missing_image_is_announced_before_building_and_returns_built_image(self) -> None:
        query = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        stdout = io.StringIO()
        built_image = docker_runtime.ImageInfo(tag=self.tag)

        def build_image(task, project_dir):
            self.assertIs(task, self.task)
            self.assertEqual(project_dir, "/project")
            self.assertIn(self.tag, stdout.getvalue())
            self.assertIn("未找到本地模型镜像", stdout.getvalue())
            self.assertIn("自动构建", stdout.getvalue())
            return built_image

        with patch.object(docker_runtime, "_run", return_value=query), patch.object(
            docker_runtime, "build_image", side_effect=build_image,
        ) as build, redirect_stdout(stdout):
            image = docker_runtime.prepare_image(self.task, "/project", reuse_existing=True)

        self.assertIs(image, built_image)
        build.assert_called_once_with(self.task, "/project")

    def test_docker_query_errors_stop_without_building(self) -> None:
        for error in ("Cannot connect to the Docker daemon", "permission denied"):
            with self.subTest(error=error):
                query = subprocess.CompletedProcess([], 1, stdout="", stderr=error)
                with patch.object(docker_runtime, "_run", return_value=query), patch.object(
                    docker_runtime, "build_image",
                ) as build, self.assertRaisesRegex(RuntimeError, error):
                    docker_runtime.prepare_image(self.task, "/project", reuse_existing=True)
                build.assert_not_called()

    def test_build_failure_is_propagated_after_missing_image(self) -> None:
        query = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch.object(docker_runtime, "_run", return_value=query), patch.object(
            docker_runtime, "build_image", side_effect=SystemExit(1),
        ) as build, self.assertRaises(SystemExit) as raised:
            docker_runtime.prepare_image(self.task, "/project", reuse_existing=True)

        self.assertEqual(raised.exception.code, 1)
        build.assert_called_once_with(self.task, "/project")

    def test_without_reuse_builds_without_querying_the_image_store(self) -> None:
        built_image = docker_runtime.ImageInfo(tag=self.tag)
        with patch.object(docker_runtime, "_run") as docker, patch.object(
            docker_runtime, "build_image", return_value=built_image,
        ) as build:
            image = docker_runtime.prepare_image(self.task, "/project")

        self.assertIs(image, built_image)
        docker.assert_not_called()
        build.assert_called_once_with(self.task, "/project")


if __name__ == "__main__":
    unittest.main()

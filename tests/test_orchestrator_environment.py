import unittest
from types import SimpleNamespace
from unittest.mock import patch

import orchestrator
from detect import TaskInfo


class DetectEnvironmentTests(unittest.TestCase):
    def test_detect_environment_windows_11_with_wsl_kernel(self) -> None:
        with patch("orchestrator.platform.system", return_value="Windows"), patch(
            "orchestrator.platform.release", return_value="11"
        ), patch.dict("orchestrator.os.environ", {}, clear=True), patch(
            "orchestrator._run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="6.6.87.2-microsoft-standard-WSL2\n",
                stderr="",
            ),
        ):
            self.assertEqual(orchestrator._detect_environment(), "windows11+wsl")

    def test_detect_environment_linux_ubuntu_without_wsl(self) -> None:
        with patch("orchestrator.platform.system", return_value="Linux"), patch(
            "orchestrator.platform.freedesktop_os_release",
            return_value={"ID": "ubuntu", "VERSION_ID": "24.04"},
        ), patch.dict("orchestrator.os.environ", {}, clear=True), patch(
            "orchestrator._run",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="docker unavailable"),
        ):
            self.assertEqual(orchestrator._detect_environment(), "ubuntu24.04")

    def test_collect_static_meta_includes_environment(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )

        with patch("orchestrator._detect_environment", return_value="windows11+wsl"), patch(
            "orchestrator._get_gpu_name", return_value="Test GPU"
        ), patch(
            "orchestrator._docker_model_weight_bytes", return_value=123
        ), patch("orchestrator._docker_image_size_bytes", return_value=456):
            meta = orchestrator.collect_static_meta(
                task_info=task_info,
                image_info=orchestrator.ImageInfo(tag="acprof-test:latest"),
                batch_size=1,
                input_scale_type="seq_length",
            )

        self.assertEqual(meta.environment, "windows11+wsl")


if __name__ == "__main__":
    unittest.main()

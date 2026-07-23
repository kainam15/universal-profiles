import inspect
import os
import tempfile
import unittest
from unittest.mock import patch

from acprof.container import download_model
from acprof.container.handlers import model_revision_kwargs, resolve_model_source
from acprof.host import orchestrator
from acprof.host.detect import TaskInfo


class ModelRevisionTests(unittest.TestCase):
    def test_download_model_passes_model_revision_to_snapshot_download(self) -> None:
        revision = "0123456789abcdef"

        def fake_snapshot_download(
            repo_id,
            cache_dir,
            revision=None,
            endpoint=None,
            max_workers=None,
            etag_timeout=None,
            resume_download=None,
        ):
            del repo_id, cache_dir, endpoint, max_workers, etag_timeout, resume_download
            return revision

        with patch.object(
            download_model,
            "snapshot_download",
            side_effect=fake_snapshot_download,
        ), patch.object(
            inspect,
            "signature",
            return_value=inspect.signature(fake_snapshot_download),
        ), patch.dict(
            download_model.os.environ,
            {"MODEL_ID": "google-bert/bert-base-uncased", "MODEL_REVISION": revision},
            clear=True,
        ):
            kwargs = download_model._build_snapshot_kwargs(
                endpoint="https://huggingface.co",
                max_workers=1,
            )

        self.assertEqual(kwargs["revision"], revision)

    def test_download_model_publishes_stable_local_snapshot_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = os.path.join(tmp_dir, "snapshots", "revision")
            local_path = os.path.join(tmp_dir, "model-snapshot")
            os.makedirs(target_dir)

            download_model._publish_local_model_path(target_dir, local_path)

            self.assertTrue(os.path.islink(local_path))
            self.assertEqual(os.path.realpath(local_path), os.path.realpath(target_dir))

    def test_local_snapshot_is_preferred_and_does_not_pass_hub_revision(self) -> None:
        with tempfile.TemporaryDirectory() as local_path:
            source = resolve_model_source(
                "google-bert/bert-base-uncased",
                local_path,
            )

            self.assertEqual(source, local_path)
            self.assertEqual(model_revision_kwargs(source, "deadbeef"), {})

    def test_missing_local_snapshot_falls_back_to_model_id_and_revision(self) -> None:
        source = resolve_model_source(
            "google-bert/bert-base-uncased",
            "/missing/model-snapshot",
        )

        self.assertEqual(source, "google-bert/bert-base-uncased")
        self.assertEqual(
            model_revision_kwargs(source, "deadbeef"),
            {"revision": "deadbeef"},
        )

    def test_build_image_passes_model_revision_build_arg(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="0123456789abcdef",
            detection_method="hub_api",
        )
        commands = []

        def fake_run(cmd, check=True, capture=True, **kwargs):
            del check, capture, kwargs
            commands.append(cmd)
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("acprof.host.orchestrator.os.path.exists", return_value=True), patch(
            "acprof.host.orchestrator._run",
            side_effect=fake_run,
        ), patch(
            "acprof.host.orchestrator._select_nlp_torch_index_url",
            return_value=orchestrator.CUDA124_NLP_TORCH_INDEX_URL,
        ):
            orchestrator.build_image(task_info, ".")

        self.assertIn("MODEL_REVISION=0123456789abcdef", commands[1])


if __name__ == "__main__":
    unittest.main()

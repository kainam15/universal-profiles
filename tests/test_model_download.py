import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.container import download_model
from acprof.container.model_files import ModelFilesError, plan_download, seal_plan, validate_plan


class ModelDownloadTests(unittest.TestCase):
    def dependency_plan(self):
        def plan(repo, revision):
            return plan_download(model_id=repo, revision=revision, family="multimodal",
                                 backend="transformers_pipeline", policy="full",
                                 files={"config.json": {"size": 2}}, read_json=lambda name: {})
        primary = plan("example/audio", "a" * 40)
        primary["dependencies"] = [{"repo_id": "example/base", "revision": "b" * 40,
                                    "allow_patterns": ["*.json"], "download": plan("example/base", "b" * 40)}]
        primary["total_selected_bytes"] = 4
        return seal_plan(primary)

    def test_dependencies_are_downloaded_verified_and_bound_to_offline_default_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.dependency_plan()
            calls = []

            def snapshot(**kwargs):
                calls.append(kwargs)
                target = root / ("models--" + kwargs["repo_id"].replace("/", "--")) / "snapshots" / kwargs["revision"]
                target.mkdir(parents=True)
                (target / "config.json").write_text("{}")
                return str(target)

            with patch.object(download_model, "_prepare_plan", return_value=plan), patch.object(
                download_model, "snapshot_download", side_effect=snapshot,
            ), patch.object(download_model, "CACHE_DIR", str(root)):
                target = download_model._download_once("https://huggingface.co", 1)
                verified = download_model.verify_download(target, download_model._LAST_PLAN)
            self.assertEqual([(call["repo_id"], call["revision"]) for call in calls],
                             [("example/audio", "a" * 40), ("example/base", "b" * 40)])
            self.assertEqual((root / "models--example--base/refs/main").read_text(), "b" * 40)
            self.assertEqual(verified["dependencies"][0]["download"]["verification"], "sha256")
            self.assertEqual(verified["total_selected_bytes"], 4)
            validate_plan(verified)

    def test_dependency_identity_mismatch_is_rejected_even_with_a_resealed_parent(self):
        plan = self.dependency_plan()
        plan["dependencies"][0]["revision"] = "c" * 40
        seal_plan(plan)
        with self.assertRaisesRegex(ModelFilesError, "dependency"):
            validate_plan(plan)

    def test_unknown_hub_sizes_are_resolved_by_download_verification(self):
        plan = self.dependency_plan()
        declaration = {key: value for key, value in plan["dependencies"][0].items() if key != "download"}
        child = plan.pop("dependencies")[0]["download"]
        plan["selected_bytes"] = None
        with patch.object(download_model, "_prepare_repository_plan", side_effect=[plan, child]), patch.object(
            download_model, "load_model_dependencies", return_value=[declaration],
        ):
            prepared = download_model._prepare_plan("https://huggingface.co")
        self.assertIsNone(prepared["total_selected_bytes"])

    def test_dependency_size_total_must_match_resealed_file_plans(self):
        plan = self.dependency_plan()
        plan["total_selected_bytes"] = 999
        seal_plan(plan)
        with self.assertRaisesRegex(ModelFilesError, "dependency.*bytes|total_selected_bytes"):
            validate_plan(plan)

    def test_standard_checkpoint_download_omits_unused_formats(self):
        """实际下载调用必须只取得加载器选中的权重，而非整个仓库。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"model_type": "bert"}))
            files = ["config.json", "model.safetensors", "pytorch_model.bin", "flax_model.msgpack"]
            info = SimpleNamespace(sha="a" * 40, siblings=[
                SimpleNamespace(rfilename=name, size=None, blob_id=None, lfs=None) for name in files
            ])
            calls = []

            def snapshot(**kwargs):
                calls.append(kwargs)
                return str(root)

            with patch.dict(os.environ, {
                "MODEL_REVISION": "a" * 40, "TASK_FAMILY": "nlp",
                "RUNTIME_BACKEND": "transformers_pipeline", "MODEL_DOWNLOAD_POLICY": "auto",
            }), patch.object(download_model, "MODEL_ID", "example/bert"), patch.object(
                download_model, "CACHE_DIR", str(root),
            ), patch.object(download_model, "snapshot_download", side_effect=snapshot), patch(
                "huggingface_hub.HfApi.model_info", return_value=info,
            ), patch("huggingface_hub.hf_hub_download", return_value=str(config)), patch.object(
                download_model, "_native_model_types", return_value={"bert"},
            ):
                download_model._download_once("https://huggingface.co", 1)
            self.assertEqual(set(calls[-1].get("allow_patterns") or files), {
                "config.json", "model.safetensors",
            })


if __name__ == "__main__":
    unittest.main()

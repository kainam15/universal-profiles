import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.container import download_model


class ModelDownloadTests(unittest.TestCase):
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

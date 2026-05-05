import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.host import detect


class DetectTaskTests(unittest.TestCase):
    def test_config_fallback_reads_config_json_without_transformers_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir, "config.json")
            config_path.write_text(
                json.dumps({"architectures": ["BertForMaskedLM"]}),
                encoding="utf-8",
            )

            with patch("huggingface_hub.hf_hub_download", return_value=str(config_path)):
                info = detect._detect_from_config("google-bert/bert-base-uncased")

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.pipeline_tag, "fill-mask")
        self.assertEqual(info.task_family, "nlp")
        self.assertEqual(info.runtime_backend, "transformers_pipeline")
        self.assertEqual(info.library_name, "transformers")
        self.assertEqual(info.detection_method, "config_infer")

    def test_detect_task_reports_auto_detection_failure_reasons(self) -> None:
        stderr = io.StringIO()

        with patch("huggingface_hub.model_info", side_effect=RuntimeError("hub timeout")), patch(
            "huggingface_hub.hf_hub_download", side_effect=OSError("config missing")
        ), patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit) as raised:
                detect.detect_task("missing/model")

        self.assertEqual(raised.exception.code, 1)
        message = stderr.getvalue()
        self.assertIn("[ERROR] Cannot auto-detect task for 'missing/model'.", message)
        self.assertIn("hub_api: RuntimeError: hub timeout", message)
        self.assertIn("config_json: OSError: config missing", message)
        self.assertIn("AutoConfig: ModuleNotFoundError", message)
        self.assertIn("Please specify --task and/or --task-family manually.", message)


if __name__ == "__main__":
    unittest.main()

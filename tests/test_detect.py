import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import detect


class DetectTaskTests(unittest.TestCase):
    def test_hub_detection_collects_model_metadata(self) -> None:
        hub_info = SimpleNamespace(
            pipeline_tag="fill-mask",
            library_name="transformers",
            sha="0123456789abcdef",
            safetensors=SimpleNamespace(
                parameters={"F32": 110_106_428},
                total=110_106_428,
            ),
            card_data={"license": "apache-2.0"},
            config={"architectures": ["BertForMaskedLM"]},
            tags=["transformers", "license:apache-2.0"],
        )

        with patch("huggingface_hub.model_info", return_value=hub_info):
            info = detect._detect_from_hub(
                "google-bert/bert-base-uncased"
            )

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.parameter_count, 110_106_428)
        self.assertEqual(info.parameter_bytes, 440_425_712)
        self.assertEqual(info.precision_dtype, "FP32")
        self.assertEqual(
            info.parameter_dtype_counts,
            {"FP32": 110_106_428},
        )
        self.assertFalse(info.quantized)
        self.assertEqual(info.model_license, "apache-2.0")
        self.assertEqual(info.model_metadata_source, "huggingface_hub")

    def test_hub_detection_preserves_quantization_metadata(self) -> None:
        hub_info = SimpleNamespace(
            pipeline_tag="text-generation",
            library_name="transformers",
            sha="fedcba9876543210",
            safetensors=SimpleNamespace(
                parameters={"I8": 7_000_000_000},
                total=7_000_000_000,
            ),
            card_data={},
            config={
                "architectures": ["ExampleForCausalLM"],
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 8,
                },
            },
            tags=["gptq", "license:mit"],
        )

        with patch("huggingface_hub.model_info", return_value=hub_info):
            info = detect._detect_from_hub("example/quantized-model")

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.parameter_bytes, 7_000_000_000)
        self.assertEqual(info.precision_dtype, "INT8")
        self.assertTrue(info.quantized)
        self.assertEqual(info.quantization_method, "gptq")
        self.assertEqual(
            info.quantization_config,
            {"quant_method": "gptq", "bits": 8},
        )
        self.assertEqual(info.model_license, "mit")

    def test_parameter_bytes_is_null_when_any_dtype_width_is_unknown(self) -> None:
        self.assertEqual(
            detect._parameter_bytes_from_dtype_counts(
                {"FP16": 10, "INT64": 2}
            ),
            36,
        )
        metadata = detect._hub_model_metadata(
            SimpleNamespace(
                safetensors=SimpleNamespace(
                    parameters={"F16": 10, "CUSTOM": 2},
                    total=12,
                ),
                card_data={},
                config={},
                tags=[],
            )
        )

        self.assertEqual(
            metadata["parameter_dtype_counts"],
            {"FP16": 10, "CUSTOM": 2},
        )
        self.assertIsNone(metadata["parameter_bytes"])

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

"""Golden host contracts and strict declaration-driven resolution."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.extensions import CATALOG, UnsupportedExtensionError, load_catalog
from acprof.host.detect import TaskInfo
from acprof.host.model_schema import _inference_precision_by_device, _model_io_formats


class ExtensionResolutionTests(unittest.TestCase):
    def test_existing_io_and_precision_contracts(self):
        golden = json.loads(Path(__file__).with_name("fixtures").joinpath("extension_contracts_v1.json").read_text())
        for row in golden["cases"]:
            with self.subTest(extension=row["extension"], task=row["task"], backend=row["backend"]):
                if row["extension"] == "torch-timeseries-legacy":
                    continue  # Rejected execution route, covered separately below.
                task = TaskInfo("fixture/model", row["task"], row["family"], row["backend"], "", "fixed", "manual",
                                model_adapter=row["adapter"])
                self.assertEqual(_inference_precision_by_device(task), row["precision"])
                digest = hashlib.sha256(json.dumps(_model_io_formats(task), sort_keys=True).encode()).hexdigest()
                self.assertEqual(digest, row["io_sha256"])

    def test_resolve_routes_and_config_predicates(self):
        cases = [
            ("text-generation", "transformers", {}, "transformers_pipeline"),
            ("feature-extraction", "sentence-transformers", {}, "sentence_transformers"),
            ("sentence-similarity", "transformers", {}, "sentence_transformers"),
            ("text-ranking", "transformers", {}, "cross_encoder"),
            ("voice-activity-detection", "", {}, "torchscript"),
            ("audio-to-audio", "transformers", {}, "transformers_model"),
            ("tabular-regression", "sklearn", {}, "skops"),
            ("tabular-regression", "", {}, "torchscript"),
            ("image-classification", "onnx", {}, "onnxruntime"),
            ("text-classification", "onnx", {}, "onnxruntime"),
            ("text-to-image", "diffusers", {}, "diffusers"),
            ("image-text-to-text", "transformers", {}, "transformers_model"),
            (None, "chronos-forecasting", {}, "chronos"),
            (None, "transformers", {"chronos_pipeline_class": "ChronosBoltPipeline"}, "chronos"),
            (None, "transformers", {"model_type": "moss_transcribe_diarize"}, "transformers_model"),
        ]
        for task, library, config, backend in cases:
            with self.subTest(task=task, library=library, config=config):
                result = CATALOG.resolve(task=task, library=library, config=config)
                self.assertEqual(result.backend, backend)
                self.assertEqual(result.family, CATALOG.task_families[result.task])
                if config.get("model_type") == "moss_transcribe_diarize":
                    self.assertEqual(result.declaration.adapter, "moss-transcribe-diarize")

    def test_unknown_and_conflicting_routes_are_explicit_errors(self):
        for kwargs in (
            {"task": "future-task", "library": "transformers"},
            {"task": "text-generation", "library": "unknown-library"},
            {"task": "text-generation", "family": "cv", "library": "transformers"},
            {"task": "text-generation", "backend": "onnxruntime"},
            {"task": "time-series-forecasting", "backend": "transformers_pipeline"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(UnsupportedExtensionError):
                CATALOG.resolve(**kwargs)
        self.assertEqual(CATALOG.resolve(task="text-generation", library="unknown-library",
                                        backend="transformers_model").backend, "transformers_model")

    def test_schema_v1_and_malformed_nested_fields_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            for payload in (
                {"schema_version": 1, "extensions": []},
                {"schema_version": True, "extensions": []},
                {"schema_version": 2, "extensions": [{"input_plan": {"text_payload": "yes"}}]},
            ):
                path.write_text(json.dumps(payload))
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    load_catalog([path])

    def test_precision_fallback_and_schema_copies(self):
        task = TaskInfo("fixture/model", "time-series-forecasting", "timeseries", "chronos", "chronos", "fixed", "manual")
        self.assertEqual(_inference_precision_by_device(replace(task, precision_dtype="BF16")),
                         {"cpu": "BF16", "gpu": "BF16"})
        first, _ = _model_io_formats(task)
        first["json_schema"]["required"].clear()
        self.assertEqual(_model_io_formats(task)[0]["json_schema"]["required"], ["context", "prediction_length"])

    def test_priority_is_order_independent_and_equal_priority_conflicts_fail(self):
        from acprof.extensions.schema import BackendRule
        catalog = load_catalog()
        catalog.backend_rules.extend([
            BackendRule(task="text-generation", backend="transformers_pipeline", priority=10),
            BackendRule(task="text-generation", backend="transformers_model", priority=20),
        ])
        self.assertEqual(catalog.resolve(task="text-generation", library="transformers").backend, "transformers_model")
        catalog.backend_rules.reverse()
        self.assertEqual(catalog.resolve(task="text-generation", library="transformers").backend, "transformers_model")
        catalog.backend_rules.append(BackendRule(task="text-generation", backend="transformers_pipeline", priority=20))
        with self.assertRaisesRegex(UnsupportedExtensionError, "ambiguous backend"):
            catalog.resolve(task="text-generation", library="transformers")

    def test_nested_capability_typo_and_non_boolean_are_rejected(self):
        raw = asdict(CATALOG.get_extension("nlp", "transformers_model"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            for capability in ({"text_payload": "yes"}, {"text_paylod": True}):
                path.write_text(json.dumps({"schema_version": 2, "extensions": [{**raw, "input_plan": capability}]}))
                with self.subTest(capability=capability), self.assertRaisesRegex(ValueError, "input_plan"):
                    load_catalog([path])

    def test_invalid_io_template_is_rejected_when_loading_manifest(self):
        raw = asdict(CATALOG.get_extension("nlp", "transformers_model"))
        raw["io_format"]["input"]["json_schema"]["required"] = ["undeclared_input"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            path.write_text(json.dumps({"schema_version": 2, "extensions": [raw]}))
            with self.assertRaisesRegex(ValueError, "io_format.*required"):
                load_catalog([path])

    def test_candidate_discovery_records_unknown_library_without_fallback(self):
        from acprof.model_resolution import discover_model_candidates
        task = TaskInfo("fixture/model", "text-generation", "nlp", "transformers_pipeline", "undeclared", "fixed", "manual")
        result = discover_model_candidates(task)
        self.assertEqual(result["status"], "needs_configuration")
        self.assertEqual(task.runtime_backend, "unknown")
        self.assertIn("undeclared", " ".join(result["missing"]))
        explicit = discover_model_candidates(task, override_backend="transformers_model")
        self.assertEqual(explicit["status"], "candidate")
        self.assertEqual(task.runtime_backend, "transformers_model")
        task = TaskInfo("fixture/multiple-heads", "text-classification", "nlp", "onnxruntime", "onnx", "fixed", "manual",
                        model_config={"auto_map": {"AutoModelForSequenceClassification": "model.Classifier",
                                                   "AutoModelForCausalLM": "model.Generator"}},
                        repository_files=("model.onnx",))
        explicit = discover_model_candidates(task, override_tag="text-classification", override_backend="onnxruntime")
        self.assertEqual(explicit["status"], "candidate")
        rejected = next(candidate for candidate in explicit["candidates"] if candidate["task"] == "text-generation")
        self.assertIn("onnxruntime", rejected["unsupported_reason"])

    def test_catalog_inference_does_not_become_author_declared_hub_task(self):
        from acprof.host.detect import _detect_from_hub
        from acprof.model_resolution import discover_model_candidates
        hub = SimpleNamespace(pipeline_tag=None, library_name="transformers", sha="a" * 40,
                              config={"architectures": ["BertForMaskedLM"], "model_type": "bert"})
        with patch("huggingface_hub.HfApi.model_info", return_value=hub):
            task = _detect_from_hub("fixture/model")
        result = discover_model_candidates(task)
        self.assertEqual(task.pipeline_tag, "fill-mask")
        self.assertEqual(result["semantics"]["status"], "inferred")
        self.assertNotIn("hub_metadata", result["candidates"][0]["evidence"])

    def test_new_family_and_non_torch_runtime_reach_existing_consumers(self):
        from acprof.host import input_plan, model_schema
        from acprof.runtime_profiles import runtime_declarations

        io_format = {"input": {"json_schema": {"type": "object", "properties": {"values": {"type": "array"}},
                                               "required": ["values"]}},
                     "output": {"json_schema": {"type": "object", "properties": {"score": {"type": "number"}},
                                                "required": ["score"]}}}
        entry = {
            "extension_id": "fixture-sensor", "family": "sensor", "runtime": "fixture-runtime",
            "backends": ["fixture-backend"], "tasks": ["sensor-score"],
            "handler_entrypoint": "uninstalled_fixture:Handler", "validation_entrypoint": "uninstalled_fixture:validate",
            "workload_entrypoint": "uninstalled_fixture:Generator", "dtypes": ["FP64"],
            "input_modalities": ["sensor"], "execution": {"cpu": "available", "cuda": "unsupported"},
            "environment": "fixture-cpu", "profile": "fixture-profile",
            "dependency_environment": {"platform": "python-cpu", "requirements_lock": "fixture.lock",
                                       "runtime": {"type": "fixture-runtime", "version": "1.2.3", "package": "fixture"}},
        }
        family = {"scaling": {"param_name": "readings", "values": [2, 4]}, "task_params": {"window": 3},
                  "precision": {"cpu": "FP64"}, "io_format": io_format,
                  "input_plan": {"accepts_workload_spec": True, "workload_scales": True}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            path.write_text(json.dumps({"schema_version": 2, "families": {"sensor": family}, "extensions": [entry],
                                        "platforms": {"python-cpu": CATALOG.platforms["python-cpu"]},
                                        "backend_rules": [{"family": "sensor", "backend": "fixture-backend", "priority": 0}]}))
            catalog = load_catalog([path])
            result = catalog.resolve(task="sensor-score")
            self.assertEqual((result.family, result.backend, result.precision), ("sensor", "fixture-backend", {"cpu": "FP64"}))
            self.assertEqual(catalog.family_scaling()["sensor"]["param_name"], "readings")
            self.assertEqual(catalog.family_task_params()["sensor"], {"window": 3})
            _, environments, profiles = runtime_declarations(catalog)
            self.assertIsNone(environments["fixture-cpu"].platform.torch_version)
            self.assertEqual(environments["fixture-cpu"].runtime.type, "fixture-runtime")
            self.assertEqual(profiles["fixture-profile"].backends, ("fixture-backend",))
            task = TaskInfo("fixture/model", "sensor-score", "sensor", "fixture-backend", "", "fixed", "manual")
            generator = SimpleNamespace(default_input_scales=lambda: [2, 4], generate=lambda scale: {"values": [1] * int(scale)},
                                        effective_input_scale=lambda scale, payload: len(payload["values"]),
                                        scale_label=str, plan_metadata=lambda: {"input_scale_type": "readings"})
            with patch.object(input_plan, "CATALOG", catalog), patch.object(model_schema, "CATALOG", catalog), patch.object(
                input_plan, "_get_task_generator", return_value=generator,
            ), patch.object(input_plan, "_start_probe_session") as start:
                planned = input_plan.plan_input_scales(task, None, [1], [2], ["off"], 1, tmp, workload_spec_path="fixture.json")
                self.assertEqual(planned.scales, [2, 4])
                self.assertEqual(json.loads(Path(planned.plan_file).read_text())["entries"][1]["payload"], {"values": [1, 1, 1, 1]})
                self.assertEqual(model_schema._model_io_formats(task), (io_format["input"], io_format["output"]))
                start.assert_not_called()


if __name__ == "__main__":
    unittest.main()

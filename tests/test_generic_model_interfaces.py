"""Shared interfaces must work without checkpoint-specific dispatch."""
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from acprof.container.handlers.nlp import NLPHandler
from acprof.container.handlers.timeseries import ChronosHandler
from acprof.host.detect import TaskInfo, detect_task
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.runtime_profiles import select_runtime_profile
from acprof.workloads import get_generator
from test_nlp_handler import FakeTokenizer


class ChronosInterfaceTests(unittest.TestCase):
    def test_configuration_dispatches_all_generations_without_trial_loads(self):
        for class_name in ("ChronosPipeline", "ChronosBoltPipeline", "Chronos2Pipeline"):
            with self.subTest(class_name=class_name):
                pipeline = type(class_name, (), {})()
                loader = Mock(return_value=pipeline)
                legacy = Mock(side_effect=AssertionError("trial loading is not dispatch"))
                with patch.dict("sys.modules", {
                    "torch": SimpleNamespace(),
                    "chronos": SimpleNamespace(
                        BaseChronosPipeline=SimpleNamespace(from_pretrained=loader),
                        ChronosPipeline=SimpleNamespace(from_pretrained=legacy),
                        ChronosBoltPipeline=SimpleNamespace(from_pretrained=legacy),
                    ),
                }):
                    context = ChronosHandler().load(
                        "arbitrary/checkpoint", "time-series-forecasting", "chronos", "cpu", "fixed",
                    )
                self.assertIs(context["pipeline"], pipeline)
                self.assertEqual(loader.call_args.kwargs["revision"], "fixed")
                self.assertTrue(loader.call_args.kwargs["local_files_only"])
                legacy.assert_not_called()

    def test_native_univariate_forecast_list_preserves_batch_and_horizon(self):
        forecasts = [np.full((1, 3, 4), value) for value in (1., 2.)]
        pipeline = SimpleNamespace(predict=Mock(return_value=forecasts))
        with patch.dict("sys.modules", {"torch": SimpleNamespace(
            inference_mode=contextlib.nullcontext, cat=lambda rows, dim: np.concatenate(rows, axis=dim),
        )}):
            output = ChronosHandler().predict(
                {"pipeline": pipeline}, {"context": np.ones((2, 8)), "prediction_length": 4},
            )
        self.assertEqual(output.shape, (2, 3, 4))
        np.testing.assert_equal(output[:, 0, 0], [1., 2.])


class SentenceTransformerInterfaceTests(unittest.TestCase):
    def test_embedding_parameters_survive_public_manual_and_auto_input_planning(self):
        from acprof.host import input_plan
        task = TaskInfo(model_id="unseen/encoder", pipeline_tag="feature-extraction", task_family="nlp",
                        runtime_backend="sentence_transformers", library_name="sentence-transformers",
                        model_revision="a" * 40, detection_method="hub_api")
        params = {"prompt": "query: ", "normalize_embeddings": True}
        def probe(session, payload, label):
            self.assertEqual(payload["params"], params)
            return {"payload": payload, "effective_input_scale": len(payload["text"].split()),
                    "truncated_by_limit": False, "reason": "within limit"}
        for scales in ("4", None):
            with self.subTest(scales=scales), tempfile.TemporaryDirectory() as directory:
                spec = Path(directory, "embedding.json")
                spec.write_text(json.dumps({"schema_version": 1, "task": task.pipeline_tag, "params": params}))
                with patch.object(input_plan, "_start_probe_session", return_value=SimpleNamespace(name="probe")), patch.object(
                    input_plan, "_stop_container_session",
                ), patch.object(input_plan, "_post_probe_payload", side_effect=probe), patch.object(
                    input_plan, "_request_nlp_scale_meta", return_value={"max_effective_input_scale": 16, "reason": "test tokenizer"},
                ):
                    planned = input_plan.plan_input_scales(task_info=task, image_info=None, cpu_list=[2], mem_list=[4],
                        gpu_list=["off"], batch_size=1, output_dir=directory, input_scales=scales, workload_spec_path=str(spec))
                saved = json.loads(Path(planned.plan_file).read_text())
                self.assertEqual(saved["workload"]["params"], params)
                self.assertTrue(saved["workload"]["workload_spec_sha256"])
                self.assertTrue(saved["entries"])
                self.assertTrue(all(entry["payload"]["params"] == params for entry in saved["entries"]))

    def test_feature_extraction_uses_complete_sentence_transformer(self):
        encoder = SimpleNamespace(encode=Mock(return_value=np.ones((2, 3))))
        constructor = Mock(return_value=encoder)
        pipeline = Mock(side_effect=AssertionError("raw token features are not sentence embeddings"))
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(float32="fp32", float16="fp16"),
            "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
            "transformers": SimpleNamespace(pipeline=pipeline),
        }):
            handler = NLPHandler()
            ctx = handler.load("arbitrary/encoder", "feature-extraction", "sentence_transformers", "cpu")
            output = handler.predict(ctx, {"text": "hello", "batch_size": 2, "params": {}})
            result = handler.postprocess(ctx, output)
        np.testing.assert_equal(output, np.ones((2, 3)))
        self.assertEqual(result["output_shape"], [2, 3])
        self.assertEqual(result["n_results"], 2)
        self.assertEqual(encoder.encode.call_args.args[0], ["hello", "hello"])
        pipeline.assert_not_called()

    def test_default_prompt_is_preserved_and_included_in_token_budget(self):
        encoder = SimpleNamespace(
            tokenizer=FakeTokenizer("[MASK]", model_max_length=8), max_seq_length=8,
            prompts={"search": "find passage "}, default_prompt_name="search",
            encode=Mock(return_value=np.eye(3)), similarity=Mock(return_value=np.array([[.2, .9]])),
        )
        context = {"pipeline": encoder, "task_type": "sentence-similarity", "backend": "sentence_transformers"}
        handler = NLPHandler()
        prepared = handler.preprocess(context, {
            "query": "query", "documents": ["one two three four five"] * 2,
        })
        self.assertEqual(prepared["documents"], ["one two three four"] * 2)
        self.assertTrue(prepared["_truncated_by_limit"])
        handler.predict(context, prepared)
        self.assertNotEqual(encoder.encode.call_args.kwargs.get("prompt"), "")
        metadata = handler.get_scale_metadata(context, {})
        self.assertEqual(metadata["max_effective_input_scale"], 4)

    def test_embedding_parameters_are_materialized_by_shared_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "input.json")
            path.write_text(json.dumps({"schema_version": 1, "task": "feature-extraction",
                                        "params": {"prompt_name": "query", "normalize_embeddings": True}}))
            generator = get_generator("nlp", "unseen/encoder", "feature-extraction", 2,
                                      workload_spec_path=str(path))
        first = generator.generate(16)
        self.assertEqual(first["params"], {"prompt_name": "query", "normalize_embeddings": True})
        first["params"]["prompt_name"] = "changed"
        self.assertEqual(generator.generate(16)["params"]["prompt_name"], "query")
        self.assertTrue(generator.plan_metadata()["workload_spec_sha256"])


class ModelDetectionInterfaceTests(unittest.TestCase):
    def test_mirror_metadata_failure_does_not_contact_undeclared_endpoint(self):
        from acprof.host.detect import _download_metadata
        from huggingface_hub.errors import FileMetadataError, LocalEntryNotFoundError
        error = LocalEntryNotFoundError("metadata request failed")
        error.__cause__ = FileMetadataError("missing X-Repo-Commit")
        with patch.dict("os.environ", {"HF_ENDPOINT": "https://mirror.example"}, clear=True), patch(
            "huggingface_hub.hf_hub_download", side_effect=error,
        ) as download, self.assertRaises(LocalEntryNotFoundError):
            _download_metadata("unseen/encoder", "config.json", "a" * 40)
        download.assert_called_once_with(repo_id="unseen/encoder", filename="config.json",
                                         revision="a" * 40, endpoint="https://mirror.example")

    def test_offline_metadata_miss_does_not_retry_another_endpoint(self):
        from acprof.host.detect import _repository_metadata
        from huggingface_hub.errors import LocalEntryNotFoundError
        hub = SimpleNamespace(siblings=[SimpleNamespace(rfilename="config.json")])
        with patch("huggingface_hub.hf_hub_download", side_effect=LocalEntryNotFoundError("offline miss")) as download:
            metadata = _repository_metadata("unseen/encoder", "c" * 40, hub)
        self.assertEqual(download.call_count, 1)
        self.assertIn("offline miss", metadata["metadata_errors"][0])

    def test_mirror_metadata_redirect_uses_official_hub_at_same_commit(self):
        from huggingface_hub.errors import FileMetadataError, LocalEntryNotFoundError
        error = LocalEntryNotFoundError("metadata request failed")
        error.__cause__ = FileMetadataError("missing X-Repo-Commit")
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "config.json")
            config.write_text(json.dumps({"model_type": "bert"}))
            hub = SimpleNamespace(pipeline_tag="fill-mask", library_name="transformers", sha="c" * 40,
                                  config={}, tags=[], siblings=[SimpleNamespace(rfilename="config.json")])
            with patch("huggingface_hub.HfApi.model_info", return_value=hub), patch.dict(
                "os.environ", {"HF_ENDPOINT": "https://hf-mirror.com", "HF_FALLBACK_ENDPOINTS": "https://huggingface.co"}, clear=True,
            ), patch("huggingface_hub.hf_hub_download", side_effect=[error, str(config)]) as download:
                info = detect_task("unseen/encoder")
        self.assertFalse(info.metadata_errors)
        self.assertEqual(info.model_config["model_type"], "bert")
        self.assertEqual(download.call_args.kwargs["endpoint"], "https://huggingface.co")
        self.assertEqual(download.call_args.kwargs["revision"], "c" * 40)

    def test_metadata_failure_is_not_reported_as_unsupported_task(self):
        info = ModelResolutionTests.task(metadata_errors=("config.json: connection failed",))
        with self.assertRaises(TaskSupportError) as caught:
            require_task_support(info)
        self.assertIn("[model-metadata][ERROR]", str(caught.exception))
        self.assertNotIn("Unsupported collection task", str(caught.exception))
        self.assertIn("HF_ENDPOINT", str(caught.exception))

    def test_feature_extraction_library_selects_sentence_encoder_for_any_id(self):
        hub = SimpleNamespace(pipeline_tag="feature-extraction", library_name="sentence-transformers",
                              sha="a" * 40, config={"model_type": "bert"}, siblings=[], tags=[])
        with patch("huggingface_hub.HfApi.model_info", return_value=hub):
            info = detect_task("unseen/encoder")
        self.assertEqual(info.runtime_backend, "sentence_transformers")
        require_task_support(info)

    def test_detection_reads_configuration_at_resolved_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "config.json")
            config.write_text(json.dumps({"model_type": "chronos2", "chronos_pipeline_class": "Chronos2Pipeline",
                                          "transformers_version": "4.57.6"}))
            hub = SimpleNamespace(pipeline_tag="time-series-forecasting", library_name="chronos-forecasting",
                                  sha="b" * 40, config={}, tags=[],
                                  siblings=[SimpleNamespace(rfilename="config.json")])
            with patch("huggingface_hub.HfApi.model_info", return_value=hub), patch(
                "huggingface_hub.hf_hub_download", return_value=str(config),
            ) as download:
                info = detect_task("unseen/forecaster")
        self.assertEqual(info.runtime_backend, "chronos")
        self.assertEqual(info.model_config["chronos_pipeline_class"], "Chronos2Pipeline")
        self.assertEqual(download.call_args.kwargs["revision"], "b" * 40)


class ModelResolutionTests(unittest.TestCase):
    @staticmethod
    def task(**kwargs):
        values = dict(model_id="unseen/checkpoint", pipeline_tag="image-text-to-text", task_family="multimodal",
                      runtime_backend="transformers_model", library_name="transformers", model_revision="a" * 40,
                      detection_method="hub_api", model_config={"model_type": "gemma4"})
        values.update(kwargs)
        return TaskInfo(**values)

    def test_native_new_architecture_selects_compatible_environment_without_id_rule(self):
        task = self.task()
        profile = select_runtime_profile(task)
        self.assertEqual(profile.environment.environment_key, "transformers560-cu128")
        self.assertEqual(profile.adapter, "family-default")

    def test_hardware_selection_preserves_compatible_transformers_version(self):
        from acprof.host.runtime_images import configure_runtime_profile
        task = self.task()
        require_task_support(task)
        with patch("acprof.host.docker_runtime._select_nlp_torch_index_url",
                   return_value="https://download.pytorch.org/whl/cpu"):
            profile = configure_runtime_profile(task)
        self.assertEqual(profile.environment.environment_key, "transformers560-cpu")

    def test_saved_version_does_not_force_upgrade_for_an_already_supported_architecture(self):
        task = self.task(model_config={"model_type": "qwen2_vl", "transformers_version": "5.6.0"})
        self.assertEqual(select_runtime_profile(task).profile_id, "multimodal-transformers4576")

    def test_explicit_incompatible_profile_cannot_bypass_architecture_check(self):
        with self.assertRaisesRegex(ValueError, "does not register"):
            select_runtime_profile(self.task(runtime_profile_id="multimodal-transformers4576"))

    def test_unsupported_native_architecture_fails_before_build(self):
        with self.assertRaisesRegex(ValueError, "unregistered_future_arch"):
            select_runtime_profile(self.task(model_config={"model_type": "unregistered_future_arch"}))

    def test_gguf_is_not_accepted_as_a_complete_diffusers_pipeline(self):
        task = self.task(pipeline_tag="text-to-video", task_family="diffusion", runtime_backend="diffusers",
                         library_name="diffusers", model_config={}, repository_files=("README.md", "model.gguf"))
        with self.assertRaisesRegex(TaskSupportError, "GGUF|gguf"):
            require_task_support(task)

    def test_foreign_library_is_not_accepted_as_a_transformers_audio_pipeline(self):
        task = self.task(pipeline_tag="automatic-speech-recognition", task_family="audio",
                         runtime_backend="transformers_pipeline", library_name="pyannote-audio", model_config={})
        with self.assertRaisesRegex(TaskSupportError, "pyannote"):
            require_task_support(task)

    def test_complete_pipeline_reports_metadata_evidence_without_claiming_execution(self):
        task = self.task(pipeline_tag="text-to-image", task_family="diffusion", runtime_backend="diffusers",
                         library_name="diffusers", model_config={}, repository_files=("model_index.json",),
                         repository_metadata={"model_index.json": {"_class_name": "StableDiffusionPipeline"}})
        require_task_support(task)
        self.assertEqual(task.model_resolution["artifact_format"], "diffusers_pipeline")
        self.assertEqual(task.model_resolution["status"], "candidate")


if __name__ == "__main__":
    unittest.main()

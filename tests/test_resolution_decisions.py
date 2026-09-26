"""Static decisions keep correlated evidence and execution observations separate."""
import copy
import unittest

from acprof.host.detect import TaskInfo
from acprof.model_resolution import discover_model_candidates, require_resolved_candidate


def candidate(*, tag="unknown", config=None, hub=None):
    task = TaskInfo("example/model", tag, "unknown", "transformers_pipeline",
                    "transformers", "a" * 40, "hub_api")
    task.model_config = config or {}
    task.repository_metadata = {"config.json": task.model_config}
    task.hub_metadata = hub or {}
    task.model_resolution = discover_model_candidates(task)
    return task


class ResolutionDecisionTests(unittest.TestCase):
    def test_transformers_info_resolves_missing_task_without_model_import(self):
        task = candidate(hub={"transformers_info": {"pipeline_tag": "text-generation",
                                                     "auto_model": "AutoModelForCausalLM"}})
        self.assertEqual(task.pipeline_tag, "text-generation")
        self.assertEqual(task.model_resolution["semantics"]["status"], "declared")
        self.assertEqual(task.model_resolution["runtime_validation"]["status"], "not_run")

    def test_conflicting_hub_observations_abstain_even_when_runtime_passed(self):
        task = candidate(tag="fill-mask", hub={"transformers_info": {"pipeline_tag": "text-generation"}})
        task.model_resolution["runtime_validation"] = {"status": "verified"}
        with self.assertRaisesRegex(ValueError, "ambiguous|conflict"):
            require_resolved_candidate(task)

    def test_explicit_choice_preserves_conflict_evidence(self):
        task = candidate(tag="fill-mask", hub={"transformers_info": {"pipeline_tag": "text-generation"}})
        task.model_resolution = discover_model_candidates(task, override_tag="text-generation")
        require_resolved_candidate(task)
        self.assertEqual(task.model_resolution["semantics"]["status"], "explicit")
        self.assertTrue(task.model_resolution["provenance"]["overridden_conflicts"])

    def test_correlated_observations_share_source_and_have_no_vote_score(self):
        task = candidate(tag="text-generation", config={"architectures": ["LlamaForCausalLM"]},
                         hub={"transformers_info": {"pipeline_tag": "text-generation",
                                                     "auto_model": "AutoModelForCausalLM"}})
        provenance = task.model_resolution["provenance"]
        hub = [item for item in provenance["observations"] if item["source_id"] == "hub"]
        self.assertGreaterEqual(len(hub), 2)
        self.assertEqual(provenance["sources"]["hub"]["derived_from"], ["repository_snapshot"])
        self.assertNotIn("confidence_score", provenance)

    def test_static_identity_changes_with_evidence_but_not_runtime(self):
        first = candidate(tag="text-generation", config={"model_type": "llama"})
        second = candidate(tag="text-generation", config={"model_type": "gpt2"})
        identity = first.model_resolution["provenance"]["identity_sha256"]
        self.assertNotEqual(identity, second.model_resolution["provenance"]["identity_sha256"])
        first.model_resolution["runtime_validation"] = {"status": "verified"}
        self.assertEqual(identity, first.model_resolution["provenance"]["identity_sha256"])

    def test_bare_auto_model_does_not_prove_a_task(self):
        task = candidate(hub={"transformers_info": {"auto_model": "AutoModel"}})
        with self.assertRaises(ValueError):
            require_resolved_candidate(task)

    def test_custom_pipeline_loader_hint_cannot_supply_missing_task_semantics(self):
        task = candidate(config={"custom_pipelines": {
            "custom-task": {"impl": "pipeline.CustomPipeline", "pt": ["AutoModel"]},
        }}, hub={"transformers_info": {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"}})
        with self.assertRaisesRegex(ValueError, "task semantics are unknown"):
            require_resolved_candidate(task)
        self.assertFalse(task.model_resolution["candidates"])

    def test_generic_tag_conflicts_without_matching_custom_pipeline_loader(self):
        for config in ({}, {"custom_pipelines": {
            "custom-task": {"impl": "pipeline.CustomPipeline", "pt": ["AutoModelForCausalLM"]},
        }}):
            with self.subTest(config=config):
                task = candidate(tag="text-generation", config=config, hub={
                    "transformers_info": {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"},
                })
                with self.assertRaisesRegex(ValueError, "conflict"):
                    require_resolved_candidate(task)

    def test_incompatible_declared_head_requires_explicit_resolution(self):
        for config, hub in (({"architectures": ["BertForMaskedLM"]}, {}),
                            ({}, {"transformers_info": {"auto_model": "AutoModelForMaskedLM"}})):
            with self.subTest(config=config, hub=hub):
                task = candidate(tag="text-generation", config=config, hub=hub)
                with self.assertRaisesRegex(ValueError, "conflict|ambiguous"):
                    require_resolved_candidate(task)

    def test_shared_loader_is_not_a_translation_semantic_conflict(self):
        task = candidate(tag="translation", hub={"transformers_info": {"auto_model": "AutoModelForSeq2SeqLM"}})
        require_resolved_candidate(task)
        self.assertEqual(task.pipeline_tag, "translation")

    def test_native_registry_reverse_lookup_resolves_architecture(self):
        task = candidate(config={"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"})
        self.assertEqual(task.pipeline_tag, "text-generation")
        self.assertTrue(any(item["field"] == "config.architectures" for item in
                            task.model_resolution["provenance"]["observations"]))

    def test_runtime_observation_cannot_rewrite_static_semantics(self):
        from acprof.model_contract import record_runtime_validation
        task = candidate(tag="text-generation")
        before = copy.deepcopy(task.model_resolution)
        record_runtime_validation(task, {"mode": "full", "status": "ok", "image_id": "sha256:" + "b" * 64,
                                         "build_fingerprint": "c" * 64, "payload_sha256": "d" * 64,
                                         "devices": {"off": {"status": "ok"}}})
        self.assertEqual(task.model_resolution["semantics"], before["semantics"])
        self.assertEqual(task.model_resolution["provenance"], before["provenance"])
        self.assertEqual(task.model_resolution["runtime_validation"]["status"], "verified")

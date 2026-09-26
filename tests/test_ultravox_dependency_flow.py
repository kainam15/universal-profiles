"""Pinned upstream source is text-only acceptance input, never imported."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from acprof.host.detect import TaskInfo
from acprof.model_contract import apply_model_contract
from acprof.model_resolution import discover_model_candidates
from acprof.model_spec import task_model_spec


FIXTURE = Path(__file__).parent / "fixtures" / "ultravox_dependency_snapshot"
MODEL = "fixie-ai/ultravox-v0_5-llama-3_2-1b"


class UltravoxDependencyFlowTests(unittest.TestCase):
    def discover(self, model=MODEL, version="4.57.6"):
        repositories = json.loads((FIXTURE / "repositories.json").read_text())
        metadata = {p.name: json.loads(p.read_text()) for p in FIXTURE.glob("*.json")
                    if p.name not in {"repositories.json", "source_sha256.json"}}
        config = metadata["config.json"]
        task = TaskInfo(model, "audio-text-to-text", "multimodal", "transformers_model", "transformers",
                        repositories[MODEL]["revision"], "hub_api", model_config=config,
                        repository_files=tuple(repositories[MODEL]["files"]), repository_metadata=metadata,
                        hub_metadata={"transformers_info": {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"}})
        task.model_resolution = discover_model_candidates(task)

        def read_text(name):
            return (FIXTURE / (name + ".txt" if name.endswith(".py") else name)).read_text()

        lookup = Mock(side_effect=lambda repo, revision: copy.deepcopy(repositories[repo]))
        with patch("acprof.runtime_profiles._transformers_version", return_value=version):
            apply_model_contract(task, read_text, resolve_repository=lookup)
        return task, lookup

    def test_snapshot_hashes_match_fixed_source_input(self):
        for name, digest in json.loads((FIXTURE / "source_sha256.json").read_text()).items():
            with self.subTest(name=name):
                path = FIXTURE / (name + ".txt" if name.endswith(".py") else name)
                data = path.read_bytes()
                # The upstream config lacks a final newline; the fixture adds
                # one for repository text hygiene without changing JSON data.
                if name == "config.json":
                    data = data.removesuffix(b"\n")
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_zero_unresolved_dependencies_without_checkpoint_name_routing(self):
        example = json.loads((Path(__file__).parents[1] / "examples/multimodal/ultravox.model.json").read_text())
        for model in (MODEL, "arbitrary/composite-audio"):
            with self.subTest(model=model):
                task, lookup = self.discover(model)
                report = task.model_resolution["contract"]
                self.assertEqual(report["unresolved_fields"], [])
                self.assertEqual(report["status"], "resolved", report["fields"].get("dependencies"))
                self.assertEqual(task.model_resolution["status"], "candidate")
                self.assertEqual(task.runtime_backend, "transformers_pipeline")
                spec = task_model_spec(task)
                for field in ("task", "pipeline_task", "format", "multimodal"):
                    self.assertEqual(spec[field], example[field])
                dependencies = {d["repo_id"]: d for d in spec["dependencies"]}
                self.assertEqual({repo: d["revision"] for repo, d in dependencies.items()},
                                 {d["repo_id"]: d["revision"] for d in example["dependencies"]})
                self.assertEqual(dependencies["meta-llama/Llama-3.2-1B-Instruct"]["allow_patterns"],
                                 ["config.json", "generation_config.json", "model.safetensors"])
                self.assertEqual(dependencies["openai/whisper-large-v3-turbo"]["allow_patterns"],
                                 ["config.json", "preprocessor_config.json"])
                candidates = report["dependency_candidates"]
                self.assertFalse(any(c["activation"] == "unknown" for c in candidates))
                self.assertTrue(any(c["dependency_kind"] == "main_model" and c["loader"] == "super()" for c in candidates))
                self.assertTrue(any(c.get("alternative", {}).get("branch") == "fallback"
                                    and c["activation"] == "inactive" for c in candidates))
                self.assertTrue(any(c["role"] == "weights" and c["activation"] == "inactive"
                                    and "audio_model_id" in c["expression"] for c in candidates))
                self.assertEqual(lookup.call_count, 2)
                self.assertEqual(report["runtime_validation"], "not_run")

    def test_unreviewed_transformers_version_keeps_weight_lifecycle_unknown(self):
        task, _ = self.discover(version="99.0.0")
        report = task.model_resolution["contract"]
        self.assertIn("dependencies", report["unresolved_fields"])
        self.assertFalse(task_model_spec(task))
        self.assertTrue(any(c["role"] == "weights" and c["activation"] == "unknown"
                            for c in report["dependency_candidates"]))
        self.assertFalse(any("model.safetensors" in d["allow_patterns"]
                             for d in report["draft_spec"].get("dependencies", [])))


if __name__ == "__main__":
    unittest.main()

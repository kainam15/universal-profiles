"""Contract synthesis must remain static, pinned, conservative and explainable."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.host.detect import detect_task
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.model_spec import task_model_spec, validate_model_spec


FIXTURE = Path(__file__).parent / "fixtures" / "custom_pipeline_audio_like"
SOURCE = (FIXTURE / "pipeline.py").read_text()
CONFIG = json.loads((FIXTURE / "config.json").read_text())
SHA = "a" * 40
EXPECTED: dict = {
    "schema_version": 1, "format": "transformers-pipeline", "task": "audio-text-to-text",
    "pipeline_task": "listen-and-answer",
    "multimodal": {"inputs": {"prompt": "text", "audio": "audio", "sampling_rate": "sampling_rate"},
                   "forward_kwargs": {"max_new_tokens": "$max_new_tokens", "temperature": 0}},
}


class ModelContractTests(unittest.TestCase):
    def discover(self, source=SOURCE, config=None, *, revision=SHA, spec=None, readme=None,
                 dependency_lookup=None, **options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = {"config.json": json.dumps(CONFIG if config is None else config), "pipeline.py": source}
            if spec is not None:
                documents["acprof_model.json"] = json.dumps(spec)
            if readme is not None:
                documents["README.md"] = readme
            for name, text in documents.items():
                (root / name).write_text(text)
            hub = SimpleNamespace(sha=revision, pipeline_tag="audio-text-to-text", library_name="transformers",
                                  config={}, siblings=[SimpleNamespace(rfilename=name) for name in documents])
            self.downloads = []

            def download(**kwargs):
                self.assertEqual(kwargs["revision"], SHA if revision == SHA else revision)
                self.downloads.append(kwargs["filename"])
                return str(root / kwargs["filename"])

            def model_info(repo_id, **kwargs):
                # Route all Hub reads through one mock so dependency responses
                # cannot be overwritten by a nested main-repository patch.
                if repo_id == "arbitrary/audio-model":
                    return hub
                if dependency_lookup is None:
                    raise AssertionError(f"Unexpected dependency lookup: {repo_id}")
                return dependency_lookup(repo_id, **kwargs)

            with patch("huggingface_hub.HfApi.model_info", side_effect=model_info), patch(
                "huggingface_hub.hf_hub_download", side_effect=download,
            ):
                return detect_task("arbitrary/audio-model", **options)

    def test_generates_v1_contract_without_checkpoint_specific_routing(self):
        task = self.discover()
        self.assertEqual(task_model_spec(task), EXPECTED)
        validate_model_spec(task_model_spec(task))
        require_task_support(task)
        self.assertEqual(task.runtime_profile_id, "custom-multimodal-cu128")
        self.assertEqual(task.model_resolution["status"], "candidate")
        report = task.model_resolution["contract"]
        self.assertEqual(report["status"], "resolved")
        self.assertEqual(report["fields"]["task"]["state"], "declared")
        self.assertEqual(report["fields"]["multimodal.inputs.prompt"]["state"], "derived")
        self.assertEqual(report["fields"]["multimodal.forward_kwargs.temperature"]["value"], 0)
        self.assertFalse(any(field["state"] == "verified" for field in report["fields"].values()))
        self.assertIn("pipeline.py", self.downloads)
        self.assertNotIn("turns", task_model_spec(task)["multimodal"]["inputs"])

    def test_required_unknown_input_is_actionable_and_not_executable(self):
        task = self.discover(SOURCE.replace('turns = inputs.get("turns", [])', 'speaker = inputs["speaker"]\n        turns = []'))
        self.assertEqual(task.model_resolution["status"], "needs_configuration")
        self.assertFalse(task_model_spec(task))
        with self.assertRaisesRegex(TaskSupportError, "speaker"):
            require_task_support(task)

    def test_dynamic_keys_and_unproven_sampling_do_not_get_guessed(self):
        for index, source in enumerate((
            SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs.get(runtime_key(), "Listen.")'),
            SOURCE.replace("temperature = temperature or None", "temperature = temperature + 1"),
            SOURCE.replace("do_sample = temperature is not None", "do_sample = True"),
            SOURCE.replace('generation_keys = ["temperature", "max_new_tokens", "repetition_penalty"]',
                           'generation_keys = get_keys()'),
        )):
            with self.subTest(case=index):
                task = self.discover(source)
                self.assertEqual(task.model_resolution["status"], "needs_configuration")
                self.assertFalse(task_model_spec(task))

    def test_dynamic_mapping_and_generation_mutations_are_unresolved(self):
        changes = (
            ('generation_kwargs = {k:', 'generation_keys.append(dynamic_key())\n        generation_kwargs = {k:'),
            ('return {}, generation_kwargs, {}', 'generation_kwargs.update(dynamic_kwargs())\n        return {}, generation_kwargs, {}'),
            ('turns = inputs.get("turns", [])', 'inputs = rewrite(inputs)\n        turns = inputs.get("turns", [])'),
            ('turns = inputs.get("turns", [])', 'eval(dynamic_code())\n        turns = inputs.get("turns", [])'),
            ('max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,',
             'max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, **dynamic_kwargs(),'),
        )
        for index, (old, new) in enumerate(changes):
            with self.subTest(case=index):
                task = self.discover(SOURCE.replace(old, new))
                self.assertEqual(task.model_resolution["status"], "needs_configuration")
                self.assertFalse(task_model_spec(task))

    def test_literal_inputs_readme_evidence_and_report_export(self):
        from acprof.model_contract import write_model_resolution
        source = SOURCE.replace('inputs.get("audio", None)', 'inputs["audio"]')
        task = self.discover(source, readme='''```python
pipe = pipeline("listen-and-answer")
pipe({"turns": turns, "audio": audio, "sampling_rate": rate}, max_new_tokens=3)
```''')
        self.assertEqual(task_model_spec(task), EXPECTED)
        report = task.model_resolution["contract"]
        self.assertTrue(report["fields"]["pipeline.inputs.audio"]["value"]["required"])
        self.assertEqual(report["fields"]["documentation.example.0"]["state"], "derived")
        with tempfile.TemporaryDirectory() as directory:
            path = write_model_resolution(task, directory)
            self.assertEqual(json.loads(path.read_text()), task.model_resolution)
            self.assertFalse(Path(directory, "acprof_model.json").exists())

    def test_config_loader_candidates_include_plain_config_and_remain_unpinned(self):
        from acprof.model_source_analysis import dependency_candidates
        source = '''AutoConfig.from_pretrained(config.audio_model_id)
AutoModel.from_pretrained(config.text_model_name_or_path)
GenerationConfig.from_pretrained("example/generation")
AutoProcessor.from_pretrained(dynamic_repo())
'''
        candidates = dependency_candidates(source, "module.py", {
            "audio_model_id": "example/audio", "text_model_name_or_path": "example/text",
        }, "example/main")
        self.assertEqual([(item["repo_id"], item["role"]) for item in candidates], [
            ("example/audio", "metadata"), ("example/text", "weights"),
            ("example/generation", "generation_metadata"), (None, "processor"),
        ])
        self.assertTrue(all("revision" not in item for item in candidates))

    def test_cache_identity_changes_with_revision_and_source_and_is_not_mutable(self):
        from acprof.model_source_analysis import analyze_pipeline
        first = self.discover().model_resolution["contract"]
        second = self.discover().model_resolution["contract"]
        self.assertEqual(first["cache_key"], second["cache_key"])
        revised = self.discover(revision="b" * 40).model_resolution["contract"]
        changed = self.discover(SOURCE + "\n# another source\n").model_resolution["contract"]
        self.assertNotEqual(first["cache_key"], revised["cache_key"])
        self.assertNotEqual(first["cache_key"], changed["cache_key"])
        result = analyze_pipeline(SOURCE, "pipeline.py", "AudioPipeline")
        result["inputs"].clear()
        self.assertIn("audio", analyze_pipeline(SOURCE, "pipeline.py", "AudioPipeline")["inputs"])

    def test_local_declaration_and_hub_conflict_stays_visible_in_provenance(self):
        spec = copy.deepcopy(EXPECTED)
        spec["task"] = "image-text-to-text"
        spec["multimodal"]["inputs"] = {"prompt": "text", "image": "image"}
        task = self.discover(spec=spec)
        self.assertEqual(task.model_resolution["status"], "ambiguous")
        self.assertEqual(task.model_resolution["contract"]["status"], "needs_confirmation")

    def test_reanalysis_cannot_reuse_a_previous_generated_spec_as_authority(self):
        from acprof.model_contract import apply_model_contract
        task = self.discover()
        changed_source = SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["messages"]')
        apply_model_contract(task, lambda _: changed_source)
        self.assertEqual(task.model_resolution["status"], "needs_configuration")
        self.assertFalse(task_model_spec(task))

    def test_structured_chat_and_oversized_source_remain_unresolved(self):
        for source in (SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs.get("prompt", [])'),
                       SOURCE + "\n#" + "x" * (256 * 1024)):
            task = self.discover(source)
            self.assertEqual(task.model_resolution["status"], "needs_configuration")
            self.assertFalse(task_model_spec(task))

    def test_explicit_sampling_parameter_is_forwarded_without_temperature_guess(self):
        source = SOURCE.replace('"temperature", "max_new_tokens"', '"do_sample", "max_new_tokens"')
        source = source.replace('temperature=None, max_new_tokens=None', 'do_sample=False, max_new_tokens=None')
        source = source.replace('        temperature = temperature or None\n        do_sample = temperature is not None\n', '')
        source = source.replace('do_sample=do_sample, temperature=temperature,', 'do_sample=do_sample,')
        task = self.discover(source)
        self.assertEqual(task_model_spec(task)["multimodal"]["forward_kwargs"],
                         {"max_new_tokens": "$max_new_tokens", "do_sample": False})

    def test_multiple_pipelines_require_selection_without_reading_code(self):
        config = copy.deepcopy(CONFIG)
        config["custom_pipelines"]["other-pipeline"] = {"impl": "other.OtherPipeline", "pt": ["AutoModel"]}
        task = self.discover(config=config)
        self.assertEqual(task.model_resolution["status"], "ambiguous")
        self.assertNotIn("pipeline.py", self.downloads)

    def test_author_spec_wins_and_no_remote_python_is_read(self):
        spec = copy.deepcopy(EXPECTED)
        spec["multimodal"]["inputs"] = {"text": "text", "audio": "audio", "rate": "sampling_rate"}
        task = self.discover(spec=spec)
        self.assertEqual(task_model_spec(task), spec)
        self.assertNotIn("pipeline.py", self.downloads)

    def test_remote_source_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory, "executed")
            source = f'open({str(marker)!r}, "w").write("bad")\n' + SOURCE
            script = '''import json, sys
sys.path.insert(0, "tests")
from test_model_contract import ModelContractTests, task_model_spec
task = ModelContractTests().discover(sys.stdin.read())
assert "torch" not in sys.modules and "transformers" not in sys.modules
print(json.dumps(task_model_spec(task)))
'''
            process = subprocess.run([sys.executable, "-c", script], input=source, capture_output=True,
                                     text=True, cwd=FIXTURE.parents[2], timeout=30, check=True)
            self.assertFalse(marker.exists())
            self.assertEqual(json.loads(process.stdout), EXPECTED)

    def test_unsafe_code_references_are_rejected_before_source_download(self):
        for reference in ("../evil.Pipeline", "other/repo--evil.Pipeline"):
            config = copy.deepcopy(CONFIG)
            config["custom_pipelines"]["listen-and-answer"]["impl"] = reference
            with self.subTest(reference=reference):
                task = self.discover(config=config)
                self.assertEqual(task.model_resolution["status"], "needs_configuration")
                self.assertNotIn("pipeline.py", self.downloads)

    def test_unpinned_revision_never_reads_contract_sources(self):
        task = self.discover(revision="main")
        self.assertEqual(self.downloads, [])
        with self.assertRaisesRegex(TaskSupportError, "SHA|revision"):
            require_task_support(task)

    def test_dependency_candidates_block_execution_but_keep_generated_draft(self):
        source = SOURCE.replace("class AudioPipeline(Pipeline):", '''class AudioPipeline(Pipeline):
    def __init__(self, model, **kwargs):
        self.processor = AutoProcessor.from_pretrained(model.config.audio_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained("example/tokenizer")
''')
        config = dict(CONFIG, audio_model_id="example/audio")
        with patch("acprof.host.detect.dependency_metadata", side_effect=OSError("metadata unavailable")):
            task = self.discover(source, config)
        self.assertEqual(task.model_resolution["status"], "needs_configuration")
        report = task.model_resolution["contract"]
        self.assertEqual(report["draft_spec"], EXPECTED)
        self.assertEqual({(d["repo_id"], d["role"]) for d in report["dependency_candidates"]},
                         {("example/audio", "processor"), ("example/tokenizer", "tokenizer")})
        self.assertFalse(task_model_spec(task))


if __name__ == "__main__":
    unittest.main()

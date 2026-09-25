"""Dependency resolution pins role-specific files without loading repository code."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import test_model_contract as fixture
from acprof.model_spec import task_model_spec


class ModelDependencyTests(unittest.TestCase):
    def discover(self, source, files, *, sha="b" * 40):
        hub = SimpleNamespace(sha=sha, siblings=[SimpleNamespace(rfilename=name) for name in files])
        lookup = Mock(return_value=hub)
        task = fixture.ModelContractTests().discover(source, dependency_lookup=lookup)
        return task, lookup

    def test_tokenizer_is_pinned_without_downloading_weights(self):
        source = fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")\n'
        task, lookup = self.discover(source, ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"])
        spec = task_model_spec(task)
        self.assertTrue(spec, task.model_resolution)
        self.assertEqual(task.model_revision, fixture.SHA)
        dep = spec["dependencies"][0]
        self.assertEqual(dep["revision"], "b" * 40)
        self.assertEqual(dep["allow_patterns"], ["config.json", "tokenizer.json", "tokenizer_config.json"])
        lookup.assert_called_once_with("example/tokenizer", revision="main", files_metadata=False)
        self.assertEqual(task.model_resolution["contract"]["runtime_validation"], "not_run")

    def test_roles_merge_but_metadata_never_selects_weights(self):
        source = fixture.SOURCE + '\nAutoConfig.from_pretrained("example/shared")\nAutoProcessor.from_pretrained("example/shared")\n'
        task, lookup = self.discover(source, ["config.json", "preprocessor_config.json", "model.safetensors"])
        self.assertEqual(task_model_spec(task)["dependencies"], [{"repo_id": "example/shared", "revision": "b" * 40,
                          "allow_patterns": ["config.json", "preprocessor_config.json"]}])
        self.assertEqual(lookup.call_count, 1)

    def test_tokenizer_prefix_does_not_include_nested_repositories_or_weights(self):
        source = fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")\n'
        files = ["tokenizer.json", "tokenizer.model", "chat_templates/default.jinja", "tokenizer_weights.bin",
                 "tokenizer/model.safetensors", "tokenizer/backups/tokenizer.json", "backup/spiece.model",
                 "chat_templates/backup/template.jinja"]
        task, _ = self.discover(source, files)
        self.assertEqual(task_model_spec(task)["dependencies"][0]["allow_patterns"],
                         ["chat_templates/default.jinja", "tokenizer.json", "tokenizer.model"])

    def test_dynamic_and_unknown_loader_never_trigger_repository_lookup(self):
        for call in ('AutoTokenizer.from_pretrained(dynamic_repo())', 'CustomLoader.from_pretrained("example/unknown")'):
            task, lookup = self.discover(fixture.SOURCE + '\n' + call, ["tokenizer.json"])
            self.assertFalse(task_model_spec(task))
            lookup.assert_not_called()

    def test_unpinned_response_and_empty_selection_remain_actionable(self):
        for sha, files in (("main", ["tokenizer.json"]), ("b" * 40, ["model.safetensors"])):
            task, _ = self.discover(fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")', files, sha=sha)
            self.assertFalse(task_model_spec(task))
            self.assertIn("dependencies", task.model_resolution["contract"]["unresolved_fields"])

    def test_dependency_commit_changes_contract_identity(self):
        source = fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")'
        first, _ = self.discover(source, ["tokenizer.json"])
        second, _ = self.discover(source, ["tokenizer.json"], sha="c" * 40)
        self.assertNotEqual(first.model_resolution["contract"]["cache_key"], second.model_resolution["contract"]["cache_key"])

    def test_author_dependencies_are_not_re_resolved(self):
        declaration = copy.deepcopy(fixture.EXPECTED)
        declaration["dependencies"] = [{"repo_id": "example/tokenizer", "revision": "c" * 40}]
        lookup = Mock(side_effect=AssertionError("Author dependencies must stay pinned"))
        task = fixture.ModelContractTests().discover(spec=declaration, dependency_lookup=lookup)
        lookup.assert_not_called()
        self.assertEqual(task_model_spec(task), declaration)

    def test_conditional_weight_load_does_not_download_a_potential_base_model(self):
        for suffix in ('\nif dynamic_training_mode():\n    AutoModel.from_pretrained("example/base")',
                       '\nmodels = [AutoModel.from_pretrained("example/base") for item in dynamic_items()]',
                       '\nmatch dynamic_mode():\n    case "train":\n        AutoModel.from_pretrained("example/base")'):
            with self.subTest(source=suffix):
                task, lookup = self.discover(fixture.SOURCE + suffix, ["config.json", "model.safetensors"])
                self.assertFalse(task_model_spec(task))
                lookup.assert_not_called()

    def test_source_revision_cannot_be_discarded_or_overridden_by_dynamic_kwargs(self):
        for arguments in ('revision="release"', '**load_options'):
            task, lookup = self.discover(fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer", ' + arguments + ')',
                                        ["tokenizer.json"])
            self.assertFalse(task_model_spec(task))
            lookup.assert_not_called()

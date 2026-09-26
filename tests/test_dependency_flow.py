"""Static dependency planning must prove branches without executing model code."""
import copy
import unittest
from unittest.mock import Mock

import test_model_contract as fixture
from acprof.host.detect import TaskInfo
from acprof.model_contract import resolve_model_contract
from acprof.model_dependencies import resolve_dependencies
from acprof.model_source_analysis import dependency_candidates


class DependencyFlowTests(unittest.TestCase):
    def analyze(self, source, *, config=None, sources=None, files=(), metadata=None):
        config = {**copy.deepcopy(fixture.CONFIG), **(config or {})}
        documents = {"pipeline.py": fixture.SOURCE + "\n" + source, **(sources or {})}
        task = TaskInfo("example/main", "audio-text-to-text", "multimodal", "transformers_model",
                        "transformers", fixture.SHA, "hub_api", model_config=config,
                        repository_metadata={"config.json": config, **(metadata or {})},
                        repository_files=tuple([*documents, "config.json", *files]))
        lookup = Mock(return_value={"revision": "b" * 40, "files": [
            "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
            "model.safetensors", "generation_config.json",
        ]})
        report = resolve_model_contract(task, documents.__getitem__, resolve_repository=lookup)
        return report, lookup

    def test_config_branches_are_role_specific(self):
        report, lookup = self.analyze('''
if config.audio_model_id is not None:
    AutoModel.from_pretrained(config.audio_model_id)
AutoFeatureExtractor.from_pretrained(config.audio_model_id or config.audio_config._name_or_path)
''', config={"audio_model_id": None, "audio_config": {"_name_or_path": "example/audio"}})
        self.assertEqual(report["status"], "resolved", report["fields"].get("dependencies"))
        candidates = report["dependency_candidates"]
        self.assertEqual([(c["role"], c["activation"]) for c in candidates],
                         [("weights", "inactive"), ("processor", "active")])
        self.assertEqual(report["draft_spec"]["dependencies"], [{"repo_id": "example/audio",
            "revision": "b" * 40, "allow_patterns": ["config.json", "preprocessor_config.json"]}])
        lookup.assert_called_once_with("example/audio", "main")

    def test_known_true_else_and_unknown_conditions_keep_distinct_states(self):
        report, lookup = self.analyze('''
if config.enabled:
    AutoTokenizer.from_pretrained("example/active")
else:
    AutoModel.from_pretrained("example/inactive")
if dynamic_training_mode():
    AutoModel.from_pretrained("example/potential")
''', config={"enabled": True})
        self.assertEqual({c["repo_id"]: c["activation"] for c in report["dependency_candidates"]},
                         {"example/active": "active", "example/inactive": "inactive", "example/potential": "unknown"})
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_called_once_with("example/active", "main")

    def test_unknown_and_false_does_not_load_weights(self):
        report, lookup = self.analyze('''
if dynamic_training_mode() and config.audio_model_id is not None:
    AutoModel.from_pretrained("example/unused")
''', config={"audio_model_id": None})
        self.assertEqual(report["status"], "resolved")
        self.assertEqual(report["dependency_candidates"][0]["activation"], "inactive")
        lookup.assert_not_called()

    def test_wrapper_arguments_and_keyword_only_revision_are_bound(self):
        revision = "c" * 40
        source = f'''
def load_tokenizer(pretrained_model_name_or_path, *, revision):
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, revision=revision)
load_tokenizer(config.text_model_id, revision="{revision}")
'''
        candidates = dependency_candidates(source, "helpers.py", {"text_model_id": "example/text"}, "example/main")
        lookup = Mock(return_value={"revision": revision, "files": ["tokenizer.json"]})
        deps, errors = resolve_dependencies(candidates, lookup)
        self.assertFalse(errors, errors)
        self.assertEqual(deps, [{"repo_id": "example/text", "revision": revision,
                                 "allow_patterns": ["tokenizer.json"]}])
        self.assertEqual(candidates[0]["activation"], "active")
        self.assertGreaterEqual(len(candidates[0]["call_chain"]), 2)

    def test_relative_wrapper_star_args_keep_call_site_identity(self):
        report, lookup = self.analyze('''
from .helpers import tokenizer as load
load(config.text_model_id)
''', config={"text_model_id": "example/text"}, sources={"helpers.py": '''
def tokenizer(*args, **kwargs):
    return AutoTokenizer.from_pretrained(*args, **kwargs)
'''})
        self.assertEqual(report["status"], "resolved", report["fields"].get("dependencies"))
        lookup.assert_called_once_with("example/text", "main")

    def test_wrapper_unknown_kwargs_do_not_discard_a_revision(self):
        report, lookup = self.analyze('''
def tokenizer(*args, **kwargs):
    return AutoTokenizer.from_pretrained(*args, **kwargs)
tokenizer("example/text", **dynamic_options())
''')
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_declared_main_model_super_forwarding_is_not_a_dependency(self):
        report, lookup = self.analyze('', config={"auto_map": {"AutoModel": "model.ExampleModel"}}, sources={"model.py": '''
from transformers import PreTrainedModel
class ExampleModel(PreTrainedModel):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)
'''})
        self.assertEqual(report["status"], "resolved", report["fields"].get("dependencies"))
        main = [c for c in report["dependency_candidates"] if c.get("dependency_kind") == "main_model"]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0]["repo_id"], "example/main")
        lookup.assert_not_called()

    def test_unrelated_super_forwarding_is_not_mistaken_for_main_model(self):
        report, lookup = self.analyze('''
class OtherLoader:
    def from_pretrained(self, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)
OtherLoader().from_pretrained("example/other")
''')
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_primary_tokenizer_files_deactivate_only_its_fallback(self):
        source = '''
try:
    tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("example/fallback")
'''
        for files, metadata, inactive in (
            (("tokenizer.json", "tokenizer_config.json"), {"tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"}}, True),
            (("tokenizer_config.json",), {"tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"}}, False),
            (("tokenizer.json", "tokenizer_config.json"), {"tokenizer_config.json": {"auto_map": {"AutoTokenizer": ["custom.Tokenizer", None]}}}, False),
        ):
            with self.subTest(files=files, metadata=metadata):
                report, lookup = self.analyze(source, files=files, metadata=metadata,
                                             sources={"custom.py": "class Tokenizer:\n    pass\n"})
                fallback = next(c for c in report["dependency_candidates"] if c["repo_id"] == "example/fallback")
                self.assertEqual(fallback["activation"], "inactive" if inactive else "unknown")
                self.assertEqual(fallback["alternative"]["branch"], "fallback")
                self.assertEqual(report["status"], "resolved" if inactive else "needs_confirmation")
                lookup.assert_not_called()

    def test_additional_fallible_statement_keeps_fallback_unknown(self):
        report, lookup = self.analyze('''
try:
    tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
    arbitrary_operation()
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("example/fallback")
''', files=("tokenizer.json", "tokenizer_config.json"),
            metadata={"tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"}})
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_transparent_wrapper_preserves_primary_fallback_proof(self):
        report, lookup = self.analyze('''
def tokenizer(*args, **kwargs):
    return AutoTokenizer.from_pretrained(*args, **kwargs)
try:
    result = tokenizer(config._name_or_path)
except Exception:
    result = tokenizer("example/fallback")
''', files=("tokenizer.json", "tokenizer_config.json"),
            metadata={"tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"}})
        self.assertEqual(report["status"], "resolved", report["fields"].get("dependencies"))
        self.assertEqual(next(c for c in report["dependency_candidates"] if c["repo_id"] == "example/fallback")["activation"], "inactive")
        lookup.assert_not_called()

    def test_primary_dynamic_keyword_or_wrapper_side_effect_keeps_fallback_unknown(self):
        for source in ('''
try:
    result = AutoTokenizer.from_pretrained(config._name_or_path, local_files_only=dynamic_option())
except Exception:
    result = AutoTokenizer.from_pretrained("example/fallback")
''', '''
def tokenizer(repo):
    result = AutoTokenizer.from_pretrained(repo)
    result.add_special_tokens(dynamic_options())
    return result
try:
    result = tokenizer(config._name_or_path)
except Exception:
    result = tokenizer("example/fallback")
'''):
            with self.subTest(source=source):
                report, lookup = self.analyze(source, files=("tokenizer.json", "tokenizer_config.json"),
                    metadata={"tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"}})
                self.assertEqual(report["status"], "needs_confirmation")
                lookup.assert_not_called()

    def test_known_config_can_be_invalidated_by_assignment(self):
        report, lookup = self.analyze('''
config.audio_model_id = dynamic_repo()
if config.audio_model_id is not None:
    AutoModel.from_pretrained(config.audio_model_id)
''', config={"audio_model_id": None})
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_recursive_wrapper_stays_unresolved_without_execution(self):
        report, lookup = self.analyze('''
def recursive(repo):
    return recursive(repo)
recursive("example/text")
''')
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_short_circuit_loader_is_not_lost_after_unknown_value(self):
        report, lookup = self.analyze('tokenizer = existing_tokenizer() or AutoTokenizer.from_pretrained("example/maybe")')
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_decorated_and_async_wrappers_cannot_prove_downloads(self):
        for header in ('@dynamic_decorator\ndef load(repo):', 'async def load(repo):'):
            with self.subTest(header=header):
                report, lookup = self.analyze(header + '\n    return AutoTokenizer.from_pretrained(repo)\nload("example/text")')
                self.assertEqual(report["status"], "needs_confirmation")
                lookup.assert_not_called()

    def test_unknown_keyword_binding_cannot_use_default_repository(self):
        report, lookup = self.analyze('''
def load(repo="example/default"):
    return AutoTokenizer.from_pretrained(repo)
load(**dynamic_options())
''')
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_unknown_loop_or_match_cannot_prove_following_repository(self):
        for statement in ('for item in dynamic_items():\n    repo = "example/text"',
                          'match dynamic_mode():\n    case "train":\n        repo = "example/text"'):
            with self.subTest(statement=statement):
                report, lookup = self.analyze('repo = None\n' + statement + '\nAutoTokenizer.from_pretrained(repo)')
                self.assertEqual(report["status"], "needs_confirmation")
                lookup.assert_not_called()

    def test_mutation_through_helper_cannot_leave_stale_config_proof(self):
        report, lookup = self.analyze('''
def change(cfg):
    if dynamic_mode():
        cfg.audio_model_id = "example/audio"
change(config)
if config.audio_model_id is not None:
    AutoModel.from_pretrained(config.audio_model_id)
''', config={"audio_model_id": None})
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_loader_name_and_main_repository_do_not_bypass_unknown_loader(self):
        for source in ('from unrelated import AutoTokenizer\nAutoTokenizer.from_pretrained("example/text")',
                       'AutoTokenizer = dynamic_loader()\nAutoTokenizer.from_pretrained("example/text")',
                       'CustomLoader.from_pretrained(config._name_or_path)',
                       'AutoTokenizer.from_pretrained(config._name_or_path, **dynamic_options())'):
            with self.subTest(source=source):
                report, lookup = self.analyze(source)
                self.assertEqual(report["status"], "needs_confirmation")
                lookup.assert_not_called()

    def test_main_repository_with_different_revision_is_external(self):
        report, lookup = self.analyze('AutoTokenizer.from_pretrained(config._name_or_path, revision="release")')
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_dict_comprehension_cannot_hide_a_loader(self):
        report, lookup = self.analyze('''
options = {key: AutoModel.from_pretrained("example/base") for key in items() if key in ["dtype"]}
''')
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()

    def test_unknown_weights_and_active_processor_in_same_repo_stay_separate(self):
        report, lookup = self.analyze('''
if dynamic_training_mode():
    AutoModel.from_pretrained("example/shared")
AutoFeatureExtractor.from_pretrained("example/shared")
''')
        self.assertEqual(report["status"], "needs_confirmation")
        self.assertEqual(report["draft_spec"]["dependencies"][0]["allow_patterns"],
                         ["config.json", "preprocessor_config.json"])
        lookup.assert_called_once_with("example/shared", "main")

    def test_transformers_phases_keep_training_unknown_and_load_configured_backbone(self):
        report, lookup = self.analyze('', config={"text_model_id": "example/text", "audio_model_id": None,
            "auto_map": {"AutoModel": "model.CompositeModel"}}, sources={"model.py": '''
import transformers
class CompositeModel(transformers.PreTrainedModel):
    def __init__(self, config):
        self.language_model = self.create_text(config)
    def _init_weights(self, module):
        if module is self:
            self.language_model = self.create_text(self.config)
    @classmethod
    def create_text(cls, config):
        if hasattr(transformers.modeling_utils, "_init_weights"):
            is_init = transformers.modeling_utils._init_weights
        else:
            is_init = dynamic_training_mode()
        if is_init and config.text_model_id is not None:
            return transformers.AutoModel.from_pretrained(config.text_model_id)
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)
'''})
        self.assertEqual(report["status"], "resolved", report["fields"].get("dependencies"))
        states = {c["activation"] for c in report["dependency_candidates"] if c["repo_id"] == "example/text"}
        self.assertEqual(states, {"active", "inactive"})
        self.assertIn("model.safetensors", report["draft_spec"]["dependencies"][0]["allow_patterns"])
        lookup.assert_called_once_with("example/text", "main")

    def test_custom_initialization_dispatch_cannot_be_bypassed(self):
        for method, body in (
            ("_initialize_weights", 'if dynamic_training_mode():\n            self._init_weights(module)'),
            ("initialize_weights", 'if dynamic_training_mode():\n            self._init_weights(self)'),
        ):
            with self.subTest(method=method):
                report, lookup = self.analyze('', config={"auto_map": {"AutoModel": "model.CompositeModel"}},
                    sources={"model.py": f'''
import transformers
class CompositeModel(transformers.PreTrainedModel):
    def {method}(self, module=None):
        {body}
    def _init_weights(self, module):
        AutoModel.from_pretrained("example/potential")
'''})
                self.assertEqual(report["status"], "needs_confirmation")
                lookup.assert_not_called()

    def test_boolean_numeric_equality_obeys_python_config_semantics(self):
        report, lookup = self.analyze('''
if config.enabled == 1:
    AutoTokenizer.from_pretrained("example/text")
''', config={"enabled": True})
        self.assertEqual(report["status"], "resolved")
        lookup.assert_called_once_with("example/text", "main")

    def test_kwargs_mutation_does_not_hide_an_unknown_revision(self):
        for mutation in ('kwargs.update(dynamic_options())', 'kwargs.update({"revision": "release"})'):
            with self.subTest(mutation=mutation):
                report, lookup = self.analyze(f'''
def load(repo, **kwargs):
    {mutation}
    return AutoTokenizer.from_pretrained(repo, **kwargs)
load("example/text")
''')
                self.assertEqual(report["status"], "needs_confirmation")
                lookup.assert_not_called()

    def test_decorated_model_class_cannot_prove_its_loading_context(self):
        report, lookup = self.analyze('', config={"auto_map": {"AutoModel": "model.CompositeModel"}}, sources={"model.py": '''
import transformers
@dynamic_class_decorator
class CompositeModel(transformers.PreTrainedModel):
    def __init__(self, config):
        AutoModel.from_pretrained("example/potential")
'''})
        self.assertEqual(report["status"], "needs_confirmation")
        lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()

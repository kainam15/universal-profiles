"""Field review preserves source evidence and cannot silently accept unrelated gaps."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_model_contract as fixture
from acprof.model_review import apply_review, review_questions
from acprof.model_spec import task_model_spec


class ModelReviewTests(unittest.TestCase):
    def test_pipeline_selection_reanalyzes_only_the_selected_interface(self):
        config = copy.deepcopy(fixture.CONFIG)
        config["custom_pipelines"]["second"] = copy.deepcopy(config["custom_pipelines"]["listen-and-answer"])
        task = fixture.ModelContractTests().discover(config=config)
        with patch("acprof.host.detect.read_model_source", return_value=fixture.SOURCE, create=True):
            reviewed = apply_review(task, {"pipeline_task": "listen-and-answer"})
        self.assertEqual(task_model_spec(reviewed)["pipeline_task"], "listen-and-answer")
        self.assertIn("second", reviewed.repository_metadata["config.json"]["custom_pipelines"])
        self.assertEqual(reviewed.model_resolution["contract"]["fields"]["pipeline_task"]["sources"][-1], "user.review")

    def test_nested_chat_requires_only_the_missing_target(self):
        task = fixture.ModelContractTests().discover(fixture.SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["turns"]'))
        questions = review_questions(task)
        self.assertEqual([item["path"] for item in questions], ["multimodal.inputs.turns"])
        answer = {"template": [{"role": "user", "content": {"from": "text"}}]}
        reviewed = apply_review(task, {"multimodal.inputs.turns": answer})
        self.assertEqual(task_model_spec(reviewed)["multimodal"]["inputs"]["turns"], answer)
        self.assertFalse(task_model_spec(task))
        self.assertEqual(reviewed.model_resolution["contract"]["fields"]["multimodal.inputs.turns"]["sources"], ["user.review"])

    def test_dependency_review_needs_repo_and_role_but_not_sha_or_patterns(self):
        task = fixture.ModelContractTests().discover(fixture.SOURCE + '\nAutoTokenizer.from_pretrained(dynamic_repo())')
        hub = SimpleNamespace(sha="b" * 40, siblings=[SimpleNamespace(rfilename="tokenizer.json")])
        with patch("huggingface_hub.HfApi.model_info", return_value=hub):
            reviewed = apply_review(task, {"dependencies": [{"repo_id": "example/tokenizer", "role": "tokenizer"}]})
        self.assertEqual(task_model_spec(reviewed)["dependencies"], [{"repo_id": "example/tokenizer", "revision": "b" * 40,
                                                                  "allow_patterns": ["tokenizer.json"]}])

    def test_confirming_dependency_preserves_the_source_revision(self):
        revision = "c" * 40
        source = fixture.SOURCE + f'\nif needs_tokenizer():\n    AutoTokenizer.from_pretrained("example/tokenizer", revision="{revision}")'
        task = fixture.ModelContractTests().discover(source)
        hub = SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename="tokenizer.json")])
        with patch("huggingface_hub.HfApi.model_info", return_value=hub) as lookup:
            reviewed = apply_review(task, {"dependencies": [{"repo_id": "example/tokenizer", "role": "tokenizer"}]})
        lookup.assert_called_once_with("example/tokenizer", revision=revision, files_metadata=False)
        self.assertEqual(task_model_spec(reviewed)["dependencies"][0]["revision"], revision)

    def test_confirming_dependency_cannot_erase_conflicting_revisions(self):
        source = fixture.SOURCE + '\nif needs_tokenizer():\n' + ''.join(
            f'    AutoTokenizer.from_pretrained("example/tokenizer", revision="{revision}")\n'
            for revision in ("b" * 40, "c" * 40))
        task = fixture.ModelContractTests().discover(source)
        with patch("huggingface_hub.HfApi.model_info") as lookup, self.assertRaisesRegex(ValueError, "conflicting"):
            apply_review(task, {"dependencies": [{"repo_id": "example/tokenizer", "role": "tokenizer"}]})
        lookup.assert_not_called()

    def test_review_cannot_change_known_fields_or_accept_empty_invalid_draft(self):
        task = fixture.ModelContractTests().discover(fixture.SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["turns"]'))
        original = copy.deepcopy(task.model_resolution)
        for answers in ({}, {"task": "text-generation"}, {"multimodal.inputs.turns": {"python": "eval(text)"}}):
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                apply_review(task, answers)
            self.assertEqual(task.model_resolution, original)

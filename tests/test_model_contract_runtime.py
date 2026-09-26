"""Generated contracts through the existing offline handler with tiny Torch weights."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acprof.host.detect import TaskInfo
from acprof.model_contract import apply_model_contract
from acprof.model_resolution import discover_model_candidates
from acprof.model_spec import encode_model_spec, task_model_spec
import test_custom_multimodal_runtime as multimodal_fixture
from test_multimodal_handler import audio_payload


MODEL = '''from transformers import BertConfig, BertForSequenceClassification

class AudioConfig(BertConfig):
    model_type = "fixture_unseen_multimodal"

class AudioModel(BertForSequenceClassification):
    config_class = AudioConfig

    def generate(self, input_ids, attention_mask, token_type_ids, audio_values,
                 max_new_tokens, do_sample, temperature, repetition_penalty):
        assert not do_sample and temperature is None
        logits = self(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).logits + audio_values.mean()
        return (logits.argmax(-1).reshape(1, 1) + 5).repeat(1, max_new_tokens)
'''

PIPELINE = '''import torch
from transformers import AutoTokenizer, Pipeline

class AudioPipeline(Pipeline):
    def __init__(self, model, tokenizer=None, **kwargs):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model.config._name_or_path, local_files_only=True)
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        generation_keys = ["temperature", "max_new_tokens", "repetition_penalty"]
        generation_kwargs = {k: kwargs[k] for k in kwargs if k in generation_keys}
        return {}, generation_kwargs, {}

    def preprocess(self, inputs):
        audio = inputs["audio"]
        sampling_rate = inputs.get("sampling_rate", 16000)
        prompt = inputs.get("prompt", "What is said?")
        assert sampling_rate == 16000
        result = dict(self.tokenizer(prompt, return_tensors="pt"))
        result["audio_values"] = torch.tensor(audio).reshape(1, -1)
        return result

    def _forward(self, model_inputs, temperature=None, max_new_tokens=None, repetition_penalty=1.1):
        temperature = temperature or None
        do_sample = temperature is not None
        return self.model.generate(**model_inputs, temperature=temperature, do_sample=do_sample,
                                   max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)

    def postprocess(self, outputs):
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
'''


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers")),
                     "requires the custom multimodal container")
class ModelContractRuntimeTests(unittest.TestCase):
    def test_basic_probe_imports_and_binds_without_loading_weights(self):
        from acprof.container.model_probe import validate_basic
        with tempfile.TemporaryDirectory(prefix="acprof_basic_contract_") as directory:
            root = Path(directory)
            multimodal_fixture.CustomMultimodalRuntimeTests.snapshot(root)
            (root / "custom_model.py").write_text(MODEL)
            (root / "custom_pipeline.py").write_text(PIPELINE)
            spec = {"schema_version": 1, "format": "transformers-pipeline", "task": "audio-text-to-text",
                    "pipeline_task": "unseen-audio-pipeline", "multimodal": {
                        "inputs": {"prompt": "text", "audio": "audio", "sampling_rate": "sampling_rate"},
                        "forward_kwargs": {"max_new_tokens": "$max_new_tokens", "temperature": 0}}}
            config = json.loads((root / "config.json").read_text())
            spec["pipeline_task"] = next(iter(config["custom_pipelines"]))
            with patch.dict(os.environ, {"ACPROF_MODEL_SPEC_B64": encode_model_spec(spec),
                                         "MODEL_LOCAL_PATH": directory, "MODEL_ID": "fixture/basic",
                                         "TASK_TYPE": "audio-text-to-text"}), patch(
                "transformers.pipeline", side_effect=AssertionError("basic probe must not load a model"),
            ):
                result = validate_basic({})
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["inference"], "not_run")
            self.assertEqual({item["stage"] for item in result["stages"]}, {"import", "signature"})

    def test_reviewed_chat_transform_runs_through_real_model(self):
        from acprof.container.runtime_validate import validate
        from acprof.model_review import apply_review
        with tempfile.TemporaryDirectory(prefix="acprof_chat_contract_") as directory:
            root = Path(directory)
            multimodal_fixture.CustomMultimodalRuntimeTests.snapshot(root)
            (root / "custom_model.py").write_text(MODEL)
            source = PIPELINE.replace('prompt = inputs.get("prompt", "What is said?")',
                                      'turns = inputs["turns"]\n        prompt = turns[0]["content"]')
            (root / "custom_pipeline.py").write_text(source)
            config = json.loads((root / "config.json").read_text())
            info = TaskInfo("fixture/chat-audio", "audio-text-to-text", "multimodal", "transformers_model",
                            "transformers", "a" * 40, "hub_api", model_config=config,
                            repository_files=tuple(path.name for path in root.iterdir()),
                            repository_metadata={"config.json": config})
            info.model_resolution = discover_model_candidates(info)
            apply_model_contract(info, lambda name: (root / name).read_text())
            self.assertEqual(info.model_resolution["contract"]["status"], "needs_confirmation")
            info = apply_review(info, {"multimodal.inputs.turns": {
                "template": [{"role": "user", "content": {"from": "text"}}]}})
            environment = {"ACPROF_MODEL_SPEC_B64": encode_model_spec(task_model_spec(info)), "MODEL_LOCAL_PATH": directory,
                           "MODEL_ID": info.model_id, "MODEL_REVISION": info.model_revision,
                           "TASK_TYPE": info.pipeline_tag, "TASK_FAMILY": "multimodal", "RUNTIME_BACKEND": "transformers_pipeline",
                           "USE_GPU": "0", "TORCH_NUM_THREADS": "1", "ACPROF_MODEL_ADAPTER": "family-default",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            with patch.dict(os.environ, environment):
                result = validate({"samples": [{"text": "What is said?", "audio_base64": audio_payload(), "sampling_rate": 16000}],
                                   "params": {"max_new_tokens": 4}})
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["response"]["output_token_count"], 4)
            self.assertEqual(result["validation"]["task"]["status"], "verified")

    def test_synthesis_load_preprocess_predict_postprocess_and_output_validation(self):
        from acprof.container.handlers.multimodal import MultimodalHandler
        from acprof.container.runtime_validate import validate

        with tempfile.TemporaryDirectory(prefix="acprof_contract_") as directory:
            root = Path(directory)
            multimodal_fixture.CustomMultimodalRuntimeTests.snapshot(root)
            (root / "custom_model.py").write_text(MODEL)
            (root / "custom_pipeline.py").write_text(PIPELINE)
            config = json.loads((root / "config.json").read_text())
            info = TaskInfo("fixture/generated-audio", "audio-text-to-text", "multimodal", "transformers_model",
                            "transformers", "a" * 40, "hub_api", model_config=config,
                            hub_metadata={"pipeline_tag": "audio-text-to-text", "transformers_info": {
                                "auto_model": "AutoModel", "pipeline_tag": "feature-extraction"}},
                            repository_files=tuple(path.name for path in root.iterdir()),
                            repository_metadata={"config.json": config})
            info.model_resolution = discover_model_candidates(info)
            apply_model_contract(info, lambda name: (root / name).read_text())
            self.assertEqual(info.model_resolution["contract"]["status"], "resolved")
            spec = task_model_spec(info)
            self.assertEqual(spec["multimodal"]["inputs"]["prompt"], "text")
            payload = {"samples": [{"text": "What is said?", "audio_base64": audio_payload(), "sampling_rate": 16000}],
                       "params": {"max_new_tokens": 4}}
            environment = {"ACPROF_MODEL_SPEC_B64": encode_model_spec(spec), "MODEL_LOCAL_PATH": directory,
                           "MODEL_ID": info.model_id, "MODEL_REVISION": info.model_revision,
                           "TASK_TYPE": info.pipeline_tag, "TASK_FAMILY": "multimodal", "RUNTIME_BACKEND": "transformers_pipeline",
                           "USE_GPU": "0", "TORCH_NUM_THREADS": "1", "ACPROF_MODEL_ADAPTER": "family-default",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            with patch.dict(os.environ, environment):
                handler = MultimodalHandler()
                context = handler.load(directory, info.pipeline_tag, "transformers_pipeline", "cpu")
                processed = handler.preprocess(context, payload)
                first = handler.postprocess(context, handler.predict(context, processed))
                second = handler.postprocess(context, handler.predict(context, processed))
                self.assertEqual(first, second)
                self.assertEqual(first["output_token_count"], 4)
                self.assertEqual(processed["_effective_input_scale"], .01)
                validation = validate(payload)
                self.assertEqual(validation["status"], "ok")
                self.assertTrue(all(stage["status"] == "verified" for stage in validation["stages"]))
            self.assertEqual(info.model_resolution["contract"]["runtime_validation"], "not_run")


if __name__ == "__main__":
    unittest.main()

"""Offline shared custom pipeline protocol with tiny random Torch weights."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_custom_multimodal import pipeline_spec
from test_multimodal_handler import audio_payload, image_payload


CUSTOM_MODEL = '''from transformers import BertConfig, BertForSequenceClassification

class AudioConfig(BertConfig):
    model_type = "fixture_unseen_multimodal"

class AudioModel(BertForSequenceClassification):
    config_class = AudioConfig
'''

CUSTOM_PIPELINE = '''import numpy as np
import torch
from transformers import AutoTokenizer, Pipeline

class AudioPipeline(Pipeline):
    def __init__(self, model, tokenizer=None, **kwargs):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model.config._name_or_path, local_files_only=True)
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.events = []

    def _sanitize_parameters(self, **kwargs):
        return {}, kwargs, {}

    def preprocess(self, payload):
        self.events.append("preprocess")
        inputs = dict(self.tokenizer(payload["question"], return_tensors="pt"))
        if "waveform" in payload:
            assert payload["rate"] == 16000
            inputs["audio_values"] = torch.tensor(payload["waveform"]).reshape(1, -1)
        elif "picture" in payload:
            inputs["pixel_values"] = torch.tensor(np.asarray(payload["picture"]).copy()).float().unsqueeze(0)
        else:
            assert payload["fps"] == 2.0
            inputs["pixel_values_videos"] = torch.tensor(payload["frames"].copy()).float()
        return inputs

    def _forward(self, inputs, limit, temperature):
        self.events.append("predict")
        assert temperature == 0.0 and limit > 0
        media = inputs.pop("audio_values", None)
        if media is None:
            media = inputs.pop("pixel_values", None)
        if media is None:
            media = inputs.pop("pixel_values_videos")
        scores = self.model(**inputs).logits + media.float().mean()
        return {"scores": scores, "samples": media.numel(), "limit": limit}

    def postprocess(self, outputs):
        self.events.append("postprocess")
        label = int(outputs["scores"].argmax())
        return f"answer {label} samples {outputs['samples']} limit {outputs['limit']}"
'''


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers")),
                     "requires the custom multimodal container")
class CustomMultimodalRuntimeTests(unittest.TestCase):
    @staticmethod
    def snapshot(root):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        torch.manual_seed(123)
        torch.set_num_threads(1)
        (root / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwhat\nis\nsaid\n")
        BertTokenizerFast(vocab_file=str(root / "vocab.txt"), model_max_length=32).save_pretrained(root)
        config = BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=32, max_position_embeddings=32, num_labels=2)
        BertForSequenceClassification(config).save_pretrained(root)
        data = json.loads((root / "config.json").read_text())
        data.update(model_type="fixture_unseen_multimodal", auto_map={
            "AutoConfig": "custom_model.AudioConfig", "AutoModel": "custom_model.AudioModel",
        }, custom_pipelines={"listen-and-answer": {
            "impl": "custom_pipeline.AudioPipeline", "pt": ["AutoModel"], "type": "multimodal",
        }})
        (root / "config.json").write_text(json.dumps(data))
        (root / "custom_model.py").write_text(CUSTOM_MODEL)
        (root / "custom_pipeline.py").write_text(CUSTOM_PIPELINE)

    def test_modalities_match_official_pipeline_and_runtime_validation(self):
        import torch
        import numpy as np
        from PIL import Image
        from acprof.container.handlers.multimodal import MultimodalHandler
        from acprof.container.runtime_validate import validate
        from acprof.model_spec import encode_model_spec

        device = os.getenv("ACPROF_TEST_DEVICE", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            self.fail("CUDA runtime verification requested but unavailable")
        for task in ("audio-text-to-text", "image-text-to-text", "video-text-to-text"):
            with self.subTest(task=task), tempfile.TemporaryDirectory(prefix="acprof_custom_") as directory:
                root = Path(directory)
                self.snapshot(root)
                spec = pipeline_spec()
                spec.pop("dependencies")
                spec["task"] = task
                sample = {"text": "What is said?"}
                if task == "audio-text-to-text":
                    sample.update(audio_base64=audio_payload(), sampling_rate=16000)
                elif task == "image-text-to-text":
                    sample["image_base64"] = image_payload()
                    spec["multimodal"]["inputs"] = {"picture": "image", "question": "text"}
                else:
                    sample.update(video_frames_base64=[image_payload(), image_payload()], fps=2.0)
                    spec["multimodal"]["inputs"] = {"frames": "video", "fps": "fps", "question": "text"}
                payload = {"samples": [sample], "params": {"max_new_tokens": 4}}
                environment = {"ACPROF_MODEL_SPEC_B64": encode_model_spec(spec), "MODEL_LOCAL_PATH": directory,
                               "MODEL_ID": "fixture/unseen", "MODEL_REVISION": "a" * 40,
                               "TASK_TYPE": task, "TASK_FAMILY": "multimodal", "RUNTIME_BACKEND": "transformers_pipeline",
                               "USE_GPU": "1" if device == "cuda" else "0", "TORCH_NUM_THREADS": "1",
                               "ACPROF_MODEL_ADAPTER": "family-default", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                with patch.dict(os.environ, environment):
                    handler = MultimodalHandler()
                    context = handler.load(directory, task, "transformers_pipeline", device)
                    self.assertEqual(type(context["pipeline"].model).__name__, "AudioModel")
                    processed = handler.preprocess(context, payload)
                    self.assertEqual(context["pipeline"].events, ["preprocess"])
                    first = handler.postprocess(context, handler.predict(context, processed))
                    second = handler.postprocess(context, handler.predict(context, processed))
                    self.assertEqual(first, second)
                    self.assertEqual(context["pipeline"].events, ["preprocess", "predict", "postprocess", "predict", "postprocess"])
                    self.assertIn("limit 4", first["texts"][0])
                    direct_payload = {"question": sample["text"]}
                    if task == "audio-text-to-text":
                        direct_payload.update(waveform=np.zeros(160, dtype=np.float32), rate=16000)
                    elif task == "image-text-to-text":
                        direct_payload["picture"] = Image.new("RGB", (8, 8), "white")
                    else:
                        direct_payload.update(frames=np.full((2, 8, 8, 3), 255, dtype=np.uint8), fps=2.0)
                    expected = context["pipeline"](direct_payload, limit=4, temperature=0.0)
                    self.assertEqual(first["texts"], [expected])
                    # Real operators remain visible to the existing Torch profiling path.
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities) as profiler:
                        handler.predict(context, processed)
                    self.assertTrue(any(event.key.startswith("aten::") for event in profiler.key_averages()))
                    if device == "cuda":
                        self.assertTrue(any(event.device_type == torch.autograd.DeviceType.CUDA for event in profiler.events()))
                    report = validate(payload)
                    self.assertEqual(report["status"], "ok")
                    self.assertEqual(report["validation"]["task"]["status"], "verified")
                    self.assertTrue(all(stage["status"] == "verified" for stage in report["stages"]))


if __name__ == "__main__":
    unittest.main()

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from acprof.container.download_model import verify_download
from acprof.container.model_files import ModelFilesError, plan_download, validate_plan


class ModelFilePlanTests(unittest.TestCase):
    def plan(self, names, metadata=None, **kwargs):
        metadata = {"config.json": {"model_type": "bert"}, **(metadata or {})}
        return plan_download(
            model_id="example/model", revision="a" * 40, family=kwargs.pop("family", "nlp"),
            backend=kwargs.pop("backend", "transformers_pipeline"),
            files={name: {"size": 1} for name in names}, read_json=metadata.__getitem__,
            native_model_types={"bert", "whisper", "t5"}, **kwargs,
        )

    def selected(self, plan):
        validate_plan(plan)
        return {entry["path"] for entry in plan["files"]}

    def test_safetensors_preserves_processor_resources_and_default_variant(self):
        plan = self.plan([
            "config.json", "model.safetensors", "model.fp16.safetensors", "pytorch_model.bin",
            "flax_model.msgpack", "tf_model.h5", "tokenizer.model", "tokenizer.json", "processor_config.json",
            "chat_template.jinja", "special_audio.bin", "LICENSE", "auxiliary/config.json",
        ])
        self.assertEqual(self.selected(plan), {
            "config.json", "model.safetensors", "tokenizer.model", "tokenizer.json", "processor_config.json",
            "chat_template.jinja", "special_audio.bin", "LICENSE", "auxiliary/config.json",
        })
        self.assertIsNone(plan["weights"][0]["variant"])

    def test_bin_only_checkpoint_is_preserved(self):
        plan = self.plan(["config.json", "pytorch_model.bin", "flax_model.msgpack"])
        self.assertEqual(self.selected(plan), {"config.json", "pytorch_model.bin"})
        self.assertEqual(plan["weights"][0]["format"], "bin")

    def test_shard_index_selects_every_referenced_shard(self):
        index = "model.safetensors.index.json"
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        plan = self.plan(["config.json", index, *shards, "pytorch_model.bin"], {
            index: {"weight_map": {"a": shards[0], "b": shards[1], "c": shards[0]}},
        })
        self.assertEqual(self.selected(plan), {"config.json", index, *shards})

    def test_missing_safe_shard_does_not_silently_fall_back_to_bin(self):
        index = "model.safetensors.index.json"
        with self.assertRaisesRegex(ModelFilesError, "missing checkpoint shards"):
            self.plan(["config.json", index, "pytorch_model.bin"], {index: {"weight_map": {"a": "missing.safetensors"}}})

    def test_invalid_shard_paths_and_empty_index_are_rejected(self):
        index = "model.safetensors.index.json"
        for weight_map in ({}, {"a": "../outside.safetensors"}, {"a": 3}):
            with self.subTest(weight_map=weight_map), self.assertRaises(ModelFilesError):
                self.plan(["config.json", index], {index: {"weight_map": weight_map}})

    def test_unknown_custom_and_quantized_models_keep_full_snapshot(self):
        names = ["config.json", "model.safetensors", "pytorch_model.bin", "custom.py", "extra.bin"]
        for config in (
            {"model_type": "new_model"}, {"model_type": "bert", "auto_map": {"AutoModel": "custom.Model"}},
            {"model_type": "bert", "quantization_config": {"quant_method": "gptq"}},
            {"model_type": ["custom", "Model"]},
        ):
            with self.subTest(config=config):
                plan = self.plan(names, {"config.json": config})
                self.assertEqual(self.selected(plan), set(names))
                self.assertEqual(plan["effective_policy"], "full")

    def test_full_policy_does_not_parse_model_configuration(self):
        names = ["config.json", "model.safetensors", "flax_model.msgpack"]
        plan = self.plan(names, {"config.json": None}, policy="full")
        self.assertEqual(self.selected(plan), set(names))
        self.assertEqual(plan["reason"], "explicit_full")

    def test_registered_custom_adapter_keeps_all_its_assets(self):
        names = ["config.json", "model.safetensors", "pytorch_model.bin", "processing.py"]
        plan = self.plan(names, adapter="moss-transcribe-diarize")
        self.assertEqual(self.selected(plan), set(names))
        self.assertEqual(plan["reason"], "custom_adapter")

    def test_sentence_transformers_keeps_dense_modules(self):
        names = ["modules.json", "0_Transformer/config.json", "0_Transformer/model.safetensors",
                 "0_Transformer/pytorch_model.bin", "1_Pooling/config.json", "2_Dense/pytorch_model.bin"]
        plan = self.plan(names, {
            "modules.json": [{"path": "0_Transformer", "type": "sentence_transformers.models.Transformer"},
                             {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                             {"path": "2_Dense", "type": "sentence_transformers.models.Dense"}],
            "0_Transformer/config.json": {"model_type": "bert"},
        }, backend="sentence_transformers")
        self.assertEqual(self.selected(plan), set(names) - {"0_Transformer/pytorch_model.bin"})

    def test_diffusers_selects_all_components_without_changing_precision_or_ema(self):
        names = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                 "unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.non_ema.bin",
                 "text_encoder/config.json", "text_encoder/pytorch_model.bin", "scheduler/scheduler_config.json",
                 "tokenizer/vocab.json", "v1-5-pruned.ckpt", "v1-5-pruned.safetensors"]
        plan = self.plan(names, {"model_index.json": {
            "_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"],
            "text_encoder": ["transformers", "CLIPTextModel"], "scheduler": ["diffusers", "PNDMScheduler"],
            "tokenizer": ["transformers", "CLIPTokenizer"], "safety_checker": [None, None],
        }}, family="diffusion", backend="diffusers")
        self.assertEqual(self.selected(plan), set(names) - {
            "unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.non_ema.bin",
            "v1-5-pruned.ckpt", "v1-5-pruned.safetensors",
        })
        self.assertEqual(len(plan["weights"]), 2)

    def test_unknown_diffusers_pipeline_keeps_complete_repository(self):
        names = ["model_index.json", "unet/diffusion_pytorch_model.safetensors", "extra.ckpt"]
        for pipeline in ("NewPipeline", ["custom", "Pipeline"]):
            with self.subTest(pipeline=pipeline):
                plan = self.plan(names, {"model_index.json": {"_class_name": pipeline}}, backend="diffusers")
                self.assertEqual(self.selected(plan), set(names))

    def test_nonstandard_diffusers_component_keeps_complete_repository(self):
        names = ["model_index.json", "unet/diffusion_pytorch_model.safetensors", "extra.ckpt"]
        for spec in ((["custom"], "Model"), ("diffusers", {"custom": "Model"}), ("custom",), {"custom": "Model"}):
            with self.subTest(spec=spec):
                plan = self.plan(names, {"model_index.json": {
                    "_class_name": "StableDiffusionPipeline",
                    "unet": ["diffusers", "UNet2DConditionModel"],
                    "custom": list(spec) if isinstance(spec, tuple) else spec,
                }}, backend="diffusers")
                self.assertEqual(self.selected(plan), set(names))
                self.assertEqual(plan["effective_policy"], "full")

    def test_structured_manifest_selects_exact_artifact(self):
        plan = self.plan(["acprof_model.json", "policy.pt", "training.pth", "README.md"], {
            "acprof_model.json": {"schema_version": 1, "format": "torchscript", "model_file": "policy.pt"},
        }, family="structured", backend="torchscript")
        self.assertEqual(self.selected(plan), {"acprof_model.json", "policy.pt", "README.md"})

    def test_plan_hash_detects_changed_selection(self):
        plan = self.plan(["config.json", "model.safetensors"])
        plan["files"].pop()
        with self.assertRaisesRegex(ModelFilesError, "hash mismatch"):
            validate_plan(plan)

    def test_verification_records_actual_hash_and_rejects_corrupt_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            content = b"weights"
            (Path(temporary) / "model.safetensors").write_bytes(content)
            plan = self.plan(["model.safetensors"], policy="full")
            plan["files"][0].update(size=len(content), lfs_sha256=hashlib.sha256(content).hexdigest())
            verified = verify_download(temporary, copy.deepcopy(plan))
            self.assertEqual(verified["files"][0]["sha256"], hashlib.sha256(content).hexdigest())
            validate_plan(verified)
            (Path(temporary) / "model.safetensors").write_bytes(b"garbage")
            with self.assertRaisesRegex(ModelFilesError, "SHA256 mismatch"):
                verify_download(temporary, plan)


if __name__ == "__main__":
    unittest.main()

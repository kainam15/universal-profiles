"""Offline shared interfaces with native architectures absent from Transformers 4."""
import importlib.util
import os
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"),
                     "requires a Transformers container")
class Transformers5RuntimeTests(unittest.TestCase):
    def test_native_vision_processor_preserves_box_and_polygon_geometry(self):
        from types import SimpleNamespace
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, PPDocLayoutV3ImageProcessor

        with tempfile.TemporaryDirectory() as directory:
            PPDocLayoutV3ImageProcessor(size={"height": 32, "width": 32}).save_pretrained(directory)
            processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
            inputs = processor(Image.new("RGB", (64, 64), "white"), return_tensors="pt")
            self.assertEqual(tuple(inputs["pixel_values"].shape), (1, 3, 32, 32))
            output = SimpleNamespace(
                pred_boxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
                logits=torch.tensor([[[10.0]]]), order_logits=torch.zeros((1, 1, 1)),
                out_masks=torch.ones((1, 1, 8, 8)) * 10,
            )
            result = processor.post_process_object_detection(output, target_sizes=[(64, 64)])[0]
            torch.testing.assert_close(result["boxes"], torch.tensor([[16., 16., 48., 48.]]))
            polygon = result["polygon_points"][0]
            self.assertGreaterEqual(len(polygon), 4)
            self.assertTrue(((polygon >= 16) & (polygon <= 48)).all())

    def test_new_native_architecture_uses_the_existing_handler(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3_5TextConfig
        from acprof.container.handlers.nlp import NLPHandler

        torch.set_num_threads(1)
        device = os.environ.get("ACPROF_RUNTIME_TEST_DEVICE", "cpu")
        self.assertIn(device, ("cpu", "cuda"))
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                                            eos_token="[EOS]", model_max_length=128)
        config = Qwen3_5TextConfig(vocab_size=5, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                                  layer_types=["linear_attention", "full_attention"], max_position_embeddings=128,
                                  linear_key_head_dim=16, linear_value_head_dim=16,
                                  linear_num_key_heads=2, linear_num_value_heads=2,
                                  bos_token_id=1, eos_token_id=2, pad_token_id=1)
        with tempfile.TemporaryDirectory() as directory:
            model = AutoModelForCausalLM.from_config(config).eval()
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            handler = NLPHandler()
            context = handler.load(directory, "text-generation", "transformers_pipeline", device)
            self.assertEqual(context["device"], device)
            prepared = handler.preprocess(context, {"text": "hello world", "params": {"max_new_tokens": 2}})
            output = handler.predict(context, prepared)
            summary = handler.postprocess(context, output)
            self.assertEqual(summary["task"], "text-generation")
            self.assertGreater(summary["actual_output_tokens"], 0)
            self.assertLessEqual(summary["actual_output_tokens"], 2)

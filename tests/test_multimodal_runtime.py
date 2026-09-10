"""Offline API smoke tests with tiny random models, without benchmark sampling.

These require the container's Transformers/Torch dependencies. They skip on the
lightweight host and never download pretrained weights or use the GPU.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "tokenizers")
)


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires container Transformers/Torch runtime")
class MultimodalRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch

        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        import torch

        state = torch.random.fork_rng(devices=[])
        state.__enter__()
        self.addCleanup(state.__exit__, None, None, None)

    def test_blip_visual_qa_load_preprocess_generate_postprocess(self):
        import torch
        from transformers import (
            BertTokenizerFast, BlipConfig, BlipForQuestionAnswering,
            BlipImageProcessor, BlipProcessor,
        )
        from acprof.container.handlers.multimodal import MultimodalHandler
        from acprof.workloads import get_generator

        with tempfile.TemporaryDirectory() as tmp:
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                     "what", "color", "is", "the", "square", "?", "red", "blue"]
            vocab_path = Path(tmp) / "vocab.txt"
            vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
            # BLIP checkpoint tokenizers omit BERT token_type_ids, which BLIP's
            # text decoder does not accept during generation.
            tokenizer = BertTokenizerFast(
                vocab_file=str(vocab_path),
                model_input_names=["input_ids", "attention_mask"],
            )
            processor = BlipProcessor(
                image_processor=BlipImageProcessor(size={"height": 16, "width": 16}),
                tokenizer=tokenizer,
            )
            config = BlipConfig(
                text_config={
                    "vocab_size": len(tokenizer), "hidden_size": 16,
                    "encoder_hidden_size": 16, "intermediate_size": 32,
                    "num_hidden_layers": 1, "num_attention_heads": 2,
                    "bos_token_id": 2, "eos_token_id": 3,
                    "sep_token_id": 3, "pad_token_id": 0,
                },
                vision_config={
                    "hidden_size": 16, "intermediate_size": 32,
                    "num_hidden_layers": 1, "num_attention_heads": 2,
                    "image_size": 16, "patch_size": 8,
                },
                projection_dim=16,
            )
            torch.manual_seed(17)
            BlipForQuestionAnswering(config).save_pretrained(tmp)
            processor.save_pretrained(tmp)
            handler = MultimodalHandler()
            ctx = handler.load(tmp, "visual-question-answering", "transformers_model", "cpu")
            payload = get_generator("multimodal", "local", "visual-question-answering", 1).generate(32)
            payload["params"] = {"max_new_tokens": 2}
            processed = handler.preprocess(ctx, payload)
            first = handler.postprocess(ctx, handler.predict(ctx, processed))
            second = handler.postprocess(ctx, handler.predict(ctx, processed))
            self.assertEqual(first, second)
            self.assertEqual(first["output_type"], "answers")
            self.assertEqual(len(first["answers"]), 1)
            self.assertIsInstance(first["answers"][0]["answer"], str)
            self.assertEqual(processed["_effective_input_scale"], 32)

    def test_colpali_local_load_encodes_query_and_document_then_scores(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            ColPaliConfig, ColPaliForRetrieval, ColPaliProcessor,
            GemmaTokenizerFast, SiglipImageProcessor,
        )
        from acprof.container.handlers.multimodal import MultimodalHandler
        from acprof.workloads import get_generator

        with tempfile.TemporaryDirectory() as tmp:
            tokenizer_backend = Tokenizer(WordLevel(
                vocab={"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
                       "<image>": 4, "invoice": 5, "total": 6, "42": 7},
                unk_token="<unk>",
            ))
            tokenizer_backend.pre_tokenizer = Whitespace()
            tokenizer = GemmaTokenizerFast(
                tokenizer_object=tokenizer_backend, pad_token="<pad>",
                eos_token="<eos>", bos_token="<bos>", unk_token="<unk>",
            )
            image_processor = SiglipImageProcessor(size={"height": 16, "width": 16})
            image_processor.image_seq_length = 4
            processor = ColPaliProcessor(image_processor=image_processor, tokenizer=tokenizer)
            config = ColPaliConfig(
                vlm_config={
                    "model_type": "paligemma",
                    "text_config": {
                        "model_type": "gemma", "vocab_size": len(tokenizer),
                        "hidden_size": 16, "intermediate_size": 32,
                        "num_hidden_layers": 1, "num_attention_heads": 2,
                        "num_key_value_heads": 1, "head_dim": 8,
                        "max_position_embeddings": 512, "pad_token_id": 0,
                        "bos_token_id": 2, "eos_token_id": 1,
                    },
                    "vision_config": {
                        "image_size": 16, "patch_size": 8,
                        "num_channels": 3, "hidden_size": 16,
                        "intermediate_size": 32, "num_hidden_layers": 1,
                        "num_attention_heads": 2,
                    },
                    "image_token_index": tokenizer.convert_tokens_to_ids("<image>"),
                    "projection_dim": 16,
                },
                embedding_dim=8,
            )
            torch.manual_seed(17)
            ColPaliForRetrieval(config).save_pretrained(tmp)
            processor.save_pretrained(tmp)
            handler = MultimodalHandler()
            ctx = handler.load(tmp, "visual-document-retrieval", "transformers_model", "cpu")
            payload = get_generator("multimodal", "local", "visual-document-retrieval", 1).generate(32)
            processed = handler.preprocess(ctx, payload)
            first = handler.postprocess(ctx, handler.predict(ctx, processed))
            second = handler.postprocess(ctx, handler.predict(ctx, processed))
            self.assertEqual(first, second)
            self.assertEqual(len(first["scores"]), 1)
            self.assertEqual(len(first["scores"][0]), 1)
            self.assertEqual(first["retrieval_scope"], "query_and_document_encoding_plus_scoring")
            self.assertEqual(processed["_effective_input_scale"], 32)

    def test_layoutlm_document_qa_reuses_all_chunks_without_ocr(self):
        import torch
        from transformers import LayoutLMConfig, LayoutLMForQuestionAnswering, LayoutLMTokenizerFast
        from acprof.container.handlers.multimodal import MultimodalHandler
        from acprof.workloads import get_generator

        with tempfile.TemporaryDirectory() as tmp:
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                     "what", "is", "the", "total", "?", "invoice", "42", "usd"]
            vocab_path = Path(tmp) / "vocab.txt"
            vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
            tokenizer = LayoutLMTokenizerFast(vocab_file=str(vocab_path), model_max_length=32)
            config = LayoutLMConfig(
                vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
            )
            torch.manual_seed(17)
            LayoutLMForQuestionAnswering(config).save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)
            handler = MultimodalHandler()
            ctx = handler.load(tmp, "document-question-answering", "transformers_model", "cpu")
            payload = get_generator("multimodal", "local", "document-question-answering", 1).generate(32)
            payload["samples"][0]["words"] = ["invoice", "total", "42", "usd"] * 10
            payload["samples"][0]["boxes"] = [[0, 0, 100, 100]] * 40
            payload["params"] = {"max_seq_len": 16, "doc_stride": 2, "top_k": 1}
            processed = handler.preprocess(ctx, payload)
            self.assertGreater(len(processed["chunks"]), 1)
            first = handler.postprocess(ctx, handler.predict(ctx, processed))
            second = handler.postprocess(ctx, handler.predict(ctx, processed))
            self.assertEqual(first, second)
            self.assertEqual(first["output_type"], "answers")
            self.assertEqual(len(first["answers"]), 1)
            self.assertEqual(processed["_effective_input_scale"], 32)


if __name__ == "__main__":
    unittest.main()

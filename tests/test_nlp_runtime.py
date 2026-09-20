"""在 NLP 镜像中执行已有的 12 种原生小模型示例。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in (
    "torch", "transformers", "sentence_transformers", "pandas",
)), "requires the NLP container")
class NLPRuntimeTests(unittest.TestCase):
    def test_sentence_embeddings_match_native_pooling_normalization_and_prompt(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from sentence_transformers import SentenceTransformer, models
        from acprof.container.handlers.nlp import NLPHandler

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "find", "hello", "world"]
            (root / "vocab.txt").write_text("\n".join(vocab))
            tokenizer = BertTokenizerFast(vocab_file=str(root / "vocab.txt"), model_max_length=32)
            model = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32))
            base = root / "base"
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            encoder = SentenceTransformer(modules=[models.Transformer(str(base), max_seq_length=32),
                                                    models.Pooling(16), models.Normalize()],
                                          prompts={"search": "find "}, default_prompt_name="search")
            snapshot = root / "encoder"
            encoder.save(str(snapshot))
            handler = NLPHandler()
            ctx = handler.load(str(snapshot), "feature-extraction", "sentence_transformers", "cpu")
            prepared = handler.preprocess(ctx, {"text": "hello world", "batch_size": 2})
            actual = handler.predict(ctx, prepared)
            expected = encoder.encode(["hello world"] * 2, convert_to_tensor=True)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.linalg.vector_norm(actual, dim=1), torch.ones(2))
            self.assertEqual(handler.postprocess(ctx, actual)["output_shape"], [2, 16])

    def test_native_tasks_offline(self):
        from examples.nlp.smoke import main
        self.assertEqual(main(), 0)

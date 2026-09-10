"""用离线随机小模型检查 12 类 NLP 任务的四阶段接口，不评测模型质量。

需要 NLP 镜像中的 PyTorch、Transformers、Sentence Transformers 和 pandas 依赖。
从仓库根目录运行：PYTHONPATH=. python examples/nlp/smoke.py
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, models
from transformers import (
    BertConfig, BertTokenizerFast, BertForSequenceClassification, BertForTokenClassification,
    BertForQuestionAnswering, BertForMaskedLM, BertModel, TapasConfig, TapasTokenizer,
    TapasForQuestionAnswering, T5Config, T5ForConditionalGeneration, GPT2Config, GPT2LMHeadModel,
)

from acprof.container.handlers.nlp import NLPHandler
from acprof.workloads.nlp import NLPWorkloadGenerator


def main() -> int:
    torch.set_num_threads(2)
    torch.manual_seed(12345)
    handler = NLPHandler()
    failures = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(
            "the quick brown fox jumps over lazy dog what is main topic this example "
            "science sports politics name value item 1 2 3 4 5 ? .".split()))
        vocab_path = root / "vocab.txt"
        vocab_path.write_text("\n".join(vocab), encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), model_max_length=128)

        def config(labels: int = 2) -> BertConfig:
            return BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=32,
                              max_position_embeddings=128, num_labels=labels)

        def saved(name: str, model, tok=tokenizer) -> str:
            path = root / name
            model.save_pretrained(path)
            tok.save_pretrained(path)
            return str(path)

        classification = saved("classification", BertForSequenceClassification(config()))
        zero_config = config(3)
        zero_config.id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        zero_config.label2id = {label: index for index, label in zero_config.id2label.items()}
        zero = saved("zero", BertForSequenceClassification(zero_config))
        tokens = saved("tokens", BertForTokenClassification(config()))
        qa = saved("qa", BertForQuestionAnswering(config()))
        masks = saved("masks", BertForMaskedLM(config()))
        features = saved("features", BertModel(config()))
        ranker = saved("ranker", BertForSequenceClassification(config(1)))
        table_tok = TapasTokenizer(vocab_file=str(vocab_path), model_max_length=128)
        table_config = TapasConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=32, max_position_embeddings=128)
        table = saved("table", TapasForQuestionAnswering(table_config), table_tok)
        t5 = saved("t5", T5ForConditionalGeneration(T5Config(
            vocab_size=len(vocab), d_model=16, d_ff=32, num_layers=1, num_heads=2,
            decoder_start_token_id=0, pad_token_id=0, eos_token_id=3)))
        causal = saved("causal", GPT2LMHeadModel(GPT2Config(
            vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2, n_positions=128,
            pad_token_id=0, bos_token_id=2, eos_token_id=3)))
        encoder = SentenceTransformer(modules=[models.Transformer(features, max_seq_length=32), models.Pooling(16)])
        encoder.save(str(root / "sentence"))
        cases = [("text-classification", classification), ("token-classification", tokens),
                 ("table-question-answering", table), ("question-answering", qa),
                 ("zero-shot-classification", zero), ("translation", t5),
                 ("summarization", t5), ("feature-extraction", features),
                 ("text-generation", causal), ("fill-mask", masks),
                 ("sentence-similarity", str(root / "sentence")), ("text-ranking", ranker)]
        for task, path in cases:
            try:
                context = handler.load(path, task, "transformers_pipeline", "cpu", model_revision="local-sha")
                payload = NLPWorkloadGenerator(path, task, 2).generate(2 if task == "table-question-answering" else 8)
                payload["params"]["max_new_tokens"] = 4
                processed = handler.preprocess(context, payload)
                with torch.inference_mode():
                    output = handler.predict(context, processed)
                result = handler.postprocess(context, output)
                assert result["n_results"] == 2, (task, result)
                print("PASS", task, result, flush=True)
            except Exception as error:
                traceback.print_exc()
                failures.append((task, str(error)))
    if failures:
        print("FAILED", failures, flush=True)
        return 1
    print("PASS all 12 NLP tasks; random tiny native models; batch_size=2; CPU; no Hub downloads", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

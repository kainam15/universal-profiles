# Ultravox 静态依赖验收输入

来源：[fixie-ai/ultravox-v0_5-llama-3_2-1b](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/tree/b95bec8ab291eeb04b5cd600dd473377f6b79026)，
固定 revision 为 `b95bec8ab291eeb04b5cd600dd473377f6b79026`。

Python 源文件原样保存为 `.py.txt`，只交给 AST 解析器作为文本输入；测试不会导入或执行。
代码来自 [fixie-ai/ultravox](https://github.com/fixie-ai/ultravox)，MIT 许可见 `LICENSE-MIT.txt`。
未收录模型权重或 tokenizer 词表。JSON 文件只作为固定 snapshot 的配置证据。

`source_sha256.json` 保存原始文件内容摘要；仅 `config.json` 补了末行换行，校验时去掉该换行。
`repositories.json` 保存主模型及两个依赖在固定 SHA
下的 Hub 文件清单，供测试替代网络元数据查询。测试同时使用任意主模型 ID，防止依赖 checkpoint 名称的特判。

验收只证明静态规划：任务、Pipeline、输入、生成参数、Llama weights、Whisper processor，
以及 inactive/main-model/fallback 的分类；不证明实际加载、tokenizer 文件内容有效、推理或 profiling。

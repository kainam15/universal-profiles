# 项目来源与演进

本仓库是原始 AC-Prof 项目的延续与大幅扩展：

[wingter562/AI-container-runtime-profiles-dataset](https://github.com/wingter562/AI-container-runtime-profiles-dataset)

原始仓库主要用于理解 AC-Prof 的早期设计、测量语义和历史实现。
涉及历史架构、默认参数来源或不明确的遗留行为时，可对照其中的文档、源码和提交记录理解设计来源。

当前仓库已经在 Workload、Runtime、Collector、模型适配、测量协议、CLI、TUI、审计和结果格式等方面进行了大量扩展与重构。

## 历史参考与当前规范

1. 原始仓库是历史参考，不是当前实现的唯一规范。
2. 遇到不明确的历史行为时，可以对照原始仓库理解设计来源；引用具体行为时记录对应文件与 commit，避免把不同版本混为一谈。
3. 当前仓库的文档、测试和测量协议优先级高于原始仓库；实际实现与默认值结合当前代码、`--help` 和验证结果核对。
4. 不应为了与原始仓库保持一致而机械恢复旧实现；变更应满足当前协议，并取得与改动范围相称的验证证据。

解释历史实验时，以当次产物的版本、计划、日志和来源为准，不用当前默认值重新解释旧数据。

## 当前实现入口

- 模块职责与模型扩展：[代码架构](Architecture.md)、[运行兼容](Runtime_Compatibility.md)。
- 采集窗口、结果字段与指标口径：[采集协议](Profiling_Protocol.md)、[指标与结果分析](Metrics.md)、[能耗测量](Energy_Measurement.md)。
- 参数与界面操作：[CLI 与设置](CLI_Reference.md)、[TUI 用户指南](TUI.md)。
- 验证要求与其他专题：[测试指南](Testing.md)、[文档导航](README.md)。

项目许可与贡献者署名见 [LICENSE](../LICENSE) 和 [NOTICE](../NOTICE)。

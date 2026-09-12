# Python 实现约束

- 根脚本与 `cli/` 处理参数和调度；业务实现不得反向导入 `acprof.cli`。新增调用直接引用职责所属模块，兼容入口继续保留。
- `analysis/` 不依赖绘图库；`client_metrics.py` 与 `pixel_metrics.py` 保持纯计算，不初始化 workload、monitor 或读取环境。`runtime_profiles.py` 保持标准库依赖。
- `tui/commands.py`、配置及包初始化不能提前加载 Textual；`profile.py` 的标准库兼容代理与 server/runner 模块启动路径需保留。
- 指标与协议变更沿生产者到消费者核对，不只改字段列表；具体语义见[采集协议](../docs/Profiling_Protocol.md#协议不变量)和[指标](../docs/Metrics.md)。
- 沿用所在模块的四空格缩进、命名和类型注解，避免无关格式调整。模块移动后检查 mock 的实际查找位置。

模块与兼容设计见[架构](../docs/Architecture.md)，验证入口见[测试指南](../docs/Testing.md)。

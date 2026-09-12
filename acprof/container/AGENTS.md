# 容器实现约束

- 模型推理依赖与自定义模型代码只在容器内加载。正式 server 从构建时准备的本地 snapshot 离线加载，不在启动或请求期间安装包、下载模型或扫描全部权重。
- handler 复用 `load / preprocess / predict / postprocess` 协议；模型特有参数通过 workload 与输入计划传递，保持 server 和 profiler 共用路由。
- 输入有效尺度、模态和输出摘要必须与计划及 schema 一致；不静默截断、丢弃模态或额外发送隐藏的推理请求。显式保留不支持的设备或 profiler 错误。
- `acprof.container.server` 及 runner 的模块启动路径保持兼容；注册入口在 `handlers/__init__.py`。

接口和设备边界见[运行兼容](../../docs/Runtime_Compatibility.md)，窗口差异见[Profiler](../../docs/Profilers.md)。

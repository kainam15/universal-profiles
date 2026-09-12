---
name: acprof-model-adaptation
description: 为 AC-Prof 新增或修复模型、adapter、任务族或 backend 适配，联动路由、依赖镜像、输入输出和实际推理验证；单纯选择已有模型时无需使用。
---

# AC-Prof 模型与后端适配

## 按需读取

先查[运行兼容与扩展契约](../../../docs/Runtime_Compatibility.md#新增一个模型适配)，按目标任务定位同文档中的支持范围。
输入尺度查 [CLI 与 workload 清单](../../../docs/CLI_Reference.md#输入规模与音频清单)，产物变更再查[采集协议](../../../docs/Profiling_Protocol.md#协议不变量)。

## 执行流程

1. **明确适配边界。** 核对上游接口与可复用实现，确定任务、输入模态、尺度、CPU/GPU dtype 和依赖组合。已支持接口直接复用；记录不适用的设备或工具。
2. **声明路由与环境。** 在 `acprof/runtime_profiles.py` 中维护环境、adapter、依赖锁和模型/架构映射。新增任务族或 backend 时同时核对 `host/detect.py`、`host/task_support.py`、`config.py`，不能靠放宽预检宣称支持。
3. **补齐推理协议。** 能复用时选择 `family-default`；需要 adapter 时实现 `BaseHandler` 四阶段协议，在 `handlers/__init__.py` 的注册机制中接入，并由 `_auto_register` 导入。提示词和参数经 workload、物化输入计划进入 server 与 profiler。
4. **覆盖失败路径。** 根据适配内容检查路由冲突、依赖不匹配、离线加载、模态丢失、尺度截断、输出序列化和镜像错配。数值字段有变化时沿生产者与消费者补齐兼容验证。
5. **构建并实测。** 依赖组合在目标 Python/CUDA 容器解析，导出精确锁并执行 `pip check`；复用已有环境时核对匹配性。构建实际镜像，用独立输出目录验证所声明设备的加载、预处理、推理和输出，再分别验证涉及的 profiler。
6. **更新覆盖说明。** 在运行兼容文档记录支持接口和限制，更新受影响的参数、字段或用户示例。报告实际设备、工具、结果与未验证范围；普通推理成功不外推为所有工具兼容。

只修复已有路径时执行受影响的步骤，不重建无关任务族。验证选择见[测试指南](../../../docs/Testing.md)。

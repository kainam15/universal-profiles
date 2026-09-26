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
2. **声明路由与环境。** 在 `acprof/extensions/*/manifest.json` v2 声明 task、architecture/model type、backend、adapter 与执行/测量能力；共同尺度、默认参数、IO、精度和输入规划能力放入 family 默认值，差异用 task/backend 覆盖。规则显式填写 `priority`，用 `CATALOG.resolve()` 的黄金用例与未知/冲突失败用例核对消费行为。平台、环境/profile 及 adapter 提示词也在声明中维护，新依赖再补精确锁；字段与优先级查[扩展声明](../../../docs/Runtime_Compatibility.md#扩展声明与按需加载)。允许无 Torch 平台，不能靠放宽预检宣称支持。
3. **补齐推理协议。** 能复用时选择 `family-default`；需要 adapter 时继续实现 `BaseHandler` 四阶段协议，并由 manifest 的 Handler entrypoint 按需导入，不增加平行抽象或静默覆盖注册。通过 `validate_output` 或声明的 validator 在独立验证进程检查真实输出；用 `_workload` 补充已观测的模态事实。提示词和参数经 workload、物化输入计划进入 server 与 profiler。
4. **覆盖失败路径。** 根据适配内容检查路由冲突、依赖不匹配、离线加载、模态丢失、尺度截断、输出序列化和镜像错配。数值字段有变化时沿生产者与消费者补齐兼容验证。
5. **构建并实测。** 依赖组合在目标 Python/CUDA 容器解析，导出精确锁并执行 `pip check`；复用已有环境时核对匹配性。构建实际镜像，用独立输出目录验证所声明设备的加载、预处理、推理和输出，再分别验证涉及的 profiler。
6. **更新覆盖说明。** 在运行兼容文档记录支持接口和限制，更新受影响的参数、字段或用户示例。报告实际设备、工具、结果与未验证范围；普通推理成功不外推为所有工具兼容。

只修复已有路径时执行受影响的步骤，不重建无关任务族。验证选择见[测试指南](../../../docs/Testing.md)。

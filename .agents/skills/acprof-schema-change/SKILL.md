---
name: acprof-schema-change
description: 用于 AC-Prof 新增或修改采集指标、CSV 字段、静态元数据或 profiler 产物协议；联动核对来源、单位、测量窗口、历史兼容、消费者和回归验证。
---

# AC-Prof 指标与产物协议变更

## 使用范围

处理字段、计算公式、聚合口径、JSON 结构或来源记录的变化。普通参数调整、只读结果解释和纯界面调整不必进入完整协议变更流程。

遵循项目 [AGENTS.md](../../../AGENTS.md)；已加载的通用规则无需重读。以当前代码核对 schema 版本及默认值，不沿用旧实验或记忆中的数值。

## 按需查阅

先按变更类型选择入口，再搜索具体字段及相关段落。字段定义、公式和兼容口径由 docs 的对应专题维护，本技能维护变更流程。

| 变更类型 | 文档入口 |
| --- | --- |
| CSV 字段、公式或聚合 | [CSV 字段分组](../../../docs/Metrics.md#result_allcsv-字段解释)，按指标组读取 |
| 静态元数据、输入计划或来源 | [输出文件](../../../docs/Profiling_Protocol.md#输出文件)、[静态元数据](../../../docs/Profiling_Protocol.md#static_metajson-字段)、[采集历史](../../../docs/Profiling_Protocol.md#collection_historyjson-字段) |
| Profiler 计划或回填协议 | [Torch 与 NCU](../../../docs/Profilers.md#torch-与-ncu-计算指标)或[执行指标](../../../docs/Profilers.md#massif-与-nsight-systems-执行指标)，再定位对应 plan 字段 |
| 镜像或运行环境元数据 | [运行兼容说明](../../../docs/Runtime_Compatibility.md#构建复用和验证)及相关静态字段 |

## 先确定指标约定

对本次变更简要说明：字段名、类型、单位、采集来源、公式、测量窗口、聚合方式、适用任务或设备、缺失值含义、历史文件如何读取。

已有需求足以确定这些信息时直接实施；只有存在会改变实验含义的未决选择时才澄清。保持 task family 的统一语义，不用某个模型的特殊结果修补公共公式。

## 按数据流定位改动

| 位置 | 检查内容 |
| --- | --- |
| [config.py](../../../acprof/config.py) | `CSV_FIELDS`、`STATIC_META_FIELDS`、schema 常量与默认值 |
| [host/client.py](../../../acprof/host/client.py)、[client_metrics.py](../../../acprof/host/client_metrics.py) | 请求窗口、指标计算、聚合和主实验行的生成 |
| [monitors/](../../../acprof/monitors) | 原始采样来源、时间边界、单位与失败处理 |
| [packet/](../../../acprof/packet) | PCAP 解析、请求匹配和结果合并 |
| [static_metadata.py](../../../acprof/host/static_metadata.py)、[input_plan.py](../../../acprof/host/input_plan.py)、[orchestrator.py](../../../acprof/host/orchestrator.py) | 静态元数据、输入计划、执行编排和结果输出 |
| [compute_profile_plan.py](../../../acprof/host/compute_profile_plan.py)、[execution_profile_plan.py](../../../acprof/host/execution_profile_plan.py) | profiler 计划、状态、复用和来源 |
| [host/posthoc/](../../../acprof/host/posthoc)、[cli/backfill_compute.py](../../../acprof/cli/backfill_compute.py) | 补采、回填、历史兼容和备份；职责见[结果分析与补采](../../../docs/Architecture.md#结果分析与补采) |
| [plotting/](../../../acprof/plotting)、[analysis/](../../../acprof/analysis) | CSV 读取、派生值、过滤条件、绘图和数值分析 |
| [collection_history.py](../../../acprof/host/collection_history.py) | 补采、重试及修复过程的来源记录 |

用 `rg` 搜索字段及相邻概念，覆盖真正受影响的生产者与消费者；不要只修改字段列表，也不要机械改动所有表中模块。

## 核对协议边界

逐项对照[协议不变量](../../../docs/Profiling_Protocol.md#协议不变量)，明确本次字段的窗口、归因和历史兼容策略。
涉及新来源时核验输入计划 hash、模型 revision 与工具采样来源；涉及补采时检查备份和采集历史。
具体数值定义留在对应专题，不在本技能另存一份。

## 验证与交付

为有行为变化的部分覆盖新数据、旧文件缺字段、不适用设备、采集失败和补采来源等相关分支。期望值独立推导，避免用被测公式计算期望值。

先运行受影响的测试文件，例如：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_posthoc.py' -v
```

再按改动范围完成项目要求的测试、编译检查和 `git diff --check`。不为文档措辞写业务测试，不把真实长矩阵当作字段变更的默认验证方式。

更新 docs 对应专题中受影响的字段和兼容约定；用户操作变化时同步 README 的示例，模块或环境变化时按[文档分工](../../../docs/README.md#文档规则与流程的分工)更新对应专题。交付时说明口径、兼容策略、验证结果和未覆盖边界，不顺便修改已有实验数据。

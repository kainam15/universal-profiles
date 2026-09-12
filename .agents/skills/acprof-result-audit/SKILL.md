---
name: acprof-result-audit
description: 用于审计 AC-Prof 实验目录、CSV、日志或错误截图，解释指标计算与空值，区分失败原因、补采来源和正式测量范围；诊断本身保持只读。
---

# AC-Prof 实验结果审计

## 目标与边界

回答“这个结果是否可信”“空值或报错是什么意思”“指标怎么算出来的”。先读取用户指定的产物并核对实际运行条件，再给结论。仅诊断时保持只读；不默认重跑实验、启动 profiler、删除容器或回填 CSV。用户已经要求修复时，依照已有授权实施可验证的修复，不重复确认。

项目约束见 [AGENTS.md](../../../AGENTS.md)；已加载的规则无需重读。大文件先读取表头、相关行或 JSON 中的必要字段；运行中的实验只做低开销、必要的检查，不增加后台监控或 GPU 负载。

## 按问题查阅

从用户指定的产物进入，按下表定位口径，再追到实际生产者；无需通读整个字段字典或结果目录。本技能维护取证流程，指标定义和结果语义以链接中的对应章节为准。

| 当前问题 | 文档入口 |
| --- | --- |
| 哪些行可用于分析、字段如何计算 | [CSV 字段分组与分析范围](../../../docs/Metrics.md#result_allcsv-字段解释)，再读取相应指标组 |
| 空值、失败或工具不适用 | [常见判断](../../../docs/Troubleshooting.md#常见判断)、[运行状态与错误](../../../docs/Metrics.md#运行状态与错误) |
| 行数、请求数或实验是否完成 | [CSV 行数与请求数](../../../docs/Profiling_Protocol.md#csv-行数与请求数)、[输出文件](../../../docs/Profiling_Protocol.md#输出文件) |
| 元数据、补采或环境来源 | [静态元数据](../../../docs/Profiling_Protocol.md#static_metajson-字段)、[采集历史](../../../docs/Profiling_Protocol.md#collection_historyjson-字段)；涉及镜像时再查[运行兼容说明](../../../docs/Runtime_Compatibility.md#构建复用和验证) |

## 建立证据链

1. **定位本次运行。** 确认实验目录、模型 ID/revision、命令、输入计划、时间和文件修改状态，不把模型 ID 当成本地结果路径。
2. **检查产物。** 读取 `result_all.csv`、`static_meta.json`、`input_scale_plan.json` 和已有的 `collection_history.json`；按问题读取相关 case CSV、日志、profiler 计划或报告。文件不存在时如实说明。
3. **核对计划与来源。** 检查输入计划 hash、workload 素材来源、采集工具、schema 版本、补采和重试历史；`disabled` 与 `posthoc_backfill` 可能共同描述先关闭再补采的历史。
4. **对照运行证据。** 根据日志确定容器及请求阶段。需要时对相关容器执行定向 `docker inspect`，只输出 `.State` 等必要字段，不打印包含凭据的完整环境。
5. **追到当前代码。** 指标公式从实际生产者及聚合逻辑确认，不能只根据列名猜测。

## 划分可分析的数据

按 [CSV 分析范围](../../../docs/Metrics.md#result_allcsv-字段解释)筛选常规性能或能耗数据。另行统计异常、预热及跳过项，解释覆盖缺口；资源失败边界分析按[运行状态约定](../../../docs/Metrics.md#运行状态与错误)保留失败记录。

缺少 `status` 或 `warmup` 的旧文件应明确说明并根据当时 schema 判断，不能未经说明就视为全部正式成功。错误行中的部分数值不自动成为有效测量。

区分计划行数、实际写入行数、成功窗口数和请求数。根据[行数与请求数约定](../../../docs/Profiling_Protocol.md#csv-行数与请求数)，读取实际输入与资源计划及剪枝信息；不要套用固定矩阵行数。若文件仍在增长或实验仍在运行，标明审计时点，不把部分写入视为正式完成。

## 分类与复核

将日志中的发生阶段、错误文本与容器状态对照[诊断证据](../../../docs/Troubleshooting.md#诊断证据)，
再按该文档中的 OOM、超时、空值或 profiler 分支解释。只依据现有证据分类，缺失证据明确记录。
需要实时检查时选择[对应命令](../../../docs/Troubleshooting.md#实时状态检查)，不套用历史机器状态。

## 指标解释的最小完整答案

给出字段、来源、公式、单位、窗口、分母及适用条件，并用用户数据中的一行演算。区分 packet/application latency、单请求与窗口统计、Torch logical FLOP 与 NCU executed FLOP、Massif 生命周期峰值与 cgroup 窗口指标。

能耗解释区分 CPU package、估算 vCPU 和 GPU；检查 idle baseline 波动、采样间隔和归因假设。不要将估算写成独立实测，也不要由一个样本声称有稳定的尾延迟统计。

按指标搜索相关入口：[host/client.py](../../../acprof/host/client.py)、[client_metrics.py](../../../acprof/host/client_metrics.py)、[monitors/](../../../acprof/monitors)、[packet/](../../../acprof/packet)、[plotting/data.py](../../../acprof/plotting/data.py)、[host/posthoc/](../../../acprof/host/posthoc)。

## 报告方式

先给结论，再列出关键证据：相关文件、字段、行或容器状态；明确哪些已证实、哪些是推断、哪些还缺证据。必要时给出范围最小的后续检查。

审计不更改原始文件。用户已要求数据修复时，保留备份、原子写入、记录来源并验证受影响字段；不把推导或补录值冒充当时实测数据。

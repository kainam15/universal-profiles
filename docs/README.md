# 项目文档导航

AC-Prof 的长期知识在本目录按主题维护。先按任务选择一篇，再搜索相关标题或符号；链接是按需阅读入口，不表示需要加载整份文档。

## 按任务查阅

| 当前任务 | 权威文档与范围 |
| --- | --- |
| 定位代码、重构模块、维护兼容入口 | [代码架构](Architecture.md)：职责、依赖方向与兼容设计 |
| 修改采集窗口、产物或冷启动 | [采集协议](Profiling_Protocol.md)：生命周期、文件与 schema、请求数、时间预算、冷启动边界 |
| 查字段、历史数据、绘图或延迟拟合 | [指标与结果分析](Metrics.md)：分析范围、单位、公式、归一化、图表与模型 |
| 查完整列协议、审计和置信区间 | [指标登记表](Metric_Reference.md)：由代码生成的单位、来源与窗口；[分析入口](Metrics.md#窗口置信区间与开销对照) |
| 解释功率、能耗、idle 或归因误差 | [能耗测量](Energy_Measurement.md)：RAPL、NVML、估算 vCPU 与适用限制 |
| 选择或排查 GPU/CPU profiler | [分析器](Profilers.md)：Torch、NCU、Massif、Nsys 的窗口、采样、成本与失败 |
| 排查 OOM、cgroup、抓包、空值或部分结果 | [运行排障](Troubleshooting.md)：证据分类、恢复入口与实时状态检查 |
| 新增模型/backend、改依赖或镜像 | [运行兼容](Runtime_Compatibility.md)：任务目录、加载接口、环境与构建契约 |
| 查参数、workload 清单、TUI 设置协议 | [CLI 与设置](CLI_Reference.md)：选项、输入规模、持久化和历史兼容 |
| 选择测试、做 TUI 回归、判断验证边界 | [测试指南](Testing.md)：代码验证、硬件冒烟、终端证据与文档检查 |

安装与用户操作示例保留在根 [README](../README.md)，详细知识通过本页进入对应专题。
实现与默认值用当前代码和 `--help` 核对；历史实验解释以当次产物的版本、计划、日志和来源为准。

## 文档、规则与流程的分工

| 层级 | 内容 | 读取时机 |
| --- | --- | --- |
| 根 [AGENTS.md](../AGENTS.md) | 项目地图、全局约束、验证入口和完成标准 | 每次任务 |
| 目录级 `AGENTS.md` | 该目录独有的实现或产物约束 | 进入相关目录时 |
| 本目录专题 | 项目是什么、为什么这样设计、协议如何定义 | 任务涉及该主题时 |
| `.agents/skills/*/SKILL.md` | 可复用的多步骤执行流程 | 符合技能描述时 |
| 实验目录、命令输出 | 该次运行的事实与当前机器状态 | 实时、定向检查 |

一个主题只维护一处完整定义，其他文档保留必要摘要或链接；相关小主题使用章节，不为每个字段创建文件。
历史审计 `reviews/` 是带日期和输入指纹的证据快照，不是现行协议，也不证明当前机器状态。
临时任务计划和验证输出放在会话或 `internal-testing/`，不要写进长期 Agent 规则。

## 可复用流程

| 任务 | Skill |
| --- | --- |
| 新增指标或产物协议 | [acprof-schema-change](../.agents/skills/acprof-schema-change/SKILL.md) |
| 审计结果、OOM 或 profiler 异常 | [acprof-result-audit](../.agents/skills/acprof-result-audit/SKILL.md) |
| TUI 布局和交互验证 | [acprof-textual-regression](../.agents/skills/acprof-textual-regression/SKILL.md) |
| 新增模型、adapter 或 backend | [acprof-model-adaptation](../.agents/skills/acprof-model-adaptation/SKILL.md) |
| 完整 profiling 或 benchmark 实验 | [acprof-profiling-workflow](../.agents/skills/acprof-profiling-workflow/SKILL.md) |

OOM 排障复用结果审计流程；benchmark 与完整 profiling 共用实验流程，不再分别创建相近技能。

## Claude 与通用 Agent 指令

根 [CLAUDE.md](../CLAUDE.md) 只用 `@AGENTS.md` 引入通用规则。Claude 特有差异才写在导入后；
以后若添加目录级 `CLAUDE.md`，同样只导入对应 `AGENTS.md`。不要在根文件批量导入全部子目录、专题或 Skills。
Markdown 导航链接不做全文导入；Claude 的 `@` 导入会直接加载正文，语义见[官方说明](https://code.claude.com/docs/en/memory#agentsmd)。

这一组织方式参考 [AGENTS.md 项目](https://github.com/agentsmd/agents.md)和
[Skill 的分层披露说明](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md#progressive-disclosure)。
两者有公开维护的仓库，许可证分别为 MIT 和 [Apache-2.0](https://github.com/anthropics/skills/blob/main/skills/skill-creator/LICENSE.txt)。
这里只采用 Markdown 导航和按需读取的组织方式，沿用现有工具；不引入文档框架、自动同步依赖或测量期进程。

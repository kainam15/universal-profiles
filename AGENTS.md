# 项目地图与协作规则

AC-Prof 对 Docker 中的 Hugging Face 推理服务进行可复现分析，输出延迟、能耗、资源指标与实验产物。

## 仓库地图

| 位置 | 职责 |
| --- | --- |
| 根 CLI → `acprof/cli/` | 采集、探测、补采、绘图与 TUI 入口 |
| `acprof/host/`、`acprof/container/`、`acprof/workloads/` | 主机编排、容器推理、确定性输入 |
| `acprof/monitors/`、`acprof/packet/` | 原始测量、抓包与合并 |
| `acprof/analysis/`、`acprof/plotting/`、`acprof/tui/` | 结果分析、绘图、终端界面 |
| `dockerfiles/`、`tests/`、`examples/` | 镜像、回归测试、手动示例 |
| `docs/`、`.agents/skills/`、`internal-testing/` | 长期知识、可复用流程、临时验证 |

## 全局规则

- 使用简体中文回复；`AGENTS.md` 的标题与说明使用简体中文，保留技术标识。
- 不修改 `docs/Original_Project_Definition.md`；用户未明确要求时不提交 Git。
- 功能或结构改动前先检索 GitHub，评估兼容性、许可证、维护、依赖成本与测量开销，说明复用取舍。
- 使用已有 `.venv`、Python 3.10+；正式采集要求原生 Linux、本机 Docker Engine、cgroup v2。
- Python 改动遵循下方语义工具与验证工作流；不得通过扩大忽略规则掩盖新问题。
- 保持指标归因和可复现口径；界面活动、绘图、通知与额外诊断不进入正式测量窗口。
- 凭据放在被 Git 忽略的 `.env.local`，不得写入文档或提交密码、令牌、webhook。
- GPU/磁盘余量、Docker 状态、Git 分支和进程按需实时检查；临时计划不进入长期 Agent 文档。

## Python 修改工作流

修改 Python 代码时优先使用 PyCharm MCP，只执行任务涉及的操作；纯文档修改按文档范围验证。

1. 使用 `search_symbol` 定位程序符号；`rg` 用于文件、普通文本和配置检索。
2. 分析调用或依赖关系优先使用 `analyze_calls`，结合源码确认动态调用；不得仅凭文本搜索推断 Python 符号关系。
3. 重命名 Python 符号优先使用 `rename_refactoring`，核对引用更新和实际差异。
4. 修改后使用 `lint_files` / `get_file_problems` 检查受影响文件的 IDE diagnostics，处理本次改动引入的问题。
5. 使用 `get_run_configurations` 选择相关的已有 Run Configuration，通过 `execute_run_configuration` 执行测试和 smoke test。
6. 最后按[验证范围](docs/Testing.md#验证范围)运行相关 unittest / evidence runner、Ruff 及真实 workload；本地 PyCharm 可用 pytest 执行同一批用例，但不替代 evidence 产物或代表 CI 已迁移。局部修改不默认跑完整测试集；只有新改动、失败或未解决问题才扩大或重复验证。
7. 使用 `git_status` 检查最终改动范围，核对新增、被忽略文件，并结合 diff 确认没有混入无关变更。

MCP 不可用、索引不完整或没有适用 Run Configuration 时，说明限制并用源码分析和项目 CLI 入口继续；不把空调用树当作没有依赖。pytest 为可选本地 runner；不可用时继续执行现有 unittest / evidence 入口，不宣称 pytest 通过。

在已授权范围内完成修改、验证和必要修复，无需逐步确认。真实 workload 缺少 Docker、GPU、模型等运行条件时，明确标为未验证，不用 IDE diagnostics 或 smoke test 代替。

## Skill 与 Agent 文档编写

后续新建 Skill 或 Agent 文档（如 `AGENTS.md`）时，参考 [OpenAI：重新思考 GPT-6 Astra 的技能与提示词](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。

- 描述简短、适用场景明确；入口保留必要规则与导航，详细资料和步骤按需加载。
- 规则聚焦任务或项目特有约束，避免重复指令、全量必读清单和不必要的固定流程。
- 明确完成标准和需要确认的边界；在已授权范围内完成实现、相关验证和修复，验证范围与改动相称。

## 按任务读取

进入目录前检查适用的局部 `AGENTS.md`。文档先用 `rg` 定位，再读取相关章节；下表不是必读清单，已加载且未变化的规则无需重读。

| 任务 | 入口 |
| --- | --- |
| 安装、运行、参数 | [快速开始](README.md#快速开始)、[CLI 与设置](docs/CLI_Reference.md) |
| 架构或模块重构 | [代码架构](docs/Architecture.md) |
| 协议、冷启动、字段变更 | [采集协议](docs/Profiling_Protocol.md)、[指标](docs/Metrics.md) → [变更流程](.agents/skills/acprof-schema-change/SKILL.md) |
| 能耗、OOM、cgroup、结果异常 | [能耗](docs/Energy_Measurement.md)、[排障](docs/Troubleshooting.md) → [审计流程](.agents/skills/acprof-result-audit/SKILL.md) |
| 模型/backend、依赖、镜像 | [运行兼容](docs/Runtime_Compatibility.md) → [适配流程](.agents/skills/acprof-model-adaptation/SKILL.md) |
| profiling、benchmark、GPU profiler | [实验流程](.agents/skills/acprof-profiling-workflow/SKILL.md)、[分析器](docs/Profilers.md) |
| TUI、焦点、日志、设置 | [交互说明](README.md#交互式终端界面) → [回归流程](.agents/skills/acprof-textual-regression/SKILL.md) |
| 绘图、测试、文档维护 | [结果分析](docs/Metrics.md#图表与延迟拟合产物)、[测试指南](docs/Testing.md)、[文档分工](docs/README.md) |

## 常用验证

从仓库根目录执行，按[改动范围](docs/Testing.md#验证范围)选择：

```bash
.venv/bin/ruff check .
.venv/bin/python -m pre_commit run --all-files --show-diff-on-failure
.venv/bin/python run.py --help
.venv/bin/python scripts/run_tests.py --pattern 'test_env_utils.py' --report internal-testing/env-tests.json
# 跨模块变更需要完整回归时：
.venv/bin/python scripts/run_tests.py --report internal-testing/host-tests.json
.venv/bin/python -m compileall -q acprof run.py probe.py plot.py profile.py tui.py
git diff --check
```

开发工具安装、hook 与版本维护见[开发质量检查](docs/Testing.md#开发质量检查)。

## 完成标准

- 处理目标行为及相关失败路径，保持当前协议；已有替代实现的历史兼容代码删除，旧协议明确报错；按改动范围取得实际验证证据。
- 更新对应 `docs/` 权威专题及受影响的摘要、示例和链接；Skill 只维护执行流程。
- 检查完整变更清单（含新增、被忽略文件）；说明验证结果和未覆盖的 Docker/GPU 或终端范围。

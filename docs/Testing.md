# 测试与验证

选择本次改动能改变的行为和失败路径。下列命令均从仓库根目录执行，使用已有 `.venv`。
测试数量、设备余量和镜像可用性由本次执行确认，不把历史通过记录作为当前验证结果。

## 验证范围

| 改动 | 应取得的证据 |
| --- | --- |
| 文档、导航或链接迁移 | 本地文件与章节锚点可达、旧入口仍可跳转、代码块与差异格式正确；无需为措辞运行模型 |
| Skill | frontmatter、名称与描述匹配、相对路径、流程边界和可用验证器的格式检查 |
| 单个逻辑或失败路径 | 能观察目标行为的相关 unittest；修复缺陷时先复现，再验证修复 |
| 模块搬迁、依赖方向或兼容入口 | 当前入口成功、已删除入口拒绝、全套 unittest、CLI 帮助和编译；mock 放到函数实际查找依赖的模块 |
| 指标或产物协议 | 独立推导的期望值、当前 schema 成功与旧 schema 拒绝、缺失/失败/不适用字段和受影响消费者 |
| 模型、backend、依赖或 Dockerfile | 路由与离线加载测试、镜像构建、所声明设备的真实推理；profiler 分别验证 |
| TUI | 受影响的交互与尺寸检查；原生终端问题还需对应终端证据 |

## 开发质量检查

Ruff、pre-commit 和锁生成工具 uv 由 [`requirements-dev.in`](../requirements-dev.in) 声明，
完整版本与制品哈希保存在 [`requirements-dev.lock`](../requirements-dev.lock)。开发锁以主机锁
为约束，避免在同一个 `.venv` 安装时引入冲突；不加入主机运行依赖或容器环境身份。

```bash
.venv/bin/python -m pip install --require-hashes -r requirements-dev.lock
.venv/bin/python -m pip check
.venv/bin/python -m pre_commit install
.venv/bin/python -m pre_commit run --all-files --show-diff-on-failure
```

`install` 仅给当前 clone 安装 Git hook；新 clone 需执行一次。手动运行和 CI 读取同一份
[`.pre-commit-config.yaml`](../.pre-commit-config.yaml)，检查尾随空白、文件末尾换行、YAML、JSON、
TOML、冲突标记、文件大小和 Python 代码。大文件检查对所有文件执行，限额为 1 MiB，覆盖现有
约 938 KiB 的固定音频输入；Markdown 的两个行尾空格保留为换行。临时证据目录和禁止修改的
`docs/Original_Project_Definition.md` 不参与 hooks。

Ruff 版本由 [`pyproject.toml`](../pyproject.toml) 的 `required-version` 强制核验，Python 目标为
3.10，显式启用 `E4`、`E7`、`E9`、`F`。第一版不启用 import 排序、`E501` 或 formatter；
`line-length = 100` 本身不检查行长。Ruff hook 只检查，不自动修复；空白和末尾换行 hooks
会修正文件并返回失败，检查 `git diff` 后重新运行。不得用扩大 `ignore` 或排除目录掩盖新问题。
公共导出用显式重导出或 `__all__` 表达；必须先设置路径、环境或验证缺失依赖的 import，
只在对应行标注具体规则及原因，不统一忽略 `__init__.py`。

本地 Python 修改先运行 Ruff，再按受影响行为选择已有 runner 的测试模式；`--pattern` 可重复。
未指定 `--pattern` 会执行完整测试集，不作为局部修改的默认要求。Git hook 不运行业务测试或硬件采集。

```bash
.venv/bin/ruff check acprof/host/env_utils.py tests/test_env_utils.py
.venv/bin/python scripts/run_tests.py --pattern 'test_env_utils.py' \
  --report internal-testing/env-tests.json
git diff --check
```

更新开发工具时，修改输入并使用开发锁中的 uv 版本重新生成；Ruff 需同步修改版本约束及 hook 的
完整 commit SHA，pre-commit 需同步最低版本。随后核对锁和 diff，重新安装开发锁并运行完整 hooks。

```bash
.venv/bin/uv pip compile requirements-dev.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header --output-file requirements-dev.lock
```

`scripts/compile_locks.py --check` 仍只验证既有容器锁与 profile 映射，不代替开发锁的重新解析。
这些开发工具只在编辑、提交和 CI 验证时运行，不进入正式测量窗口。

### PyCharm MCP 的验证边界

使用项目的 `.venv` 解释器；符号搜索按需限定 `paths=["acprof/**", "tests/**"]`，
避免将临时虚拟环境中的第三方代码视为项目实现。调用分析无法解析已找到的 Python 符号时，
结合符号文档和源码核对调用者；空结果不能证明没有依赖。

PyCharm 2026.2.3（build `262.10968.92`）已复现一种 MCP 兼容问题：
`analyze_calls` 的 `isCallableSymbol` 依赖显示文本中的 `name(...)`，
而 Python 函数的 Usage View 文本只有名称，因此真实 `PyFunction` 也会被过滤。
这类错误应修复 IDE 工具的函数识别，不需要改动业务函数或重建 Python 环境。
修复验收应包含真实函数的入向、出向调用和不存在符号的错误路径；
本机兼容补丁还需验证版本匹配、撤销与启动加载，并区分独立 JVM 检查和完整 IDE 重启。
实现依据见 [JetBrains Call Hierarchy](https://github.com/JetBrains/intellij-community/blob/master/plugins/mcp-server/mcpserver.toolsets/src/general/CallHierarchyAnalysisSupport.kt)
与 [Python Usage View](https://github.com/JetBrains/intellij-community/blob/master/python/src/com/jetbrains/python/findUsages/PyElementDescriptionProvider.java)。

项目使用 unittest。若从代码位置创建的 Run Configuration 自动选择 pytest，而解释器没有
安装 pytest，应选择 unittest 配置，或通过 IDE 终端运行本页的 `scripts/run_tests.py` 入口；
无需为该 IDE 默认值引入另一套测试依赖。执行证据必须包含实际输出和退出码。
`build_project` 若提示无法收集构建诊断，不能替代 Python 编译和相关测试；
依赖查询返回空列表也不能证明 Python 环境没有安装依赖。
临时重命名和工具测试文件放在任务独立的 `internal-testing/` 子目录中。

## 自动化验证入口

安装分发修改除普通回归外，还需构建 sdist/wheel，在隔离环境从空目录执行
`scripts/check_distribution.py`；standalone 用 `--binary <path>` 执行相同验收。
该脚本核对全部公共帮助入口、参数错误、无 Docker 时的 doctor JSON、内置资源和 packet worker 的实际输出。
CI 另执行真实 `uv tool install`，构建步骤见[发行包说明](Distribution.md#linux-standalone)。
这些检查不代替 Docker/GPU 推理和完整 profiling。

初始化入口修改运行 `test_setup.py` 和 `test_tui_onboarding.py`，覆盖缺失 uv、安装/诊断失败、
重复执行、非交互终端、首次配置与已有设置保留；使用隔离 uv 目录实际执行
`./setup.sh --no-tui --no-modify-path`。真实推理另用安装后的命令运行最小 basic CPU 实验并审计结果。

测试使用 `unittest`、`unittest.mock` 和临时目录；文件名为 `test_*.py`，方法名以 `test_` 开头。
模拟网络、Docker、硬件与通知边界；避免测试触发真实采集或改写用户设置。

```bash
# 按受影响的行为选择测试文件
.venv/bin/python -m unittest discover -s tests -p 'test_env_utils.py' -v
# 跨模块变更的全套回归
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py --help
.venv/bin/python -m compileall -q acprof run.py probe.py plot.py profile.py tui.py
git diff --check
```

| 实现范围 | 测试入口示例 |
| --- | --- |
| 模块依赖与导入副作用 | `tests/test_architecture.py` |
| 环境、依赖与镜像 | `tests/test_runtime_profiles.py`、`tests/test_environment_identity.py`、`tests/test_lock_compiler.py`、`tests/test_image_layers.py`、`tests/test_runtime_image_build.py`、`tests/test_image_reuse.py`、`tests/test_runtime_validation.py` |
| 模型文件与离线加载 | `tests/test_model_files.py`、`tests/test_model_download.py`、`tests/test_offline_model_loading.py` |
| 字段、能耗、资源与补采 | `tests/test_energy_cpu.py`、`tests/test_resource_usage.py`、`tests/test_posthoc.py` |
| 页面、设置与焦点 | `tests/test_tui_layout_settings.py`、`tests/test_tui_interaction.py`、`tests/test_tui_input.py` |
| 固定页头、底栏与操作颜色 | `tests/test_tui_page_chrome.py`；中英文、三种尺寸、滚动与缩窗、命令框显隐、八种主题和按钮状态 |
| 日志、语言与命令 | `tests/test_tui_log_view.py`、`tests/test_tui_i18n.py`、`tests/test_tui.py` |
| 终端色深与 RGB 输出 | `tests/test_tui_colors.py`；检查缺失/空 `COLORTERM`、真彩色、256 色与自动检测 |
| 中文浮层缺字 | `tests/test_tui_cjk_rendering.py`；通知覆盖按钮、真实遮挡、宽字符两半的局部刷新、中英文和奇偶列宽缩放 |
| 统计报告、异步读取与计算 | `tests/test_tui_reports.py`、`tests/test_report_views.py`、`tests/test_uncertainty.py` |
| Docker 镜像树、层空间、标签删除与采集互斥 | `tests/test_image_management.py`、`tests/test_tui_images.py` |
| 表头拖动、固定列、滚动范围与鼠标释放 | `tests/test_tui_table_resize.py`、`tests/test_tui_all_tables.py`、`tests/test_tui_images.py` |

以上是定位入口，不是每次必须运行的清单。先用 `rg --files tests` 查实际受影响的测试；新改动、失败或未解决问题才需要扩大或重复验证。

`test_runtime_image_build.py` 在 Docker 边界模拟环境中验证四层构建、跨 profile 的环境共享、
模型 commit、Torch 来源、BuildKit secret、标签错配、额外包、输入变化及构建失败停止。
环境身份测试覆盖 40 个 profile / 27 个环境、无 Torch 环境、跨任务族精确共享及 cu128 的版本差异；CV 新增 timm 后按新包集计算身份。注释、锁文件名
和条目顺序不影响身份，版本、制品、来源、平台和系统锁影响身份。配方变化改变构建缓存，业务
代码变化只重建服务层。它们不能替代实际容器构建与推理验证。

镜像分层测试的临时目录需包含指纹依赖的全部文件，包括 `acprof/model_spec.py`；
修改模型文件筛选或声明解析代码时，断言模型层和服务层指纹变化、依赖层身份不变。
模型接口新增解析字段时，同步更新 CLI 委派测试中独立声明的完整 `model_resolution`
期望值，保留完整对象比较，不从被测函数的返回值生成期望。

## CI 与环境测试

独立 `lint` job 在 Python 3.10 上只安装开发锁，执行 `pip check` 和完整 pre-commit hooks；
不安装推理依赖，也不需要 Docker/GPU。检查内容及 hook 版本与本地一致。

`.github/workflows/ci.yml` 在 Python 3.10 / 3.12 上安装哈希锁并执行主机回归；
每个版本将完整测试集按排序后的 test ID 轮转分成四片，保留 20 分钟作业超时。
所有分片都执行完整 discovery，新增测试会自动分配；不使用手写文件白名单。
每片分别上传 `host.json` 和实时保存的 `host.log`，失败时继续执行其他分片。
同时运行 `compile_locks.py --check`。七个任务族分别执行 CPU 接口测试，以随机小模型或明确导出的
样例验证真实加载与推理；audio 和 multimodal 在同一作业共享一个 CPU 依赖环境，仍分别执行测试。
网络在容器测试期间关闭。CI Actions 固定为已核验的 commit SHA，作业只授予仓库读取权限。
另有 `nlp-transformers560-cpu` 矩阵项运行新版原生架构与图像 processor 测试，避免主机缺少推理依赖的 skip 掩盖环境回归。
`custom-multimodal-cpu` 矩阵项运行 `test_custom_multimodal_runtime.py`：未知 `auto_map` 架构的
随机小模型经音频、图像、视频输入映射执行，比较共享四阶段与上游 pipeline 结果，并检查
重复推理、真实 Torch profiler 算子及独立 runtime validation；容器禁止网络和跳过。
两个 Transformers 版本线均执行 `test_audio_generation_runtime.py`：保存小型原生模型 snapshot，
再经过共享 Auto 加载、原生音频消息、真实 `generate` 和输出验证；逐参数核对加载权重，防止
组合模型的前缀处理丢失权重。4.57.6 验证 Voxtral/Qwen2 Audio；5.6.0 另验证 Omni 文本子模型。
这些随机权重验证接口，不证明完整 checkpoint 的容量、质量或正式 profiling 指标。

本地入口：

```bash
.venv/bin/python scripts/run_tests.py --report internal-testing/host-tests.json
.venv/bin/python scripts/compile_locks.py --check
.venv/bin/python scripts/check_runtime.py --family audio --variant cpu --output-dir internal-testing/audio-runtime
.venv/bin/python scripts/check_runtime.py --profile moss-transformers560 --build-only --output-dir internal-testing/moss-dependencies
.venv/bin/python scripts/check_runtime.py --profile nlp-transformers560-cpu --test-pattern test_transformers5_runtime.py --output-dir internal-testing/transformers5-runtime
.venv/bin/python scripts/check_runtime.py --profile multimodal-transformers560-cpu --test-pattern test_audio_generation_runtime.py --output-dir internal-testing/audio-generation-runtime
.venv/bin/python scripts/render_metric_reference.py --check
```

`run_tests.py` 保留 unittest 输出，并将每项测试的结果、失败/跳过原因、版本及耗时写入 JSON。
本地可用 `--shard-index 0 --shard-count 4` 重现一个 CI 分片；省略参数执行完整测试集。
报告的 `shard` 记录编号、总片数、完整发现数、选中数和排序后 test ID 列表的 SHA256。
汇总时须确认同一 Python 版本的分片齐全且测试集摘要相同，所有 test ID 无重复，
总执行数等于完整发现数；单片通过不代表主机回归完成。
空测试集必定失败；容器作业带 `--require-no-skips`，跳过或 expected failure 都不算环境验证通过。
普通主机测试允许缺少推理依赖时跳过，报告明确列出范围。`check_runtime.py` 的目录必须为空；
`runtime.json` 另记录逻辑 profile、环境 ID、平台/环境 image ID、完整运行清单和退出结果，不覆盖旧验证。
`--build-only` 仅证明依赖构建和清单核验；`--family` / `--profile` 共用主构建的环境准备入口。
普通接口测试固定使用 CPU，即使选择 CUDA wheel；GPU 和自定义 MOSS adapter 由完整服务镜像的
独立 `runtime_validation` 验证。依赖迁移需逐一构建全部唯一环境，再分别验证共享环境的各 profile。

更新锁前后比较原完整包版本，验证目标 wheel 的 ABI/平台和全部制品 SHA256。默认锁生成保留原
版本，显式 `--upgrade` 才更新环境包；uv 固定 0.12.13，生成 wheel 锁的脚本使用 Python 3.11+，
只读检查兼容 Python 3.10+。系统锁可指定同一 snapshot 再生成并比较，过程只修改一次性容器和
指定输出锁。具体命令见[当前配置](Runtime_Compatibility.md#当前配置)。

源代码、模型权重和依赖层保持分离，CPU 容器测试不下载 Hub 模型，不代替真实 GPU/PMU/抓包实验。

通用接口回归见 `test_generic_model_interfaces.py`：任意 checkpoint 名称的路由、固定 revision
元数据读取、任务／架构到版本线的选择、制品拒绝、prompt 预算及 workload 参数重放。
`test_cv_runtime.py` 在离线容器比较两类原生 timm 快照与官方模型输出；`test_timeseries_runtime.py`
比较 Chronos-Bolt／Chronos-2 的多序列预测；`test_nlp_runtime.py` 比较完整句向量模块图、默认 prompt
和归一化数值。`test_transformers5_runtime.py` 用旧环境没有的 Qwen3.5 原生小配置验证共享 NLP 入口，
并验证新版图像 processor 的输入形状、检测框及依赖 OpenCV 的多边形输出。该测试需在
`nlp-transformers560-cpu` 共享环境显式运行；随机小配置不是热门 checkpoint 的采集或准确率证据。

候选发现、冲突处理、本地模型声明、镜像／恢复身份和输入宽度回归见
`test_model_discovery.py`；`test_validation_stages.py` 检查成功步骤、失败位置及原异常保留。
`test_custom_pipeline_runtime.py` 在断网 NLP 容器创建微型随机 BERT snapshot 与自定义 pipeline，
比较原生接口和自定义接口的 label/score，并执行真实独立验证、失败阶段检查及自定义 `auto_map`
架构加载。它已加入默认 NLP 容器测试集，由现有 CI 的 NLP job 执行；也可单独运行：

```bash
.venv/bin/python scripts/check_runtime.py --family nlp --variant cpu \
  --test-pattern test_custom_pipeline_runtime.py \
  --output-dir internal-testing/custom-pipeline-runtime
```

该证据覆盖锁定环境中的标准文本分类桥接，不证明任意自定义模型、其它任务或 GPU 兼容。
多模态共享接口的离线测试可运行：

```bash
.venv/bin/python scripts/check_runtime.py --profile custom-multimodal-cpu \
  --test-pattern test_custom_multimodal_runtime.py \
  --output-dir internal-testing/custom-multimodal-runtime
```

主机协议／环境选择回归在 `test_custom_multimodal.py`；依赖 commit、文件哈希与镜像复用检查在
`test_model_download.py`、`test_image_reuse.py`；TUI 输入、命令、持久化和语言切换在
`test_tui_model_spec.py`。真实 Ultravox checkpoint 还需对应基础仓库访问权限和设备容量。

自动契约的主机回归使用 `test_model_contract.py` 和
`tests/fixtures/custom_pipeline_audio_like/`：覆盖 SHA 绑定、来源状态、输入映射、确定性参数、
动态表达式／冲突拒绝、依赖候选、缓存身份和报告导出；恶意源码检查在隔离进程中确认没有执行
模型顶层代码，也没有导入 Torch／Transformers。生成契约与通用 handler 的真实衔接使用随机小权重：

```bash
.venv/bin/python scripts/run_tests.py --pattern test_model_contract.py \
  --report internal-testing/model-contract-tests.json
.venv/bin/python scripts/check_runtime.py --profile custom-multimodal-cpu \
  --test-pattern test_model_contract_runtime.py \
  --output-dir internal-testing/model-contract-runtime
```

第二条命令核验锁定环境并在断网 CPU 容器执行加载、预处理、推理及输出验证，拒绝跳过。
同一容器测试还覆盖 basic 仅导入／签名、不加载权重，以及经显式审阅生成的嵌套 `turns` 模板。
M4～M6 的主机回归使用 `test_model_dependencies.py`、`test_model_review.py`、`test_model_transforms.py`、
`test_model_probe.py`、`test_model_inspection.py`：覆盖依赖角色／SHA／过滤、条件和动态路径、逐字段决策、
DSL 深度和引用限制、只读断网命令、证据完整性、CLI 导出及失败状态。TUI 的字段编辑、导出、
Probe 子进程交接、语言／resize 和测量禁用由 `test_tui_model_resolution.py` 在三种终端尺寸验证。
真实断网容器检查之外，外部依赖验收应保留 Hub SHA、实际选择文件和缓存内容，确认未下载无关权重；
真实只读 Probe 应使用独立测试镜像和目录，不能只用 mock Docker 命令代替。
真实 Ultravox 的静态 draft 验证不下载权重、不证明依赖完备或大模型推理成功；GPU／profiler
需各自取得运行证据，不能从这项 CPU fixture 验证外推。

生产模型声明与服务镜像的完整链路可使用 [Iris 示例](Runtime_Compatibility.md#本地模型声明与自定义-pipeline)，
按实际结果分别验收 basic／full；接口检查、预热、正式行和 profiler 结果分别计数。

### 无 Torch 运行时验收

```bash
.venv/bin/python scripts/check_runtime.py --profile onnxruntime-cpu \
  --basic-e2e --output-dir internal-testing/onnx-runtime
```

ONNX 的三个 profile 按 runtime 选择专用测试集，不按 family 误选 Torch 用例。
`.github/workflows/ci.yml` 的独立 ONNX CPU job 在 GitHub-hosted runner 必跑上述命令；
它检查 Torch、Transformers **未安装**，并要求 ORT/ONNX/Pillow/Tokenizers 等实际可导入。
任何指定测试模式为空、skip、expected failure、依赖缺失、超时、输出／必需字段错误或清理失败
都会失败。普通宿主测试仍可明确 skip 缺失的可选推理依赖，不代表容器通过。
`--test-pattern` 可重复覆盖默认测试，报告记录每个 pattern 的发现数；专用 CI 不覆盖默认集。

离线 fixture 固定 IR10/opset17，真实执行表格、图像、多输入文本，覆盖数值参考、动态维度、
样本顺序、dtype/shape 与固定形状拒绝；不会下载 Hub 模型。basic 回归复用生产 client、
ResourceUsageMonitor、CSV 合并及 audit，验证应用延迟、原有 batch/latency 吞吐口径、真实
cgroup CPU/内存、actual workload、输出验证和能力状态。合成图只证明接口与执行链路。
`--basic-e2e` 依次执行表格、图像、文本三种场景，每种包含一次 warmup 和两次正式请求；
任一场景失败即返回失败。图像检查原始与处理后尺寸，文本检查具名整数输入及实际 token 数。
真实 ORT、WordPiece tokenizer、HTTP probe 与既有规划器的联合回归还覆盖超限拒绝和自动尺度规划。
真实 HTTP 回归还验证异步等待、后台失败、超时及请求内不执行输出验证。
服务只发布 loopback 端口，不使用 privileged、Docker socket 挂载或硬件 runner；完成或失败后
清理本次容器。已有 `.github/workflows/hardware.yml` 仍仅手动触发。

两个预训练小模型可另行验证；第一步联网准备，第二步在同一无 Torch 环境断网执行。
以下目录需尚未存在或为空，重复检查请换新目录：

```bash
.venv/bin/python examples/onnxruntime/real_models.py prepare mnist \
  --directory internal-testing/pretrained-mnist
.venv/bin/python examples/onnxruntime/real_models.py prepare bert-tiny \
  --directory internal-testing/pretrained-bert-tiny
runtime_image=$(.venv/bin/python -c 'import json; print(json.load(open("internal-testing/onnx-runtime/runtime.json"))["image_id"])')
docker run --rm --network none --cpus 2 --memory 2g \
  -e ACPROF_RUNTIME_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:ro" -w /workspace "$runtime_image" \
  python examples/onnxruntime/real_models.py validate mnist \
  --directory internal-testing/pretrained-mnist
docker run --rm --network none --cpus 2 --memory 2g \
  -e ACPROF_RUNTIME_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:ro" -w /workspace "$runtime_image" \
  python examples/onnxruntime/real_models.py validate bert-tiny \
  --directory internal-testing/pretrained-bert-tiny
```

脚本保留来源、许可证说明、revision 和制品哈希，参考检查使用独立的 ONNX ReferenceEvaluator。
它验证一个样例的数值一致性，不执行训练数据准确率评测，也不输出正式性能结果。
真实 full 矩阵仍需相应硬件和权限；basic 成功不能替代 full 指标验收。

## TUI 与终端证据

终端兼容验收面向 Windows Terminal、PyCharm / JetBrains Terminal、VS Code Terminal、
Linux 原生终端、SSH 会话及浏览器 Web Terminal。SSH 只传输终端数据，记录结果时仍需注明
客户端终端、版本、字体和窗口列数 / 行数；每种环境分别标记通过、失败或未验证。
至少检查非浮层控件不重叠、边框可辨认、中文无乱码、`Tab` / `Input` / `Select` 可操作，
以及缩放后的布局和焦点可达性。字形能力有限的终端应检查基础 Unicode 线框和键盘路径，
不能依赖 `tall` 块状边框无缝拼接；纯 ASCII 终端不属于当前中文 TUI 的支持范围。

使用 `unittest.IsolatedAsyncioTestCase`、Textual `run_test()` / `Pilot` 和临时 `settings_path`。
尺寸覆盖用户报告的场景，并按布局变更检查 `80×24`、`120×30`、`150×45` 及运行中 resize。
七个页面的标题或状态摘要和底部操作栏应保持可见；主要操作在右下角，内容滚动、切换语言和隐藏快捷命令框后仍可点击。
按钮颜色检查主题切换、悬停、聚焦、禁用与恢复；删除和终止的红色、补采的黄色不能在交互中丢失，服务就绪和保存成功使用绿色。
镜像树路径高亮检查最终屏幕的连接线颜色，覆盖点击、方向键、折叠、搜索、失焦和中英文/深浅主题切换，防止祖先线未重绘或其它分支误亮。
镜像详情检查摘要与诊断分离、默认折叠、长包清单的末项可达、点击/Enter 展开，以及换行选择、空筛选、语言切换和缩放时的折叠状态；这些操作不得触发额外 Docker 查询。
详情分隔条检查上下拖动的实际高度变化、拖出边界后的最小可见区域、按键调整与默认值恢复、缩窗后再放大的手动高度和阅读位置；切换页面、采集开始、失去捕获、缩放或按 `Esc` 后均须释放鼠标，三个镜像视图与中英文均应可用。
镜像自动刷新检查首次打开、定时更新、失败重试与有效勾选/浏览位置保留；验证后台页面、确认框、并发操作和测量窗口不启动扫描，以及任务结束后恢复。
退出阶段还需验证已排队的 timer 和 worker 回调：Textual 停止应用后、控件部分卸载而 `on_unmount` 尚未执行时，不再扫描或访问页面控件。
页面切换、挂载和布局更新后等待框架处理事件，再判断点击和焦点，不用堆叠固定 `sleep` 掩盖竞态。
Textual 8.2.8 的 `Pilot.pause()` 可能在返回前的布局刷新中才排入 `Hide`；隐藏后的鼠标捕获断言需继续等待目标控件的 `wait_for_refresh()`，并设置超时，确保已排队的事件完成处理。
拖动中断的各个场景使用独立 `run_test()` 应用实例，避免某次断言失败遗留的页面或 busy 状态引发连带失败。
光标阶段与点击、编辑后恢复的测试只将光标闪烁 Timer 以 `pause=True` 创建，由测试显式推进阶段，
并等待输入框刷新完成后断言；重复点击同一位置也应恢复可见阶段。真实 0.5 秒闪烁间隔及无额外重绘
由独立的 idle-cursor 测试验证，避免较慢的 `Pilot` 操作跨过闪烁周期而误报失败。

Headless 能检查布局、键盘路径和输出状态；SVG、tmux 与真实 VS Code/SSH 终端是不同证据。
通知回归需显式使用 `run_test(notifications=True)`；默认测试模式不显示通知，无法发现浮层缺字。
隐藏快捷命令栏后让通知覆盖底部按钮，核对最终 compositor 输出和终端更新字节中的汉字完整性；
只断言通知原文或 Toast 自身的内容不能证明叠加后的显示正确。
颜色回归还需检查实际渲染器输出的 SGR 颜色序列：Textual 的 SVG 导出固定使用真彩色，
单看 SVG 无法发现终端输出被降级为 256 色的问题。终端颜色变更覆盖输入选中、下拉菜单、
确认按钮与深浅主题；PTY 输出仍不能代替用户客户端实际显示的验收。
原生光标、剪贴板、闪烁和宿主快捷键路由只能在相应终端确认。交付注明实际验证环境；
流程见 [TUI 回归 Skill](../.agents/skills/acprof-textual-regression/SKILL.md)，模块分工见[架构](Architecture.md#tui-与兼容维护)。

## 真实采集与实验隔离

`.github/workflows/hardware.yml` 只支持手动触发，在带 `acprof` 标签的专用 Linux x86_64 runner
上使用预先准备的 `.venv`、本机 Docker/cgroup v2、RAPL、perf 和抓包权限。不会由 PR 自动触发。
本地同一入口为：

```bash
.venv/bin/python scripts/check_hardware.py --model hf-internal-testing/tiny-random-bert \
  --task fill-mask --gpus off,on --output-dir internal-testing/hardware-smoke
```

默认每个设备 1 个 case、warmup=1、repeat=3、2 秒请求窗口及 2 秒 idle，属于短 smoke。
`command.json`、`run.log`、原始结果及 `audit.json` 一同保留；验收要求正式行为 `ok`、计划完整，
且两种延迟、CPU 能量、PMU instructions 和 GPU 模式下的 GPU 能量均为有效数值。
未通过不自动更改原参数或覆盖产物。`--compute-profile-tool` / `--execution-profile-tool`
可另测指定工具，工具字段仍需根据其计划和错误列验收，不能用主采集成功代替工具成功。

采样线程及 CLI/TUI 开销的独立对照入口和统计假设见[指标分析](Metrics.md#窗口置信区间与开销对照)。
`compare_ui.py --ui terminal` 继承当前终端，要求 stdout 为 TTY；自动化可用 `script` 分配 PTY
并保存会话。PTY、headless 和用户的 VS Code/SSH 终端须分别标明，不能互相替代。

镜像依赖变化后，主机 `.venv` 测试不能证明容器已更新；构建与复用契约见[运行兼容](Runtime_Compatibility.md#构建复用和验证)。
最小采集示例见[运行指南](Getting_Started.md#3-跑一个最小-smoke-test)。用独立输出目录运行验证，保留模型 revision、输入计划与日志。
`examples/` 下脚本是手动接口示例，不会自动运行，也不产生与正式 `run.py` 等价的测量证据。

`internal-testing/` 用于本地临时验证和截图；原始实验结果留在对应结果目录。
普通推理成功不能证明 Torch/NCU/Massif/Nsys 都支持；每种设备、dtype 和工具分别报告实际覆盖范围。

## 参考实现与复用取舍

表单边框复用 [Textual 8.2.8 的 `solid` 字符集](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/_border.py)，
并根据 [Select 上游说明](https://github.com/Textualize/textual/discussions/4061)覆盖 `SelectCurrent`
及[源码中的展开菜单 `SelectOverlay`](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_select.py)。
Textual 为 MIT 许可且由上游维护；这里只覆盖现有 TCSS，
保留控件尺寸和事件处理，不增加依赖、终端自动探测或测量期间的后台处理。

开发检查复用 [Ruff 官方 hook](https://github.com/astral-sh/ruff-pre-commit)
和 [pre-commit 官方基础 hooks](https://github.com/pre-commit/pre-commit-hooks)（MIT，持续维护，
所选版本支持 Python 3.10）。参考 [HTTPX 的工具配置](https://github.com/encode/httpx/blob/master/pyproject.toml)
把 Ruff 规则放在 `pyproject.toml`，保留本项目的 unittest 与 evidence runner。
hooks 固定完整 commit SHA，CI 直接执行同一份配置，避免维护第二份检查清单；只新增开发依赖，
没有采集期间的后台进程或测量开销。

结果原子发布采用 [CPython 的 tempfile](https://github.com/python/cpython/blob/main/Lib/tempfile.py)
和标准库文件同步、替换机制；实验身份参考 [ASV 的结果管理](https://github.com/airspeed-velocity/asv/blob/main/asv/results.py)。
两者的通用做法与现有 CSV/目录协议兼容，恢复仍按 AC-Prof 的 case 与测量窗口实现。
依赖解析复用持续维护的 [uv](https://github.com/astral-sh/uv)（MIT / Apache-2.0），只在更新锁时使用。
指标元数据参考 [Prometheus Python client](https://github.com/prometheus/client_python)（Apache-2.0）的类型与单位声明，
窗口区间参考 [SciPy bootstrap](https://github.com/scipy/scipy/blob/main/scipy/stats/_resampling.py)（BSD-3-Clause）的重采样方法。
本项目只需离线登记表和均值区间，使用标准库实现，无需在采集服务加入 exporter 或 SciPy 依赖。
profiler 调研了 [NVIDIA nsight-python](https://github.com/NVIDIA/nsight-python)（Apache-2.0）；其 kernel profiling 接口
不替代现有完整请求和旁路 probe 契约，因此保留 CLI/CSV 集成，提取纯解析与环境发现模块。
这些选择不增加正式测量窗口内的服务或网络调用，工具和环境验证均在采集前后进行。

主机分片沿用 [CPython unittest 的测试集与 fixture 机制](https://github.com/python/cpython/blob/3.12/Lib/unittest/suite.py)
和 [GitHub Actions matrix](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations)，
保留各分片内的模块、类排序，不引入并行测试插件。涉及 `sys.modules` 隔离的测试，先用
`importlib.import_module` 获取真实目标再 `patch.object`，避免 Python 3.10 的字符串 patch
沿父包属性找到已脱离导入缓存的模块；请求完成、异常和超时断言保持原语义。

统计报告页复用 [Textual 官方 DataTable](https://github.com/Textualize/textual/blob/main/docs/widgets/data_table.md)
及 [Worker API](https://github.com/Textualize/textual/blob/main/docs/guide/workers.md)（MIT，官方持续维护），
已在项目使用的 Textual 8.2.8 中验证；不增加表格库或统计依赖。
窗口统计调用既有 CLI，JSON 读取在后台执行；只在用户操作和任务完成时更新表格，采集期间禁止启动，
避免给正式窗口增加轮询或统计计算。回归覆盖三种终端尺寸、中英文切换、失败恢复、原 CSV 不变及测量互斥。

表头拖动复用 [Textual DataTable](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_data_table.py)
的列元数据、渲染与鼠标捕获。沿用官方维护的 MIT 依赖，无额外包、后台轮询或测量窗口内的诊断。
当前 8.2.8 没有公开的列宽 setter；兼容逻辑集中在 `tui/table.py`，调整列宽时清理渲染缓存并更新虚拟尺寸。
升级 Textual 时需运行拖动回归，覆盖中英文 cell 宽度、固定列与横向滚动、表头点击排序、释放后的 Click、
拖出表格、禁用/隐藏/清空时的鼠标释放，以及三种终端尺寸下的会话列宽保留和采集锁定。
末列右沿在完整显示、横向滚动及单列表格中均不显示手柄或捕获拖动，普通表头点击仍有效。
报告页覆盖数据更新、报告重建和语言切换；镜像树另检查父子行的大小、未知值和容器数量与表头左对齐、横向滚动同步和折叠状态保留。

## 文档与 Skill 检查

检查新增文件也包括被 Git 忽略的文件；`git diff --check` 只覆盖已跟踪差异，不能替代完整文件清单。
迁移章节时核对原有锚点、相对链接、代码中的文档引用及字段表是否有遗漏；代码示例中的路径以注明的执行目录为准。
技能格式可用已安装 `skill-creator` 的 `scripts/quick_validate.py <skill-dir>` 检查；该工具是开发辅助，不是项目运行依赖。

交付说明实际执行的命令、结果、跳过原因及未验证范围。只改文档时，不宣称完成真实 Docker/GPU 或用户终端验证。

# 测试与验证

选择本次改动能改变的行为和失败路径。下列命令均从仓库根目录执行，使用已有 `.venv`。
测试数量、设备余量和镜像可用性由本次执行确认，不把历史通过记录作为当前验证结果。

## 验证范围

| 改动 | 应取得的证据 |
| --- | --- |
| 文档、导航或链接迁移 | 本地文件与章节锚点可达、旧入口仍可跳转、代码块与差异格式正确；无需为措辞运行模型 |
| Skill | frontmatter、名称与描述匹配、相对路径、流程边界和可用验证器的格式检查 |
| 单个逻辑或失败路径 | 能观察目标行为的相关 unittest；修复缺陷时先复现，再验证修复 |
| 模块搬迁、依赖方向或兼容入口 | 相关测试、全套 unittest、CLI 帮助和编译；mock 放到函数实际查找依赖的模块 |
| 指标或产物协议 | 独立推导的期望值、新旧 schema、缺失/失败/不适用字段和受影响消费者 |
| 模型、backend、依赖或 Dockerfile | 路由与离线加载测试、镜像构建、所声明设备的真实推理；profiler 分别验证 |
| TUI | 受影响的交互与尺寸检查；原生终端问题还需对应终端证据 |

## 自动化验证入口

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
| 环境、依赖与镜像 | `tests/test_runtime_profiles.py`、`tests/test_image_layers.py`、`tests/test_runtime_validation.py` |
| 模型文件与离线加载 | `tests/test_model_files.py`、`tests/test_model_download.py`、`tests/test_offline_model_loading.py` |
| 字段、能耗、资源与补采 | `tests/test_energy_cpu.py`、`tests/test_resource_usage.py`、`tests/test_posthoc.py` |
| 页面、设置与焦点 | `tests/test_tui_layout_settings.py`、`tests/test_tui_interaction.py`、`tests/test_tui_input.py` |
| 日志、语言与命令 | `tests/test_tui_log_view.py`、`tests/test_tui_i18n.py`、`tests/test_tui.py` |
| 统计报告、异步读取与计算 | `tests/test_tui_reports.py`、`tests/test_report_views.py`、`tests/test_uncertainty.py` |
| Docker 镜像管理、标签删除与采集互斥 | `tests/test_image_management.py`、`tests/test_tui_images.py` |

以上是定位入口，不是每次必须运行的清单。先用 `rg --files tests` 查实际受影响的测试；新改动、失败或未解决问题才需要扩大或重复验证。

## CI 与环境测试

`.github/workflows/ci.yml` 在 Python 3.10 / 3.12 上安装哈希锁并执行主机回归；
七个任务族另构建 Python 3.10 CPU 镜像，以随机小模型或明确导出的样例验证真实加载与推理。
网络在容器测试期间关闭。CI Actions 固定为已核验的 commit SHA，作业只授予仓库读取权限。

本地入口：

```bash
.venv/bin/python scripts/run_tests.py --report internal-testing/host-tests.json
.venv/bin/python scripts/check_runtime.py --family audio --variant cpu --output-dir internal-testing/audio-runtime
.venv/bin/python scripts/render_metric_reference.py --check
```

`run_tests.py` 保留 unittest 输出，并将每项测试的结果、失败/跳过原因、版本及耗时写入 JSON。
空测试集必定失败；容器作业带 `--require-no-skips`，跳过或 expected failure 都不算环境验证通过。
普通主机测试允许缺少推理依赖时跳过，报告明确列出范围。`check_runtime.py` 的目录必须为空；
`runtime.json` 另记录锁定 profile、实际 image ID 和退出结果，不覆盖旧验证。

源代码、模型权重和依赖层保持分离，CPU 容器测试不下载 Hub 模型，不代替真实 GPU/PMU/抓包实验。

## TUI 与终端证据

使用 `unittest.IsolatedAsyncioTestCase`、Textual `run_test()` / `Pilot` 和临时 `settings_path`。
尺寸覆盖用户报告的场景，并按布局变更检查 `80×24`、`120×30`、`150×45` 及运行中 resize。
页面切换、挂载和布局更新后等待框架处理事件，再判断点击和焦点，不用堆叠固定 `sleep` 掩盖竞态。

Headless 能检查布局、键盘路径和输出状态；SVG、tmux 与真实 VS Code/SSH 终端是不同证据。
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
最小采集示例见 [README](../README.md#3-跑一个最小-smoke-test)。用独立输出目录运行验证，保留模型 revision、输入计划与日志。
`examples/` 下脚本是手动接口示例，不会自动运行，也不产生与正式 `run.py` 等价的测量证据。

`internal-testing/` 用于本地临时验证和截图；原始实验结果留在对应结果目录。
普通推理成功不能证明 Torch/NCU/Massif/Nsys 都支持；每种设备、dtype 和工具分别报告实际覆盖范围。

## 参考实现与复用取舍

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

统计报告页复用 [Textual 官方 DataTable](https://github.com/Textualize/textual/blob/main/docs/widgets/data_table.md)
及 [Worker API](https://github.com/Textualize/textual/blob/main/docs/guide/workers.md)（MIT，官方持续维护），
已在项目使用的 Textual 8.2.8 中验证；不增加表格库或统计依赖。
窗口统计调用既有 CLI，JSON 读取在后台执行；只在用户操作和任务完成时更新表格，采集期间禁止启动，
避免给正式窗口增加轮询或统计计算。回归覆盖三种终端尺寸、中英文切换、失败恢复、原 CSV 不变及测量互斥。

## 文档与 Skill 检查

检查新增文件也包括被 Git 忽略的文件；`git diff --check` 只覆盖已跟踪差异，不能替代完整文件清单。
迁移章节时核对原有锚点、相对链接、代码中的文档引用及字段表是否有遗漏；代码示例中的路径以注明的执行目录为准。
技能格式可用已安装 `skill-creator` 的 `scripts/quick_validate.py <skill-dir>` 检查；该工具是开发辅助，不是项目运行依赖。

交付说明实际执行的命令、结果、跳过原因及未验证范围。只改文档时，不宣称完成真实 Docker/GPU 或用户终端验证。

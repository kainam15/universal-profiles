# AC-Prof 实验与指标参考

完整内容已按主题迁入 [docs/](docs/README.md)。本文件仅保留旧章节锚点和跳转，
新增或修改知识请维护对应专题；安装与运行示例见 [README](README.md)。

| 原章节 | 详细文档 |
| --- | --- |
| <a id="输出文件"></a>输出文件 | [查看](docs/Profiling_Protocol.md#输出文件) |
| <a id="static_metajson-字段"></a>`static_meta.json` 字段 | [查看](docs/Profiling_Protocol.md#static_metajson-字段) |
| <a id="collection_historyjson-字段"></a>`collection_history.json` 字段 | [查看](docs/Profiling_Protocol.md#collection_historyjson-字段) |
| <a id="最大输入探测结果"></a>最大输入探测结果 | [查看](docs/Profiling_Protocol.md#最大输入探测结果) |
| <a id="图表与延迟拟合产物"></a>图表与延迟拟合产物 | [查看](docs/Metrics.md#图表与延迟拟合产物) |
| <a id="result_allcsv-字段解释"></a>result_all.csv 字段解释 | [查看](docs/Metrics.md#result_allcsv-字段解释) |
| <a id="资源配置输入与网络"></a>资源配置、输入与网络 | [查看](docs/Metrics.md#资源配置输入与网络) |
| <a id="延迟与吞吐"></a>延迟与吞吐 | [查看](docs/Metrics.md#延迟与吞吐) |
| <a id="torch-与-ncu-计算指标"></a>Torch 与 NCU 计算指标 | [查看](docs/Profilers.md#torch-与-ncu-计算指标) |
| <a id="massif-与-nsight-systems-执行指标"></a>Massif 与 Nsight Systems 执行指标 | [查看](docs/Profilers.md#massif-与-nsight-systems-执行指标) |
| <a id="gpu-功率与能耗"></a>GPU 功率与能耗 | [查看](docs/Energy_Measurement.md#gpu-功率与能耗) |
| <a id="cpu-package-功率与能耗"></a>CPU package 功率与能耗 | [查看](docs/Energy_Measurement.md#cpu-package-功率与能耗) |
| <a id="估算-vcpu-能耗与派生能效"></a>估算 vCPU 能耗与派生能效 | [查看](docs/Energy_Measurement.md#估算-vcpu-能耗与派生能效) |
| <a id="像素归一化口径"></a>像素归一化口径 | [查看](docs/Metrics.md#像素归一化口径) |
| <a id="cpu-资源频率与-pmu"></a>CPU 资源、频率与 PMU | [查看](docs/Metrics.md#cpu-资源频率与-pmu) |
| <a id="容器内存swapio-与-pid"></a>容器内存、swap、I/O 与 PID | [查看](docs/Metrics.md#容器内存swapio-与-pid) |
| <a id="gpu-资源与运行状态"></a>GPU 资源与运行状态 | [查看](docs/Metrics.md#gpu-资源与运行状态) |
| <a id="冷启动"></a>冷启动 | [查看](docs/Profiling_Protocol.md#冷启动) |
| <a id="运行状态与错误"></a>运行状态与错误 | [查看](docs/Metrics.md#运行状态与错误) |
| <a id="latency_s-和-latency_app_s-的区别"></a>`latency_s` 和 `latency_app_s` 的区别 | [查看](docs/Metrics.md#latency_s-和-latency_app_s-的区别) |
| <a id="结果行数和时间成本估算"></a>结果行数和时间成本估算 | [查看](docs/Profiling_Protocol.md#结果行数和时间成本估算) |
| <a id="csv-行数与请求数"></a>CSV 行数与请求数 | [查看](docs/Profiling_Protocol.md#csv-行数与请求数) |
| <a id="每行测量窗口"></a>每行测量窗口 | [查看](docs/Profiling_Protocol.md#每行测量窗口) |
| <a id="整条命令的耗时"></a>整条命令的耗时 | [查看](docs/Profiling_Protocol.md#整条命令的耗时) |
| <a id="可选分析器成本"></a>可选分析器成本 | [查看](docs/Profilers.md#可选分析器成本) |
| <a id="常见判断"></a>常见判断 | [查看](docs/Troubleshooting.md#常见判断) |
| <a id="先区分预期空值与失败"></a>先区分预期空值与失败 | [查看](docs/Troubleshooting.md#先区分预期空值与失败) |
| <a id="任务尚未适配的预检退出"></a>任务尚未适配的预检退出 | [查看](docs/Troubleshooting.md#任务尚未适配的预检退出) |
| <a id="图像描述输出与兼容范围"></a>图像描述输出与兼容范围 | [查看](docs/Runtime_Compatibility.md#图像描述输出与兼容范围) |
| <a id="启动-oom-与剪枝占位"></a>启动 OOM 与剪枝占位 | [查看](docs/Troubleshooting.md#启动-oom-与剪枝占位) |
| <a id="运行期-oom"></a>运行期 OOM | [查看](docs/Troubleshooting.md#运行期-oom) |
| <a id="gpu-energy-字段全是-nan"></a>GPU energy 字段全是 `nan` | [查看](docs/Energy_Measurement.md#gpu-energy-字段全是-nan) |
| <a id="cpu--vcpu-energy-字段全是-nan"></a>CPU / vCPU energy 字段全是 `nan` | [查看](docs/Energy_Measurement.md#cpu--vcpu-energy-字段全是-nan) |
| <a id="cpu-idle-baseline-波动-warning"></a>CPU idle baseline 波动 warning | [查看](docs/Energy_Measurement.md#cpu-idle-baseline-波动-warning) |
| <a id="gpu-idle-baseline-波动-warning"></a>GPU idle baseline 波动 warning | [查看](docs/Energy_Measurement.md#gpu-idle-baseline-波动-warning) |
| <a id="资源占用率字段全是-nan"></a>资源占用率字段全是 `nan` | [查看](docs/Troubleshooting.md#资源占用率字段全是-nan) |
| <a id="mipscache-miss-与-dtlb-miss"></a>MIPS、cache miss 与 dTLB miss | [查看](docs/Troubleshooting.md#mipscache-miss-与-dtlb-miss) |
| <a id="cpu--vcpu-peak-power-看起来异常"></a>CPU / vCPU peak power 看起来异常 | [查看](docs/Energy_Measurement.md#cpu--vcpu-peak-power-看起来异常) |
| <a id="mflops--compute-profiling-字段全是-nan"></a>MFLOPS / compute profiling 字段全是 `nan` | [查看](docs/Profilers.md#mflops--compute-profiling-字段全是-nan) |
| <a id="massif--nsight-systems-execution-profiling-字段全是-nan"></a>Massif / Nsight Systems execution profiling 字段全是 `nan` | [查看](docs/Profilers.md#massif--nsight-systems-execution-profiling-字段全是-nan) |
| <a id="tui-本地设置"></a>TUI 本地设置 | [查看](docs/CLI_Reference.md#tui-本地设置) |
| <a id="cli-参数"></a>CLI 参数 | [查看](docs/CLI_Reference.md#cli-参数) |
| <a id="runpy"></a>`run.py` | [查看](docs/CLI_Reference.md#runpy) |
| <a id="模型与资源矩阵"></a>模型与资源矩阵 | [查看](docs/CLI_Reference.md#模型与资源矩阵) |
| <a id="请求窗口与采样"></a>请求窗口与采样 | [查看](docs/CLI_Reference.md#请求窗口与采样) |
| <a id="输入"></a>输入 | [查看](docs/CLI_Reference.md#输入) |
| <a id="计算分析器"></a>计算分析器 | [查看](docs/CLI_Reference.md#计算分析器) |
| <a id="执行分析器"></a>执行分析器 | [查看](docs/CLI_Reference.md#执行分析器) |
| <a id="输出与运行环境"></a>输出与运行环境 | [查看](docs/CLI_Reference.md#输出与运行环境) |
| <a id="输入规模与音频清单"></a>输入规模与音频清单 | [查看](docs/CLI_Reference.md#输入规模与音频清单) |
| <a id="真实音频-workload"></a>真实音频 workload | [查看](docs/CLI_Reference.md#真实音频-workload) |
| <a id="probepy"></a>`probe.py` | [查看](docs/CLI_Reference.md#probepy) |
| <a id="profilepy"></a>`profile.py` | [查看](docs/CLI_Reference.md#profilepy) |
| <a id="其他入口"></a>其他入口 | [查看](docs/CLI_Reference.md#其他入口) |

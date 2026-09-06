# AC-Prof 实验与指标参考

本文用于查阅实验产物、CSV 字段、时间预算、结果判断和命令行参数。
安装、运行、TUI、通知与补采操作见 [README](README.md)。

- [输出文件](#输出文件)
- [result_all.csv 字段解释](#result_allcsv-字段解释)
- [结果行数和时间成本估算](#结果行数和时间成本估算)
- [常见判断](#常见判断)
- [CLI 参数](#cli-参数)

## 输出文件

输出目录为 `<output-dir>/<model-dir>/`；模型 ID 中的 `/` 替换为 `--`。
下表列出可能生成的文件；probe、补采、调试与绘图产物仅在执行对应操作时出现。

| 文件 | 说明 |
| --- | --- |
| `result_case_*.csv` | 采集期间逐资源配置写入的可恢复中间结果；成功合并后清理。 |
| `result_all.csv` | 动态测量结果。每一行对应一个 resource config、一个 input scale、一次 warmup/repeat iteration，并记录归一化指标、PCAP 网络字节、cold-start phases，以及该窗口的 cgroup memory/stat/PID、swap、块 I/O 与压力/事件。 |
| `static_meta.json` | 单个 JSON object 的静态元数据。记录模型版本、参数/精度/量化/许可证、输入输出格式、per-scale 静态逻辑 FLOPs、推理后端、镜像、GPU/主机 RAM、主机 swap、Docker 存储和环境信息。 |
| `collection_history.json` | schema v1 的采集/修复 provenance。分别记录 post-hoc profiler 补采、timeout retry、quality retry 和静态元数据回填历史；最新一次状态由对应 history 的最后一项得到。 |
| `input_scale_plan.json` | 所有任务族共用的 input scale/payload 计划。schema v2 额外记录 workload provenance、per-scale 输入元数据和模型约束；读取端继续兼容无版本字段的 v1 计划。主采集和 compute profiler 复用同一份 payload。 |
| `startup_oom_pruning.json` | 仅启用 `--prune-startup-oom` 时生成。记录最低参考 CPU、执行顺序、逐 GPU mode 的实测启动 OOM 前缀、最低启动可行内存、推断跳过 case、排除范围与资源单调性假设。 |
| `compute_profile_plan.json` | per-scale FLOP profiling 结果。每个 CPU/GPU scale 可同时记录独立的 `torch_profiler_eager` 与 `ncu` profile；NCU 只存在于 GPU profile。失败信息按工具保存，只读取当前按 profiler 分层的 plan 结构。 |
| `execution_profile_plan.json` | 显式 execution profiling 的采样与 per-resource-config/per-scale 汇总。Massif 条目对应 `gpu_mode=off`，Nsight Systems 条目对应 `gpu_mode=on`；复用 entry 记录实际 source resource 与 sampling strategy，失败按工具记录且不阻断主实验。 |
| `compute_profiles/` | 默认保留的原始 compute profiler artifacts；`--discard-compute-profiles` 可在汇总后删除。 |
| `posthoc_profiles/` | `profile.py` 生成的补采 plan、原始报告与可恢复 checkpoint。 |
| `posthoc_backups/<timestamp>/` | 成功补采替换文件前保留的原始 CSV、静态元数据与已有历史记录备份。 |
| `probes/largest_scale_<timestamp>_<pid>/` | `probe.py` 的独立输入计划与 `largest_scale_probe.json`，不含正式 CSV。 |
| `execution_profiles/` | 默认保留 raw Massif `.out` 与 Nsight Systems `.nsys-rep`；stats 导出的 `.sqlite` 缓存会自动删除。传入 `--discard-execution-profiles` 时 raw artifacts 也会在汇总后删除。 |
| `tmux_all.log` | 在 tmux pane 内运行 `run.py` 时自动记录的完整终端显示。实验正常结束或报错退出时落盘，不受 tmux 历史行数上限影响。 |
| `latency_model/latency_model_report.json` | `plot.py` 生成的 latency 拟合报告。包含分 CPU/GPU 的正值模型、整配置留一与最大尺度外推指标、质量门槛、系数和训练范围。 |
| `latency_model/latency_model_residuals.csv` | `plot.py` 生成的 case-level residual。每个 `GPU mode × CPU × memory × input scale` 聚合 case 一行，包含重复数/离散度、full-fit、resource-config OOF 和最大尺度 holdout 预测。 |
| `latency_model/latency_model_fit_curves.png` | `plot.py` 生成的 full-fit 曲线图。横轴为 input scale，CPU-off 与 GPU-on 分面展示，每个 `CPU × memory` 资源配置一条拟合曲线，并叠加实测 case 中位数。 |
| `latency_model/latency_model_residuals.png` | `plot.py` 在 residual CSV 有有效数据时生成的模型诊断图，包含 OOF 实际值/预测值、相对残差分布及残差随预测延迟和输入尺度的变化。 |
| `debug_idle_diag/result_case_*.csv.idle_diag.jsonl` | 仅 `--idle-debug` 时生成。每行对应一个 workload window 的 idle 诊断记录，包含 GPU NVML idle power trace、`nvidia-smi` GPU/process 快照、CPU idle window 内 RAPL 子窗口功率、host/container CPU delta、top proc CPU delta，以及 after-idle 快照，用于定位 `gpu_idle_power_w` / `cpu_idle_power_w` case 内波动来源。 |
| `cpu/*.png` | `plot.py` 生成的 CPU-only 图表。 |
| `gpu/*.png` | `plot.py` 生成的 GPU-only 图表。 |
| `gpu+cpu/*.png` | `plot.py` 生成的 GPU/CPU 对比图表。 |

中间文件 `result_case_*.csv`、`result_case_*.csv.sniff_groups.jsonl`、`lat_case_*.json`、`sniff_case_*.pcap` 会在 `result_all.csv` 成功 merge 后自动清理。若运行被中断，这些中间文件可能保留。

### `static_meta.json` 字段

`static_meta.json` 是一个 model/image/run-level JSON object，只保存描述实验对象、环境和最终 profiling 配置/口径的相对稳定信息。补采、重试、回填与备份过程记录放在独立的 `collection_history.json`。数组、布尔值、数字和 `null` 均保留 JSON 原生类型，不再编码成 CSV 字符串。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | `static_meta.json` schema 版本；新增 cgroup 版本与采集模式后的当前版本为 `6`。 |
| `model_name` | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `model_revision` | 实际解析到的 model revision / commit hash。 |
| `parameter_count` | Hugging Face Hub SafeTensors metadata 的参数总数；Hub 未提供时为 `null`。 |
| `parameter_bytes` | 根据 `parameter_dtype_counts` 的各 dtype 元素数量与字节宽度精确求和得到的逻辑 tensor payload 大小，不含序列化 header；没有 dtype 统计或存在未知 dtype 时为 `null`。 |
| `precision_dtype` | SafeTensors 参数中数量占主导的权重精度，例如 `FP32`、`FP16`、`BF16`、`INT8`；无法确认时为 `null`。 |
| `parameter_dtype_counts` | 按 dtype 统计的参数/张量元素数量，保留混合精度与少量整型 buffer 信息。 |
| `inference_precision_by_device` | 当前 handler 明确请求的 CPU/GPU 推理精度。Transformers NLP/CV/audio handler 当前为 `{"cpu":"FP32","gpu":"FP16"}`。 |
| `static_flops` | Torch eager profiler 得到的逻辑 shape FLOPs，按 `input_scale` 保存 `flops_per_request`；未采集成功时为 `null`。 |
| `static_macs` | 静态 MACs。当前不做不可靠的 FLOPs/2 推断，因此未单独采集时为 `null`。 |
| `input_format` | 实际 `/predict` HTTP JSON 输入协议及其 JSON Schema。 |
| `output_format` | 实际 `/predict` HTTP JSON 响应协议及其 JSON Schema。 |
| `quantized` | 是否检测到量化配置、量化 tag 或量化权重 dtype；无法确认时为 `null`。 |
| `quantization_method` | 量化方法，例如 `gptq`、`awq`；不适用或未知时为 `null`。 |
| `quantization_config` | Hub model config 中的完整量化配置；没有时为空 object。 |
| `model_license` | Hugging Face model card 许可证，例如 `apache-2.0`、`mit`；无法确认时为 `null`。 |
| `model_metadata_source` | 参数量、参数 payload、精度、量化和许可证的元数据来源，当前在线 Hub 检测成功时为 `huggingface_hub`。 |
| `task_family` | 任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`。 |
| `pipeline_tag` | Hugging Face pipeline tag，例如 `fill-mask`、`image-classification`。 |
| `runtime_backend` | 容器内使用的 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`。 |
| `image_tag` | 本次使用的 Docker image tag。 |
| `batch_size` | 本次 profiling 的 batch size。 |
| `input_scale_type` | `result_all.csv/input_scale` 的语义名，例如 `seq_length`。 |
| `workload` | workload 清单的可复现元数据，包括素材 SHA256、来源、变换、推理模式以及模型侧输入约束。 |
| `input_scale_plan_sha256` | 本次实际执行的 `input_scale_plan.json` SHA256。 |
| `run_command` | 启动本次 profiling 的 `python run.py ...` 命令，便于复现实验参数。 |
| `model_download_url` | Hugging Face model page URL。 |
| `gpu` | host device 0 的 GPU 名称；没有可见 NVIDIA GPU 时为 `unknown`。 |
| `gpu_mem_total_bytes` | host device 0 的 total VRAM，单位 bytes；无法读取时为 `null`。 |
| `host_mem_total_bytes` | Host 物理 RAM 总量，单位 bytes；无法读取时为 `null`。 |
| `host_swap_total_bytes` | 实验启动时 host 已启用 swap 的总容量，单位 bytes；无法读取时为 `null`，未启用时为 `0`。 |
| `host_swap_used_bytes_at_start` | 静态元数据采集时 host 已使用的 swap 快照，单位 bytes；无法读取时为 `null`。 |
| `host_swap_type` | `/proc/swaps` 中 active swap 的 backing 类型：`none`、`file`、`partition`、`zram`、`mixed` 或无法识别时的 `unknown`。 |
| `host_vm_swappiness` | 实验启动时 `/proc/sys/vm/swappiness` 的整数值；无法读取时为 `null`。 |
| `model_cache_bytes` | Docker image 内 `/models/hf` 下唯一普通文件的逻辑字节数总和；跳过符号链接并按 inode 去重。包括缓存中的全部权重格式、配置、tokenizer 等 artifacts，不代表单一权重文件大小、文件系统实际占用块或 Docker 下载体积。 |
| `docker_image_bytes` | `docker image inspect <image_tag> --format "{{.Size}}"` 返回的本地 image size，单位 bytes。 |
| `docker_storage_total_bytes` | Docker daemon `DockerRootDir` 所在文件系统的总容量，单位 bytes；无法访问 daemon 路径时为 `null`。 |
| `docker_storage_available_bytes_at_start` | 静态元数据采集时 `DockerRootDir` 所在文件系统对当前用户可用的容量快照，单位 bytes；该值会随磁盘使用变化。 |
| `docker_storage_filesystem` | `DockerRootDir` 所在文件系统类型，例如 `ext4`；无法识别时为 `unknown`。 |
| `docker_storage_device` | 承载 `DockerRootDir` 的 mount source，例如 `/dev/nvme0n1p2`；无法识别时为 `unknown`。 |
| `docker_storage_type` | 根据 `lsblk` transport/rotational 信息得到的 `nvme_ssd`、`ssd`、`hdd` 或内存文件系统 `memory`；证据不足时为 `unknown`。 |
| `environment` | 自动检测的运行环境标签，例如 `ubuntu24.04`；历史文件也可能包含 WSL/macOS 标签，当前正式采集会拒绝这些环境。 |
| `cgroup_version` | 本次 preflight 实际检测到的 hierarchy：正式数据应为 `v2`；显式兼容旧环境时可为 `v1`。 |
| `cgroup_collection_mode` | `strict_v2` 表示默认正式采集策略；`legacy_compatible` 表示用户显式启用了 `--allow-cgroup-v1`。分析正式数据集时应同时要求 `cgroup_version=v2` 和 `cgroup_collection_mode=strict_v2`。 |
| `cpu_power_source` | CPU package 功耗来源。`rapl` 表示使用 Linux RAPL powercap 真实计数器；`unavailable` 表示当前环境没有可用 RAPL。 |
| `vcpu_power_method` | estimated vCPU 功耗计算方法。`rapl_cgroup_cpu_share` 表示用 RAPL package energy 乘以 container cgroup CPU share；`unavailable` 表示无法估算。 |
| `cpu_governor` | Host CPU frequency governor 汇总值，例如 `performance`、`powersave`、`schedutil`；如果各 CPU policy 不一致，会写成 `mixed:<governor>=<count>,...`；无法读取时为 `unavailable`。 |
| `cpu_boost` | Host CPU boost / turbo 状态。`on` 表示 boost 可用，`off` 表示关闭；无法读取时为 `unavailable`。 |
| `compute_profile_tools` | 本次启用/实际记录的 profiler 列表，例如 `["torch_profiler_eager","ncu"]`。 |
| `torch_profiler_eager_flop_semantics` | Torch eager FLOP 的统计口径说明。 |
| `torch_profiler_eager_attention_implementation` | 独立 Torch probe 强制并验证的 attention 实现，当前为 `eager`。 |
| `torch_profiler_eager_repeat_cpu` | CPU Torch eager probe 的 repeat；未采 CPU profile 时为 `null`。 |
| `torch_profiler_eager_repeat_gpu` | GPU Torch eager probe 的 repeat；未采 GPU profile 时为 `null`。 |
| `ncu_flop_semantics` | NCU GPU 实际执行 FLOP 的计数器、Tensor/Scalar 分类口径说明。 |
| `ncu_repeat` | NCU probe repeat；所有 NCU per-request 指标据此归一化。 |
| `ncu_fma_flop_weight` | NCU FMA 指令的 FLOP 权重，当前为 `2`。 |
| `ncu_metrics` | 本次 NCU 实际请求/解析的 metric 列表。 |
| `torch_version` | Torch probe 使用的 PyTorch 版本。 |
| `transformers_version` | Torch probe 使用的 Transformers 版本。 |
| `ncu_version` | Host NCU 版本；历史回填无法可靠确认时为 `unknown`。 |
| `gpu_compute_capability` | profile 使用 GPU 的 compute capability。 |
| `gpu_sm_count` | profile 使用 GPU 的 SM 数。 |
| `compute_profiles_retained` | raw profiler artifact 是否保留。 |
| `compute_profile_provenance` | profile 来源，例如本次直接采集或历史 `posthoc_backfill`。 |
| `execution_profile_schema_version` | execution profile plan schema 版本。 |
| `execution_profile_tools` | 本次显式启用且适用于所选 GPU modes 的 execution profiler 列表，例如 `["massif","nsys"]`；工具缺失时仍列出，并通过对应 error 字段诊断；默认关闭时为空列表。 |
| `massif_peak_semantics` | Massif peak 的 process-lifetime 口径，明确包含模型加载与预热。 |
| `massif_repeat` | 每个 Massif probe 内的 inference repeat；peak bytes 不按 repeat 归一化。 |
| `massif_version` | 实际执行分析的 container image 中的 Valgrind/Massif 版本；镜像可以是共享运行依赖的模型镜像或旧模型兼容镜像，未启用或无法确认时为 `unknown`。 |
| `massif_sampling_strategy` | `representative_per_scale` 或 `full_resource_matrix`。 |
| `massif_reference_cpu_cores` / `massif_reference_mem_cap_gb` | 缩减采样实际使用的代表资源；完整矩阵时为 `null`。 |
| `massif_reused_across_resource_cases` | Massif entry 是否从代表资源复用到其他结果行。 |
| `nsys_timeline_semantics` | Nsight Systems timeline 的 NVTX range 与汇总口径。 |
| `nsys_repeat` | 每个 Nsight Systems NVTX range 内的 inference repeat；动态汇总据此归一化到单 request。 |
| `nsys_version` | Host Nsight Systems 版本；未启用或无法确认时为 `unknown`。 |
| `nsys_sampling_strategy` | `representative_per_cpu_scale`、`representative_per_scale` 或 `full_resource_matrix`。 |
| `nsys_reference_cpu_cores` / `nsys_reference_mem_cap_gb` | Nsys 缩减采样使用的代表资源；per-CPU 策略的代表 CPU 为 `null`。 |
| `nsys_reused_across_resource_cases` | Nsys entry 是否从采样资源复用到其他结果行。 |
| `execution_profiles_retained` | raw Massif / Nsight Systems artifacts 是否保留。 |
| `execution_profile_provenance` | execution profile 的来源；默认关闭时为 `disabled`。 |

`schema_version=3` 的历史文件使用 `model_weight_bytes` 表示上述完整 cache artifacts 大小；v4 以 `model_cache_bytes` 替代该旧字段；v5 新增 host swap 字段；v6 新增 cgroup 版本与采集模式。历史文件不会自动伪造当时的 swap 或 cgroup 环境；无法回溯的值应保持 `null`，补录时在 `collection_history.json` 记录来源。

这些字段在 profiling 后原子补写，原始 `run_command` 保持不变。`static_flops` 只保存不依赖硬件计数器的 Torch 逻辑 shape FLOPs，并按 input scale 展开；NCU 实际执行 FLOPs、吞吐率以及 execution 数值仍保存在 `result_all.csv`，execution 字段是否来自代表资源由上述 sampling metadata 和 plan entry provenance 说明。

### `collection_history.json` 字段

| 字段 | 含义 |
| --- | --- |
| `schema_version` | `collection_history.json` schema 版本，当前为 `1`。 |
| `posthoc_profile_history` | `profile.py` 事后补采记录，包括工具、采样策略、完成时间与备份位置。 |
| `timeout_retry_history` | 请求超时后的重采/合并记录。当前仓库没有自动生成该记录的入口，但会迁移和保留已有数据。 |
| `quality_retry_history` | 质量检查后的定向重采/合并记录。当前仓库没有自动生成该记录的入口，但会迁移和保留已有数据。 |
| `static_meta_backfill_history` | 对历史结果补充静态元数据时的来源、字段、备份位置及无法回溯的字段。 |

不再重复保存 `posthoc_profile_last_run`、`timeout_retry_last_run` 或 `quality_retry_last_run`；需要最新记录时读取对应 `*_history[-1]`。旧版 `static_meta.json` 中已有的六个 history/last-run 字段，会在首次成功 post-hoc 更新时无损迁移并去重。

### 最大输入探测结果

`probe.py` 每次生成独立的 `input_scale_plan.json` 和 `largest_scale_probe.json`。
后者当前为 schema v3，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `memory_probe.candidate_order_gb` | 去重、升序排列的候选内存上限。 |
| `memory_probe.attempts` | 每档的 `startup_oom`、`runtime_oom`、`cuda_oom`、`timeout`、`error` 或 `ok` 结果，以及错误与分段计时。 |
| `memory_probe.minimum_viable_mem_gb` | 完整返回且有效尺度与计划一致的第一个候选；没有成功档时为空。 |
| `timing.request_s` | 成功档 `/predict` 的 host 端到端耗时。 |
| `cold_start.total_s` | 成功档全新容器启动到 ready 的耗时。 |
| `timing.ready_plus_request_s` | 上述冷启动与请求耗时之和。 |
| `timing.command_s` | 包含预检、检测、可选构建、输入规划、失败候选和清理的整条命令耗时。 |
| `timing.request_timeout_s` | 默认无限等待时为 `null`；显式设置 `--timeout-seconds` 时为对应秒数。旧 schema v2 始终记录有限值。 |

探测不会写入或修改 `result_case_*.csv`、`result_all.csv`、`static_meta.json` 或
`collection_history.json`，结果不包含 idle、能耗或网络测量。用法见 [README](README.md#先探测最大输入)。

### 图表与延迟拟合产物

`plot.py` 默认读取同目录下的 `static_meta.json`，用其中的 `input_scale_type` 作为横轴语义名；读取历史结果时仍兼容旧的 `static_meta.csv`。图片会写入结果目录下的三个子目录：

- `cpu/`：只使用 `gpu_mode=off` 的 CPU 数据
- `gpu/`：只使用 `gpu_mode=on` 的 GPU 数据
- `gpu+cpu/`：同时包含 GPU 和 CPU 数据，用于对比

每个有对应数据的目录按可用指标生成总览图、能耗图和专项分析图；Massif 采用独立图表，以保留其进程生命周期内存口径：

- `latency_overview_vs_scale.png`
- `service_efficiency_overview_vs_scale.png`
- `packet_overview_vs_scale.png`
- `torch_compute_overview_vs_scale.png`
- `ncu_arithmetic_overview_vs_scale.png`
- `ncu_runtime_overview_vs_scale.png`
- `nsys_timing_overview_vs_scale.png`
- `container_cpu_overview_vs_scale.png`
- `cpu_execution_overview_vs_scale.png`
- `cpu_memory_behavior_overview_vs_scale.png`
- `container_memory_process_overview_vs_scale.png`
- `container_io_overview_vs_scale.png`
- `gpu_resource_overview_vs_scale.png`
- `gpu_energy_power_overview_vs_scale.png`
- `cpu_package_energy_power_overview_vs_scale.png`
- `vcpu_estimated_energy_power_overview_vs_scale.png`
- `massif_cpu_heap_peak_total_vs_scale.png`
- `cold_start_bar.png`
- `resource_feasibility_heatmap.png`
- `tail_latency_overview_vs_scale.png`
- `latency_energy_pareto.png`
- `cold_start_breakdown.png`

13 张通用总览图统一使用 input scale 横轴、配置颜色和共享图例；每张最多 6 个子图。只有单位、语义和数值尺度都适合直接比较的子图才共享纵轴，例如 packet/application latency、NCU application/packet MFLOPS 以及 cache/dTLB 的同类比率。cache miss 和 dTLB miss 的单 request 计数可能相差多个数量级，因此使用独立纵轴。某个指标没有数据时，对应位置显示 `No data`；整张总览图的全部指标都没有数据时才跳过该 PNG。延迟总览使用 `3×2` 布局，同时展示 packet/application 的原始延迟、每 input unit 延迟和 CV；service efficiency 总览集中展示吞吐、每 CPU core 吞吐、container-attributed energy、每 input unit 能耗和 samples/J。

三张 energy/power 总览图分别对应 GPU board、CPU package 和 estimated vCPU-attributed 口径。每张 PNG 使用 `3×2` 子图：三行依次为 energy/request、average power 和 peak power，左列展示扣除 idle baseline 的 effective 指标，右列展示保留 idle baseline 的 total 指标；同行共享纵轴，所有子图共享 input-scale 横轴、配置颜色与图例。图例下方的灰色信息框同时给出 idle 平均功率和最大的 case 内相对极差 `(max-min)/mean`；CPU package 按 CPU-only / GPU-enabled 分开，estimated vCPU 则标明由 CPU package baseline 按 interval CPU share 归因，避免把估算值误解为独立实测。这样可以直接观察 idle 对各指标的影响，不再单独生成 effective 或 total PNG。CPU package 来自 RAPL，estimated vCPU 则按 container cgroup CPU share 估算，两者不能混作同一测量口径。

`cpu_memory_behavior_overview_vs_scale.png` 使用 `2×2` 布局汇总 Linux `perf` generic PMU event，只表示 cache / dTLB miss 的单 request 计数和 miss rate，用于观察访存局部性及地址转换开销；它们不是 DRAM read/write traffic，也不是实际内存带宽 GB/s。不同 CPU 架构、虚拟化环境或 kernel PMU 可能不提供相同事件；部分字段不可用时仍会保留其余可用子图。

Massif 图使用 `cpu_heap_peak_total_bytes_massif / 1024^3` 得到绘图期派生列 `cpu_heap_peak_total_gib_massif`；不会改写原始 CSV。它衡量进程生命周期内的 heap peak，与 container cgroup 测量窗口指标口径不同，因此保留为独立 PNG。`nsys_timing_overview_vs_scale.png` 使用 `2×2` 布局展示 host wall、CUDA API sum、GPU kernel sum 和 GPU memcpy sum。未启用对应 probe、字段不存在或整组字段均为 `nan` 时，这些可选图表会自动跳过。

四张论文分析图使用独立口径：

- `resource_feasibility_heatmap.png` 是唯一保留 `status=warn/error` 行的图。每个 `GPU mode × CPU × memory × input scale` 单元格把正式测量重复折叠为 `OK`、warning、partial failure、timeout、startup/runtime OOM、timeout 后跳过、其他错误或未知。只要同一单元格同时出现成功与失败，就标记为 partial failure，不会被成功行掩盖。
- `tail_latency_overview_vs_scale.png` 为每个 GPU mode 和 CPU 数选择有成功数据的最大 memory cap，先对 repeat window 的 P50/P90/P95 取中位数，再绘制 P50–P95 区间和 `P95/P50`。它同时展示 packet/application 口径，但不会用窗口分位数伪造 request-level violin distribution。
- `latency_energy_pareto.png` 按 input scale 分面并在 log-log 坐标中标出同时最小化 latency 与 container-attributed effective energy 的非支配前沿。延迟列依次优先使用 application P95、packet P95、application mean、packet mean；同一面板不会混合不同 input scale。历史 CSV 没有 `container_attributed_energy_eff_j` 时，只在 source fields 可用的行按现有口径重建：CPU-only 使用 estimated vCPU effective energy，GPU 行使用 estimated vCPU 与 GPU effective energy 之和。
- `cold_start_breakdown.png` 仅在五个阶段字段完整时生成，并为每个 GPU mode/CPU 数选择最大 memory cap。同一张 PNG 使用上下两个子图，共享配置横轴、独立缩放纵轴：上图堆叠 container launch、server setup、CUDA init、model load 和 ready wait，并用 `cold_start_s` 独立标记核对阶段和；下图单独展示 first-predict application latency，不计入 `/ready` 前的堆叠总量，缺少该指标时显示 `No data`。旧结果缺少阶段列时继续保留 `cold_start_bar.png`，并自动跳过分解图。

延迟建模产物统一写入结果目录下的 `latency_model/`：

- `latency_model/latency_model_report.json`
- `latency_model/latency_model_residuals.csv`
- `latency_model/latency_model_fit_curves.png`
- `latency_model/latency_model_residuals.png`

建模前会先按 `GPU mode × CPU × memory × input scale` 对正式测量重复取中位数，确保同一个 case 的重复不会被拆到训练与测试两侧。CPU-off 使用对数空间二次响应面；GPU-on 使用连续分段 log-linear 主模型，并为不稳定的上边界配置连续 affine latency tail：

```text
CPU: latency_s = exp(
  intercept + log(input_scale) + log(input_scale)²
  + log(cpu_cores) + log(cpu_cores)² + log(mem_cap_gb)
  + log(input_scale) × log(cpu_cores)
  + log(input_scale) × log(mem_cap_gb)
  + log(cpu_cores) × log(mem_cap_gb)
)

GPU within training range: latency_s = exp(
  intercept + log(input_scale)
  + Σ hinge_k × max(0, log(input_scale) - log(k))
  + 1/cpu_cores + 1/cpu_cores²
  + log(mem_cap_gb) + log(input_scale) × 1/cpu_cores
  + log(input_scale) × 1/cpu_cores²
)

GPU activated upper tail:
  latency_s(x) = spline_latency_s(x_max)
    + affine_tail(x) - affine_tail(x_max)
```

CPU 模型除共同的二次 log input-scale 项外，还使用二次 log CPU 项及 input-scale/CPU/memory 两两交互，以表达 CPU 饱和、memory cap 影响随规模变化等非线性资源响应。GPU 延迟可能因 kernel、attention 实现或内存执行区间切换而在相邻输入规模间改变斜率，因此 GPU 模型把每个内部实测 input scale `k` 作为共享的 log-space 线性样条结点；结点之间连续插值。若嵌套的一步前向检查 MAPE 超过 5%、训练尺度跨度至少为 10 倍，且 affine tail 在全部训练资源配置上的斜率为正，则上边界外推改用与样条边界连续的 latency-space affine tail，避免把单个不稳定末段斜率无限延长。否则继续使用 log-log 样条边界段。GPU 模型仍使用一阶和二阶逆 CPU 特征，以表达主机侧开销随 CPU 增加快速下降、随后进入 GPU 主导平台区的形状。所有输出都经过正值路径或回退到指数链接；CPU/GPU 独立系数等价于在联合模型中加入 GPU 相关交互。报告分别执行两种组外验证：

- resource configuration holdout：逐次完整留出一个 `(cpu_cores, mem_cap_gb)` 配置及其全部 input scale；
- input scale holdout：仅使用较小尺度训练，完整留出最大 input scale，检验向前外推；至少需要 3 个尺度，保证留出最大值后训练侧仍有 2 个不同尺度。

报告逐 CPU/GPU 模型给出 R²、MAE、RMSE、relative MAE、MAPE、SMAPE、非正预测数、系数、数值秩和训练范围。resource configuration holdout 要求 `R² >= 0.80`；两种验证都要求总体 `relative MAE <= 0.20`、总体 `MAPE <= 0.20`、任一留出资源配置的 `relative MAE` 与 `MAPE <= 0.30`、任一验证 case 的相对误差 `<= 0.30`，且预测有限为正。input scale holdout 的测试行全部处于同一个尺度，其 R² 只衡量该固定尺度内很小的资源配置差异，不能衡量尺度水位外推是否准确，因此仍在报告中保留但不作为质量门槛。全部适用门槛通过时顶层才写入 `status=ok` 和 `prediction_ready=true`；否则使用 `poor_fit`、`unvalidated` 或 `skipped`，不会把“求解成功”误报为“可用于预测”。

## result_all.csv 字段解释

每行对应一个资源配置、一个 input scale 和一次 warmup/repeat 请求窗口。
常规性能分析只取 `status=ok` 且 `warmup=0`；错误行中的部分数值不作为正式测量。
不可用或不适用的数值为 `nan`，历史 CSV 缺少新字段时不能补成 `0`。

字段按用途分组，实际列顺序以 [acprof/config.py](acprof/config.py) 的 `CSV_FIELDS` 为准。
`*_per_request` 和能量列按窗口内请求数归一化；`*_delta` 若未注明归一化，则表示整个窗口的增量。

### 资源配置、输入与网络

| 字段 | 含义 |
| --- | --- |
| `cpu_cores` | 当前 Docker container 的 CPU core 限制，来自 `--cpus`。 |
| `mem_cap_gb` | 当前 Docker container 的 memory cap，单位 GB，来自 `--mems`。 |
| `gpu_mode` | `on` 或 `off`。`on` 表示容器使用 Docker `--gpus all`。 |
| `input_scale` | 本行实际执行的主输入尺度。语义见 `static_meta.json/input_scale_type`。 |
| `input_units_per_request` | `effective input_scale × batch_size`。input unit 沿用任务族尺度：NLP/时序通常是 token/context step，Audio 是秒，CV 是缩放倍率而不是像素数。 |
| `input_num_samples` | 音频 payload 的实际采样点数；其他任务或旧计划无法推导时为 `nan`。它是诊断字段，音频主尺度仍为 `duration_s`。 |
| `request_payload_bytes` | `requests` 实际发送的 prepared HTTP JSON body 字节数，在同一 workload window 内取平均。 |
| `packet_request_wire_bytes_per_request` / `packet_response_wire_bytes_per_request` | 从同一 PCAP 中属于该 `/predict` TCP stream 的 client→server / server→client captured `frame.len` 总和，再对本行请求求平均。客户端使用 `Connection: close`，因此每个 stream 对应一个请求；握手、ACK、关闭包和重传都保留。 |
| `packet_total_wire_bytes_per_request` | 上述请求与响应 captured frame bytes 之和。它不含 capture 未保留的 Ethernet FCS，也不包含物理层 preamble / inter-frame gap，不能直接当作插座侧链路能耗输入。 |
| `packet_tcp_payload_bytes_per_request` | 同一 TCP stream 内 `tcp.len` 的总和，再对本行请求求平均；重传 payload 会按实际捕获次数计入。 |
| `packet_protocol_overhead_bytes_per_request` | `packet_total_wire_bytes_per_request - packet_tcp_payload_bytes_per_request`，表示捕获到的 L2/L3/L4 header、ACK/握手/关闭等开销；不拆分 TCP payload 内的 HTTP header 与 JSON body。 |
| `packet_protocol_overhead_ratio` | 本行所有请求的 protocol overhead bytes 总和 / total wire bytes 总和；分母无效时为 `nan`。 |
| `task_param` | 本行 payload 真正发送给 handler 的二级参数，使用稳定排序的 JSON 字符串；通常来自 `params`，时序任务同时记录顶层 `prediction_length`，不再记录未执行的任务族默认值。 |
| `output_length_avg` | 同一 workload window 内响应文本字符数的平均值；不适用时为 `nan`。 |
| `output_token_count_avg` | 同一 workload window 内响应文本 tokenizer token 数的平均值；用于解释 ASR 解码器工作量，不适用时为 `nan`。 |
| `repeat_idx` | 当前 warmup 或 repeat phase 内的 0-based iteration index。 |
| `warmup` | `1` 表示 warmup 行，`0` 表示正式测量行。`plot.py` 默认排除 warmup 行。 |
| `repeat_in_window` | 本行内部连续发送的 request 数量。`latency_app_s` 和 `latency_s` 都是该 window 内 request 的平均值。 |

### 延迟与吞吐

| 字段 | 含义 |
| --- | --- |
| `latency_s` | packet-level latency，来自 `tcpdump` PCAP + `tshark` 解析 + `acprof.packet.merge_packet_latency` merge。当前默认要求该字段完整；抓包不可用、PCAP 为空、解析为空或 merge 后仍有缺失时，程序会退出并给出恢复提示。 |
| `latency_s_per_input_unit` | `latency_s / input_units_per_request`，在 packet merge 阶段更新。 |
| `latency_request_count` | 本行实际合并到 packet-level latency 分布中的有效 request 数。可与 `repeat_in_window` 对照检查抓包完整性。 |
| `latency_p50_s` / `latency_p90_s` / `latency_p95_s` | packet-level latency 在本行 request window 内的 empirical nearest-rank 分位数，由 merge 阶段从同一组 packet latency 明细计算。 |
| `latency_std_s` / `latency_cv` / `latency_iqr_s` / `latency_max_s` | 同一 packet-level request window 的总体标准差、变异系数 `std / mean`、nearest-rank `P75 - P25` 和最大值。少于 2 个有效 request 时，std、CV 和 IQR 为 `nan`；max 仍保留。 |
| `latency_slow_ratio` | packet-level latency 中超过 `SLOW_LATENCY_THRESHOLD_S` 的 request 比例，默认阈值为 `0.06` 秒，用于观察尾延迟或双峰分布。 |
| `latency_app_s` | host-side application latency。`acprof.host.client` 用 `requests.post()` 外层 `time.perf_counter()` 测得，通常比 `latency_s` 更容易稳定产出。 |
| `latency_app_s_per_input_unit` | `latency_app_s / input_units_per_request`。 |
| `latency_app_request_count` | 本行 application latency 分布中的有效 request 数；正常成功窗口通常等于 `repeat_in_window`。 |
| `latency_app_p50_s` / `latency_app_p90_s` / `latency_app_p95_s` | host-side application latency 在本行 request window 内的 empirical nearest-rank 分位数。 |
| `latency_app_std_s` / `latency_app_cv` / `latency_app_iqr_s` / `latency_app_max_s` | application latency 的总体标准差、变异系数、nearest-rank IQR 和最大值；少于 2 个有效 request 时 std、CV 和 IQR 为 `nan`。 |
| `latency_app_slow_ratio` | host-side application latency 中超过 `SLOW_LATENCY_THRESHOLD_S` 的 request 比例，默认阈值为 `0.06` 秒。 |
| `throughput_samples_per_s` | 吞吐量，约等于 `batch_size / latency`。如果 `latency_s` 成功 merge，会优先按 `latency_s` 更新；否则按 `latency_app_s` 计算。 |
| `throughput_samples_per_s_per_cpu_core` | `throughput_samples_per_s / cpu_cores`；packet latency merge 后会与 throughput 一起重算。它表示按配置 CPU quota 归一化的吞吐，不是实际 CPU utilization 归一化值。 |

### Torch 与 NCU 计算指标

Torch eager 记录模型逻辑计算量，NCU 记录 GPU 实际执行量；两者分别使用
`*_torch_profiler_eager` 和 `*_ncu` 列。历史 plan / CSV 中的五个通用 compute 字段
仍可由绘图和迁移入口读取，新结果不再重复写入。
从已有计划生成派生 CSV 时，匹配键为 GPU mode 与 input scale（绝对容差 `1e-6`）；
未匹配或失败的工具字段保持 `nan`，对应错误列记录原因。操作见
[README](README.md#从已有计划生成派生-csv)。

| 字段 | 含义 |
| --- | --- |
| `model_logical_mflop_per_request_torch_profiler_eager` | `torch_profiler_eager` 根据 eager operator shape 统计的单 request 模型逻辑计算量，单位 MFLOP。 |
| `model_logical_mflops_app_torch_profiler_eager` | Torch eager 逻辑 MFLOP 除以 `latency_app_s`，单位 MFLOPS。 |
| `model_logical_mflops_packet_torch_profiler_eager` | Torch eager 逻辑 MFLOP 除以 packet latency，单位 MFLOPS；packet latency 无效时回退到 application latency。 |
| `compute_profile_error_torch_profiler_eager` | Torch eager probe 的独立诊断；正常为空，初始化、eager attention 验证或 profiler 失败时记录原因。 |
| `gpu_executed_mflop_per_request_ncu` | NCU 计数器得到的单 request GPU 实际执行总量，单位 MFLOP，等于 Tensor 与 Scalar 两列之和。只填充 `gpu_mode=on` 行。 |
| `gpu_executed_tensor_mflop_per_request_ncu` | NCU 统计的单 request Tensor FLOP，单位 MFLOP。 |
| `gpu_executed_scalar_mflop_per_request_ncu` | NCU 统计的单 request Scalar FLOP，单位 MFLOP；FMA 按 2 FLOP 计。 |
| `gpu_executed_tensor_share_pct_ncu` | `Tensor MFLOP / total MFLOP * 100`。 |
| `gpu_executed_mflops_app_ncu` | NCU GPU 实际执行 MFLOP 除以 `latency_app_s`，单位 MFLOPS。 |
| `gpu_executed_mflops_packet_ncu` | NCU GPU 实际执行 MFLOP 除以 packet latency，单位 MFLOPS；packet latency 无效时回退到 application latency。 |
| `gpu_kernel_launch_count_per_request_ncu` | NCU 报告中的 kernel launch 数，按 `--ncu-repeat` 归一化到单 request。 |
| `gpu_kernel_time_sum_ms_per_request_ncu` | NCU 报告中的 kernel duration 总和，换算为 ms 并按 `--ncu-repeat` 归一化到单 request。 |
| `compute_profile_error_ncu` | NCU probe 的独立诊断；正常为空，工具缺失、性能计数器受限或报告解析失败时记录原因。 |

### Massif 与 Nsight Systems 执行指标

| 字段 | 含义 |
| --- | --- |
| `cpu_heap_peak_bytes_massif` | CPU-only Massif 全部 snapshot 中 useful heap 的独立最大值，单位 bytes。该 process lifetime 包含模型加载、预热与 inference，不是单 request 内存增量。 |
| `cpu_heap_extra_peak_bytes_massif` | 全部 Massif snapshot 中 allocator bookkeeping、alignment 等 heap extra 的独立最大值，单位 bytes。 |
| `cpu_stack_peak_bytes_massif` | 全部 Massif snapshot 中 stack 的独立最大值，单位 bytes。 |
| `cpu_heap_peak_total_bytes_massif` | 全部 Massif snapshot 中 `heap + heap extra + stack` 总量的最大值，单位 bytes。三个 component 的独立 maxima 可能来自不同 snapshot，不保证三者相加等于该 total。`plot.py` 会据此派生 GiB 图，但不改写 CSV。 |
| `cpu_heap_peak_at_ms_massif` | 上述 total 最大值 snapshot 相对被剖析进程启动的时间，单位 ms；包含加载/预热阶段，不能当作单 request latency。 |
| `compute_profile_error_massif` | Massif execution probe 的独立诊断；正常为空，未安装、执行失败或输出解析失败时记录原因。虽然沿用 `compute_profile_error_*` CSV 命名，它与 FLOP compute probe 独立。 |
| `host_inference_wall_time_ms_per_request_nsys` | Nsight Systems probe 中同步 inference window 的 host wall time，按 `--nsys-repeat` 归一化为 ms/request。 |
| `cuda_api_time_sum_ms_per_request_nsys` | `acprof_compute` NVTX range 内 CUDA API activity duration 总和，单位 ms/request。它是 activity sum，不是 wall time。 |
| `cuda_api_call_count_per_request_nsys` | 同一 NVTX range 内 CUDA API call 数，按 request 归一化。 |
| `gpu_kernel_time_sum_ms_per_request_nsys` | 同一 NVTX range 内 GPU kernel duration 总和，单位 ms/request。并行 stream 上的 kernel 可能重叠。 |
| `gpu_kernel_launch_count_per_request_nsys` | 同一 NVTX range 内 GPU kernel launch 数，按 request 归一化。 |
| `gpu_memcpy_time_sum_ms_per_request_nsys` | 同一 NVTX range 内 GPU memcpy duration 总和，单位 ms/request；copy 与 kernel 或其他 copy 可能重叠。 |
| `gpu_memcpy_count_per_request_nsys` | 同一 NVTX range 内 GPU memcpy activity 数，按 request 归一化。 |
| `gpu_memcpy_bytes_per_request_nsys` | 同一 NVTX range 内 GPU memcpy bytes 总量，按 request 归一化。 |
| `compute_profile_error_nsys` | Nsight Systems execution probe 的独立诊断；正常为空，工具/driver 不兼容、capture 失败或 SQLite report 解析失败时记录原因。 |

### GPU 功率与能耗

| 字段 | 含义 |
| --- | --- |
| `gpu_idle_power_w` | GPU matched-control baseline power，单位 W。仅 `gpu_mode=on` 且 NVML 可用时有值；使用与 workload 相同的 NVML monitor 生命周期，由 control samples 梯形积分后的能量 / 实际 duration 得到。 |
| `gpu_idle_measured_at` | `--idle-debug` 开启且 `gpu_mode=on` 时，matched control window 测量完成时的本地 ISO-8601 时间戳；未开启或 GPU 不可用时为 `nan`。CPU/GPU control 同时采集，因此两者共用同一时间戳。 |
| `gpu_idle_rel_range_so_far` | `--idle-debug` 开启时，截至本行为止当前 case 内有效 `gpu_idle_power_w` 的相对极差，公式为 `(max - min) / mean`；`0.05` 表示 5%。未开启时为 `nan`。 |
| `gpu_energy_iters` | GPU energy measurement 内部采样窗口中的 iteration 数。 |
| `gpu_avg_power_total_w` | 测量窗口内 GPU total average power，单位 W。 |
| `gpu_peak_power_total_w` | 测量窗口内 GPU total peak power，单位 W。 |
| `gpu_energy_total_j` | 本行平均到单 request 的 total GPU energy，单位 J。 |
| `gpu_avg_power_eff_w` | 扣除 idle baseline 后的 effective average power，单位 W。 |
| `gpu_peak_power_eff_w` | 扣除 idle baseline 后的 effective peak power，单位 W。 |
| `gpu_energy_eff_j` | 本行平均到单 request 的 effective GPU energy，单位 J。 |

### CPU package 功率与能耗

| 字段 | 含义 |
| --- | --- |
| `cpu_idle_power_w` | CPU package matched-control baseline power，单位 W。仅主机暴露可读的 Linux RAPL `/sys/class/powercap/*/energy_uj` 时有值；由与 workload 相同 monitor 生命周期下整段 control window 的 RAPL 能耗差 / 实际 duration 得到。 |
| `cpu_idle_measured_at` | `--idle-debug` 开启时，CPU idle baseline 测量完成时的本地 ISO-8601 时间戳；未开启时为 `nan`。 |
| `cpu_idle_rel_range_so_far` | `--idle-debug` 开启时，截至本行为止当前 case 内有效 `cpu_idle_power_w` 的相对极差，公式为 `(max - min) / mean`；`0.05` 表示 5%。未开启时为 `nan`。 |
| `cpu_energy_iters` | CPU package energy measurement 采样窗口中的 sample 数。 |
| `cpu_avg_power_total_w` | 测量窗口内 host CPU package total average power，单位 W。 |
| `cpu_peak_power_total_w` | 测量窗口内 host CPU package total peak power，单位 W。peak 取相邻 RAPL 采样区间功率的最大值，低于 `0.5 / sample_hz` 的尾部短区间不参与 peak 计算。 |
| `cpu_energy_total_j` | 本行平均到单 request 的 total CPU package energy，单位 J。 |
| `cpu_avg_power_eff_w` | 扣除 CPU idle baseline 后的 CPU package effective average power，单位 W。 |
| `cpu_peak_power_eff_w` | 扣除 CPU idle baseline 后的 CPU package effective peak power，单位 W。peak 口径同 `cpu_peak_power_total_w`。 |
| `cpu_energy_eff_j` | 本行平均到单 request 的 effective CPU package energy，单位 J。 |

### 估算 vCPU 能耗与派生能效

| 字段 | 含义 |
| --- | --- |
| `vcpu_cpu_share` | container cgroup CPU time delta / host active CPU time delta，用于估算本 container 占 host active CPU 的比例。 |
| `vcpu_cpu_time_s` | 本行平均到单 request 的 container CPU time，单位秒。 |
| `vcpu_avg_power_total_w` | 按 `vcpu_cpu_share` 分摊后的 estimated vCPU total average power，单位 W。 |
| `vcpu_peak_power_total_w` | 按相邻采样区间 container cgroup CPU share 分摊后的 estimated vCPU total peak power，单位 W。低于 `0.5 / sample_hz` 的尾部短区间不参与 peak 计算。 |
| `vcpu_energy_total_j` | 本行平均到单 request 的 estimated vCPU total energy，单位 J。 |
| `vcpu_avg_power_eff_w` | 按 `vcpu_cpu_share` 分摊后的 estimated vCPU effective average power，单位 W。 |
| `vcpu_peak_power_eff_w` | 按相邻采样区间 container cgroup CPU share 分摊后的 estimated vCPU effective peak power，单位 W。peak 口径同 `vcpu_peak_power_total_w`。 |
| `vcpu_energy_eff_j` | 本行平均到单 request 的 estimated vCPU effective energy，单位 J。 |
| `container_attributed_energy_eff_j` | 不增加采集轮次的派生值。CPU-only 行等于 estimated `vcpu_energy_eff_j`；GPU 行等于 `vcpu_energy_eff_j + gpu_energy_eff_j`。任一必需分量缺失或为负时为 `nan`。它只覆盖已采集并归因的 vCPU/GPU 分量，不代表 wall-plug system energy。 |
| `container_attributed_samples_per_j` | `batch_size / container_attributed_energy_eff_j`，单位 samples/J；能量不为正或缺失时为 `nan`。 |
| `container_attributed_edp_app_js` | `container_attributed_energy_eff_j * latency_app_s`，即 application-latency energy-delay product，单位 J·s/request。 |
| `output_tokens_per_s_app` | `output_token_count_avg / latency_app_s`。只在 handler 能可靠返回 output token count 时有值，当前主要用于 ASR。 |
| `container_attributed_j_per_output_token` | `container_attributed_energy_eff_j / output_token_count_avg`；没有可靠 output token count 时为 `nan`。 |
| `container_attributed_j_per_input_unit` | `container_attributed_energy_eff_j / input_units_per_request`；沿用上述任务族 input unit 语义，不增加能耗采集窗口。 |

### CPU 资源、频率与 PMU

| 字段 | 含义 |
| --- | --- |
| `resource_usage_iters` | resource usage monitor 在本行测量窗口内保留的 sample 数。 |
| `container_cpu_util_avg_pct` | 当前 Docker container 在测量窗口内的平均 CPU 占用率，按 `container_cpu_time_delta / (elapsed_seconds * cpu_cores) * 100` 计算。 |
| `container_cpu_util_peak_pct` | 当前 Docker container 在相邻采样间隔中的峰值 CPU 占用率，单位 `%`。 |
| `container_cpu_nr_periods_delta` / `container_cpu_nr_throttled_delta` | workload 窗口首尾 cgroup `cpu.stat` 的调度周期数和被 quota throttled 周期数之差。支持 cgroup v2，也兼容 v1 `cpu.stat`。 |
| `container_cpu_throttled_period_ratio_pct` | `nr_throttled_delta / nr_periods_delta * 100`；用于判断 `--cpus` quota 是否实际成为瓶颈。没有有效 period 时为 `nan`。 |
| `container_cpu_throttled_time_s_per_request` | cgroup CPU throttled time 的窗口增量换算成秒后除以 `repeat_in_window`。v2 读取 `throttled_usec`，v1 读取 `throttled_time` 纳秒值；它不是单纯的 request wall latency。 |
| `container_cpu_pressure_some_stall_pct` / `container_cpu_pressure_full_stall_pct` | cgroup v2 `cpu.pressure` 的 `some/full total` 在 workload 窗口内的增量除以窗口时长。使用累计 stall time，不使用瞬时 `avg10/60/300`；系统不提供 per-cgroup PSI 时为 `nan`。 |
| `cpu_freq_avg_hz` | 测量窗口内 host online CPU 当前频率的平均值，单位 Hz。每个 sample 先对 online CPU 求平均，最终再对窗口内 sample 求平均；优先读取 Linux cpufreq sysfs，失败时回退到 `/proc/cpuinfo`。 |
| `cpu_freq_peak_hz` | 测量窗口内 host online CPU 当前频率的峰值，单位 Hz。每个 sample 取 online CPU 的最高当前频率，最终再取窗口内最大值。 |
| `cpu_cycles_est_app` | 基于 application latency 的 estimated CPU cycles，公式为 `latency_app_s * cpu_freq_avg_hz * cpu_cores * container_cpu_util_avg_pct / 100`。这是利用率与频率推导值，不是硬件 PMU retired instructions / cycles 计数。 |
| `cpu_cycles_est_packet` | 基于 packet-level `latency_s` 的 estimated CPU cycles，公式同 `cpu_cycles_est_app`，但在 `acprof.packet.merge_packet_latency` 成功回填 `latency_s` 后才会更新；merge 前或 packet latency 缺失时为 `nan`。 |
| `cpu_instructions_per_request` | Linux `perf stat -e instructions` 采集到的 retired instructions，按本行 `repeat_in_window` 平均到单 request。MIPS 采集失败会中止实验而不是写入静默 `nan`。 |
| `cpu_cache_references_per_request` | Linux `perf` generic event `cache-references` 的窗口计数，按本行 `repeat_in_window` 平均到单 request。其对应的 cache level 由 CPU 架构和 kernel PMU 映射决定。 |
| `cpu_cache_misses_per_request` | Linux `perf` generic event `cache-misses` 的窗口计数，按本行 `repeat_in_window` 平均到单 request；不能跨架构固定解释为某一级 cache miss。 |
| `cpu_cache_miss_rate_pct` | `cache-misses / cache-references * 100`。用于观察 cache access locality，不表示实际内存带宽；分母无效或为 0 时为 `nan`。 |
| `cpu_dtlb_loads_per_request` | Linux `perf` event `dTLB-loads` 的窗口计数，按本行 `repeat_in_window` 平均到单 request。 |
| `cpu_dtlb_load_misses_per_request` | Linux `perf` event `dTLB-load-misses` 的窗口计数，按本行 `repeat_in_window` 平均到单 request，用于观察数据地址转换未命中。 |
| `cpu_dtlb_load_miss_rate_pct` | `dTLB-load-misses / dTLB-loads * 100`。用于观察数据地址转换开销；分母无效或为 0 时为 `nan`。 |
| `cpu_mips_app` | 基于 `latency_app_s` 的真实 retired-instruction MIPS，公式为 `cpu_instructions_per_request / latency_app_s / 1e6`。 |
| `cpu_mips_packet` | 基于 packet-level `latency_s` 的真实 retired-instruction MIPS，在 `acprof.packet.merge_packet_latency` 成功回填 `latency_s` 后更新；merge 前或 packet latency 缺失时为 `nan`。 |
| `cpu_perf_elapsed_s` | perf 统计窗口报告的 elapsed time，单位秒，用于诊断 perf 窗口是否覆盖本行 workload。 |

### 容器内存、swap、I/O 与 PID

| 字段 | 含义 |
| --- | --- |
| `container_mem_usage_avg_bytes` | 当前 Docker container 在测量窗口内的平均 memory usage，单位 bytes，来自 cgroup memory 文件。 |
| `container_mem_usage_peak_bytes` | 当前 Docker container 在测量窗口内的峰值 memory usage，单位 bytes。 |
| `container_mem_util_avg_pct` | 当前 Docker container 平均 memory usage / `mem_cap_gb` 的百分比。 |
| `container_mem_util_peak_pct` | 当前 Docker container 峰值 memory usage / `mem_cap_gb` 的百分比。 |
| `container_mem_peak_cgroup_bytes` | workload 窗口结束时读取 cgroup v2 `memory.peak`。它是该新建 container cgroup 自创建以来的内存峰值，因此能捕获 `sample_hz` 之间的瞬时峰值，也可能包含模型加载期；不是单个 workload window 可重置的峰值。文件不存在或 cgroup v1 时为 `nan`。 |
| `container_mem_anon_bytes_end` / `container_mem_file_bytes_end` / `container_mem_slab_bytes_end` | workload 窗口结束时 cgroup v2 `memory.stat` 的匿名内存、文件页和 slab 当前字节数。`slab` 缺失时使用 `slab_reclaimable + slab_unreclaimable`；无法读取时为 `nan`。 |
| `container_mem_pgfault_delta` / `container_mem_pgmajfault_delta` | cgroup v2 `memory.stat` 的 page fault / major page fault 计数器在 workload 窗口首尾的增量。它们是整个 cgroup 的事件数，不按 request 归一化。 |
| `container_mem_workingset_refault_delta` | cgroup v2 `memory.stat` 的 workingset refault 窗口增量；内核只提供 anon/file 分项时取两者之和，用于观察页被回收后再次访问。 |
| `container_mem_high_events_delta` / `container_mem_max_events_delta` | cgroup v2 `memory.events` 的 `high` 和 `max` 计数器在 workload 窗口内的增量，分别表示 memory high 边界触发和 memory max 边界命中次数。cgroup v1 无同口径字段时为 `nan`。 |
| `container_mem_oom_events_delta` / `container_mem_oom_kill_events_delta` | cgroup v2 `memory.events` 的 `oom` 与 `oom_kill` 窗口增量；前者表示 cgroup 内分配进入 OOM，后者表示实际发生进程 OOM kill。 |
| `container_mem_pressure_some_stall_pct` / `container_mem_pressure_full_stall_pct` | cgroup v2 `memory.pressure` 的 `some/full total` 窗口增量占窗口时长的比例。`full` 表示窗口内所有相关任务同时因内存压力停顿。 |
| `container_swap_limit_bytes` | 当前 container cgroup 的独立 swap hard limit。cgroup v2 来自 `memory.swap.max`；cgroup v1 由 mem+swap limit 减去 memory limit 得到。`-1` 表示 cgroup 未设上限，无法读取时为 `nan`。 |
| `container_swap_usage_avg_bytes` | 本行 workload 测量窗口内 container swap 使用量的平均值。cgroup v2 读取 `memory.swap.current`；cgroup v1 由 mem+swap usage 减去 memory usage，按现有 `sample_hz` 采样。 |
| `container_swap_usage_peak_bytes` | 同一测量窗口内采样到的 container swap 使用量峰值，单位 bytes；它不是 host 全局 swap 使用量。 |
| `container_io_read_bytes_per_request` | 本行测量窗口首尾 container cgroup block-I/O read bytes 计数器之差，再除以实际 `repeat_in_window`；聚合 cgroup 报告的全部设备，无法读取或本行未完成请求时为 `nan`。 |
| `container_io_write_bytes_per_request` | 本行测量窗口首尾 container cgroup block-I/O write bytes 计数器之差，再除以实际 `repeat_in_window`；page-cache hit 不产生块设备读取，因此该值不等于应用读取的文件字节数。 |
| `container_io_read_ops_per_request` / `container_io_write_ops_per_request` | cgroup v2 `io.stat` 中全部设备 `rios/wios` 的窗口增量除以实际 `repeat_in_window`。它们表示 block-I/O operation 数，不是 POSIX read/write syscall 数；cgroup v1 兼容模式为 `nan`。 |
| `container_io_pressure_some_stall_pct` / `container_io_pressure_full_stall_pct` | cgroup v2 `io.pressure` 的 `some/full total` 窗口增量占窗口时长的比例，用于区分真实 I/O stall 与仅有块 I/O 字节增量的情况。 |
| `container_pids_current_end` | workload 窗口结束时 cgroup v2 `pids.current`，统计该 cgroup 当前 tasks 数。 |
| `container_pids_peak_cgroup` | workload 窗口结束时读取 cgroup v2 `pids.peak`；与 `memory.peak` 一样是新建 container cgroup 生命周期峰值。旧内核未提供该文件时为 `nan`。 |
| `container_pids_max_events_delta` | cgroup v2 `pids.events/max` 在 workload 窗口内的增量，表示因 PID hard limit 拒绝创建 task 的次数。 |

### GPU 资源与运行状态

| 字段 | 含义 |
| --- | --- |
| `gpu_sm_clock_mhz` | NVML device 0 在 workload 测量窗口内的 SM clock 成功采样值算术平均，单位 MHz。仅 `gpu_mode=on` 且 NVML 支持该查询时有值；不采集 graphics clock。 |
| `gpu_memory_clock_mhz` | NVML device 0 在 workload 测量窗口内的 memory clock 成功采样值算术平均，单位 MHz。 |
| `gpu_pstate` | NVML device 0 在 workload 测量窗口内出现次数最多的 performance state（`P0`–`P15`）；次数并列时取性能等级更高的较小编号。它是运行状态解释变量，不代表锁频。 |
| `gpu_temp_c` | NVML device 0 在 workload 测量窗口内的 GPU temperature 成功采样值算术平均，单位 °C。 |
| `gpu_util_avg_pct` | NVML device 0 在测量窗口内的平均 GPU utilization，单位 `%`。这是 device-level 口径，不做 container process attribution。 |
| `gpu_util_peak_pct` | NVML device 0 在测量窗口内的峰值 GPU utilization，单位 `%`。 |
| `gpu_mem_used_avg_bytes` | NVML device 0 在测量窗口内的平均 used VRAM，单位 bytes。 |
| `gpu_mem_used_peak_bytes` | NVML device 0 在测量窗口内的峰值 used VRAM，单位 bytes。 |
| `gpu_mem_util_avg_pct` | NVML device 0 平均 used VRAM / total VRAM 的百分比。 |
| `gpu_mem_util_peak_pct` | NVML device 0 峰值 used VRAM / total VRAM 的百分比。 |

### 冷启动

| 字段 | 含义 |
| --- | --- |
| `cold_start_started_at` / `cold_start_ready_at` | host 侧开始执行 `docker run` 与首次成功收到 `/ready` 的本地 ISO-8601 时间戳。 |
| `cold_start_container_launch_s` | host 开始 `docker run` 到 container 内 Python server process 开始执行的时间；使用共享系统 wall clock 对齐。旧镜像未返回启动时间戳时为 `nan`。 |
| `cold_start_server_setup_s` | server process 开始到 model load 开始之间的 setup 时间，扣除单独列出的显式 CUDA 初始化，包括 Python/framework/handler import 与配置。 |
| `cold_start_cuda_init_s` | 现有启动路径中 `torch.cuda.is_available()` 与 `torch.cuda.init()` 的显式计时；CPU case 为 `0`，不发送 probe inference。 |
| `cold_start_model_load_s` | container handler `load()` 的持续时间，使用 `time.perf_counter()`。 |
| `cold_start_ready_wait_s` | model load 完成到 host 首次收到成功 `/ready` 的时间，包括 Flask 开始监听、poll interval 和本地 HTTP 往返。 |
| `cold_start_first_predict_app_s` | workload client 发出的第一个成功 `/predict` 的 application latency；默认 auto-window 模式下通常是已有 auto-warmup 请求，固定窗口模式下是已有首个 warmup/measurement 请求，不会为此新增请求。 |
| `cold_start_s` | 当前 container 从 `docker run` 到 `/ready` 成功的时间，单位秒。 |

### 运行状态与错误

| 字段 | 含义 |
| --- | --- |
| `status` | `ok`、`warn` 或 `error`。`warn` 标记存在异常的行；常规性能图与延迟模型只使用 `ok` 行，资源可行性图保留失败状态。 |
| `error` | 错误或 warning 文本。正常行为空；`status=error` 时强制非空。请求超时会区分实际发出但未完成的请求（`client_request_timeout`）与未发请求、因前序超时跳过的计划行（`not_measured_after_timeout`），并记录触发尺度、timeout 下界、请求阶段和 request ID。 |

### `latency_s` 和 `latency_app_s` 的区别

- `latency_app_s` 是 client 侧应用层计时，只要 `/predict` 请求成功，一般就能写出。
- `latency_s` 是 packet-level 计时，需要完整完成 `tcpdump` capture、`acprof.packet.sniff_parse_pcap` parse、`acprof.packet.merge_packet_latency` merge。
- 当前默认行为是严格模式：如果无法保证 `latency_s` 有值，`run.py` 会退出，不继续 merge 最终结果。

## 结果行数和时间成本估算

### CSV 行数与请求数

完整计划的行数为：

```text
资源 case 数 = len(cpus) × len(mems) × len(gpus)
总行数 = 资源 case 数 × 实际 input scale 数 × (warmup + repeat)
正式测量行数 = 资源 case 数 × 实际 input scale 数 × repeat
```

这里使用 `input_scale_plan.json` 中实际确定的档数；自定义音频清单可能不是 6 档。
默认 6 档时，计划行数为 `4 × 4 × 2 × 6 × (2 + 5) = 1,344`，其中 warmup 384 行、
正式测量 960 行。OOM、超时或剪枝可能生成错误占位，异常中断也可能留下部分结果，
因此计划行数不等于成功实测行数。

一行 CSV 是一个请求窗口，不等于一个 batch 或一次 `/predict`：

- 固定 `--repeat-in-window N`：每个成功窗口发送 `N` 次请求，完整成功矩阵的窗口内请求数为 `总行数 × N`。
- 默认 `--repeat-in-window 0`：每行至少发送一次请求，累计请求耗时达到 `--repeat-window-seconds` 后停止；CSV 的 `repeat_in_window` 记录该行实际请求数。
- `--batch-size` 决定一次请求的样本数，不乘入 CSV 行数。样本总数还需按任务支持情况乘以 batch size。

auto 模式在每个资源 case / input scale 开始前默认额外发送 5 次 auto warmup 请求；
它们不写成 CSV 行，也不同于 `--warmup 2` 的两个完整测量窗口。默认完整矩阵另有
`32 × 6 × 5 = 960` 次 auto warmup 请求。固定 `--repeat-in-window N` 时不执行这段预热。

### 每行测量窗口

当前 client 按“冷却 → 无请求对照 → workload”执行；CPU、GPU 与资源监控共享对照窗口：

```text
row_window_s ≈ idle_cooldown_s + matched_control_s + active_workload_s
               + monitor_start_stop_and_csv_overhead_s

默认快速请求场景：5 + 20 + 10 = 35 秒/行
默认完整矩阵：1,344 × 35 / 3,600 ≈ 13.07 小时
```

GPU 开启和关闭都使用同一组 `5 + 20` 秒默认基线开销，不会再为 GPU 单独追加一个对照窗口。
`10` 秒是 auto workload 的目标，结束条件在完整请求返回后判断：若单次请求需要 120 秒，
该行至少约 `5 + 20 + 120 = 145` 秒。它不会在第 10 秒截断请求。
固定请求数时，active workload 约为 `N × 平均请求延迟`；已有成功 CSV 可用
`latency_app_s × repeat_in_window` 估算该行累计请求耗时。

单请求超时默认是 `--request-timeout-seconds 300`；超时会形成错误行，不能当作一次
成功的 300 秒测量。上述 13.07 小时仅适用于完整成功矩阵、请求足够快且默认窗口接近目标的情况，
还未包含 auto warmup、启动、抓包解析、构建、分析器和清理等开销。

### 整条命令的耗时

```text
main_collection_s ≈ sum_over_cases(
  cold_start_s
  + sum_over_scales(auto_warmup_s)
  + sum_over_scales((warmup + repeat) × row_window_s)
  + sniff_parse_and_container_cleanup_s
) + final_merge_s

total_wall_s ≈ preflight_s + model_detection_s + docker_build_and_download_s
               + input_scale_planning_s + compute_profile_s
               + execution_profile_s + main_collection_s
```

auto warmup 约为每个 case/scale 的 `5 × 平均请求延迟`，长耗时模型应单独计入。
镜像复用、失败 case、启动 OOM 剪枝和续跑会改变实际工作量；启用通知和日志收尾也有额外耗时。
内存/PID 峰值、cgroup 增量、冷启动分解和派生能效使用已有采样路径，不增加 `/predict` 数量；
网络字节复用已有 PCAP，但仍需离线解析时间。

### 可选分析器成本

compute 与 execution profiler 默认都是 `none`。显式启用 compute `both` 时：

```text
compute_profile_s ≈ 实际 input scale 数 × (
  [启用 CPU-only] × torch_cpu_probe_s
  + [启用 GPU] × (torch_gpu_probe_s + ncu_gpu_probe_s)
)
```

单工具模式只计所选工具。compute probe 按尺度和 CPU/GPU device class 采集，不按主矩阵
CPU × memory 全量展开；`--torch-profiler-repeat` 和 `--ncu-repeat` 会增加各 probe 工作量，
NCU kernel replay 还可能显著拉长单次分析。

execution profiler 的实际来源配置数取决于采样策略：

| 工具 | 默认来源配置数 | 其他策略 |
| --- | --- | --- |
| Massif | `per-scale`：1 个代表 CPU/内存 | `full`：`len(cpus) × len(mems)`。 |
| Nsys | `per-cpu-scale`：`len(cpus)`，共用代表内存 | `per-scale`：1；`full`：`len(cpus) × len(mems)`。 |

```text
execution_profile_s ≈ 实际 input scale 数 × (
  massif_source_cases × massif_probe_s
  + nsys_source_cases × nsys_probe_s
)
```

未选择对应工具或 GPU mode 时，该项来源配置数为 0。Massif 包含模型加载与预热，
Nsys 还需生成和解析 timeline；repeat 参数会进一步增加工作量。默认保留的原始报告
也需要磁盘空间，使用 `full` 前可先用单个资源配置与尺度估算。

## 常见判断

本节解释结果值、失败边界与采样限制。安装、主机权限、抓包和旧镜像的操作排查见
[README 常见问题](README.md#常见问题)。

### 先区分预期空值与失败

- 常规性能分析筛选 `status=ok` 且 `warmup=0`；`status=error` 行中的部分数值不视为完整测量。
- 关闭的 profiler、不适用的任务/GPU mode 和旧版本缺失字段可以为 `nan`，不能直接解释为采集失败或数值为零。
- 一个窗口只有 1 个有效请求时，标准差、CV、IQR 为 `nan`；增加 `--repeat` 只增加窗口数，不保证每个窗口有多个请求。
- 比较 profiler 数值前检查 plan 的采样来源；代表资源复用值不代表该 CPU/内存配置被独立分析过。

### 启动 OOM 与剪枝占位

- 容器在模型加载期间触达 `--memory` cgroup 上限并被内核终止；错误会同时记录 memory cap、Docker 状态和 exit code。
- 该 case 的占位行保留为 `status=error`，latency、throughput、energy 和 resource usage 等未执行指标保持 `nan`。`plot.py` 的性能图与 latency model 只使用 `status=ok` 行；资源可行性热力图会单独读取这些占位行，用来展示失败边界。
- 增大 memory cap，或改用更小/量化模型；不要用推测值回填失败 case 的指标。
- 默认的启动 OOM 剪枝不会改变任何可运行 case 的 warmup、repeat 或指标，只在后续 CPU 上为已确认的连续启动 OOM 前缀写入未执行占位行。热力图中实测启动 OOM 为 `OOM-S`，剪枝推断为 `P-OOM`；论文中必须区分两者。需要每个资源格独立实测时使用 `--no-prune-startup-oom`。

### 运行期 OOM

- 容器已完成启动，但在 workload 期间被 memory cgroup OOM kill。Docker 的 `OOMKilled=true` 优先于客户端的 MIPS、HTTP 断连或其他次生退出码进行分类。
- 已成功写入的测量行原样保留；已有失败行追加 Docker OOM 上下文；其余计划行写为 `status=error`，未采指标保持 `nan`。该 case 返回后矩阵继续执行。
- 运行期 OOM 在资源可行性热力图中显示为 `OOM-R`，但不会用于启动 OOM 剪枝，也不能从一个 CPU 配置外推到其他 CPU 配置。

### GPU energy 字段全是 `nan`

- `gpu_mode=off` 时这是正常结果。
- `gpu_mode=on` 时检查 NVIDIA driver、NVIDIA Container Toolkit、`pynvml` 和容器 GPU 可见性。

### CPU / vCPU energy 字段全是 `nan`

- 当前版本会在 task detection 前检查 RAPL；计数器不存在或不可读时，`run.py` 会退出并给出权限修复步骤，不会继续生成新的完整结果。
- 历史结果或中断产生的 CSV 仍可能包含 `nan`。这类缺失值不应使用 TDP 或 CPU utilization 猜测回填。
- `cpu_*` 字段是 host CPU package/root domain 的真实 RAPL 测量值，不累加 `intel-rapl:*:*` 这类 core 子 domain；`vcpu_*` 字段是在同一窗口内按 container cgroup CPU share 分摊出来的估计值。

### CPU idle baseline 波动 warning

- `cpu_idle_power_w` 的 case 级相对极差达到或超过 5% 时，case 结束后会输出 warning，实验继续运行。常见原因是 host 后台进程、IDE/远程桌面、Docker 其他容器、系统索引或 CPU 温度/频率策略变化。
- 可先增大 `--idle-cooldown-seconds` 让上一轮 workload 的瞬态结束；如果单次 idle 窗口本身仍然抖动，再增大 `--idle-seconds`。同时关闭非必要后台进程，并检查 `static_meta.json` 中的 `cpu_governor` / `cpu_boost` 是否符合实验设置。
- 需要定位具体时间点和进程时，加 `--idle-debug` 重跑；优先查看 `debug_idle_diag/result_case_*.csv.idle_diag.jsonl` 中同一 row 的 `rapl_trace.top_power_windows`、`idle_host_active_delta_s`、`idle_container_cpu_delta_s`，再结合 control window 结束后采集的 loadavg、top CPU processes 和 Docker 快照。`idle_proc_cpu_top` 在 matched control 模式下保持为空，避免 `/proc` 全量遍历进入 baseline 能量。

### GPU idle baseline 波动 warning

- `gpu_idle_power_w` 的 case 级相对极差达到或超过 5% 时，case 结束后会输出 warning，实验继续运行。常见原因是桌面显示栈、其他 GPU 进程、P-state/clock 调整、温度或电源管理状态变化。
- 可先增大 `--idle-cooldown-seconds`，等待上一轮 workload 后的 GPU clock/power 状态回落；如果 idle trace 内部仍抖动，再增大 `--idle-seconds`。
- 需要定位具体时间点和进程时，加 `--idle-debug` 重跑；优先查看同一 row 的 `gpu_idle_power_samples`、`nvidia_smi_gpu`、`nvidia_smi_pmon` 和 `nvidia_smi_compute_apps`。

### 资源占用率字段全是 `nan`

- `container_*` usage/I/O 字段依赖被测容器的 cgroup CPU、memory、swap 与 I/O 文件；如果 Docker inspect、`/proc/<pid>/cgroup` 或 `/sys/fs/cgroup` 不可读，对应字段会保持 `nan`，不会影响其他可用指标。
- 默认正式模式已在启动阶段强制 cgroup v2。只有显式使用 `--allow-cgroup-v1` 的兼容运行才会走 v1 fallback；其 `memory.peak` / `memory.stat` 分解与计数器、`io.stat` 操作数、`pids.*`、`memory.events` 和 per-cgroup PSI 字段保持 `nan`，不会使用 host 全局数据冒充 container 指标。
- `cpu_freq_*` 字段依赖 Linux cpufreq sysfs 或 `/proc/cpuinfo`。如果当前内核、虚拟化环境或权限不暴露当前频率，会保持 `nan`。
- `cpu_cycles_est_app` 依赖 `latency_app_s`、`cpu_freq_avg_hz`、`cpu_cores` 和 `container_cpu_util_avg_pct` 都有效；`cpu_cycles_est_packet` 还额外依赖 packet latency merge 成功回填 `latency_s`。
- `gpu_*` utilization / VRAM 字段仅在 `gpu_mode=on` 且 NVML 可用时采集；口径是 NVML device-level，可能包含同一 GPU 上其他进程的占用。

### MIPS、cache miss 与 dTLB miss

- `cpu_mips_*` 字段来自 Linux `perf` 的 `instructions` 硬件事件，不是 CPU frequency 推导值。`run.py` 会在 task detection 前检查 `perf` 权限，失败时打印 `[mips][ERROR]`、当前 `perf_event_paranoid`、sudo 状态和恢复步骤。
- `cpu_cache_*` 和 `cpu_dtlb_*` 字段来自 Linux `perf` generic PMU events。可先用 `perf list` 和 `perf stat -e cache-references,cache-misses,dTLB-loads,dTLB-load-misses -- true` 检查当前 CPU / kernel 是否支持；事件不支持不表示 miss 为 0。
- 这些 cache / dTLB 字段用于描述访存行为，不提供 DRAM GB/s。实际 read/write bandwidth 需要 uncore memory-controller、Intel PCM、AMD IBS/DF 或其他硬件专用计数器，不能由 miss 数直接换算。

### CPU / vCPU peak power 看起来异常

- `cpu_peak_power_*` 和 `vcpu_peak_power_*` 是相邻采样区间功率的最大值，不是整段平均功率。增大 `--sample-hz` 会缩短区间、提高捕捉短峰值的能力，也可能让峰值更敏感。
- 为避免停止监控时的极短尾部区间放大 peak，当前实现会排除短于半个采样周期的 peak 区间；avg power 和 energy 仍按完整测量窗口计算。

### MFLOPS / compute profiling 字段全是 `nan`

- 默认 `--compute-profile-tool none` 不采 FLOP，因此这些字段为 `nan` 属于预期结果。显式启用 `both` 后，Torch eager 与 NCU 是独立 probe；先分别查看 `compute_profile_error_torch_profiler_eager` 和 `compute_profile_error_ncu`，一个失败不会令另一套指标或主实验失败。
- NCU 字段只在 `gpu_mode=on` 行有值；CPU-only 行为 `nan` 是预期行为。
- `gpu_mode=on` 时检查 `ncu` 是否安装、`--ncu-root` 是否正确、NVIDIA driver 是否允许 performance counters。若被测镜像裸跑 CUDA 正常、仅 NCU 下报错，结合工具日志核对 NCU、CUDA 和 driver 的兼容性，不能只凭一个错误码确定版本问题。
- 如果显式使用 `--compute-profile-tool vendor`，`gpu_mode=off` 时检查 Intel Advisor 是否安装，以及 `--advisor-root` 是否指向可在容器中 bind mount 的 Advisor root 或 executable。
- compute profiling 与正常 workload 分离；失败不会影响 latency / energy / resource usage 采集。完整状态与静态口径见 `compute_profile_plan.json` 和 `static_meta.json`。

### Massif / Nsight Systems execution profiling 字段全是 `nan`

- 默认 `--execution-profile-tool none` 不采 execution profile，这是预期结果；需要显式选择 `massif`、`nsys` 或 `both`。
- Massif 字段只填充 `gpu_mode=off` 行，Nsight Systems 字段只填充 `gpu_mode=on` 行；不适用的另一组字段保持 `nan`。
- 分别查看 `compute_profile_error_massif` 和 `compute_profile_error_nsys`。Massif 检查模型镜像中预装的 Valgrind；旧模型兼容构建失败时检查 Docker build/apt 网络与 `dockerfiles/massif.Dockerfile` 日志，不要求 host 安装 `valgrind`。Nsight Systems 检查 `nsys`、`--nsys-root`、实际分析镜像的 importer runtime preflight、NVIDIA driver / Container Toolkit 兼容性，以及 raw `.nsys-rep` 是否可导出；旧模型的依赖由 `dockerfiles/nsys.Dockerfile` 补齐。出现 `nsys_importer_unavailable` 时，优先检查分析镜像中 `libdw.so.1` 等动态库；probe 阶段只出现 `.qdstrm` 而没有 `.nsys-rep` 属于 importer 失败。
- 两者是显式 opt-in 的独立 probe；一个失败不会影响另一个 execution probe、FLOP compute profiling 或主实验。完整状态与静态口径见 `execution_profile_plan.json` 和 `static_meta.json`。

## CLI 参数

以下参数表对应 `run.py`。示例命令见 [README](README.md#运行正式实验)，
默认值与实际选项以当前入口的 `--help` 和 [acprof/config.py](acprof/config.py) 为准。

### `run.py`

#### 模型与资源矩阵

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | required | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `--task` | auto | 覆盖 `pipeline_tag`，例如 `fill-mask`、`text-generation`。 |
| `--task-family` | auto | 覆盖任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`。 |
| `--backend` | auto | 覆盖 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`。 |
| `--cpus` | `1,2,4,8` | CPU core 限制列表。 |
| `--mems` | `2,4,8,16` | Memory cap GB 列表。 |
| `--gpus` | `off,on` | GPU mode 列表。`on` 会用 Docker `--gpus all`。 |
| `--prune-startup-oom` / `--no-prune-startup-oom` | enabled | 默认以最低选中 CPU 为参考，按内存升序完整采集；仅把 Docker 明确 `OOMKilled` 的连续低内存启动失败前缀推断到后续更高 CPU。跳过 case 保留占位行和独立 provenance。运行期/CUDA OOM、timeout 与普通启动失败不触发剪枝。使用 `--no-prune-startup-oom` 可恢复逐格独立尝试。 |

#### 请求窗口与采样

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--batch-size` | `1` | 每个 request 的 batch size。 |
| `--warmup` | `2` | 每个资源配置、每个 input scale 的 warmup 行数。 |
| `--repeat` | `5` | 每个资源配置、每个 input scale 的正式测量行数。 |
| `--repeat-in-window` | `0` | 每一行内部连续发送的 `/predict` request 数量。`0` 表示 auto 模式：每行至少发送 1 个请求，并持续到累计 `latency_app_s` 达到 `--repeat-window-seconds`。 |
| `--repeat-window-seconds` | `10.0` | `--repeat-in-window 0` 时的目标 workload window 秒数。auto 模式不再额外跑一个 10 秒校准窗口。 |
| `--request-timeout-seconds` | `300.0` | 正式矩阵中每个 `/predict` 请求的最大等待秒数，必须是大于 0 的有限值；它适用于 warmup、auto-window warmup 和正式请求，不限制整行、整个 case 或整条命令的总运行时间。超时后保留已完成行，并将触发请求及后续未测计划行分别写成可诊断的 error 占位。 |
| `--sample-hz` | `20.0` | GPU power sampling rate，单位 Hz；CPU workload 和 matched control window 期间也用它控制 RAPL、container cgroup、CPU frequency 和 GPU/resource usage 的采样间隔，以估计 average/peak power、vCPU share、CPU utilization 和 CPU cycles。perf MIPS 使用独立的 `perf stat` 窗口，不受该采样率影响。 |
| `--idle-seconds` | `20.0` | 每个 workload window 前 matched control window 的目标时长。CPU、GPU、resource usage 以及启用时的 perf MIPS monitor 会按与 workload 相同的 `start()` / `stop()` 生命周期同时运行，但 control window 内不发送 `/predict` 请求。CPU baseline 为整段 RAPL 能耗 / 实际 duration；GPU baseline 为 NVML samples 的时间加权平均功率。case 结束后会复查该 case CSV 中所有有效 CPU/GPU baseline 的相对极差，达到或超过 5% 会输出 warning，实验继续运行。 |
| `--idle-cooldown-seconds` | `5.0` | 每个 workload window 采集 idle baseline 前的统一冷却等待时间。CPU-only 和 GPU+CPU case 都使用同一个值，避免上一轮推理刚结束后的短时热状态、Docker/server 收尾或 GPU clock/power 瞬态直接进入 idle baseline。 |
| `--idle-debug` | false | 开启 baseline 调试输出。主 CSV 会填充 GPU 的 `gpu_idle_measured_at` / `gpu_idle_rel_range_so_far` 和 CPU 的 `cpu_idle_measured_at` / `cpu_idle_rel_range_so_far`，并写出 `debug_idle_diag/result_case_*.csv.idle_diag.jsonl`。诊断文件记录 matched control window 的 GPU NVML trace、CPU RAPL 子窗口、host/container CPU delta，以及 control 结束后的 `nvidia-smi`、loadavg、top CPU processes、Docker 容器和 `docker stats` 快照。为避免诊断本身污染 baseline，逐进程 `/proc` 快照移到 control window 外，不再归入 RAPL control 能量。 |

#### 输入

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-scales` | auto | 手动覆盖 input scale 列表；未提供时通常自动规划 6 档，自定义音频清单按声明档数。 |
| `--workload-spec` | task default | workload 清单 JSON。ASR 默认使用仓库内置的 LibriSpeech 英文短音频清单；其他音频任务必须显式提供清单。 |

#### 计算分析器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--compute-profile-tool` | `none` | 默认跳过全部 compute probe；`both` 独立采集 `torch_profiler_eager` 逻辑 FLOP，并在 `gpu_mode=on` 时采集 NCU GPU 实际执行 FLOP。`auto` 是 `both` 的弃用别名；`torch`、`ncu`、`vendor` 用于单工具诊断或旧流程兼容。 |
| `--advisor-root` | auto | Host Intel Advisor install root or executable；显式值优先于自动检测。 |
| `--ncu-root` | auto | Host Nsight Compute install root or `ncu` executable；显式值优先于自动检测。 |
| `--advisor-repeat` | `20` | 旧 `vendor` CPU Advisor probe 的推理重复次数；最终 FLOP 会除回单 request。 |
| `--torch-profiler-repeat` | `1` | `torch_profiler_eager` probe 的推理重复次数；CPU/GPU 结果分别除回单 request。 |
| `--ncu-repeat` | `1` | NCU GPU probe 的推理重复次数；FLOP、kernel 数和 kernel 时间最终都除回单 request。 |
| `--compute-profile-cpus` | host logical CPUs | 临时 compute profiler container 的 CPU core cap。 |
| `--compute-profile-mem` | 75% host memory | 临时 compute profiler container 的 memory cap，单位 GB。 |
| `--keep-compute-profiles` | true | 保留 raw profiler artifacts；这是默认行为。artifact 位于模型结果目录的 `compute_profiles/`，路径不写入结果行。 |
| `--discard-compute-profiles` | false | 汇总完成后删除 raw profiler artifacts。 |

#### 执行分析器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--execution-profile-tool` | `none` | 显式启用高开销 execution profiler：`massif` 用于 CPU-only、`nsys` 用于 GPU，`both` 同时选择两者；默认 `none` 不运行。 |
| `--massif-sampling` | `per-scale` | `per-scale` 使用一个代表 CPU/内存逐 input scale 采集并复用；`full` 采完整 CPU × memory 矩阵。 |
| `--massif-reference-cpu` / `--massif-reference-mem` | 最大选中值 | Massif `per-scale` 的代表 CPU 与内存；必须存在于本次 `--cpus` / `--mems` 中。 |
| `--massif-repeat` | `1` | 每个 Massif probe 内执行的 inference 次数。Massif peak 仍是包含加载和预热的 process-lifetime peak，不按此值归一化。 |
| `--nsys-sampling` | `per-cpu-scale` | `per-cpu-scale` 保留全部 CPU、只用一个代表内存；`per-scale` 只用一个代表 CPU/内存；`full` 采完整矩阵。 |
| `--nsys-reference-cpu` / `--nsys-reference-mem` | 最大选中值 | Nsys 缩减采样的代表资源；`per-cpu-scale` 只使用代表内存，`per-scale` 同时使用两者。 |
| `--nsys-repeat` | `1` | 每个 Nsight Systems `acprof_compute` NVTX range 内的 inference 次数；time、count 和 bytes 汇总会除回单 request。 |
| `--nsys-root` | auto | Host Nsight Systems install root 或 `nsys` executable；显式值优先于自动检测。 |
| `--keep-execution-profiles` | true | 保留 `execution_profiles/` 下的 raw Massif `.out` 与 Nsight Systems `.nsys-rep`；这是默认行为。stats 导出的 `.sqlite` 缓存会自动删除。 |
| `--discard-execution-profiles` | false | 汇总成功后删除 raw execution-profiler artifacts，保留 plan、CSV 数值与错误诊断。 |

#### 输出与运行环境

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sniff-iface` | `docker0` | 本机 Docker 默认 bridge 对应的 `tcpdump` 抓包网卡。只有 daemon 改过 bridge 名时才覆盖。 |
| `--output-dir` | `results` | 输出根目录。最终还会追加 model name 子目录。 |
| `--skip-build` | false | 提前检查并优先复用本地模型镜像；不存在时提示并自动构建。 |
| `--notify` | `auto` | `auto` 在配置 Webhook 后启用企业微信；`none` 关闭，`wecom` 显式选择企业微信。配置见 [README](README.md#企业微信通知)。 |
| `--help` | — | 显示此入口的全部公开参数后退出。 |
| `--allow-cgroup-v1` | false | 仅用于旧主机诊断的兼容开关。默认正式模式要求 cgroup v2；启用后允许 v1，但会记录 `legacy_compatible`，且 memory peak/stat、I/O 操作数、PID、memory events 与 per-cgroup PSI 不具备同等口径。 |

结果目录存在异常中断留下的 `result_case_*.csv` 时，`run.py` 会先读取同目录 `static_meta.json/cgroup_version`。只有版本与当前 host 一致才允许续写；版本不同、缺失或元数据不可读时会退出，避免把 v1/v2 窗口合并到同一结果文件。

### 输入规模与音频清单

`input_scale` 是每个任务族的主输入尺度，语义由 `static_meta.json` 的 `input_scale_type` 决定：

| task family | `input_scale_type` | 含义 |
| --- | --- | --- |
| `nlp` | `seq_length` | 输入 token length。 |
| `cv` | `resolution_scale` | 图像基础尺寸的缩放倍率。 |
| `audio` | `duration_s` | 输入音频时长，单位秒。 |
| `timeseries` | `context_length` | 时间序列 context length。 |
| `diffusion` | `resolution_px` | 方形输出图像边长，单位像素。 |

未提供 `--input-scales` 时，当前内置 workload/legacy 配置通常会为一次 profiling run 规划 6 档 input scale；自定义音频清单则使用清单中声明的档数：

- `nlp` 会启动容器读取 tokenizer / handler 的可用最大输入长度，最后一档尽量贴近有效上限。
- `audio` 从 workload 清单读取默认尺度；内置英文 ASR 清单固定为 `1,2,5,10,20,30` 秒。`cv`、`timeseries` 仍根据各自 generator 的最大尺度或配置默认值生成。
- `diffusion` 使用固定的 `128,192,256,320,384,512` 像素输出边长；提示词、随机种子、guidance scale 和去噪步数在各尺度间保持不变。
- 同一次 run 的所有资源配置共用同一组 scale。
- 所有任务族都会把已确定尺度的 payload 写入唯一的 `input_scale_plan.json`；主采集与 compute profiler 共同读取该文件，保证实际执行 payload、FLOP profiling 和 CSV 中记录的 `input_scale` 一致。
- 手动传入 `--input-scales` 时以手动值为准；`nlp`、`audio`、`timeseries` 和 `diffusion` 会在 sweep 前验证合法性（文生图分辨率至少为 64 且必须是 8 的倍数）。

#### 真实音频 workload

`automatic-speech-recognition` 默认使用 `assets/audio/librispeech-clean-test-en-30s/source.json`。该清单引用 LibriSpeech `clean/test` 中同一说话人、同一章节的三条连续语音，按固定顺序拼接后截取前 30 秒；素材是单声道 16 kHz PCM16 WAV，许可证为 CC BY 4.0。每一档输入都从同一个 30 秒基准音频取前缀，不做逐档归一化、补全或循环。

音频请求采用 JSON 内的 Base64 WAV；handler 仍能读取历史 `audio_samples` 浮点数组。短音频模式会读取模型 feature extractor 的约束并拒绝超过 receptive field 的尺度。对于 Whisper，30 秒是音频 receptive field；当前 feature extractor 会把接受的短音频补齐为固定的 480,000 samples / 3,000 frames，`/scale_meta` 会显式记录这一点。`max_target_positions=448` 是解码器输出 token 上限，不是音频输入上限，因此框架不会把 latency 必须随 `duration_s` 单调增加作为正确性条件。

自定义素材时可复制内置 `source.json`，设置新的 `workload_id`，再修改相对素材路径、SHA256、provenance 和 inference 字段。自定义 provenance 可以描述单条录音或既有重采样/增益流程；运行时仍会严格验证派生 WAV 本身是单声道、16 kHz、PCM16 且哈希匹配。然后传入：

```bash
python run.py --model openai/whisper-large-v3 \
  --workload-spec /path/to/source.json
```

当前音频 request 只实现 `batch_size=1` 和 `short_form`。清单会拒绝非空的 `chunk_length_s` / `stride_length_s`；长音频 sequential/chunked 应使用独立 workload，不能通过把本清单尺度直接扩展到 30 秒以上来混测。

### `probe.py`

复用 `--model`、`--task`、`--task-family`、`--backend`、`--batch-size`、`--workload-spec`、
`--output-dir`、`--skip-build` 和 `--allow-cgroup-v1` 的参数及默认值。
资源列表与超时的用途如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--cpus` | `1,2,4,8` | 只选择列表中的最小 CPU。 |
| `--mems` | `2,4,8,16` | 从低到高实测的候选内存上限。 |
| `--gpus` | `off,on` | 包含 `off` 时优先 CPU-only，否则使用 `on`。 |
| `--input-scales` | 自动规划 | 只探测已确定尺度中的最大值。 |
| `--timeout-seconds` | 不设超时 | 单次探测请求的等待上限；显式值必须有限且大于 0。 |

`probe.py` 不接收 `run.py` 的 `--request-timeout-seconds`、warmup/repeat、能耗采样或 profiler 参数。
详细用法见 [README](README.md#先探测最大输入)。

### `profile.py`

位置参数 `result_dir` 是已完成的模型结果目录。操作和恢复规则见
[README 补采说明](README.md#补采已有结果)。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tools` | `torch,ncu,nsys,massif` | 选择需要补齐的工具，逗号分隔。 |
| `--dry-run` | 关闭 | 验证已有文件并展示计划，不启动分析器或改写结果。 |
| `--force-reprofile` | 关闭 | 强制重采并替换所选工具已经成功的值。 |
| `--massif-sampling` | `per-scale` | 可选 `per-scale`、`full`。 |
| `--nsys-sampling` | `per-cpu-scale` | 可选 `per-cpu-scale`、`per-scale`、`full`。 |
| `--massif-reference-cpu` / `--massif-reference-mem` | 结果矩阵最大值 | 在已有 CPU-only 资源配置中选择代表值。 |
| `--nsys-reference-cpu` / `--nsys-reference-mem` | 结果矩阵最大值 | 在已有 GPU 资源配置中选择代表值；`per-cpu-scale` 只用代表内存。 |
| `--torch-profiler-repeat` / `--torch-repeat` | `1` | 同一参数的两个名称；控制 Torch probe 内推理次数。 |
| `--ncu-repeat` / `--nsys-repeat` / `--massif-repeat` | `1` | 对应工具的 probe 内推理次数，归一化口径同 `run.py`。 |
| `--ncu-root` / `--nsys-root` | 自动检测 | Host 工具安装目录或可执行文件。 |
| `--compute-profile-cpus` / `--compute-profile-mem` | host 逻辑 CPU / 75% host memory | 临时 compute profiler 的 CPU/内存上限，内存单位 GB。 |

### 其他入口

`plot.py` 接收结果 CSV 路径，`tui.py` 可用 `--model` 预填模型、用 `--preset` 选择预设。
各入口的完整帮助可直接运行：

```bash
.venv/bin/python run.py --help
.venv/bin/python probe.py --help
.venv/bin/python profile.py --help
.venv/bin/python plot.py --help
.venv/bin/python tui.py --help
```

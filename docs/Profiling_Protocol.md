# 采集协议与实验产物

修改请求窗口、输入计划、冷启动或产物来源时查阅。字段计算见 [指标](Metrics.md)，运行步骤见 [profiling 流程](../.agents/skills/acprof-profiling-workflow/SKILL.md)。

[文档导航](README.md)

## 采集生命周期

根 CLI 先完成环境与任务预检、镜像准备、输入计划和独立接口验证，再运行所选 profiler 与资源矩阵。
每个正式 case 创建新容器；client 控制已有预热、冷却、无请求对照与 workload 窗口，
监控停止后才计算派生值和写行。case 产物经抓包解析与校验后合并；图表及通知属于测量之外的操作。
模块顺序见[主机编排](Architecture.md#主机编排与测量)，镜像与独立验证见[运行兼容](Runtime_Compatibility.md#构建复用和验证)。

独立接口验证可能预热宿主机文件缓存。冷启动描述全新容器的进程和模型初始化，
不承诺磁盘冷缓存；`cold_start_first_predict_app_s` 不计入 `/ready` 前的分段和，也不新增推理请求。

## 协议不变量

- CPU package、估算 vCPU、GPU device 与 container-attributed 数据分别命名和解释，整机测量不能静默替换容器归因。
- 单请求、workload window、进程生命周期、独立 profiler 使用不同窗口与分母；比较前核对请求数、尺度单位及来源资源。
- 数值 `0`、CSV `nan`、JSON `null` 和字段缺失含义不同；未知历史数据不补成零，推导或代表资源复用不冒充独立实测。
- JSON 保留原生类型；schema 版本由协议与兼容策略决定。历史文件缺少可选字段时，消费者按相应约定降级。
- 输入计划、`input_scale_plan_sha256`、workload 素材与模型 revision 必须一致；派生字段复用已有采样，不额外发起请求。
- 静态对象描述与采集过程来源分开保存；补采和修复记录进入 `collection_history.json`，保留原始备份和可回溯信息。

## 输出文件

输出目录为 `<output-dir>/<model-dir>/`；模型 ID 中的 `/` 替换为 `--`。
下表列出可能生成的文件；probe、补采、调试与绘图产物仅在执行对应操作时出现。

| 文件 | 说明 |
| --- | --- |
| `result_case_*.csv` | 采集期间逐资源配置写入的可恢复中间结果；成功合并后清理。 |
| `result_all.csv` | 动态测量结果。每一行对应一个 resource config、一个 input scale、一次 warmup/repeat iteration，并记录归一化指标、PCAP 网络字节、cold-start phases，以及该窗口的 cgroup memory/stat/PID、swap、块 I/O 与压力/事件。 |
| `run_state.json` | 主实验状态 schema v1，记录实验 ID、参数、主机与源码/依赖指纹、绑定的镜像和输入计划、case 完成状态与 CSV SHA256、启动/恢复记录、最终完成状态。 |
| `interrupted_cases/` | 恢复时保存中断 case 的原始 CSV、PCAP 与关联 sidecar；备份完成后才开始该 case 的新测量。 |
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

### 结果完整性与断点续跑

主实验默认拒绝已有实验产物的目录，避免重写静态元数据或追加上次遗留的测量行。
重新执行原命令并加 `--resume`，或在 TUI 高级参数勾选“恢复未完成实验”，可以恢复由当前
状态协议创建的实验。参数与输出根目录保持一致；`--notify`、`--skip-build` 不影响测量身份。
只有 probe 产物的目录仍可用于首次正式采集。旧实验没有 `run_state.json` 时继续支持读取、
绘图及补采，但不能仅凭残留 CSV 推断恢复状态；新的主实验须使用另一输出目录。

恢复检查主机、Python/依赖、AC-Prof 源码及继承的测量环境参数。已经完成准备的实验直接
使用保存的模型 revision、不可变 image ID、输入计划与 profiler 汇总，不重新查询 Hub、构建
镜像或生成另一组输入。原镜像必须存在。输入计划、静态元数据及 profiler 计划的 hash
不匹配时退出；准备阶段尚未完成、尚无 case 时可重新准备。

case 只有在抓包回填和原有校验结束，且其测量唯一键完整匹配计划后才记为完成。唯一键为
`CPU × memory × GPU mode × input scale × warmup × repeat_idx`。恢复会复用校验通过的完整
case；中断 case 先备份，再创建新容器，重新执行该 case 的原 warmup/repeat 协议。已记录完整
错误占位的 OOM/timeout case 也属于已完成的尝试，不在恢复时自动改变超时或重测条件。
OOM pruning 继续按原有参考 CPU/内存顺序重建证据，复用与推断不会增加正式请求。

合并拒绝缺失/空 case、重复文件、重复测量、截断行和与计划不符的行；保留历史扩展列，
历史缺失的可选指标保持 `nan`。全部校验通过后，在同一目录写临时文件并 flush/fsync，
再用原子替换发布 `result_all.csv`。发布前发生写入错误时保留已有最终文件和 case 产物。
完成状态先持久化，再清理中间文件。`status=complete` 表示计划已执行并完成合并，
`outcome=partial` 表示其中包含错误行，两者不能等同于全部测量成功。

`.acprof-result.lock` 是采集和补采共享的目录锁；进程退出自动释放锁，锁文件本身可以保留。
系统临时目录中的 `acprof-measurement-<uid>.lock` 还会串行化本机同一用户发起的实验，
防止不同输出目录争用固定端口、容器名称和主机计数器；它不协调其它用户或外部负载。
`--resume` 对已完成实验只检查结果结构并报告完成，不重新测量或覆盖后续补采的指标。
所有状态、备份与校验操作均在准备阶段、case 边界或最终发布阶段执行。

### `static_meta.json` 字段

`static_meta.json` 是一个 model/image/run-level JSON object，只保存描述实验对象、环境和最终 profiling 配置/口径的相对稳定信息。补采、重试、回填与备份过程记录放在独立的 `collection_history.json`。数组、布尔值、数字和 `null` 均保留 JSON 原生类型，不再编码成 CSV 字符串。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | `static_meta.json` schema 版本；新增运行环境绑定与验证记录后的当前版本为 `7`。 |
| `model_name` | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `model_revision` | 实际解析到的 model revision / commit hash。 |
| `parameter_count` | Hugging Face Hub SafeTensors metadata 的参数总数；Hub 未提供时为 `null`。 |
| `parameter_bytes` | 根据 `parameter_dtype_counts` 的各 dtype 元素数量与字节宽度精确求和得到的逻辑 tensor payload 大小，不含序列化 header；没有 dtype 统计或存在未知 dtype 时为 `null`。 |
| `precision_dtype` | SafeTensors 参数中数量占主导的权重精度，例如 `FP32`、`FP16`、`BF16`、`INT8`；无法确认时为 `null`。 |
| `parameter_dtype_counts` | 按 dtype 统计的参数/张量元素数量，保留混合精度与少量整型 buffer 信息。 |
| `inference_precision_by_device` | 当前 handler 明确请求的 CPU/GPU 推理精度。通常 Transformers NLP/CV/audio 为 CPU FP32、GPU FP16；MOSS adapter 为 CPU FP32、GPU BF16；Encodec/DAC 两者均 FP32。TorchScript 保留导出权重精度、输入 FP32；skops 保留 estimator 内部精度、输入 FP32。Silero 和 skops 仅声明 CPU。 |
| `static_flops` | Torch eager profiler 得到的逻辑 shape FLOPs，按 `input_scale` 保存 `flops_per_request`；未采集成功时为 `null`。 |
| `static_macs` | 静态 MACs。当前不做不可靠的 FLOPs/2 推断，因此未单独采集时为 `null`。 |
| `input_format` | 实际 `/predict` HTTP JSON 输入协议及其 JSON Schema。 |
| `output_format` | 实际 `/predict` HTTP JSON 响应协议及其 JSON Schema。 |
| `quantized` | 是否检测到量化配置、量化 tag 或量化权重 dtype；无法确认时为 `null`。 |
| `quantization_method` | 量化方法，例如 `gptq`、`awq`；不适用或未知时为 `null`。 |
| `quantization_config` | Hub model config 中的完整量化配置；没有时为空 object。 |
| `model_license` | Hugging Face model card 许可证，例如 `apache-2.0`、`mit`；无法确认时为 `null`。 |
| `model_metadata_source` | 参数量、参数 payload、精度、量化和许可证的元数据来源，当前在线 Hub 检测成功时为 `huggingface_hub`。 |
| `task_family` | 任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`、`multimodal`、`structured`。 |
| `pipeline_tag` | Hugging Face pipeline tag，例如 `fill-mask`、`image-classification`。 |
| `runtime_backend` | 容器内使用的 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`。 |
| `image_tag` | 本次传给 Docker 的镜像引用；v7 新采集使用不可变 `sha256:` image ID，历史文件可能为可变 tag。 |
| `image_id` | 经 Docker inspect 核验的不可变镜像 ID；补采优先使用该字段。历史文件无法确认时不补造。 |
| `image_name` | 便于查看的构建标签，含模型名和构建指纹前缀；执行仍使用 `image_id`。 |
| `runtime_environment` | 镜像内生成的环境清单：profile、adapter、构建指纹、模型及实际 snapshot revision、Python 和已安装包版本、依赖锁与包清单 SHA256、自定义 Python 源码 SHA256，以及可选的 `model_download` 文件选择与完整性清单。普通包版本是实测清单；当前默认 profile 均记录完整依赖锁，历史或自定义未锁定环境的 `dependency_lock_sha256` 可以为空。 |
| `runtime_validation` | 独立容器验证报告。保存实际 image ID、输入尺度／payload SHA256、每个设备的状态、dtype、attention 实现及输出摘要。验证推理接口，不替代 profiler 兼容性检查，也不计入请求或性能测量。失败的完整报告另见 `runtime_validation.json`。 |
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
| `vcpu_power_method` | estimated vCPU 功耗计算方法。`rapl_cgroup_cpu_share` 表示对 RAPL package energy 逐采样区间按 container cgroup CPU share 归因；`unavailable` 表示无法估算。 |
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

`schema_version=3` 的历史文件使用 `model_weight_bytes` 表示上述完整 cache artifacts 大小；v4 以 `model_cache_bytes` 替代该旧字段；v5 新增 host swap 字段；v6 新增 cgroup 版本与采集模式；v7 新增运行环境和验证记录。历史文件不会自动伪造当时的 swap、cgroup 或依赖环境；无法回溯的值应保持未知，补录时在 `collection_history.json` 记录来源。

`runtime_environment.model_download` 是可选的独立 schema v1 清单，历史结果可缺失。`requested_policy` 保存 `auto/full`，`effective_policy` 保存实际 `selected/full`，`reason` 说明筛选或回退原因；`weights` 记录组件、格式、variant 和索引／分片文件。`files` 保存路径、实际逻辑大小和构建时计算的 SHA256，另保留 Hub 提供的 Git blob／LFS 标识；`excluded_files` 是未下载文件的远端元数据。`verification=sha256` 表示构建阶段已完成完整性检查，`plan_sha256` 校验规范化 JSON（不含自身字段）。`selected_bytes` 按清单路径求和，不对相同内容的多个路径去重，因此不等同于 `model_cache_bytes`、镜像大小或释放的磁盘空间。新增清单不改变 CSV 字段和历史指标定义。

`runtime_validation.json` 使用独立 schema v1：`devices.off/on` 分别保存 CPU／GPU 的 `ok`、`error` 或明确 cgroup OOM 的 `resource_limit`；总状态为 `ok`、`error` 或 `resource_limited`。每个模式只执行一次最小计划输入，资源上限为本次配置的最大 CPU／内存。错误会在矩阵之前退出；资源限制允许正式矩阵继续测定 OOM 边界。stdout/stderr 保存在 `runtime_validation_off/on.log`，超时也清理验证容器。它们不是 warmup、测量行或 profiler 结果。验证前已有的结果不因此变为本次成功结果。

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
`collection_history.json`，结果不包含 idle、能耗或网络测量。用法见 [README](../README.md#先探测最大输入)。

## 冷启动

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
               + input_scale_planning_s + runtime_validation_s + compute_profile_s
               + execution_profile_s + main_collection_s
```

auto warmup 约为每个 case/scale 的 `5 × 平均请求延迟`，长耗时模型应单独计入。
镜像复用、失败 case、启动 OOM 剪枝和续跑会改变实际工作量；启用通知和日志收尾也有额外耗时。
内存/PID 峰值、cgroup 增量、冷启动分解和派生能效使用已有采样路径，不增加 `/predict` 数量；
网络字节复用已有 PCAP，但仍需离线解析时间。

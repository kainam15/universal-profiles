# 指标与结果分析

查字段、分析过滤、历史 CSV 兼容或图表时查阅。能耗详见 [能耗测量](Energy_Measurement.md)，独立工具详见 [Profiler](Profilers.md)，产物结构见 [采集协议](Profiling_Protocol.md)。

[文档导航](README.md)

## result_all.csv 字段解释

每行对应一个资源配置、一个 input scale 和一次 warmup/repeat 请求窗口。
常规性能分析只取 `status=ok` 且 `warmup=0`；错误行中的部分数值不作为正式测量。
不可用或不适用的数值为 `nan`，历史 CSV 缺少新字段时不能补成 `0`。

字段按用途分组，实际列顺序以 [acprof/config.py](../acprof/config.py) 的 `CSV_FIELDS` 为准。
`*_per_request` 和能量列按窗口内请求数归一化；`*_delta` 若未注明归一化，则表示整个窗口的增量。

| 查阅方向 | 字段组 |
| --- | --- |
| 配置、输入与请求 | [资源配置、输入与网络](#资源配置输入与网络)、[延迟与吞吐](#延迟与吞吐)、[两种延迟的区别](#latency_s-和-latency_app_s-的区别) |
| Profiler | [Torch 与 NCU](Profilers.md#torch-与-ncu-计算指标)、[Massif 与 Nsight Systems](Profilers.md#massif-与-nsight-systems-执行指标) |
| 能耗与归一化 | [GPU](Energy_Measurement.md#gpu-功率与能耗)、[CPU package](Energy_Measurement.md#cpu-package-功率与能耗)、[估算 vCPU](Energy_Measurement.md#估算-vcpu-能耗与派生能效)、[像素口径](#像素归一化口径) |
| 资源与 PMU | [CPU](#cpu-资源频率与-pmu)、[容器内存、swap、I/O 与 PID](#容器内存swapio-与-pid)、[GPU 资源](#gpu-资源与运行状态) |
| 生命周期与失败 | [冷启动](Profiling_Protocol.md#冷启动)、[运行状态与错误](#运行状态与错误) |

### 资源配置、输入与网络

| 字段 | 含义 |
| --- | --- |
| `cpu_cores` | 当前 Docker container 的 CPU core 限制，来自 `--cpus`。 |
| `mem_cap_gb` | 当前 Docker container 的 memory cap，单位 GB，来自 `--mems`。 |
| `gpu_mode` | `on` 或 `off`。`on` 表示容器使用 Docker `--gpus all`。 |
| `input_scale` | 本行实际执行的主输入尺度。语义见 `static_meta.json/input_scale_type`。 |
| `input_units_per_request` | `effective input_scale × batch_size`。保留历史尺度语义：NLP/时序通常是 token/context step，Audio 是秒，CV 是缩放倍率，分辨率型 Diffusion/多模态是边长。这些图像尺度都不是像素总数。 |
| `input_num_samples` | 音频 payload 的实际采样点数；其他任务或旧计划无法推导时为 `nan`。它是诊断字段，音频主尺度仍为 `duration_s`。 |
| `input_pixels_per_request` | CV/多模态每请求输入像素总数，单位 pixel/request。由物化计划中的宽高计算；相同尺寸视频为 `batch_size × width × height × frame_count`，多模态中同时存在的图像与视频像素相加。不乘颜色通道数，指 processor 处理前的素材尺寸。 |
| `output_pixels_per_request` | Diffusion 每请求输出像素总数，单位 pixel/request。单张方形图像为 `batch_size × resolution_px²`；视频为 `batch_size × output_pixel_count_per_video`，帧数只计一次。来自输入计划中受 handler 尺寸协议约束的输出几何，不新增采样。不适用或不能确认几何时为 `nan`。 |
| `request_payload_bytes` | `requests` 实际发送的 prepared HTTP JSON body 字节数，在同一 workload window 内取平均。 |
| `packet_request_wire_bytes_per_request` / `packet_response_wire_bytes_per_request` | 从同一 PCAP 中属于该 `/predict` TCP stream 的 client→server / server→client captured `frame.len` 总和，再对本行请求求平均。客户端使用 `Connection: close`，因此每个 stream 对应一个请求；握手、ACK、关闭包和重传都保留。 |
| `packet_total_wire_bytes_per_request` | 上述请求与响应 captured frame bytes 之和。它不含 capture 未保留的 Ethernet FCS，也不包含物理层 preamble / inter-frame gap，不能直接当作插座侧链路能耗输入。 |
| `packet_tcp_payload_bytes_per_request` | 同一 TCP stream 内 `tcp.len` 的总和，再对本行请求求平均；重传 payload 会按实际捕获次数计入。 |
| `packet_protocol_overhead_bytes_per_request` | `packet_total_wire_bytes_per_request - packet_tcp_payload_bytes_per_request`，表示捕获到的 L2/L3/L4 header、ACK/握手/关闭等开销；不拆分 TCP payload 内的 HTTP header 与 JSON body。 |
| `packet_protocol_overhead_ratio` | 本行所有请求的 protocol overhead bytes 总和 / total wire bytes 总和；分母无效时为 `nan`。 |
| `task_param` | 本行 payload 真正发送给 handler 的二级参数，使用稳定排序的 JSON 字符串；通常来自 `params`，时序任务同时记录顶层 `prediction_length`，不再记录未执行的任务族默认值。 |
| `output_length_avg` | 同一 workload window 内响应长度的平均值：文字任务为该请求所有返回文本的 Unicode 字符数之和（不是 UTF-8 字节）；Diffusers 图像生成为图像数，视频生成为总帧数；检索等不适用任务为 `nan`。 |
| `output_token_count_avg` | 同一 workload window 内每次响应文本 tokenizer token 数的平均值；ASR/图像描述按输出文本重新分词，`add_special_tokens=False`。图像描述先逐条分词再求和，不拼接 caption，也不等于实际生成 token ID 数或解码步数。tokenizer 不可用或统计失败时响应为 `null`；窗口内全部不可得时 CSV 为 `nan`，有效空文本为 `0`。 |
| `repeat_idx` | 当前 warmup 或 repeat phase 内的 0-based iteration index。 |
| `warmup` | `1` 表示 warmup 行，`0` 表示正式测量行。`plot.py` 默认排除 warmup 行。 |
| `repeat_in_window` | 本行内部连续发送的 request 数量。`latency_app_s` 和 `latency_s` 都是该 window 内 request 的平均值。 |

### 延迟与吞吐

| 字段 | 含义 |
| --- | --- |
| `latency_s` | packet-level latency，来自 `tcpdump` PCAP + `tshark` 解析 + `acprof.packet.merge_packet_latency` merge。当前默认要求该字段完整；抓包不可用、PCAP 为空、解析为空或 merge 后仍有缺失时，程序会退出并给出恢复提示。 |
| `latency_s_per_input_unit` | `latency_s / input_units_per_request`，在 packet merge 阶段更新。 |
| `latency_s_per_input_megapixel` / `latency_s_per_output_megapixel` | `latency_s × 1,000,000 / 对应的 pixels_per_request`，单位 s/Mpixel；沿用 packet 请求响应窗口，在 packet merge 后更新。 |
| `latency_request_count` | 本行实际合并到 packet-level latency 分布中的有效 request 数。可与 `repeat_in_window` 对照检查抓包完整性。 |
| `latency_p50_s` / `latency_p90_s` / `latency_p95_s` | packet-level latency 在本行 request window 内的 empirical nearest-rank 分位数，由 merge 阶段从同一组 packet latency 明细计算。 |
| `latency_std_s` / `latency_cv` / `latency_iqr_s` / `latency_max_s` | 同一 packet-level request window 的总体标准差、变异系数 `std / mean`、nearest-rank `P75 - P25` 和最大值。少于 2 个有效 request 时，std、CV 和 IQR 为 `nan`；max 仍保留。 |
| `latency_slow_ratio` | packet-level latency 中超过 `SLOW_LATENCY_THRESHOLD_S` 的 request 比例，默认阈值为 `0.06` 秒，用于观察尾延迟或双峰分布。 |
| `latency_app_s` | host-side application latency。`acprof.host.client` 用 `requests.post()` 外层 `time.perf_counter()` 测得，通常比 `latency_s` 更容易稳定产出。 |
| `latency_app_s_per_input_unit` | `latency_app_s / input_units_per_request`。 |
| `latency_app_s_per_input_megapixel` / `latency_app_s_per_output_megapixel` | `latency_app_s × 1,000,000 / 对应的 pixels_per_request`，单位 s/Mpixel；包含原 application 请求的预处理、推理、后处理和传输，不是仅模型算子的耗时。 |
| `latency_app_request_count` | 本行 application latency 分布中的有效 request 数；正常成功窗口通常等于 `repeat_in_window`。 |
| `latency_app_p50_s` / `latency_app_p90_s` / `latency_app_p95_s` | host-side application latency 在本行 request window 内的 empirical nearest-rank 分位数。 |
| `latency_app_std_s` / `latency_app_cv` / `latency_app_iqr_s` / `latency_app_max_s` | application latency 的总体标准差、变异系数、nearest-rank IQR 和最大值；少于 2 个有效 request 时 std、CV 和 IQR 为 `nan`。 |
| `latency_app_slow_ratio` | host-side application latency 中超过 `SLOW_LATENCY_THRESHOLD_S` 的 request 比例，默认阈值为 `0.06` 秒。 |
| `throughput_samples_per_s` | 吞吐量，约等于 `batch_size / latency`。如果 `latency_s` 成功 merge，会优先按 `latency_s` 更新；否则按 `latency_app_s` 计算。 |
| `throughput_samples_per_s_per_cpu_core` | `throughput_samples_per_s / cpu_cores`；packet latency merge 后会与 throughput 一起重算。它表示按配置 CPU quota 归一化的吞吐，不是实际 CPU utilization 归一化值。 |

### 像素归一化口径

`Mpixel` 表示一百万像素。新字段使用每百万像素单位，避免 CSV 六位小数把很小的
每像素延迟舍入为零。所有分子已经平均到单 request；分母计入该 request 的完整 batch，
不再乘除 `repeat_in_window`。绘图先对每行计算比值，再沿用同一资源配置和尺度内的
mean/median 聚合，默认仅纳入 `status=ok` 且 `warmup=0` 的行。

例如 batch=2、输出为 128×128 时，每请求有 32768 像素；若能耗为 327.68 J，
则为 10000 J/Mpixel。输出改为 256×256、能耗增为 1310.72 J 时，仍为
10000 J/Mpixel。旧字段 `E / (batch_size × resolution_px)` 会从 1.28 升至 2.56，
因此它的上升不能用于判断每像素能效变差。

像素归一化使用 2 个像素计数和 6 个比值字段；旧 partial case 缺少这些
可选列时仍可保留原测量行并补写失败记录，新列填 `nan`。静态 schema 保持 v7，
输入计划保持兼容的 v2，多模态视频计划另行记录 `video_frame_width / video_frame_height`。
实时采集在 monitor 停止后从物化计划计算像素数；若服务返回的有效尺度与计划不一致，
像素计数保持 `nan`。packet 回填只更新 packet 像素延迟，不改变 application 延迟或能耗。

历史文件读取时优先使用 CSV 已有的显式像素计数；没有计数列或列为空/`nan` 时，可按同目录输入计划的
精确尺度匹配宽高/像素数，并结合 batch 元数据派生。旧多模态视频也可使用该计划对应的
`workload.fixed_media.image_resolution` 与已记录帧数。有 `input_scale_plan_sha256`
时必须匹配，同时核对计划中的模型和任务族；不匹配或损坏时警告并跳过计划派生。
旧 schema 未记录 hash 时仅使用同目录、模型/任务族未冲突的计划，不补造 hash。
缺少计划、对应尺度、batch 或尺寸时保持未知；不根据模型名、边长或默认 224 像素猜测。
读取时从原始能耗/延迟和有效像素数重新计算比值，避免使用回填前的过期派生值。
整个过程不改写历史 CSV、静态元数据、输入计划及其 hash。

输入像素数不等同于模型实际处理的 patch/token 数或 FLOP：processor 可能缩放、切块或
固定尺寸。像素指标描述给定工作负载的成本，比较时仍需核对模型、推理步数、精度和
输出质量；token、音频秒数、context step、去噪步数等尺度不适用像素面积归一化。

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

### 运行状态与错误

| 字段 | 含义 |
| --- | --- |
| `status` | `ok`、`warn` 或 `error`。`warn` 标记存在异常的行；常规性能图与延迟模型只使用 `ok` 行，资源可行性图保留失败状态。 |
| `error` | 错误或 warning 文本。正常行为空；`status=error` 时强制非空。请求超时会区分实际发出但未完成的请求（`client_request_timeout`）与未发请求、因前序超时跳过的计划行（`not_measured_after_timeout`），并记录触发尺度、timeout 下界、请求阶段和 request ID。 |

### `latency_s` 和 `latency_app_s` 的区别

- `latency_app_s` 是 client 侧应用层计时，只要 `/predict` 请求成功，一般就能写出。
- `latency_s` 是 packet-level 计时，需要完整完成 `tcpdump` capture、`acprof.packet.sniff_parse_pcap` parse、`acprof.packet.merge_packet_latency` merge。
- 当前默认行为是严格模式：如果无法保证 `latency_s` 有值，`run.py` 会退出，不继续 merge 最终结果。

## 图表与延迟拟合产物

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

13 张通用总览图统一使用 input scale 横轴、配置颜色和共享图例；每张最多 6 个子图。只有单位、语义和数值尺度都适合直接比较的子图才共享纵轴，例如 packet/application latency、NCU application/packet MFLOPS 以及 cache/dTLB 的同类比率。cache miss 和 dTLB miss 的单 request 计数可能相差多个数量级，因此使用独立纵轴。某个指标没有数据时，对应位置显示 `No data`；整张总览图的全部指标都没有数据时才跳过该 PNG。延迟总览使用 `3×2` 布局，同时展示 packet/application 的原始延迟、归一化延迟和 CV；service efficiency 总览集中展示吞吐、每 CPU core 吞吐、container-attributed energy、归一化能耗和 samples/J。分辨率类型的归一化子图使用每百万像素指标（Diffusion 为输出像素，CV/多模态为输入像素），其他尺度仍使用原来的每 input unit 指标。分辨率图缺少可靠像素数时显示 `No data`，不退回边长分母。

通用总览图和 energy/power 总览图的 `Configuration` 图例按运行模式、CPU 核数分列：`GPU+CPU1` 表示启用 GPU、CPU 配额为 1 核，`CPU1` 表示仅使用 CPU、配额为 1 核；列内的 `Mem2`、`Mem4` 等对应 `mem_cap_gb`，按数值从小到大排列并对齐。GPU 组在前，CPU-only 组在后，各组 CPU 核数递增；本图没有有效数据的配置留空，不生成额外曲线。宽图最多并排 8 组，单列子图最多并排 4 组，更多配置整组换行；画布按实际图例高度预留空间，避免遮挡标题、idle 说明或子图。曲线颜色和聚合口径保持原有规则。

三张 energy/power 总览图分别对应 GPU board、CPU package 和 estimated vCPU-attributed 口径。每张 PNG 使用 `3×2` 子图：三行依次为 energy/request、average power 和 peak power，左列展示扣除 idle baseline 的 effective 指标，右列展示保留 idle baseline 的 total 指标；同行共享纵轴，所有子图共享 input-scale 横轴、配置颜色与图例。图例下方的灰色信息框同时给出 idle 平均功率和最大的 case 内相对极差 `(max-min)/mean`；CPU package 按 CPU-only / GPU-enabled 分开，estimated vCPU 则标明由 CPU package baseline 按 interval CPU share 归因，避免把估算值误解为独立实测。这样可以直接观察 idle 对各指标的影响，不再单独生成 effective 或 total PNG。CPU package 来自 RAPL，estimated vCPU 则按 container cgroup CPU share 估算，两者不能混作同一测量口径。

`cpu_memory_behavior_overview_vs_scale.png` 使用 `2×2` 布局汇总 Linux `perf` generic PMU event，只表示 cache / dTLB miss 的单 request 计数和 miss rate，用于观察访存局部性及地址转换开销；它们不是 DRAM read/write traffic，也不是实际内存带宽 GB/s。不同 CPU 架构、虚拟化环境或 kernel PMU 可能不提供相同事件；部分字段不可用时仍会保留其余可用子图。

`container_io_overview_vs_scale.png` 的 block read/write operation 数可能相差多个数量级，因此两个子图分别按各自数据自动缩放纵轴，并各自标注 `Operations/request`。比较读写数量时应读取刻度值，不能直接比较曲线高度；零值和原始聚合数值保持不变。

`container_memory_process_overview_vs_scale.png` 中 GPU 曲线统一使用绿色，按本图 GPU 配置的 `mem_cap_gb` 从小到大由浅变深；相同 memory cap 的不同 CPU 配置同色。六个子图和配置图例使用同一映射，适用于 `gpu/` 和 `gpu+cpu/`。CPU-only 曲线沿用 CPU 色相与 memory cap 深浅，每个资源配置仍独立聚合和绘制。

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

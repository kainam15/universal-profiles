# 计算与执行分析器

选择 Torch、NCU、Massif、Nsys，解释工具指标或采样来源时查阅。补采操作见 [README](../README.md#补采已有结果)，参数见 [CLI 参考](CLI_Reference.md#计算分析器)。

[文档导航](README.md)

## 选择性能分析器

[计算分析器](#计算分析器--compute-profile-tool) · [执行分析器](#执行分析器--execution-profile-tool) · [补采已有结果](../README.md#补采已有结果) · [从计划生成派生 CSV](../README.md#从已有计划生成派生-csv)

### 计算分析器：`--compute-profile-tool`

FLOP profiling 和主 latency / energy workload 相互独立：

| 选项 | 采集内容 | 适合场景 |
| --- | --- | --- |
| `--compute-profile-tool none` | 不运行 compute probe | 默认；smoke test、先完成主矩阵 |
| `torch` | Torch eager 逻辑 FLOP | 只关心模型算子形状对应的理论工作量 |
| `ncu` | GPU 实际执行的 Tensor / Scalar FLOP | 单独诊断 NVIDIA GPU |
| `both` | Torch eager；GPU 行再运行 NCU | 显式启用完整采集 |

Torch probe 会强制并验证 eager attention，正式请求仍使用正常运行时的 attention 实现。
NCU 需要主机上的 `ncu` 和可用的 GPU 性能计数器，可用 `--ncu-root` 指定工具位置。

### 执行分析器：`--execution-profile-tool`

Execution profiling 用于分析内存峰值与执行时间线，默认关闭，与 FLOP profiling 独立选择。
显式启用后，在主 latency / energy 矩阵开始前运行独立分析探针：

| 选项 | 采集内容 | 适合场景 |
| --- | --- | --- |
| `--execution-profile-tool none` | 不运行 execution probe | 默认；smoke test、先完成主矩阵 |
| `massif` | Valgrind Massif 的 heap、heap extra、stack 峰值，以及三者总量峰值和出现时间 | 分析 CPU-only 模型进程的内存占用 |
| `nsys` | Nsight Systems 的推理窗口 host wall time，以及 CUDA API、GPU kernel、memcpy 时间线与汇总 | 分析 NVIDIA GPU 推理耗时、调用次数和数据搬运 |
| `both` | CPU-only 行采 Massif，GPU 行采 Nsys | 同时分析 CPU 内存与 GPU 执行行为 |

工具按本次 `--gpus` 选择的模式生效：Massif 只用于 `off`，Nsys 只用于 `on`；
`both` 在 `--gpus off,on` 时才会运行两种工具。

Massif 的内存峰值覆盖模型加载、预热和推理的整个进程生命周期。
`--massif-repeat` 默认 `1`，只控制探针内的推理次数，峰值不按次数平均；
解读时应与正式测量窗口的容器内存指标区分。

Nsys 在预热后的推理窗口采集，时间、调用次数及 memcpy 字节数按
`--nsys-repeat`（默认 `1`）归一化为单 request。
CUDA API、kernel 和 memcpy 时间各自是活动时长之和，活动可能重叠，不能相加当作请求总延迟。
完整字段口径见 [Massif 与 Nsight Systems 执行指标](#massif-与-nsight-systems-执行指标)。

显式启用后默认采用缩减采样，并把来源资源与复用策略记录到 plan 与静态元数据：

- Massif 默认 `--massif-sampling per-scale`：最大 CPU × 最大内存 × 每个 input scale，结果复用到其他 CPU-only 资源配置。
- Nsys 默认 `--nsys-sampling per-cpu-scale`：全部 CPU × 最大内存 × 每个 input scale，结果复用到相同 CPU、不同内存上限的其他 GPU 结果行。
- Nsys 也支持 `--nsys-sampling per-scale`：只采一个代表 CPU/内存 × 每个 input scale，结果复用到其他 GPU 资源配置。
- 两者均支持 `full`：在各自适用的 GPU 模式下，逐 CPU × 内存 × input scale 采集。

锁定环境的模型镜像预装 Valgrind 和 Nsys 运行库；分析直接使用原模型的不可变 image ID。
缺少对应能力标记时立即报错，要求用当前构建流程重建；不再为旧模型按需派生兼容镜像。
Nsys 主程序仍从宿主机挂载，可用 `--nsys-root` 指定；host 无需安装 Valgrind。
两个工具只在独立分析探针中运行。

需要严格采完整资源矩阵时显式传入：

```bash
python run.py --model google-bert/bert-base-uncased \
  --gpus off,on \
  --execution-profile-tool both \
  --massif-sampling full --nsys-sampling full
```

代表资源默认取本次 `--cpus` / `--mems` 中的最大值，也可用
`--massif-reference-cpu`、`--massif-reference-mem`、
`--nsys-reference-cpu`、`--nsys-reference-mem` 显式选择，取值必须在本次资源矩阵中；
Nsys 的 `per-cpu-scale` 只使用代表内存，`per-scale` 同时使用代表 CPU 和内存。

## Torch 与 NCU 计算指标

NCU 仅选择当前 SASS 和浮点 Tensor counters，不使用旧 `flop_count_*`。查询失败会报告原因，
不会猜测默认 counter 列表后继续。

Torch eager 记录模型逻辑计算量，NCU 记录 GPU 实际执行量；两者分别使用
`*_torch_profiler_eager` 和 `*_ncu` 列。历史 plan / CSV 中的五个通用 compute 字段
仍可由绘图和迁移入口读取，新结果不再重复写入。
从已有计划生成派生 CSV 时，匹配键为 GPU mode 与 input scale（绝对容差 `1e-6`）；
未匹配或失败的工具字段保持 `nan`，对应错误列记录原因。操作见
[README](../README.md#从已有计划生成派生-csv)。

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

## Massif 与 Nsight Systems 执行指标

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

## 可选分析器成本

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

### MFLOPS / compute profiling 字段全是 `nan`

- 默认 `--compute-profile-tool none` 不采 FLOP，因此这些字段为 `nan` 属于预期结果。显式启用 `both` 后，Torch eager 与 NCU 是独立 probe；先分别查看 `compute_profile_error_torch_profiler_eager` 和 `compute_profile_error_ncu`，一个失败不会令另一套指标或主实验失败。
- NCU 字段只在 `gpu_mode=on` 行有值；CPU-only 行为 `nan` 是预期行为。
- `gpu_mode=on` 时检查 `ncu` 是否安装、`--ncu-root` 是否正确、NVIDIA driver 是否允许 performance counters。若被测镜像裸跑 CUDA 正常、仅 NCU 下报错，结合工具日志核对 NCU、CUDA 和 driver 的兼容性，不能只凭一个错误码确定版本问题。
- 如果显式使用 `--compute-profile-tool vendor`，`gpu_mode=off` 时检查 Intel Advisor 是否安装，以及 `--advisor-root` 是否指向可在容器中 bind mount 的 Advisor root 或 executable。
- compute profiling 与正常 workload 分离；失败不会影响 latency / energy / resource usage 采集。完整状态与静态口径见 `compute_profile_plan.json` 和 `static_meta.json`。

### Massif / Nsight Systems execution profiling 字段全是 `nan`

- 分别查看 `compute_profile_error_massif` 和 `compute_profile_error_nsys`。出现
  `massif_runtime_unavailable` / `nsys_runtime_unavailable` 时，使用当前锁定环境重建镜像。
  Massif 要求镜像内预装 Valgrind；Nsys 还需主机 `nsys`、`--nsys-root` 和 importer runtime preflight。
  `nsys_importer_unavailable` 表示镜像动态库或 importer 运行失败；仅产生 `.qdstrm` 而没有 `.nsys-rep` 不算成功。
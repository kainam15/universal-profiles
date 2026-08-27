# AC-Prof 完整参考

> 这里保留 AC-Prof 的完整参数、指标口径、输出字段与故障排查说明。第一次使用请先阅读[项目首页](README.md)。

快速定位：

- [环境、安装与运行](#1-环境要求)
- [CLI 参数与 Input Scale](#5-cli-参数)
- [输出文件与字段字典](#7-输出文件)
- [时间成本估算](#10-结果行数和时间成本估算)
- [故障排查](#troubleshooting)
- [扩展新模型或任务族](#12-扩展新模型或任务族)

AC-Prof 是一个面向 containerized HuggingFace inference service 的运行时 profiling 工具。它会把模型权重 bake 进 Docker image，在不同 CPU / Memory / GPU 资源限制和不同 input scale 下运行推理 workload，并输出 latency、latency variability、归一化 throughput/energy、cold-start phase、GPU / CPU power 与 energy、container-attributed energy efficiency、container CPU / memory / swap usage、cgroup memory/stat/PID/throttling/events/PSI、block I/O、CPU frequency、estimated CPU cycles、perf retired-instruction MIPS、CPU cache / dTLB miss behavior，以及 packet-level latency/bytes 等指标。

本项目采用保守包化结构：核心代码位于 `acprof/`，根目录保留 [run.py](run.py)、[plot.py](plot.py) 和 [profile.py](profile.py) 三个用户入口；内部工具统一通过 `python -m acprof...` 调用。当前不引入 `pyproject.toml` / `setup.py`，仍通过 `.venv` + `requirements.txt` 运行。

常用验证命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q acprof run.py plot.py
```

核心目录：

```text
acprof/
  cli/          # run / plot CLI 实现
  host/         # host 侧检测、编排、client、compute profile
  container/    # 容器内 server、模型下载、compute runner、handlers
  workloads/    # host 侧任务族 workload generator
  monitors/     # GPU/CPU/resource/perf side-channel monitors
  packet/       # packet latency parse / merge 工具
```

## 1. 环境要求

基础要求：

- 原生 Linux 主机；当前推荐并验证的环境是 Ubuntu 24.04。WSL、Windows host 和 macOS host 不作为实验采集环境。
- Python 3.10+
- 原生 Docker Engine，并通过本机 `unix:///var/run/docker.sock` 访问；运行用户需要属于 `docker` 组。
- 统一 cgroup v2 hierarchy；正式采集默认在模型检测前强制检查。旧 cgroup v1 只能通过 `--allow-cgroup-v1` 进入诊断兼容模式。
- Hugging Face Hub 网络访问，或可用的 `HF_TOKEN`（仅用于 host 检测和镜像构建）
- `pip install -r requirements.txt`

采集依赖（是否必需按各项说明）：

- NVIDIA GPU + NVIDIA Container Toolkit：用于 `--gpus on`、GPU energy metrics、GPU utilization 和 VRAM metrics。
- Linux RAPL powercap（`/sys/class/powercap/*/energy_uj`）：必需，用于 CPU package power / energy 和 estimated vCPU energy metrics；不可读时 preflight 会退出。
- Docker cgroup v2 CPU / memory / swap / I/O / PID、`memory.peak`、`memory.stat`、`memory.events`、`io.stat`、`pids.*` 与 per-cgroup PSI files：用于 container CPU utilization、vCPU CPU time/share、memory/swap footprint、内存构成/缺页/refault、throttling、压力/事件、窗口 block-I/O 字节/操作数、PID footprint，以及 estimated CPU cycles 的 CPU utilization 输入。
- Linux CPU frequency sysfs 或 `/proc/cpuinfo`：用于 `cpu_freq_*` 和 estimated CPU cycles。优先读取 `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` / `cpuinfo_cur_freq`，不可用时回退到 `/proc/cpuinfo` 的 `cpu MHz`。
- Linux `perf` + PMU hardware events：`instructions` 用于真实 retired-instruction MIPS；`cache-references`、`cache-misses`、`dTLB-loads` 和 `dTLB-load-misses` 用于 CPU bandwidth behavior 的 cache / address-translation miss 计数和比例。`run.py` 启动时会先做 preflight；如果 `perf_event_paranoid`、sudo 或 PMU 权限不足，会直接退出并给出修复命令。generic cache 事件的具体硬件含义由 CPU 架构和 kernel PMU 映射决定，部分架构可能不支持全部事件。
- `tcpdump` + `tshark`：必需，用于填充 `result_all.csv` 的 `latency_s` packet-level latency，并从同一 PCAP 派生每请求 captured frame bytes、TCP payload 和 L2–L4 overhead。`run.py` 启动时会先做 preflight；不满足条件会直接退出并给出恢复提示。
- PyTorch profiler：显式选择 `torch` 或 `both` 时独立采集 `torch_profiler_eager` 逻辑 FLOP。该 probe 会强制并验证 eager attention，只影响临时 profiler container，不改变正常 latency / energy workload 的运行时 attention 实现。
- NVIDIA Nsight Compute CLI (`ncu`)：显式选择 `ncu` 或 `both` 时独立采集 `gpu_mode=on` 行的 GPU 实际执行 FLOP，并把 Tensor 与 Scalar FLOP 分列。通过 `--ncu-root` 挂载到临时 profiler container；不可用、不兼容当前 CUDA/driver、或性能计数器被限制时只会令 NCU 字段为 `nan`，不会影响 Torch probe 或主采集。Ubuntu multiverse 的 `nsight-compute` 可能过旧，推荐使用 NVIDIA CUDA apt 源里的版本化包，例如 `/opt/nvidia/nsight-compute/<version>/ncu`。
- Valgrind Massif (`valgrind`)：仅在显式启用 execution profiling 时用于 `gpu_mode=off` 的 process-lifetime CPU memory peak。AC-Prof 会基于模型镜像构建 `dockerfiles/massif.Dockerfile` 派生镜像并在其中安装 Valgrind，不要求 host 预装；首次启用需要 Docker build 的 apt 网络访问。未启用或 probe 失败不会影响主采集。
- NVIDIA Nsight Systems CLI (`nsys`)：仅在显式启用 execution profiling 时用于 `gpu_mode=on` 的 CUDA API、kernel 和 memcpy timeline 汇总。可通过 `--nsys-root` 指定 host 安装根目录或 executable；也会自动搜索 `/opt/nvidia/nsight-systems` 与 NVIDIA Nsight Compute 安装中附带的 `nsys`。AC-Prof 会基于模型镜像构建 `dockerfiles/nsys.Dockerfile` 派生镜像，补齐 host `QdstrmImporter` 在 slim container 中需要的 `libdw.so.1`，并在资源矩阵开始前验证 importer；首次启用需要 Docker build 的 apt 网络访问。需要 NVIDIA driver、NVIDIA Container Toolkit 与被测 CUDA workload 兼容。
- Intel Advisor：只保留给显式 `--compute-profile-tool vendor` 的兼容/诊断流程，用于 `gpu_mode=off` 行。通过 `--advisor-root` 挂载到临时 profiler container。
- 原生 Docker 的 `docker0` bridge：packet sniffing 默认监听 `docker0`。如果抓包条件不足、PCAP 为空或 merge 后仍有 `latency_s=nan`，程序会退出，不会产出看似完整但缺少 packet latency 的结果。

Hugging Face token 可以放在项目根目录 `.env.local`：

```env
HF_TOKEN=hf_xxx
# 可选：当 perf/tcpdump 需要 sudo 且 sudo -n 不可用时使用
ACPROF_SUDO_PASSWORD=your_sudo_password
```

`run.py` 会自动读取 `.env` 和 `.env.local`，并把 `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` 用于 host 检测；Docker build 通过 BuildKit secret 临时挂载令牌，不会把令牌写入构建参数或镜像历史。正式 inference、scale probe 和 compute profiling 容器不会接收 Hugging Face token。`ACPROF_SUDO_PASSWORD` 只在 host 侧用于 `sudo -S perf` / `setcap tcpdump` 这类 preflight，不会传入被测容器。

模型 snapshot 会在镜像构建阶段下载并通过 `/models/model-snapshot` 暴露为稳定本地路径。正式 inference、scale probe 和 compute profiling 容器统一启用 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，因此资源矩阵运行阶段不会向 Hugging Face Hub 或镜像站查询模型文件。使用 `--skip-build` 的旧镜像如果还没有稳定路径，会回退到镜像内 Hugging Face cache，但仍保持离线解析。

## 2. 安装

```bash
cd universal-profiles
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. 快速运行

完整默认矩阵：

```bash
python run.py --model google-bert/bert-base-uncased
```

默认会运行：

- CPU: `1,2,4,8`
- Memory: `2,4,8,16` GB
- GPU mode: `off,on`
- input scale: 自动规划 6 档
- warmup: 2
- repeat: 5
- repeat-in-window: auto 模式下每行连续发送请求，直到本行累计 application latency 达到目标 workload window，默认约 10 秒

默认完整矩阵较慢。开发和验证建议先在 Ubuntu shell 中跑小矩阵：

```bash
python run.py --model google-bert/bert-base-uncased \
  --cpus 1 --mems 4 --gpus off \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --output-dir results/smoke
```

### 最低配置的最大 input scale 探测

在启动完整矩阵前，可以先用最重输入逐档确认最低可用内存：

```bash
.venv/bin/python probe.py --model google-bert/bert-base-uncased \
  --cpus 1,2,4,8 --mems 2,4,8,16 --gpus off,on \
  --skip-build
```

`probe.py` 复用正式流程的任务识别、镜像和 input-scale materialization，但不运行主矩阵。
最大尺度取 materialized `input_scale_plan.json` 中的最大 `input_scale`。CPU 固定为所选列表
中的最小值；所选 GPU 模式包含 `off` 时使用 CPU-only，否则使用 `on`；内存候选则去重、
升序逐档实测。尺度规划仍在正常的规划资源容器中完成。实际探测为每个内存候选启动
全新容器，服务 ready 后最多发送一次最大的 materialized payload。明确的 Docker/cgroup
启动 OOM、运行期 OOM，或 CPU allocator/`MemoryError` 会进入下一档；第一档完整返回且
`effective_input_scale` 与计划一致的候选，才记为最低可用内存，同时其请求就是最终计时值。

CUDA OOM 是 GPU 显存不足，改变 `--mems` 的主机内存 cap 无法解决，因此立即停止；timeout、
尺度不一致和其他非 OOM 错误也会停止并保持最低内存为空，不能越过不确定档位继续宣称
更大的候选是“最低值”。整个流程不做 warmup/repeat、idle baseline、能耗、PMU、PCAP 或
profiler 采集。

探测请求默认不设超时，会一直等待模型返回或发生明确错误；如需为自动化任务设置上限，
可显式传入 `--timeout-seconds <正数>`。每次探测写入独立目录：

```text
results/<model-dir>/probes/largest_scale_<timestamp>_<pid>/
├── input_scale_plan.json
└── largest_scale_probe.json
```

摘要 schema v3 的 `memory_probe.candidate_order_gb` 保存候选顺序，
`memory_probe.attempts` 保存每档的 `startup_oom`、`runtime_oom`、`cuda_oom`、`timeout`、
`error` 或 `ok` 结果及其错误和分段耗时，`memory_probe.minimum_viable_mem_gb` 只在成功时
写入。顶层 `timing.request_s` 是最低成功档 `/predict` 的 host 端到端耗时；
`cold_start.total_s` 是该档新容器从 `docker run` 到 ready 的耗时；
`timing.ready_plus_request_s` 是二者之和；`timing.command_s` 还包含 preflight、模型识别、
可选镜像构建、尺度规划、失败候选尝试和清理。`timing.request_timeout_s` 在默认无限等待时
为 `null`，仅在显式设置 `--timeout-seconds` 时记录对应的秒数；旧 schema v2 摘要中的该字段
始终是有限秒数。探测目录与正式模型结果共用输出根，但不会创建或修改
`result_case_*.csv`、`result_all.csv`、`static_meta.json` 或 `collection_history.json`。
它适合估算可行性，不应当作包含 idle/energy/network 口径的正式测量行。

FLOP profiling 默认关闭（`--compute-profile-tool none`）。显式选择 `both` 后，每个适用的 input scale 会运行相互独立的 probe：

- `torch_profiler_eager` 根据 eager operator shape 统计模型逻辑 FLOP；CPU/GPU 行分别使用对应 device 的独立 eager probe。它回答“模型按 eager 算子形状应执行多少计算”。
- `ncu` 只用于 `gpu_mode=on`，根据 GPU 性能计数器统计实际执行的 Tensor / Scalar FLOP。它会受 kernel 实现、Tensor Core tile 和 padding 影响，不能与逻辑 FLOP 混作同一口径。

启用后，两套采集不会污染正常 latency / energy window，也不会互相覆盖。任一 probe 失败时只写对应的错误列和 `nan`，另一套 probe 与主实验仍继续。临时 profiler container 默认使用 host 逻辑 CPU 全量和 host memory 的 75%；可用 `--compute-profile-cpus` / `--compute-profile-mem` 覆盖。raw NCU/Advisor artifact 默认保留，只有显式传入 `--discard-compute-profiles` 才删除；也可在主矩阵完成后通过 `profile.py` 补采。

默认 GPU profiling 会自动查找 `ncu`；如果工具不在默认 `PATH`，显式指定安装根目录：

```bash
python run.py --model google-bert/bert-base-uncased \
  --ncu-root /opt/nvidia/nsight-compute/2024.1.1 \
  --compute-profile-tool both \
  --compute-profile-cpus 8 --compute-profile-mem 16 \
  --cpus 1 --mems 4 --gpus off,on
```

`--compute-profile-tool none` 跳过全部 compute probe；`auto` 是 `both` 的弃用兼容别名。`torch`、`ncu` 和 `vendor` 只用于单工具诊断或旧流程兼容；如需 CPU 行使用 Intel Advisor，可显式传 `--compute-profile-tool vendor --advisor-root /opt/intel/oneapi/advisor/latest`。

Massif / Nsight Systems execution profiling 与上述 FLOP profiling 相互独立，并且默认关闭。需要时显式 opt in：

```bash
python run.py --model google-bert/bert-base-uncased \
  --execution-profile-tool both \
  --massif-repeat 1 --nsys-repeat 1 \
  --nsys-root /opt/nvidia/nsight-systems \
  --cpus 1,2 --mems 4,8 --gpus off,on
```

`massif` 只匹配 CPU-only 配置，`nsys` 只匹配 GPU 配置，`both` 同时选择两者。默认采样矩阵为：Massif 选择本次资源列表中的最大 CPU/内存并逐 input scale 采集；Nsys 选择最大内存、保留全部 CPU 并逐 input scale 采集。采样结果会展开到完整结果行，但每个 entry 都记录实际 `profile_source_cpu_cores`、`profile_source_mem_cap_gb` 和 `profile_sampling_strategy`，不会丢失 provenance。显式传入 `--massif-sampling full --nsys-sampling full` 才会为每个完整资源配置另起 profiler container。

Massif 的 peak 是被剖析进程整个生命周期的峰值，包含模型加载、预热和随后执行的 inference，不能解释成“单次 request 新增了多少内存”，`--massif-repeat` 也不会把 peak bytes 除回单 request。Nsight Systems 只在 `acprof_compute` NVTX range 内追踪 CUDA 与 NVTX，汇总 CUDA API、GPU kernel 和 memcpy，并用 `--nsys-repeat` 归一化为单 request；host inference wall timer 只包围正式推理循环与其末尾 CUDA synchronize，不计入 NVTX capture 的启动/停止握手。不同 host/device timeline 和 CUDA stream 上的活动可能重叠，因此各项 `*_time_sum_*` 彼此相加不要求等于 host wall time。中间 `.qdstrm` 不属于保留产物；若 importer 失败，程序会删除该中间文件并记录 `nsys_import_failed`，避免批量采集反复堆积巨型 raw stream。

两个 execution probe 的失败互不影响，也不影响 FLOP probe 或正常 latency / energy / resource-usage 窗口。raw Massif `.out` 与 Nsight Systems `.nsys-rep` 默认保留；`nsys stats` 导出的 `.sqlite` 只作为解析缓存，在指标汇总完成或 stats 报错后自动删除。显式传入 `--discard-execution-profiles` 会在汇总后连同 raw artifacts 一并删除。

### 原生 Ubuntu 下采集 `latency_s` 的推荐启动方式

`run.py` 现在明确要求“原生 Linux 进程 + 本机 Docker daemon”。启动后会依次检查 host 是否为 WSL、本机 Docker endpoint、`tcpdump`、`tshark`、抓包网卡、`tcpdump` capability、RAPL 和 `perf`；任一必需条件不满足都会在模型检测/build 前退出。WSL、Docker Desktop、远程 Docker 和旧的 `/var/run/docker-native.sock` 会被主动拒绝，避免长时间实验结束后才发现部分字段无法采集。

首次配置或切换环境后，先确认基础链路：

```bash
cd universal-profiles
source .venv/bin/activate

unset DOCKER_HOST DOCKER_CONTEXT
docker context use default
docker context inspect default --format '{{(index .Endpoints "docker").Host}}'
docker info --format 'OperatingSystem={{.OperatingSystem}}'
test -f /sys/fs/cgroup/cgroup.controllers
cat /proc/self/cgroup

command -v tcpdump
command -v tshark
getcap "$(command -v tcpdump)"
ip link show docker0
```

Docker endpoint 应输出 `unix:///var/run/docker.sock`。`tcpdump` capability 应包含 `cap_net_raw` 和 `cap_net_admin`；缺少时执行一次：

```bash
sudo apt-get install -y tcpdump tshark
sudo setcap cap_net_raw,cap_net_admin=eip "$(command -v tcpdump)"
```

随后直接从 Ubuntu shell 启动，不再使用 `wsl.exe`、`/mnt/...` 路径或 `DOCKER_HOST=unix:///var/run/docker-native.sock`：

```bash
python run.py --model google-bert/bert-base-uncased \
  --skip-build \
  --output-dir results/test
```

前提是：

- 当前 shell 的 `uname -r` 不包含 `microsoft`，Docker daemon 的操作系统为本机 Ubuntu。
- Docker context 指向 `unix:///var/run/docker.sock`，且当前普通用户可以直接执行 `docker info`。
- `/sys/fs/cgroup/cgroup.controllers` 存在，`/proc/self/cgroup` 使用 `0::...` 的统一 v2 hierarchy。
- 默认 bridge 名为 `docker0`。如果 daemon 明确改过 bridge 名，使用 `docker network inspect bridge` 核对后传 `--sniff-iface <实际网卡>`。
- `--skip-build` 只有在本机 `/var/lib/docker` 对应的 image store 已存在目标 image 时才可用。
- 实验需要完整跑到 merge 阶段，并且所有结果行都成功回填 `latency_s`；否则本次 run 会失败退出并保留错误提示。

如果 Docker image 已经存在，可以跳过 build：

```bash
python run.py --model google-bert/bert-base-uncased --skip-build
```

运行完成后输出目录为：

```text
results/<model-name-with-slash-replaced-by-double-dash>/
```

例如：

```text
results/google-bert--bert-base-uncased/
```

如果指定 `--output-dir results/test`，则输出为：

```text
results/test/google-bert--bert-base-uncased/
```

在 tmux pane 内启动实验时，`run.py` 会从 preflight 开始持续记录该 pane
的全部显示输出，并在实验正常结束或报错退出时自动关闭记录，将文件原子保存为
结果目录下的 `tmux_all.log`。该机制不依赖 tmux 的 `history-limit`，无需再手动
执行 `tmux capture-pane`。如果当前 pane 已经配置了其他 `pipe-pane`，程序会保留
原有 pipe 并打印警告，不会擅自覆盖。

## 4. 常用命令

指定资源矩阵：

```bash
python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
```

指定 task family / backend：

```bash
python run.py --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
```

手动指定 input scale：

```bash
python run.py --model google-bert/bert-base-uncased --input-scales 64,128,256,512
```

生成图表：

```bash
python plot.py results/google-bert--bert-base-uncased/result_all.csv
```

`plot.py` 默认读取同目录下的 `static_meta.json`，用其中的 `input_scale_type` 作为横轴语义名；读取历史结果时仍兼容旧的 `static_meta.csv`。图片会写入结果目录下的三个子目录：

- `cpu/`：只使用 `gpu_mode=off` 的 CPU 数据
- `gpu/`：只使用 `gpu_mode=on` 的 GPU 数据
- `gpu+cpu/`：同时包含 GPU 和 CPU 数据，用于对比

每个有对应数据的目录会按可用指标生成以下图表。原先 50 种通用单指标图所覆盖的指标中，49 项已归入 13 张总览图，只有与 cgroup 窗口内存口径不同的 Massif 保持独立：

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
- `cold_start_breakdown.png` 仅在五个阶段字段完整时生成，并为每个 GPU mode/CPU 数选择最大 memory cap，堆叠 container launch、server setup、CUDA init、model load 和 ready wait。`cold_start_s` 以独立标记核对阶段和，first-predict application latency 也只作为独立标记，不计入 `/ready` 前的堆叠总量。旧结果缺少阶段列时继续保留 `cold_start_bar.png`，并自动跳过分解图。

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

## 5. CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | required | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `--task` | auto | 覆盖 `pipeline_tag`，例如 `fill-mask`、`text-generation`。 |
| `--task-family` | auto | 覆盖任务族：`nlp`、`cv`、`audio`、`timeseries`。 |
| `--backend` | auto | 覆盖 runtime backend，例如 `transformers_pipeline`、`chronos`。 |
| `--cpus` | `1,2,4,8` | CPU core 限制列表。 |
| `--mems` | `2,4,8,16` | Memory cap GB 列表。 |
| `--gpus` | `off,on` | GPU mode 列表。`on` 会用 Docker `--gpus all`。 |
| `--prune-startup-oom` / `--no-prune-startup-oom` | enabled | 默认以最低选中 CPU 为参考，按内存升序完整采集；仅把 Docker 明确 `OOMKilled` 的连续低内存启动失败前缀推断到后续更高 CPU。跳过 case 保留占位行和独立 provenance。运行期/CUDA OOM、timeout 与普通启动失败不触发剪枝。使用 `--no-prune-startup-oom` 可恢复逐格独立尝试。 |
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
| `--input-scales` | auto | 手动覆盖 input scale 列表。未提供时自动规划 6 档。 |
| `--workload-spec` | task default | workload 清单 JSON。ASR 默认使用仓库内置的 LibriSpeech 英文短音频清单；其他音频任务必须显式提供清单。 |
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
| `--sniff-iface` | `docker0` | 本机 Docker 默认 bridge 对应的 `tcpdump` 抓包网卡。只有 daemon 改过 bridge 名时才覆盖。 |
| `--output-dir` | `results` | 输出根目录。最终还会追加 model name 子目录。 |
| `--skip-build` | false | 跳过 Docker build，直接使用已存在的 image tag。 |
| `--allow-cgroup-v1` | false | 仅用于旧主机诊断的兼容开关。默认正式模式要求 cgroup v2；启用后允许 v1，但会记录 `legacy_compatible`，且 memory peak/stat、I/O 操作数、PID、memory events 与 per-cgroup PSI 不具备同等口径。 |

结果目录存在异常中断留下的 `result_case_*.csv` 时，`run.py` 会先读取同目录 `static_meta.json/cgroup_version`。只有版本与当前 host 一致才允许续写；版本不同、缺失或元数据不可读时会退出，避免把 v1/v2 窗口合并到同一结果文件。

## 6. Input Scale 规则

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

### 真实音频 workload

`automatic-speech-recognition` 默认使用 `assets/audio/librispeech-clean-test-en-30s/source.json`。该清单引用 LibriSpeech `clean/test` 中同一说话人、同一章节的三条连续语音，按固定顺序拼接后截取前 30 秒；素材是单声道 16 kHz PCM16 WAV，许可证为 CC BY 4.0。每一档输入都从同一个 30 秒基准音频取前缀，不做逐档归一化、补全或循环。

音频请求采用 JSON 内的 Base64 WAV；handler 仍能读取历史 `audio_samples` 浮点数组。短音频模式会读取模型 feature extractor 的约束并拒绝超过 receptive field 的尺度。对于 Whisper，30 秒是音频 receptive field；当前 feature extractor 会把接受的短音频补齐为固定的 480,000 samples / 3,000 frames，`/scale_meta` 会显式记录这一点。`max_target_positions=448` 是解码器输出 token 上限，不是音频输入上限，因此框架不会把 latency 必须随 `duration_s` 单调增加作为正确性条件。

自定义素材时可复制内置 `source.json`，设置新的 `workload_id`，再修改相对素材路径、SHA256、provenance 和 inference 字段。自定义 provenance 可以描述单条录音或既有重采样/增益流程；运行时仍会严格验证派生 WAV 本身是单声道、16 kHz、PCM16 且哈希匹配。然后传入：

```bash
python run.py --model openai/whisper-large-v3 \
  --workload-spec /path/to/source.json
```

当前音频 request 只实现 `batch_size=1` 和 `short_form`。清单会拒绝非空的 `chunk_length_s` / `stride_length_s`；长音频 sequential/chunked 应使用独立 workload，不能通过把本清单尺度直接扩展到 30 秒以上来混测。

## 7. 输出文件

每个模型输出目录通常包含：

| 文件 | 说明 |
| --- | --- |
| `result_all.csv` | 动态测量结果。每一行对应一个 resource config、一个 input scale、一次 warmup/repeat iteration，并记录归一化指标、PCAP 网络字节、cold-start phases，以及该窗口的 cgroup memory/stat/PID、swap、块 I/O 与压力/事件。 |
| `static_meta.json` | 单个 JSON object 的静态元数据。记录模型版本、参数/精度/量化/许可证、输入输出格式、per-scale 静态逻辑 FLOPs、推理后端、镜像、GPU/主机 RAM、主机 swap、Docker 存储和环境信息。 |
| `collection_history.json` | schema v1 的采集/修复 provenance。分别记录 post-hoc profiler 补采、timeout retry、quality retry 和静态元数据回填历史；最新一次状态由对应 history 的最后一项得到。 |
| `input_scale_plan.json` | 所有任务族共用的 input scale/payload 计划。schema v2 额外记录 workload provenance、per-scale 输入元数据和模型约束；读取端继续兼容无版本字段的 v1 计划。主采集和 compute profiler 复用同一份 payload。 |
| `startup_oom_pruning.json` | 仅启用 `--prune-startup-oom` 时生成。记录最低参考 CPU、执行顺序、逐 GPU mode 的实测启动 OOM 前缀、最低启动可行内存、推断跳过 case、排除范围与资源单调性假设。 |
| `compute_profile_plan.json` | per-scale FLOP profiling 结果。每个 CPU/GPU scale 可同时记录独立的 `torch_profiler_eager` 与 `ncu` profile；NCU 只存在于 GPU profile。失败信息按工具保存，只读取当前按 profiler 分层的 plan 结构。 |
| `execution_profile_plan.json` | 显式 execution profiling 的采样与 per-resource-config/per-scale 汇总。Massif 条目对应 `gpu_mode=off`，Nsight Systems 条目对应 `gpu_mode=on`；复用 entry 记录实际 source resource 与 sampling strategy，失败按工具记录且不阻断主实验。 |
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

<a id="posthoc-profiling"></a>

### 已有结果后补采 Torch / NCU / Nsight Systems / Massif

高开销 profiler 可以与正常 latency / energy / resource-usage 矩阵分开执行。若希望 input-scale 计划生成后直接进入正常矩阵，先关闭全部 compute profiling 与 execution profiling：

```bash
python run.py \
  --model openai/whisper-large-v3 \
  --gpus off,on \
  --compute-profile-tool none \
  --execution-profile-tool none
```

`--compute-profile-tool torch` 仍会在正常矩阵之前逐个 input scale 运行 Torch eager probe，因此不能用于“直接进入 result CSV 矩阵”。使用 `none` 后，Torch logical FLOP 与 NCU 字段暂时为空，下面的独立命令会与 Nsys、Massif 一起补采。

正常矩阵运行期间会先写 `result_case_*.csv`。只有全部 case 完成后，`run.py` 才把这些中间文件合并成最终的 `result_all.csv`，因此启动主矩阵并不代表目录中会立刻出现 `result_all.csv`。

正常运行结束且结果目录中已经同时存在 `result_all.csv`、`static_meta.json`、`input_scale_plan.json` 后，在项目根目录执行。新结果还会包含 `collection_history.json`；旧结果没有该文件时，首次成功补采会自动创建并迁移 `static_meta.json` 中的旧 history 字段：

```bash
python profile.py results/openai--whisper-large-v3
```

命令默认补采所有适用且尚未成功的 `torch,ncu,nsys,massif`：Torch 按已有的 CPU/GPU device class 和 input scale 采集，NCU 与 Nsys 只匹配 `gpu_mode=on` 行，Massif 只匹配 `gpu_mode=off` 行。它不会启动主 workload matrix，不会重新采 latency、energy、MIPS、resource usage 或 packet latency。也可以只选部分工具：

```bash
python profile.py results/openai--whisper-large-v3 --tools torch,ncu,nsys
python profile.py results/openai--whisper-large-v3 --tools massif
```

Massif 默认使用 `--massif-sampling per-scale`：自动选择结果矩阵中最大的 CPU/memory case，每个 input scale 运行一个探针，再把带有明确 representative provenance 的结果用于同尺度 CPU 行。需要严格按完整 CPU × memory 矩阵采集时使用：

```bash
python profile.py results/google-bert--bert-base-uncased \
  --tools massif --massif-sampling full
```

Nsys 默认使用 `--nsys-sampling per-cpu-scale`：保留结果矩阵中的全部 CPU，自动选择最大 memory，每个 CPU/input scale 运行一个探针，再复用到同 CPU/scale 的其他内存行。可以进一步缩成一个代表资源，或恢复完整矩阵：

```bash
python profile.py results/google-bert--bert-base-uncased \
  --tools nsys --nsys-sampling per-scale
python profile.py results/google-bert--bert-base-uncased \
  --tools nsys --nsys-sampling full
```

四个 `--*-reference-cpu/mem` 参数可覆盖默认最大资源，但显式值必须存在于已有结果矩阵中。已有成功结果默认会复用；需要按新采样参数重采时同时传入 `--force-reprofile`。

Post-hoc NCU 与 Massif 补采支持按 input scale 断点续跑。NCU 每个成功尺度会在 raw CSV 旁原子写入 `ncu_scale_<scale>.checkpoint.json`；重新执行同一命令时，会依次尝试复用匹配的 checkpoint、校验并迁移旧版 CSV、从已存在的 `.ncu-rep` 重新导出 CSV，只有无法恢复的尺度才会重新采集。Massif 同样为每个完成的 `.out` 写 checkpoint，并可迁移补采中断前留下的完整旧报告。checkpoint 会校验模型 revision、镜像、资源、repeat，以及 NCU metrics（适用时）；`--force-reprofile` 会禁用恢复路径并强制重新采集。

常用安全/诊断选项：

```bash
# 只检查适用工具、输入计划和缺失字段，不启动 profiler、不修改文件
python profile.py results/google-bert--bert-base-uncased --dry-run

# 显式重新采集并替换已有成功 profiler 字段
python profile.py results/google-bert--bert-base-uncased --force-reprofile
```

回填后仍使用原文件名 `result_all.csv` 与 `static_meta.json`，并把本次操作追加到 `collection_history.json/posthoc_profile_history`。发布新文件前，命令会把旧版本复制到 `posthoc_backups/<timestamp>/`，三个临时文件全部验证通过后才替换；失败时从备份恢复。非 profiler 字段原样保留。raw report 与补采 plan 位于 `posthoc_profiles/`。若检测到 `run.py`、Torch/NCU/Nsys/Massif 或另一个 `profile.py` 正在使用相同结果目录，命令会拒绝启动。完整成功的已有 plan 会直接复用；已有成功 CSV 行默认不会被覆盖。

### 已有 CSV 回填 compute profile

如果 latency 已采集完成、之后才生成 `compute_profile_plan.json`，可以写出一份带 FLOP/MFLOPS 的新 CSV：

```bash
python -m acprof.cli.backfill_compute \
  results/google-bert--bert-base-uncased/result_all.csv \
  results/google-bert--bert-base-uncased/compute_profile_plan.json \
  --output results/google-bert--bert-base-uncased/result_all.with_compute.csv
```

工具按 `gpu_mode` 和 `input_scale`（绝对容差 `1e-6`）匹配 CPU/GPU plan 条目，并分别回填 Torch eager 逻辑 FLOP 与 NCU GPU 实际执行 FLOP。两套 `*_mflops_app_*` 都使用 `latency_app_s`，`*_mflops_packet_*` 优先使用有效的 `latency_s`，否则回退到 application latency。未匹配或失败的工具只把自身字段保留为 `nan`，原因写入该工具的错误列。输出采用原子写入，且默认拒绝覆盖已有文件；确需替换显式输出路径时追加 `--overwrite`。

旧 plan / CSV 的五个通用 compute 字段仍可由绘图和迁移入口读取，但新结果不再写入这些重复列。Torch eager 与 NCU 指标分别写入名称明确的 `*_torch_profiler_eager` 和 `*_ncu` 列。

## 8. `result_all.csv` 字段解释

`result_all.csv` 只放 per-case / per-iteration 的动态字段。

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
| `cpu_idle_power_w` | CPU package matched-control baseline power，单位 W。仅 Linux/WSL 暴露 RAPL `/sys/class/powercap/*/energy_uj` 时有值；由与 workload 相同 monitor 生命周期下整段 control window 的 RAPL 能耗差 / 实际 duration 得到。 |
| `cpu_idle_measured_at` | `--idle-debug` 开启时，CPU idle baseline 测量完成时的本地 ISO-8601 时间戳；未开启时为 `nan`。 |
| `cpu_idle_rel_range_so_far` | `--idle-debug` 开启时，截至本行为止当前 case 内有效 `cpu_idle_power_w` 的相对极差，公式为 `(max - min) / mean`；`0.05` 表示 5%。未开启时为 `nan`。 |
| `cpu_energy_iters` | CPU package energy measurement 采样窗口中的 sample 数。 |
| `cpu_avg_power_total_w` | 测量窗口内 host CPU package total average power，单位 W。 |
| `cpu_peak_power_total_w` | 测量窗口内 host CPU package total peak power，单位 W。peak 取相邻 RAPL 采样区间功率的最大值，低于 `0.5 / sample_hz` 的尾部短区间不参与 peak 计算。 |
| `cpu_energy_total_j` | 本行平均到单 request 的 total CPU package energy，单位 J。 |
| `cpu_avg_power_eff_w` | 扣除 CPU idle baseline 后的 CPU package effective average power，单位 W。 |
| `cpu_peak_power_eff_w` | 扣除 CPU idle baseline 后的 CPU package effective peak power，单位 W。peak 口径同 `cpu_peak_power_total_w`。 |
| `cpu_energy_eff_j` | 本行平均到单 request 的 effective CPU package energy，单位 J。 |
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
| `cold_start_started_at` / `cold_start_ready_at` | host 侧开始执行 `docker run` 与首次成功收到 `/ready` 的本地 ISO-8601 时间戳。 |
| `cold_start_container_launch_s` | host 开始 `docker run` 到 container 内 Python server process 开始执行的时间；使用共享系统 wall clock 对齐。旧镜像未返回启动时间戳时为 `nan`。 |
| `cold_start_server_setup_s` | server process 开始到 model load 开始之间的 setup 时间，扣除单独列出的显式 CUDA 初始化，包括 Python/framework/handler import 与配置。 |
| `cold_start_cuda_init_s` | 现有启动路径中 `torch.cuda.is_available()` 与 `torch.cuda.init()` 的显式计时；CPU case 为 `0`，不发送 probe inference。 |
| `cold_start_model_load_s` | container handler `load()` 的持续时间，使用 `time.perf_counter()`。 |
| `cold_start_ready_wait_s` | model load 完成到 host 首次收到成功 `/ready` 的时间，包括 Flask 开始监听、poll interval 和本地 HTTP 往返。 |
| `cold_start_first_predict_app_s` | workload client 发出的第一个成功 `/predict` 的 application latency；默认 auto-window 模式下通常是已有 auto-warmup 请求，固定窗口模式下是已有首个 warmup/measurement 请求，不会为此新增请求。 |
| `cold_start_s` | 当前 container 从 `docker run` 到 `/ready` 成功的时间，单位秒。 |
| `status` | `ok`、`warn` 或 `error`。`warn` 常用于可继续分析但存在异常值的行。 |
| `error` | 错误或 warning 文本。正常行为空；`status=error` 时强制非空。请求超时会区分实际发出但未完成的请求（`client_request_timeout`）与未发请求、因前序超时跳过的计划行（`not_measured_after_timeout`），并记录触发尺度、timeout 下界、请求阶段和 request ID。 |

### `latency_s` 和 `latency_app_s` 的区别

- `latency_app_s` 是 client 侧应用层计时，只要 `/predict` 请求成功，一般就能写出。
- `latency_s` 是 packet-level 计时，需要完整完成 `tcpdump` capture、`acprof.packet.sniff_parse_pcap` parse、`acprof.packet.merge_packet_latency` merge。
- 当前默认行为是严格模式：如果无法保证 `latency_s` 有值，`run.py` 会退出，不继续 merge 最终结果。

## 9. `static_meta.json` 字段解释

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
| `task_family` | 任务族：`nlp`、`cv`、`audio`、`timeseries`。 |
| `pipeline_tag` | Hugging Face pipeline tag，例如 `fill-mask`、`image-classification`。 |
| `runtime_backend` | 容器内使用的 runtime backend，例如 `transformers_pipeline`、`chronos`。 |
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
| `environment` | 自动检测的运行环境标签，例如 `windows11+wsl`、`ubuntu24.04+wsl`、`ubuntu24.04`、`macos15`。 |
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
| `massif_version` | Massif 派生 container image 中的 Valgrind/Massif 版本；未启用或无法确认时为 `unknown`。 |
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

## 10. 结果行数和时间成本估算

`result_all.csv` 行数大致为：

```text
len(cpus) * len(mems) * len(gpus) * len(input_scales) * (warmup + repeat)
```

每一行内部还会发送：

```text
repeat_in_window
```

个 `/predict` request。显式指定 `--repeat-in-window N` 时，每行固定发送 `N` 个请求；默认 `--repeat-in-window 0` 时，每行会连续发送请求，直到本行累计 `latency_app_s` 达到 `--repeat-window-seconds`，并把本行实际完成的请求数写入 CSV 的 `repeat_in_window` 字段。因此 auto 模式下每行请求数会随 resource case、input scale 和当时 latency 波动而变化。手动指定固定窗口时，请按下式估算：

```text
len(cpus) * len(mems) * len(gpus) * len(input_scales) * (warmup + repeat) * repeat_in_window
```

每一行 CSV 对应一个 workload window。窗口 wall time 可以按下式粗估：

```text
row_window_s ~= active_workload_s
              + idle_cooldown_s
              + matched_control_s
              + monitor_stop_overhead_s
```

其中：

- `active_workload_s`：正式连续发送 `/predict` 的时间。`--repeat-in-window 0` 时约等于 `--repeat-window-seconds`，默认约 `10s`；固定 `--repeat-in-window N` 时约等于 `N * mean_latency_app_s`。已生成 CSV 后，也可以用 `latency_app_s * repeat_in_window` 反推该行的实际 active window。
- `idle_cooldown_s`：每行采集 matched control 前的统一等待，约等于 `--idle-cooldown-seconds`，默认 `5s`。
- `matched_control_s`：CPU/GPU/resource usage monitor 同时运行的无请求对照窗口，约等于 `--idle-seconds`，默认 `20s`；GPU 关闭时不会额外增加一段 GPU baseline。
- `monitor_stop_overhead_s`：停止 `perf`、energy/resource monitor、写 CSV 等小额开销，通常按秒级以内预留；慢机器或 `perf`/权限异常时可能更高。

因此在默认 `--repeat-in-window 0 --repeat-window-seconds 10 --idle-seconds 20 --idle-cooldown-seconds 5` 下，单行主采集窗口通常可按以下方式估算：

```text
gpu_mode=off 且 CPU RAPL 可用: 约 10 + 5 + 20 = 35s / row
gpu_mode=on  且 CPU RAPL 和 NVML 可用: 约 10 + 5 + 20 = 35s / row
```

auto 模式会在每个 resource case / input scale 前发送少量 auto warmup 请求；这部分不写入 CSV 行数，也不会再跑一个 `--repeat-window-seconds` 长度的校准窗口：

```text
auto_warmup_s_per_case_scale ~= 5 * mean_latency_app_s
```

如果显式指定 `--repeat-in-window N`，这段 auto warmup 成本为 `0`。

本节新增的低成本字段不会扩大上述实验矩阵：`memory.peak`、`memory.stat`、`io.stat` 操作数与 `pids.*` 只在既有 workload window 首尾各读一次小型 cgroup 文件；归一化指标只做算术运算；网络字节复用已有 PCAP；cold-start phases 复用已有启动和首个推理路径。它们不增加 `/predict` 数量。PCAP parser 多做一次离线 `tshark` fields pass，通常只增加与 PCAP 大小相关的解析时间，不增加模型运行时间。

整体主采集时间可以按 resource case 累加：

```text
main_collection_s ~= sum_over_cases(
  cold_start_s
  + sum_over_scales(auto_warmup_s_per_case_scale)
  + sum_over_scales((warmup + repeat) * row_window_s)
  + sniff_parse_merge_overhead_s
)
```

如果把 build 和 compute profiling 也算入端到端 wall time，可再外加：

```text
total_wall_s ~= docker_build_s + model_detection_s + compute_profile_s + main_collection_s
```

默认完整矩阵是 `4 CPU * 4 MEM * 2 GPU * 6 scales * (2 warmup + 5 repeat) = 1344` 行。若 CPU RAPL 和 NVML 都可用、模型 latency 足够低使 auto window 接近 `10s`，仅主采集窗口就大约是：

```text
gpu off 行: 16 cases * 6 scales * 7 rows * 13s ~= 2.4h
gpu on  行: 16 cases * 6 scales * 7 rows * 19s ~= 3.5h
auto warmup: 32 cases * 6 scales * 5 requests，耗时取决于单 request latency
```

也就是说，默认主矩阵通常至少按 `6h+` 预留；最终 wall time 还要额外加上 auto warmup、Docker build、模型检测、每个 case 的 cold start、tcpdump/tshark parse/merge 和 container cleanup。计算分析器默认关闭；显式启用 `both` 时，额外成本可粗估为：

```text
compute_profile_s ~= len(input_scales) * (
  [gpu off enabled] * torch_profiler_eager_cpu_probe_s
  + [gpu on enabled] * (
      torch_profiler_eager_gpu_probe_s
      + ncu_gpu_probe_s
    )
)
```

probe 次数按 input scale 和启用的 GPU mode 增长，不按 CPU × memory 主矩阵展开；`--torch-profiler-repeat` 与 `--ncu-repeat` 会进一步增加各自 workload，NCU 还可能对 kernel 做 replay，因此常是最昂贵的 probe，在部分模型上可能从数分钟增加到更久。启用时默认保留 raw NCU report，也会增加磁盘占用。保持默认的 `--compute-profile-tool none` 可先完成主矩阵，之后再用 `profile.py` 补采；单工具模式只运行所选 probe。

Execution profiling 默认 `none`，所以不进入上述默认时间预算。显式启用后，先按采样策略计算实际 source case 数：

```text
massif_source_cases = 1                         # per-scale 默认
                      或 len(cpus) * len(mems) # full
nsys_source_cases   = len(cpus)                 # per-cpu-scale 默认
                      或 1                      # per-scale
                      或 len(cpus) * len(mems) # full

execution_profile_s ~= len(input_scales) * (
  massif_source_cases * massif_probe_s
  + nsys_source_cases * nsys_probe_s
)
```

未选择对应工具/GPU mode 时，该项 source case 数按 0 计算。Massif 会放慢整个被剖析进程生命周期，Nsight Systems 还需要生成和解析 timeline；`--massif-repeat` / `--nsys-repeat` 会增加每个 probe 的 workload。默认保留 `execution_profiles/` raw artifacts 也会增加磁盘占用。使用 `full` 前建议先用一个 CPU、一个 memory cap 和一个 input scale 做 smoke。

<a id="troubleshooting"></a>

## 11. 常见判断

`run.py` 启动时报 `[sniff][ERROR]`，或 case 阶段因 packet latency 失败退出：

- 按错误里的恢复提示检查 `tcpdump`、`tshark`、`getcap $(command -v tcpdump)`、`--sniff-iface` 和 Docker bridge。
- 常用修复命令：`sudo setcap cap_net_raw,cap_net_admin=eip $(command -v tcpdump)`。

`run.py` 启动时报 `[infra][ERROR]`：

- 确认命令是在原生 Ubuntu shell 中执行，而不是 WSL。
- 执行 `unset DOCKER_HOST DOCKER_CONTEXT && docker context use default`，然后确认 Docker endpoint 是 `unix:///var/run/docker.sock`。
- 不要使用 Docker Desktop、远程 Docker context 或旧的 `/var/run/docker-native.sock`；这些 daemon 与宿主侧抓包、cgroup、RAPL、perf PID 和 profiler 观察到的对象不一致。

case CSV 的 `error` 包含 `container_oom_killed during startup`：

- 容器在模型加载期间触达 `--memory` cgroup 上限并被内核终止；错误会同时记录 memory cap、Docker 状态和 exit code。
- 该 case 的占位行保留为 `status=error`，latency、throughput、energy 和 resource usage 等未执行指标保持 `nan`。`plot.py` 的性能图与 latency model 只使用 `status=ok` 行；资源可行性热力图会单独读取这些占位行，用来展示失败边界。
- 增大 memory cap，或改用更小/量化模型；不要用推测值回填失败 case 的指标。
- 默认的启动 OOM 剪枝不会改变任何可运行 case 的 warmup、repeat 或指标，只在后续 CPU 上为已确认的连续启动 OOM 前缀写入未执行占位行。热力图中实测启动 OOM 为 `OOM-S`，剪枝推断为 `P-OOM`；论文中必须区分两者。需要每个资源格独立实测时使用 `--no-prune-startup-oom`。

case CSV 的 `error` 包含 `container_runtime_oom`：

- 容器已完成启动，但在 workload 期间被 memory cgroup OOM kill。Docker 的 `OOMKilled=true` 优先于客户端的 MIPS、HTTP 断连或其他次生退出码进行分类。
- 已成功写入的测量行原样保留；已有失败行追加 Docker OOM 上下文；其余计划行写为 `status=error`，未采指标保持 `nan`。该 case 返回后矩阵继续执行。
- 运行期 OOM 在资源可行性热力图中显示为 `OOM-R`，但不会用于启动 OOM 剪枝，也不能从一个 CPU 配置外推到其他 CPU 配置。

GPU energy 字段全是 `nan`：

- `gpu_mode=off` 时这是正常结果。
- `gpu_mode=on` 时检查 NVIDIA driver、NVIDIA Container Toolkit、`pynvml` 和容器 GPU 可见性。

CPU / vCPU energy 字段全是 `nan`：

- 当前版本会在 task detection 前检查 RAPL；计数器不存在或不可读时，`run.py` 会退出并给出权限修复步骤，不会继续生成新的完整结果。
- 历史结果或中断产生的 CSV 仍可能包含 `nan`。这类缺失值不应使用 TDP 或 CPU utilization 猜测回填。
- `cpu_*` 字段是 host CPU package/root domain 的真实 RAPL 测量值，不累加 `intel-rapl:*:*` 这类 core 子 domain；`vcpu_*` 字段是在同一窗口内按 container cgroup CPU share 分摊出来的估计值。

CPU idle baseline 波动 warning：

- `cpu_idle_power_w` 的 case 级相对极差达到或超过 5% 时，case 结束后会输出 warning，实验继续运行。常见原因是 host 后台进程、IDE/远程桌面、Docker 其他容器、系统索引或 CPU 温度/频率策略变化。
- 可先增大 `--idle-cooldown-seconds` 让上一轮 workload 的瞬态结束；如果单次 idle 窗口本身仍然抖动，再增大 `--idle-seconds`。同时关闭非必要后台进程，并检查 `static_meta.json` 中的 `cpu_governor` / `cpu_boost` 是否符合实验设置。
- 需要定位具体时间点和进程时，加 `--idle-debug` 重跑；优先查看 `debug_idle_diag/result_case_*.csv.idle_diag.jsonl` 中同一 row 的 `rapl_trace.top_power_windows`、`idle_host_active_delta_s`、`idle_container_cpu_delta_s`，再结合 control window 结束后采集的 loadavg、top CPU processes 和 Docker 快照。`idle_proc_cpu_top` 在 matched control 模式下保持为空，避免 `/proc` 全量遍历进入 baseline 能量。

GPU idle baseline 波动 warning：

- `gpu_idle_power_w` 的 case 级相对极差达到或超过 5% 时，case 结束后会输出 warning，实验继续运行。常见原因是桌面显示栈、其他 GPU 进程、P-state/clock 调整、温度或电源管理状态变化。
- 可先增大 `--idle-cooldown-seconds`，等待上一轮 workload 后的 GPU clock/power 状态回落；如果 idle trace 内部仍抖动，再增大 `--idle-seconds`。
- 需要定位具体时间点和进程时，加 `--idle-debug` 重跑；优先查看同一 row 的 `gpu_idle_power_samples`、`nvidia_smi_gpu`、`nvidia_smi_pmon` 和 `nvidia_smi_compute_apps`。

资源占用率字段全是 `nan`：

- `container_*` usage/I/O 字段依赖被测容器的 cgroup CPU、memory、swap 与 I/O 文件；如果 Docker inspect、`/proc/<pid>/cgroup` 或 `/sys/fs/cgroup` 不可读，对应字段会保持 `nan`，不会影响其他可用指标。
- 默认正式模式已在启动阶段强制 cgroup v2。只有显式使用 `--allow-cgroup-v1` 的兼容运行才会走 v1 fallback；其 `memory.peak` / `memory.stat` 分解与计数器、`io.stat` 操作数、`pids.*`、`memory.events` 和 per-cgroup PSI 字段保持 `nan`，不会使用 host 全局数据冒充 container 指标。
- `cpu_freq_*` 字段依赖 Linux cpufreq sysfs 或 `/proc/cpuinfo`。如果当前内核、虚拟化环境或权限不暴露当前频率，会保持 `nan`。
- `cpu_cycles_est_app` 依赖 `latency_app_s`、`cpu_freq_avg_hz`、`cpu_cores` 和 `container_cpu_util_avg_pct` 都有效；`cpu_cycles_est_packet` 还额外依赖 packet latency merge 成功回填 `latency_s`。
- `gpu_*` utilization / VRAM 字段仅在 `gpu_mode=on` 且 NVML 可用时采集；口径是 NVML device-level，可能包含同一 GPU 上其他进程的占用。

`perf` preflight 或采集失败：

- `cpu_mips_*` 字段来自 Linux `perf` 的 `instructions` 硬件事件，不是 CPU frequency 推导值。`run.py` 会在 task detection 前检查 `perf` 权限，失败时打印 `[mips][ERROR]`、当前 `perf_event_paranoid`、sudo 状态和恢复步骤。
- `cpu_cache_*` 和 `cpu_dtlb_*` 字段来自 Linux `perf` generic PMU events。可先用 `perf list` 和 `perf stat -e cache-references,cache-misses,dTLB-loads,dTLB-load-misses -- true` 检查当前 CPU / kernel 是否支持；事件不支持不表示 miss 为 0。
- 这些 cache / dTLB 字段用于描述访存行为，不提供 DRAM GB/s。实际 read/write bandwidth 需要 uncore memory-controller、Intel PCM、AMD IBS/DF 或其他硬件专用计数器，不能由 miss 数直接换算。
- 常见临时修复：`echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid`。如果不想手动调整，也可以在 `.env.local` 设置 `ACPROF_SUDO_PASSWORD`，让 AC-Prof 使用 `sudo -S perf`。
- 不建议用 `sudo python run.py ...`，这会让结果文件可能变成 root 所有；修复 perf 权限后用普通用户重跑。

CPU / vCPU peak power 看起来异常：

- `cpu_peak_power_*` 和 `vcpu_peak_power_*` 是相邻采样区间功率的最大值，不是整段平均功率。增大 `--sample-hz` 会缩短区间、提高捕捉短峰值的能力，也可能让峰值更敏感。
- 为避免停止监控时的极短尾部区间放大 peak，当前实现会排除短于半个采样周期的 peak 区间；avg power 和 energy 仍按完整测量窗口计算。

MFLOPS / compute profiling 字段全是 `nan`：

- 默认 `--compute-profile-tool none` 不采 FLOP，因此这些字段为 `nan` 属于预期结果。显式启用 `both` 后，Torch eager 与 NCU 是独立 probe；先分别查看 `compute_profile_error_torch_profiler_eager` 和 `compute_profile_error_ncu`，一个失败不会令另一套指标或主实验失败。
- NCU 字段只在 `gpu_mode=on` 行有值；CPU-only 行为 `nan` 是预期行为。
- `gpu_mode=on` 时检查 `ncu` 是否安装、`--ncu-root` 是否正确、NVIDIA driver 是否允许 performance counters。若 `ncu` 下 CUDA 初始化报 `Error 36` 或没有 kernel，被测镜像裸跑 CUDA 正常但 ncu 下不正常，通常是 Nsight Compute 版本过旧；安装 NVIDIA CUDA apt 源里的较新版本后再试。
- 如果显式使用 `--compute-profile-tool vendor`，`gpu_mode=off` 时检查 Intel Advisor 是否安装，以及 `--advisor-root` 是否指向可在容器中 bind mount 的 Advisor root 或 executable。
- compute profiling 与正常 workload 分离；失败不会影响 latency / energy / resource usage 采集。完整状态与静态口径见 `compute_profile_plan.json` 和 `static_meta.json`。

Massif / Nsight Systems execution profiling 字段全是 `nan`：

- 默认 `--execution-profile-tool none` 不采 execution profile，这是预期结果；需要显式选择 `massif`、`nsys` 或 `both`。
- Massif 字段只填充 `gpu_mode=off` 行，Nsight Systems 字段只填充 `gpu_mode=on` 行；不适用的另一组字段保持 `nan`。
- 分别查看 `compute_profile_error_massif` 和 `compute_profile_error_nsys`。Massif 检查派生镜像的 Docker build/apt 网络与 `dockerfiles/massif.Dockerfile` 日志，不要求 host 安装 `valgrind`；Nsight Systems 检查 `nsys`、`--nsys-root`、`dockerfiles/nsys.Dockerfile` 的 importer runtime preflight、NVIDIA driver / Container Toolkit 兼容性，以及 raw `.nsys-rep` 是否可导出。出现 `nsys_importer_unavailable` 时，优先检查派生镜像中 `libdw.so.1` 等动态库；probe 阶段只出现 `.qdstrm` 而没有 `.nsys-rep` 属于 importer 失败。
- 两者是显式 opt-in 的独立 probe；一个失败不会影响另一个 execution probe、FLOP compute profiling 或主实验。完整状态与静态口径见 `execution_profile_plan.json` 和 `static_meta.json`。

`--skip-build` 后 `/scale_meta` 或 `/probe` 报错：

- 可能使用了旧 image。重新运行一次不带 `--skip-build` 的 build。

## 12. 扩展新模型或任务族

通常不需要为新 Hugging Face model 写代码。`acprof/host/detect.py` 会尽量自动识别任务和 backend。

新增 task family 时，需要同时补齐：

- `acprof/container/handlers/`：容器内 model load / preprocess / predict / postprocess。
- `acprof/workloads/`：host 侧 workload generator。
- `dockerfiles/`：对应 task family 的 Dockerfile。
- `acprof/config.py`：`PIPELINE_TAG_TO_FAMILY` 和 `SCALING_DIMENSIONS`。

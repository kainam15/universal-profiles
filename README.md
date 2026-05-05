# AC-Prof Universal Profiler

AC-Prof 是一个面向 containerized HuggingFace inference service 的运行时 profiling 工具。它会把模型权重 bake 进 Docker image，在不同 CPU / Memory / GPU 资源限制和不同 input scale 下运行推理 workload，并输出 latency、throughput、cold start、GPU / CPU power 与 energy、container CPU / memory usage、CPU frequency、estimated CPU cycles、perf retired-instruction MIPS、packet-level latency 等指标。

本项目采用保守包化结构：核心代码位于 `acprof/`，根目录的 [run.py](run.py)、[plot.py](plot.py)、[client.py](client.py) 等文件是兼容旧命令的薄入口。当前不引入 `pyproject.toml` / `setup.py`，仍通过 `.venv` + `requirements.txt` 运行。

常用验证命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q acprof run.py plot.py client.py server.py compute_profile_runner.py sniff_parse_pcap.py merge_packet_latency.py
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

- Python 3.10+
- Docker
- Hugging Face Hub 网络访问，或可用的 `HF_TOKEN`
- `pip install -r requirements.txt`

可选但会影响字段完整性：

- NVIDIA GPU + NVIDIA Container Toolkit：用于 `--gpus on`、GPU energy metrics、GPU utilization 和 VRAM metrics。
- Linux RAPL powercap（`/sys/class/powercap/*/energy_uj`）：用于 CPU package power / energy 和 estimated vCPU energy metrics。
- Docker cgroup CPU / memory files：用于 container CPU utilization、vCPU CPU time/share、memory footprint metrics，以及 estimated CPU cycles 的 CPU utilization 输入。
- Linux CPU frequency sysfs 或 `/proc/cpuinfo`：用于 `cpu_freq_*` 和 estimated CPU cycles。优先读取 `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` / `cpuinfo_cur_freq`，不可用时回退到 `/proc/cpuinfo` 的 `cpu MHz`。
- Linux `perf` + hardware event `instructions`：必需，用于真实 retired-instruction MIPS。`run.py` 启动时会先做 preflight；如果 `perf_event_paranoid`、sudo 或 PMU 权限不足，会直接退出并给出修复命令。
- `tcpdump` + `tshark`：必需，用于填充 `result_all.csv` 的 `latency_s` packet-level latency。`run.py` 启动时会先做 preflight；不满足条件会直接退出并给出恢复提示。
- Intel Advisor：仅在显式 `--compute-profile-tool vendor` 时用于 `gpu_mode=off` 行的 CPU FLOP / MFLOPS profiling。通过 `--advisor-root` 挂载到临时 profiler container。
- NVIDIA Nsight Compute CLI (`ncu`)：默认用于 `gpu_mode=on` 行的 GPU FLOP / MFLOPS profiling。通过 `--ncu-root` 挂载到临时 profiler container；不可用、不兼容当前 CUDA/driver、或性能计数器被限制时 compute 字段为 `nan`。Ubuntu multiverse 的 `nsight-compute` 可能过旧，推荐使用 NVIDIA CUDA apt 源里的版本化包，例如 `/opt/nvidia/nsight-compute/<version>/ncu`。
- Linux / WSL native Docker 的 `docker0` bridge：packet sniffing 默认监听 `docker0`。如果抓包条件不足、PCAP 为空或 merge 后仍有 `latency_s=nan`，程序会退出，不会产出看似完整但缺少 packet latency 的结果。

Hugging Face token 可以放在项目根目录 `.env.local`：

```env
HF_TOKEN=hf_xxx
# 可选：当 perf/tcpdump 需要 sudo 且 sudo -n 不可用时使用
ACPROF_SUDO_PASSWORD=your_sudo_password
```

`run.py` 会自动读取 `.env` 和 `.env.local`，并把 `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` 传给 Docker build 和 runtime。`ACPROF_SUDO_PASSWORD` 只在 host 侧用于 `sudo -S perf` / `setcap tcpdump` 这类 preflight，不会传入被测容器。

## 2. 安装

```bash
pip install -r requirements.txt
```

如果在 Windows 上遇到 `pip.exe` 被拦截，优先使用：

```bash
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
- repeat-in-window: 自动按当前 case / input scale 的单次推理时间校准，默认目标约 10 秒 workload window

默认完整矩阵较慢。开发和验证建议先跑小矩阵：

```bash
python run.py --model google-bert/bert-base-uncased ^
  --cpus 1 --mems 4 --gpus off ^
  --warmup 0 --repeat 1 --repeat-in-window 1 ^
  --output-dir results/smoke
```

Linux / WSL shell 使用反斜杠换行：

```bash
python run.py --model google-bert/bert-base-uncased \
  --cpus 1 --mems 4 --gpus off \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --output-dir results/smoke
```

MFLOPS profiling 默认启用，默认 `--compute-profile-tool auto`：CPU 行使用 PyTorch profiler，GPU 行使用 NVIDIA Nsight Compute `ncu`。采集不会污染正常 latency / energy window；临时 profiler container 默认使用 host 逻辑 CPU 全量和 host memory 的 75%。如果需要更保守的资源占用，可用 `--compute-profile-cpus` / `--compute-profile-mem` 显式覆盖。开发 smoke 可以先用 `--no-compute-profile` 验证主流程。只想跑原始 latency/energy 时可加 `--no-compute-profile`。

默认 GPU profiling 会自动查找 `ncu`；如果工具不在默认 `PATH`，显式指定安装根目录：

```bash
python run.py --model google-bert/bert-base-uncased \
  --ncu-root /opt/nvidia/nsight-compute/2024.1.1 \
  --compute-profile-cpus 8 --compute-profile-mem 16 \
  --cpus 1 --mems 4 --gpus off,on
```

如需 CPU 行也使用 Intel Advisor，可额外传 `--compute-profile-tool vendor --advisor-root /opt/intel/oneapi/advisor/latest`。

### 采集 `latency_s` 的推荐启动方式

`run.py` 默认要求完整采集 `latency_s`。启动后会先检查 `tcpdump`、`tshark`、抓包网卡和 `tcpdump` capability；任一条件不满足都会在模型检测/build 前退出。推荐从 WSL 启动，并显式使用 WSL native Docker daemon，不要落回 Docker Desktop daemon：

```bash
wsl.exe sh -lc "cd /mnt/d/DOR/universal-profiles && DOCKER_HOST=unix:///var/run/docker-native.sock python3 run.py --model google-bert/bert-base-uncased --skip-build --output-dir results/test"
```

这条路径用于让 `tcpdump` 在 WSL Docker 的 `docker0` bridge 上抓到容器流量。前提是：

- WSL native Docker daemon 已启动，并监听 `unix:///var/run/docker-native.sock`。
- WSL 内已安装 `tcpdump` 和 `tshark`。
- `--skip-build` 只有在该 native daemon 的 image store 里已经有对应 image 时才可用；Docker Desktop 和 WSL native Docker 的 image store 不是同一个。
- 实验需要完整跑到 merge 阶段，并且所有结果行都成功回填 `latency_s`。否则本次 run 会失败退出并保留错误提示。

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

`plot.py` 默认读取同目录下的 `static_meta.csv`，用其中的 `input_scale_type` 作为横轴语义名，并在结果目录生成：

- `latency_vs_scale.png`
- `latency_app_vs_scale.png`
- `gpu_avg_power_vs_scale.png`
- `gpu_energy_vs_scale.png`
- `cpu_avg_power_vs_scale.png`
- `cpu_energy_vs_scale.png`
- `vcpu_avg_power_vs_scale.png`
- `vcpu_energy_vs_scale.png`
- `throughput_vs_scale.png`
- `compute_mflops_vs_scale.png`
- `container_cpu_util_vs_scale.png`
- `container_mem_util_vs_scale.png`
- `container_mem_usage_vs_scale.png`
- `gpu_util_vs_scale.png`
- `gpu_mem_util_vs_scale.png`
- `gpu_mem_used_vs_scale.png`
- `cold_start_bar.png`

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
| `--batch-size` | `1` | 每个 request 的 batch size。 |
| `--warmup` | `2` | 每个资源配置、每个 input scale 的 warmup 行数。 |
| `--repeat` | `5` | 每个资源配置、每个 input scale 的正式测量行数。 |
| `--repeat-in-window` | `0` | 每一行内部连续发送的 `/predict` request 数量。`0` 表示按单次推理时间自动校准。 |
| `--repeat-window-seconds` | `10.0` | `--repeat-in-window 0` 时的目标 workload window 秒数。 |
| `--sample-hz` | `20.0` | GPU power sampling rate，单位 Hz；CPU workload 期间也用它控制 RAPL、container cgroup、CPU frequency 和 GPU/resource usage 的采样间隔，以估计 peak power、vCPU share、CPU utilization 和 CPU cycles。perf MIPS 使用独立的 `perf stat` 窗口，不受该采样率影响。CPU idle baseline 不使用该 20Hz 小区间 median。 |
| `--idle-seconds` | `3.0` | 每个 workload window 前的 idle baseline 测量时长。CPU idle baseline 用整段 RAPL `energy_uj` 差值 / 实际 duration；`gpu_mode=on` 时 GPU idle baseline 仍来自 NVML power samples。case 结束后会复查该 case CSV 中所有有效 `cpu_idle_power_w` 的相对极差；`gpu_mode=on` 时也会复查 `gpu_idle_power_w`。达到或超过 5% 会退出实验并提示增大该值或稳定主机/GPU 环境。 |
| `--idle-debug` | false | 开启 idle baseline 调试输出。主 CSV 会填充 GPU 的 `gpu_idle_measured_at` / `gpu_idle_rel_range_so_far` 和 CPU 的 `cpu_idle_measured_at` / `cpu_idle_rel_range_so_far`，并写出 `result_case_*.csv.idle_diag.jsonl`。诊断文件会记录 GPU idle 期间的 NVML power sample trace、`nvidia-smi` GPU/process 快照、CPU idle window 内 0.1s RAPL 子窗口、start/end `/proc` CPU delta、container CPU delta，以及测完后的 loadavg、top CPU processes、Docker 容器和 `docker stats` 快照。 |
| `--input-scales` | auto | 手动覆盖 input scale 列表。未提供时自动规划 6 档。 |
| `--no-compute-profile` | false | 禁用 MFLOPS compute profiling。 |
| `--compute-profile-tool` | `auto` | `auto` 下 CPU 使用 PyTorch profiler、GPU 使用 ncu；`torch` 全部使用 PyTorch profiler；`vendor` 使用 Intel Advisor / ncu。 |
| `--advisor-root` | auto | Host Intel Advisor install root or executable；显式值优先于自动检测。 |
| `--ncu-root` | auto | Host Nsight Compute install root or `ncu` executable；显式值优先于自动检测。 |
| `--advisor-repeat` | `20` | CPU compute probe 中重复推理次数；最终 FLOP 会除回单 request。`auto` / `torch` 下对应 CPU PyTorch profiler，`vendor` 下对应 Advisor。 |
| `--ncu-repeat` | `1` | GPU compute probe 中重复推理次数；最终 FLOP 会除回单 request。`auto` / `vendor` 下对应 ncu，`torch` 下对应 GPU PyTorch profiler。 |
| `--compute-profile-cpus` | host logical CPUs | 临时 compute profiler container 的 CPU core cap。 |
| `--compute-profile-mem` | 75% host memory | 临时 compute profiler container 的 memory cap，单位 GB。 |
| `--keep-compute-profiles` | false | 保留 raw profiler artifacts。默认 GPU / vendor GPU 包括 ncu report/CSV，vendor CPU 包括 Advisor project。 |
| `--sniff-iface` | `docker0` | `tcpdump` 抓包网卡。 |
| `--output-dir` | `results` | 输出根目录。最终还会追加 model name 子目录。 |
| `--skip-build` | false | 跳过 Docker build，直接使用已存在的 image tag。 |

## 6. Input Scale 规则

`input_scale` 是每个任务族的主输入尺度，语义由 `static_meta.csv` 的 `input_scale_type` 决定：

| task family | `input_scale_type` | 含义 |
| --- | --- | --- |
| `nlp` | `seq_length` | 输入 token length。 |
| `cv` | `resolution_scale` | 图像基础尺寸的缩放倍率。 |
| `audio` | `duration_s` | 输入音频时长，单位秒。 |
| `timeseries` | `context_length` | 时间序列 context length。 |

未提供 `--input-scales` 时，当前实现会为一次 profiling run 自动规划 6 档 input scale：

- `nlp` 会启动容器读取 tokenizer / handler 的可用最大输入长度，最后一档尽量贴近有效上限。
- `cv`、`audio`、`timeseries` 会根据各自 workload generator 的最大尺度或配置默认值生成 6 档。
- 同一次 run 的所有资源配置共用同一组 scale。
- 自动规划的 payload 会写入 `input_scale_plan.json`，`client.py` 用它保证实际执行 payload 与 CSV 中记录的 `input_scale` 一致。
- 手动传入 `--input-scales` 时以手动值为准；`nlp` 和 `timeseries` 会在 sweep 前验证合法性。

## 7. 输出文件

每个模型输出目录通常包含：

| 文件 | 说明 |
| --- | --- |
| `result_all.csv` | 动态测量结果。每一行对应一个 resource config、一个 input scale、一次 warmup/repeat iteration。 |
| `static_meta.csv` | 一行静态元数据。记录模型、镜像、batch、input scale 语义、GPU、环境和大小信息。 |
| `input_scale_plan.json` | 自动 input scale 规划文件。手动 `--input-scales` 时可能不存在。 |
| `compute_profile_plan.json` | per-scale FLOP profiling 结果。默认 CPU 来自 PyTorch profiler、GPU 来自 ncu；显式 `--compute-profile-tool torch` 时全部来自 PyTorch profiler，显式 `vendor` 时来自 Intel Advisor / ncu。失败时也会写入错误信息，供 `result_all.csv` compute 字段引用。 |
| `result_case_*.csv.idle_diag.jsonl` | 仅 `--idle-debug` 时生成。每行对应一个 workload window 的 idle 诊断记录，包含 GPU NVML idle power trace、`nvidia-smi` GPU/process 快照、CPU idle window 内 RAPL 子窗口功率、host/container CPU delta、top proc CPU delta，以及 after-idle 快照，用于定位 `gpu_idle_power_w` / `cpu_idle_power_w` case 内波动来源。 |
| `*.png` | `plot.py` 生成的图表。 |

中间文件 `result_case_*.csv`、`result_case_*.csv.sniff_groups.jsonl`、`lat_case_*.json`、`sniff_case_*.pcap` 会在 `result_all.csv` 成功 merge 后自动清理。若运行被中断，这些中间文件可能保留。

## 8. `result_all.csv` 字段解释

`result_all.csv` 只放 per-case / per-iteration 的动态字段。

| 字段 | 含义 |
| --- | --- |
| `cpu_cores` | 当前 Docker container 的 CPU core 限制，来自 `--cpus`。 |
| `mem_cap_gb` | 当前 Docker container 的 memory cap，单位 GB，来自 `--mems`。 |
| `gpu_mode` | `on` 或 `off`。`on` 表示容器使用 Docker `--gpus all`。 |
| `input_scale` | 本行实际执行的主输入尺度。语义见 `static_meta.csv/input_scale_type`。 |
| `task_param` | 任务族的 secondary parameter，通常是 JSON 字符串。生成式 NLP 常见为 `{"max_new_tokens": 64}`，timeseries 常见为 `{"prediction_length": 64}`；不适用时为空。 |
| `repeat_idx` | 当前 warmup 或 repeat phase 内的 0-based iteration index。 |
| `warmup` | `1` 表示 warmup 行，`0` 表示正式测量行。`plot.py` 默认排除 warmup 行。 |
| `repeat_in_window` | 本行内部连续发送的 request 数量。`latency_app_s` 和 `latency_s` 都是该 window 内 request 的平均值。 |
| `latency_s` | packet-level latency，来自 `tcpdump` PCAP + `tshark` 解析 + `merge_packet_latency.py` merge。当前默认要求该字段完整；抓包不可用、PCAP 为空、解析为空或 merge 后仍有缺失时，程序会退出并给出恢复提示。 |
| `latency_p50_s` / `latency_p90_s` / `latency_p95_s` | packet-level latency 在本行 request window 内的 empirical nearest-rank 分位数，由 merge 阶段从同一组 packet latency 明细计算。 |
| `latency_slow_ratio` | packet-level latency 中超过 `SLOW_LATENCY_THRESHOLD_S` 的 request 比例，默认阈值为 `0.06` 秒，用于观察尾延迟或双峰分布。 |
| `latency_app_s` | host-side application latency。`client.py` 用 `requests.post()` 外层 `time.perf_counter()` 测得，通常比 `latency_s` 更容易稳定产出。 |
| `latency_app_p50_s` / `latency_app_p90_s` / `latency_app_p95_s` | host-side application latency 在本行 request window 内的 empirical nearest-rank 分位数。 |
| `latency_app_slow_ratio` | host-side application latency 中超过 `SLOW_LATENCY_THRESHOLD_S` 的 request 比例，默认阈值为 `0.06` 秒。 |
| `throughput_samples_per_s` | 吞吐量，约等于 `batch_size / latency`。如果 `latency_s` 成功 merge，会优先按 `latency_s` 更新；否则按 `latency_app_s` 计算。 |
| `compute_profile_tool` | FLOP profiling 工具。默认 CPU 行为 `torch_profiler`、GPU 行为 `ncu`；显式 `--compute-profile-tool torch` 时全部为 `torch_profiler`；显式 `vendor` 时 CPU-only 行为 `intel_advisor`、GPU 行为 `ncu`；未采集时为 `nan`。 |
| `model_mflop_per_request` | profiler 采集到的单 request 模型计算量，单位 MFLOP。默认 CPU 根据 PyTorch profiler operator shapes 统计，GPU 来自 ncu FLOP / tensor operation counters；vendor 模式下 CPU 来自 Intel Advisor `Self GFLOP`。 |
| `compute_mflops_app` | `model_mflop_per_request / latency_app_s`，单位 MFLOPS。 |
| `compute_mflops` | 默认等于 `compute_mflops_app`；如果 `latency_s` 成功 merge，则用 packet-level `latency_s` 重算。 |
| `compute_profile_error` | compute profiling 的诊断信息。正常为空；工具缺失、性能计数器受限、报告解析失败时记录原因。 |
| `gpu_idle_power_w` | GPU idle baseline power，单位 W。仅 `gpu_mode=on` 且 NVML 可用时有值；由 NVML power samples 取中位数得到。 |
| `gpu_idle_measured_at` | `--idle-debug` 开启且 `gpu_mode=on` 时，GPU idle baseline 测量完成时的本地 ISO-8601 时间戳；未开启或 GPU 不可用时为 `nan`。GPU idle 先于 CPU idle 测量，因此该时间戳与 `cpu_idle_measured_at` 分开记录。 |
| `gpu_idle_rel_range_so_far` | `--idle-debug` 开启时，截至本行为止当前 case 内有效 `gpu_idle_power_w` 的相对极差，公式为 `(max - min) / mean`；`0.05` 表示 5%。未开启时为 `nan`。 |
| `gpu_energy_iters` | GPU energy measurement 内部采样窗口中的 iteration 数。 |
| `gpu_avg_power_total_w` | 测量窗口内 GPU total average power，单位 W。 |
| `gpu_peak_power_total_w` | 测量窗口内 GPU total peak power，单位 W。 |
| `gpu_energy_total_j` | 本行平均到单 request 的 total GPU energy，单位 J。 |
| `gpu_avg_power_eff_w` | 扣除 idle baseline 后的 effective average power，单位 W。 |
| `gpu_peak_power_eff_w` | 扣除 idle baseline 后的 effective peak power，单位 W。 |
| `gpu_energy_eff_j` | 本行平均到单 request 的 effective GPU energy，单位 J。 |
| `cpu_idle_power_w` | CPU package idle baseline power，单位 W。仅 Linux/WSL 暴露 RAPL `/sys/class/powercap/*/energy_uj` 时有值；由整段 idle window 的 RAPL 能耗差 / 实际 duration 得到。 |
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
| `resource_usage_iters` | resource usage monitor 在本行测量窗口内保留的 sample 数。 |
| `container_cpu_util_avg_pct` | 当前 Docker container 在测量窗口内的平均 CPU 占用率，按 `container_cpu_time_delta / (elapsed_seconds * cpu_cores) * 100` 计算。 |
| `container_cpu_util_peak_pct` | 当前 Docker container 在相邻采样间隔中的峰值 CPU 占用率，单位 `%`。 |
| `cpu_freq_avg_hz` | 测量窗口内 host online CPU 当前频率的平均值，单位 Hz。每个 sample 先对 online CPU 求平均，最终再对窗口内 sample 求平均；优先读取 Linux cpufreq sysfs，失败时回退到 `/proc/cpuinfo`。 |
| `cpu_freq_peak_hz` | 测量窗口内 host online CPU 当前频率的峰值，单位 Hz。每个 sample 取 online CPU 的最高当前频率，最终再取窗口内最大值。 |
| `cpu_cycles_est_app` | 基于 application latency 的 estimated CPU cycles，公式为 `latency_app_s * cpu_freq_avg_hz * cpu_cores * container_cpu_util_avg_pct / 100`。这是利用率与频率推导值，不是硬件 PMU retired instructions / cycles 计数。 |
| `cpu_cycles_est_packet` | 基于 packet-level `latency_s` 的 estimated CPU cycles，公式同 `cpu_cycles_est_app`，但在 `merge_packet_latency.py` 成功回填 `latency_s` 后才会更新；merge 前或 packet latency 缺失时为 `nan`。 |
| `cpu_instructions_per_request` | Linux `perf stat -e instructions` 采集到的 retired instructions，按本行 `repeat_in_window` 平均到单 request。MIPS 采集失败会中止实验而不是写入静默 `nan`。 |
| `cpu_mips_app` | 基于 `latency_app_s` 的真实 retired-instruction MIPS，公式为 `cpu_instructions_per_request / latency_app_s / 1e6`。 |
| `cpu_mips_packet` | 基于 packet-level `latency_s` 的真实 retired-instruction MIPS，在 `merge_packet_latency.py` 成功回填 `latency_s` 后更新；merge 前或 packet latency 缺失时为 `nan`。 |
| `cpu_perf_elapsed_s` | perf 统计窗口报告的 elapsed time，单位秒，用于诊断 perf 窗口是否覆盖本行 workload。 |
| `container_mem_usage_avg_bytes` | 当前 Docker container 在测量窗口内的平均 memory usage，单位 bytes，来自 cgroup memory 文件。 |
| `container_mem_usage_peak_bytes` | 当前 Docker container 在测量窗口内的峰值 memory usage，单位 bytes。 |
| `container_mem_util_avg_pct` | 当前 Docker container 平均 memory usage / `mem_cap_gb` 的百分比。 |
| `container_mem_util_peak_pct` | 当前 Docker container 峰值 memory usage / `mem_cap_gb` 的百分比。 |
| `gpu_util_avg_pct` | NVML device 0 在测量窗口内的平均 GPU utilization，单位 `%`。这是 device-level 口径，不做 container process attribution。 |
| `gpu_util_peak_pct` | NVML device 0 在测量窗口内的峰值 GPU utilization，单位 `%`。 |
| `gpu_mem_used_avg_bytes` | NVML device 0 在测量窗口内的平均 used VRAM，单位 bytes。 |
| `gpu_mem_used_peak_bytes` | NVML device 0 在测量窗口内的峰值 used VRAM，单位 bytes。 |
| `gpu_mem_util_avg_pct` | NVML device 0 平均 used VRAM / total VRAM 的百分比。 |
| `gpu_mem_util_peak_pct` | NVML device 0 峰值 used VRAM / total VRAM 的百分比。 |
| `cold_start_s` | 当前 container 从 `docker run` 到 `/ready` 成功的时间，单位秒。 |
| `status` | `ok`、`warn` 或 `error`。`warn` 常用于可继续分析但存在异常值的行。 |
| `error` | 错误或 warning 文本。正常行为空。 |

### `latency_s` 和 `latency_app_s` 的区别

- `latency_app_s` 是 client 侧应用层计时，只要 `/predict` 请求成功，一般就能写出。
- `latency_s` 是 packet-level 计时，需要完整完成 `tcpdump` capture、`sniff_parse_pcap.py` parse、`merge_packet_latency.py` merge。
- 当前默认行为是严格模式：如果无法保证 `latency_s` 有值，`run.py` 会退出，不继续 merge 最终结果。

## 9. `static_meta.csv` 字段解释

`static_meta.csv` 是一行 model/image/run-level 静态元数据。

| 字段 | 含义 |
| --- | --- |
| `model_name` | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `model_revision` | 实际解析到的 model revision / commit hash。 |
| `task_family` | 任务族：`nlp`、`cv`、`audio`、`timeseries`。 |
| `pipeline_tag` | Hugging Face pipeline tag，例如 `fill-mask`、`image-classification`。 |
| `runtime_backend` | 容器内使用的 runtime backend，例如 `transformers_pipeline`、`chronos`。 |
| `image_tag` | 本次使用的 Docker image tag。 |
| `batch_size` | 本次 profiling 的 batch size。 |
| `input_scale_type` | `result_all.csv/input_scale` 的语义名，例如 `seq_length`。 |
| `model_download_url` | Hugging Face model page URL。 |
| `gpu` | host device 0 的 GPU 名称；没有可见 NVIDIA GPU 时为 `unknown`。 |
| `gpu_mem_total_bytes` | host device 0 的 total VRAM，单位 bytes；无法读取时为空。 |
| `model_weight_bytes` | Docker image 内 `/models/hf` 下 Hugging Face cache artifacts 的总字节数，不是严格的单一权重文件大小。 |
| `docker_image_bytes` | `docker image inspect <image_tag> --format "{{.Size}}"` 返回的本地 image size，单位 bytes。 |
| `environment` | 自动检测的运行环境标签，例如 `windows11+wsl`、`ubuntu24.04+wsl`、`ubuntu24.04`、`macos15`。 |
| `cpu_power_source` | CPU package 功耗来源。`rapl` 表示使用 Linux RAPL powercap 真实计数器；`unavailable` 表示当前环境没有可用 RAPL。 |
| `vcpu_power_method` | estimated vCPU 功耗计算方法。`rapl_cgroup_cpu_share` 表示用 RAPL package energy 乘以 container cgroup CPU share；`unavailable` 表示无法估算。 |
| `cpu_governor` | Host CPU frequency governor 汇总值，例如 `performance`、`powersave`、`schedutil`；如果各 CPU policy 不一致，会写成 `mixed:<governor>=<count>,...`；无法读取时为 `unavailable`。 |
| `cpu_boost` | Host CPU boost / turbo 状态。`on` 表示 boost 可用，`off` 表示关闭；无法读取时为 `unavailable`。 |

## 10. 结果行数和时间成本估算

`result_all.csv` 行数大致为：

```text
len(cpus) * len(mems) * len(gpus) * len(input_scales) * (warmup + repeat)
```

每一行内部还会发送：

```text
repeat_in_window
```

个 `/predict` request。默认 `repeat_in_window=0` 会先对每个 resource case / input scale 发 5 个 calibration warmup 请求（不计入统计），再连续发送校准请求直到至少完成 9 个请求且累计 application latency 达到 `--repeat-window-seconds`，用 sustained mean latency 计算实际 request 数；因此默认完整 run 的 request 数取决于模型在持续请求窗口中的平均延迟。手动指定固定窗口时，请按下式估算：

```text
len(cpus) * len(mems) * len(gpus) * len(input_scales) * (warmup + repeat) * repeat_in_window
```

每一行 CSV 对应一个 workload window。窗口 wall time 可以按下式粗估：

```text
row_window_s ~= active_workload_s
              + cpu_idle_baseline_s
              + gpu_idle_baseline_s
              + gpu_cooldown_s
              + monitor_stop_overhead_s
```

其中：

- `active_workload_s`：正式连续发送 `/predict` 的时间。`--repeat-in-window 0` 时约等于 `--repeat-window-seconds`，默认约 `10s`；固定 `--repeat-in-window N` 时约等于 `N * mean_latency_app_s`。已生成 CSV 后，也可以用 `latency_app_s * repeat_in_window` 反推该行的实际 active window。
- `cpu_idle_baseline_s`：CPU RAPL 可用时约等于 `--idle-seconds`，默认 `3s`；RAPL 不可用时约为 `0s`。
- `gpu_idle_baseline_s`：`gpu_mode=on` 且 NVML 可用时约等于 `--idle-seconds`，默认 `3s`；`gpu_mode=off` 时为 `0s`。
- `gpu_cooldown_s`：`gpu_mode=on` 且 NVML 可用时每行前固定约 `3s`，用于保留已有 GPU cooldown 行为。
- `monitor_stop_overhead_s`：停止 `perf`、energy/resource monitor、写 CSV 等小额开销，通常按秒级以内预留；慢机器或 `perf`/权限异常时可能更高。

因此在默认 `--repeat-in-window 0 --repeat-window-seconds 10 --idle-seconds 3` 下，单行主采集窗口通常可按以下方式估算：

```text
gpu_mode=off 且 CPU RAPL 可用: 约 10 + 3 = 13s / row
gpu_mode=on  且 CPU RAPL 和 NVML 可用: 约 10 + 3 + 3 + 3 = 19s / row
```

自动 `repeat-in-window` 还会在每个 resource case / input scale 前做一次校准；这部分不写入 CSV 行数，但会占用时间：

```text
auto_calibration_s_per_case_scale ~= 5 * mean_latency_app_s
                                  + max(9 * mean_latency_app_s, repeat_window_seconds)
```

如果显式指定 `--repeat-in-window N`，这段自动校准成本为 `0`。

整体主采集时间可以按 resource case 累加：

```text
main_collection_s ~= sum_over_cases(
  cold_start_s
  + sum_over_scales(auto_calibration_s_per_case_scale)
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
auto calibration: 32 cases * 6 scales * 约 10s ~= 0.5h
```

也就是说，默认主矩阵通常至少按 `6.5h+` 预留；最终 wall time 还要额外加上 Docker build、模型检测、每个 case 的 cold start、tcpdump/tshark parse/merge、container cleanup，以及默认启用的 compute profiling。`--no-compute-profile` 可以去掉 compute profiling 成本；否则 compute profiling 约按 `len(input_scales) * (CPU probe + GPU probe)` 追加，`ncu` / Advisor 在部分模型上可能从数分钟增加到更久。

## 11. 常见判断

`run.py` 启动时报 `[sniff][ERROR]`，或 case 阶段因 packet latency 失败退出：

- 按错误里的恢复提示检查 `tcpdump`、`tshark`、`getcap $(command -v tcpdump)`、`--sniff-iface` 和 Docker bridge。
- 常用修复命令：`sudo setcap cap_net_raw,cap_net_admin=eip $(command -v tcpdump)`。

GPU energy 字段全是 `nan`：

- `gpu_mode=off` 时这是正常结果。
- `gpu_mode=on` 时检查 NVIDIA driver、NVIDIA Container Toolkit、`pynvml` 和容器 GPU 可见性。

CPU / vCPU energy 字段全是 `nan`：

- 这是当前环境没有暴露 RAPL `/sys/class/powercap/*/energy_uj` 时的正常结果，不会影响 latency / throughput 采集。
- `cpu_*` 字段是 host CPU package/root domain 的真实 RAPL 测量值，不累加 `intel-rapl:*:*` 这类 core 子 domain；`vcpu_*` 字段是在同一窗口内按 container cgroup CPU share 分摊出来的估计值。
- 本项目不会用 TDP 或 CPU utilization 造功耗值；没有 RAPL 时保持 `nan`。

CPU idle baseline 波动导致实验退出：

- `cpu_idle_power_w` 的 case 级相对极差达到或超过 5% 时，实验会退出并提示稳定主机环境。常见原因是 host 后台进程、IDE/远程桌面、Docker 其他容器、系统索引或 CPU 温度/频率策略变化。
- 可先增大 `--idle-seconds`，关闭非必要后台进程，并检查 `static_meta.csv` 中的 `cpu_governor` / `cpu_boost` 是否符合实验设置。
- 需要定位具体时间点和进程时，加 `--idle-debug` 重跑；优先查看 `result_case_*.csv.idle_diag.jsonl` 中同一 row 的 `rapl_trace.top_power_windows`、`idle_proc_cpu_top`、`idle_container_cpu_delta_s`，再结合 `snapshot_scope=after_idle` 的 loadavg、top CPU processes 和 Docker 快照。

GPU idle baseline 波动导致实验退出：

- `gpu_idle_power_w` 的 case 级相对极差达到或超过 5% 时，实验会退出并提示稳定 GPU 环境。常见原因是桌面显示栈、其他 GPU 进程、P-state/clock 调整、温度或电源管理状态变化。
- 需要定位具体时间点和进程时，加 `--idle-debug` 重跑；优先查看同一 row 的 `gpu_idle_power_samples`、`nvidia_smi_gpu`、`nvidia_smi_pmon` 和 `nvidia_smi_compute_apps`。

资源占用率字段全是 `nan`：

- `container_*` usage 字段依赖被测容器的 cgroup CPU / memory 文件；如果 Docker inspect、`/proc/<pid>/cgroup` 或 `/sys/fs/cgroup` 不可读，会保持 `nan`。
- `cpu_freq_*` 字段依赖 Linux cpufreq sysfs 或 `/proc/cpuinfo`。如果当前内核、虚拟化环境或权限不暴露当前频率，会保持 `nan`。
- `cpu_cycles_est_app` 依赖 `latency_app_s`、`cpu_freq_avg_hz`、`cpu_cores` 和 `container_cpu_util_avg_pct` 都有效；`cpu_cycles_est_packet` 还额外依赖 packet latency merge 成功回填 `latency_s`。
- `gpu_*` utilization / VRAM 字段仅在 `gpu_mode=on` 且 NVML 可用时采集；口径是 NVML device-level，可能包含同一 GPU 上其他进程的占用。

MIPS preflight 或采集失败：

- `cpu_mips_*` 字段来自 Linux `perf` 的 `instructions` 硬件事件，不是 CPU frequency 推导值。`run.py` 会在 task detection 前检查 `perf` 权限，失败时打印 `[mips][ERROR]`、当前 `perf_event_paranoid`、sudo 状态和恢复步骤。
- 常见临时修复：`echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid`。如果不想手动调整，也可以在 `.env.local` 设置 `ACPROF_SUDO_PASSWORD`，让 AC-Prof 使用 `sudo -S perf`。
- 不建议用 `sudo python run.py ...`，这会让结果文件可能变成 root 所有；修复 perf 权限后用普通用户重跑。

CPU / vCPU peak power 看起来异常：

- `cpu_peak_power_*` 和 `vcpu_peak_power_*` 是相邻采样区间功率的最大值，不是整段平均功率。增大 `--sample-hz` 会缩短区间、提高捕捉短峰值的能力，也可能让峰值更敏感。
- 为避免停止监控时的极短尾部区间放大 peak，当前实现会排除短于半个采样周期的 peak 区间；avg power 和 energy 仍按完整测量窗口计算。

MFLOPS / compute profiling 字段全是 `nan`：

- 默认 `--compute-profile-tool auto` 下 CPU 使用 PyTorch profiler，GPU 使用 ncu；先看 `compute_profile_error` 里的 `torch_profiler_*` 或 `ncu_*` 诊断。
- `gpu_mode=on` 时检查 `ncu` 是否安装、`--ncu-root` 是否正确、NVIDIA driver 是否允许 performance counters。若 `ncu` 下 CUDA 初始化报 `Error 36` 或没有 kernel，被测镜像裸跑 CUDA 正常但 ncu 下不正常，通常是 Nsight Compute 版本过旧；安装 NVIDIA CUDA apt 源里的较新版本后再试。
- 如果显式使用 `--compute-profile-tool vendor`，`gpu_mode=off` 时检查 Intel Advisor 是否安装，以及 `--advisor-root` 是否指向可在容器中 bind mount 的 Advisor root 或 executable。
- compute profiling 是独立 probe；失败不会影响 latency / energy / resource usage 采集。具体原因看 `compute_profile_error` 和 `compute_profile_plan.json`。

`--skip-build` 后 `/scale_meta` 或 `/probe` 报错：

- 可能使用了旧 image。重新运行一次不带 `--skip-build` 的 build。

## 12. 扩展新模型或任务族

通常不需要为新 Hugging Face model 写代码。`acprof/host/detect.py` 会尽量自动识别任务和 backend。

新增 task family 时，需要同时补齐：

- `acprof/container/handlers/`：容器内 model load / preprocess / predict / postprocess。
- `acprof/workloads/`：host 侧 workload generator。
- `dockerfiles/`：对应 task family 的 Dockerfile。
- `acprof/config.py`：`PIPELINE_TAG_TO_FAMILY` 和 `SCALING_DIMENSIONS`。

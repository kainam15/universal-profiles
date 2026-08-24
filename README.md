# AC-Prof

AC-Prof 是一个面向 Hugging Face 推理服务的零侵入运行时分析工具。给它一个模型 ID，它会构建包含模型权重的 Docker 镜像，在不同 CPU、内存、GPU 和输入规模下运行同一组 workload，最后输出可复现的 CSV、静态元数据和图表。

`Hugging Face 模型 → Docker 镜像 → 资源矩阵实验 → 延迟 / 能耗 / 资源 / FLOP 指标 → CSV 与图表`

[快速开始](#快速开始) · [正式实验](#运行正式实验) · [查看结果](#查看结果) · [高级分析](#选择性能分析器) · [排查问题](#常见问题) · [完整参考](REFERENCE.md)

## 先看这两点

> **运行环境：** AC-Prof 只支持原生 Linux 主机和本机 Docker Engine，正式采集默认强制使用统一 cgroup v2。WSL、Docker Desktop、远程 Docker daemon、Windows 和 macOS 不能作为实验采集环境。当前推荐并验证的是 Ubuntu 24.04。

> **时间成本：** 默认 6 档 input scale 时，完整命令会运行 1,344 行主实验，通常至少需要 6 小时，且还不包含模型下载、镜像构建和计算分析器的额外耗时。第一次使用请先跑下面的最小 smoke test。

## AC-Prof 会采集什么

- 性能：application / packet-level latency、P50/P90/P95、标准差/CV/IQR/最大值、吞吐量、每 input unit 延迟、每 CPU core 吞吐，以及容器启动、server setup、CUDA 初始化、模型加载、ready wait 和首次推理的冷启动分解。
- 能耗：CPU package、估算 vCPU 和 GPU 的 idle、平均/峰值功率与能量，以及不增加采集轮次的 container-attributed 能效派生值（包括 J/input unit）。
- 资源：容器 CPU / 内存、cgroup CPU throttling、memory events、CPU/内存/I/O PSI、`memory.peak`、anon/file/slab、page fault/refault、块 I/O 字节与操作数、PID 当前值/峰值/上限事件、CPU 频率与估算 cycles，以及 GPU utilization、VRAM、SM/显存时钟、P-state 和温度。
- 网络：从同一份 PCAP 派生每请求的请求/响应 frame bytes、TCP payload 和 L2–L4 协议开销，不增加抓包轮次。
- PMU：retired-instruction MIPS、cache miss 和 dTLB miss。
- 计算：PyTorch eager 逻辑 FLOP，以及 NVIDIA Nsight Compute 实际 GPU FLOP。
- 可选 execution profile：Valgrind Massif 内存峰值、Nsight Systems CUDA timeline。

目前支持的任务族：

| 任务族 | `input_scale` 的含义 | 示例 |
| --- | --- | --- |
| NLP | token 序列长度 | BERT、文本生成、问答 |
| CV | 基础图像尺寸的缩放倍率 | 图像分类、目标检测 |
| Audio | 音频时长（秒） | Whisper ASR、音频分类 |
| Time series | context length | Chronos 时间序列预测 |
| Diffusion | 方形输出图像边长（像素） | Stable Diffusion 文生图 |

大多数 Hugging Face 模型会自动识别任务族和后端；识别失败时再使用 `--task`、`--task-family` 或 `--backend` 覆盖。

`text-to-image` 模型会自动选择 `diffusion` 任务族和 `diffusers` 后端。内置 workload 固定提示词、随机种子、guidance scale 和 20 个去噪步，只改变输出分辨率；服务端仅返回生成图像的数量与尺寸元数据，避免图片响应体影响网络和应用延迟测量。

## 快速开始

### 1. 检查主机环境

必需条件：

- Python 3.10+。
- 当前用户可以直接访问 `unix:///var/run/docker.sock`，无需使用 `sudo docker`。
- Host 使用统一 cgroup v2；`/sys/fs/cgroup/cgroup.controllers` 必须存在。
- Hugging Face Hub 可访问；私有或 gated 模型还需要 `HF_TOKEN`。
- Linux RAPL powercap 可读。
- Linux `perf` 可以访问硬件 `instructions` 事件。
- `tcpdump`、`tshark` 和本机 Docker bridge 可用。
- 运行 `--gpus on` 时，还需要 NVIDIA driver 和 NVIDIA Container Toolkit。

先确认 Docker 指向本机 daemon：

```bash
unset DOCKER_HOST DOCKER_CONTEXT
docker context use default
docker context inspect default --format '{{(index .Endpoints "docker").Host}}'
docker info --format 'OperatingSystem={{.OperatingSystem}}'
test -f /sys/fs/cgroup/cgroup.controllers
cat /proc/self/cgroup
```

`docker context inspect` 应输出 `unix:///var/run/docker.sock`。再检查采集工具：

```bash
command -v perf tcpdump tshark
getcap "$(command -v tcpdump)"
ip link show docker0
find /sys/class/powercap -name energy_uj -readable -print -quit
perf stat -e instructions -- true
```

如果 `tcpdump` 尚未配置：

```bash
sudo apt-get install -y tcpdump tshark libcap2-bin
sudo setcap cap_net_raw,cap_net_admin=eip "$(command -v tcpdump)"
```

`run.py` 会在下载模型之前执行完整 preflight，并在条件不满足时给出对应修复命令。

### 2. 安装 Python 依赖

```bash
cd universal-profiles
# 仅在 .venv 不存在时执行下一行
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

私有或 gated 模型可在项目根目录创建 `.env.local`：

```env
HF_TOKEN=hf_xxx
```

令牌只用于 host 检测和镜像构建；构建时通过 BuildKit secret 临时挂载，不会写入镜像历史，也不会传给正式推理容器。

### 可选：企业微信采集通知

先在企业微信群中添加群机器人，把完整 Webhook 只保存在项目根目录的
`.env.local`（该文件已被 Git 忽略）：

```env
ACPROF_WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
```

建议限制本地配置文件权限：

```bash
chmod 600 .env.local
```

配置 Webhook 后，`run.py` 默认启用企业微信通知，无需额外参数：

```bash
python run.py --model google-bert/bert-base-uncased
```

如需临时关闭：

```bash
python run.py --model google-bert/bert-base-uncased --notify none
```

程序会在环境预检和 Docker 构建前先发送实验开始通知，内容包含当前运行指令和
启动后的累计耗时。每个 CPU × 内存 × GPU 资源 case 完成且对应容器、监控器和
抓包进程停止后，再同步发送一次进度，包含累计耗时、已完成 case/总 case、百分比、
刚完成的资源配置以及当前 case 的结果行/异常行；全部结果合并和 tmux 日志收尾后
发送最终总结。为了不干扰实验测量，input scale、warmup 和 repeat 窗口内部不发送
网络通知。最终总结区分成功、部分成功、无结果、失败和用户取消。发送请求超时为
5 秒并最多尝试两次；通知失败只产生警告，不改变采集结果或原退出码。不要把
Webhook 放进命令行、提交到 Git 或粘贴到日志中。

### 3. 跑一个最小 smoke test

下面只运行一个 CPU、一个内存限制、一个输入尺度和一个请求，并暂时关闭高开销 profiler：

```bash
python run.py --model google-bert/bert-base-uncased \
  --cpus 1 --mems 4 --gpus off \
  --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none \
  --execution-profile-tool none \
  --output-dir results/smoke
```

Stable Diffusion 建议先做单 GPU、单分辨率 smoke test：

```bash
python run.py --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --cpus 4 --mems 16 --gpus on --input-scales 256 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --output-dir results/sd-smoke
```

首次运行仍需下载模型并构建镜像。成功后，主要结果位于：

```text
results/smoke/google-bert--bert-base-uncased/
├── result_all.csv
├── static_meta.json
├── collection_history.json
└── input_scale_plan.json
```

### 4. 生成图表

```bash
python plot.py \
  results/smoke/google-bert--bert-base-uncased/result_all.csv
```

图表会写回模型结果目录下的 `cpu/`、`gpu/`、`gpu+cpu/` 和 `latency_model/`；没有适用数据的分组会自动跳过。除原有指标总览外，还会按可用字段生成资源失败边界、P50/P90/P95 尾延迟、延迟–能耗 Pareto 前沿和冷启动阶段分解图。历史 CSV 缺少新字段时只跳过对应图，不影响其余图表。

## 运行正式实验

建议先逐步扩大规模：最小 smoke test → 单个资源配置的全部 input scale → 不带 profiler 的目标资源矩阵 → 最后补采高开销 profiler。

### CPU-only 矩阵

```bash
python run.py --model google-bert/bert-base-uncased \
  --cpus 1,2,4 --mems 4,8 --gpus off \
  --compute-profile-tool none \
  --output-dir results/bert-cpu
```

### CPU / GPU 对比矩阵

```bash
python run.py --model google-bert/bert-base-uncased \
  --cpus 1,2,4 --mems 4,8 --gpus off,on \
  --compute-profile-tool none \
  --output-dir results/bert-cpu-gpu
```

上面两个例子先完成主矩阵，之后可用 `profile.py` 补采计算指标。若希望在矩阵开始前直接采集 Torch / NCU，删除 `--compute-profile-tool none` 即可。`--execution-profile-tool` 默认已经是 `none`。

如果已有对应模型镜像，可加 `--skip-build`。该选项只复用当前本机 Docker image store 中已存在的镜像。

### 默认完整矩阵

```bash
python run.py --model google-bert/bert-base-uncased
```

默认配置如下：

| 维度 | 默认值 |
| --- | --- |
| CPU | `1,2,4,8` |
| 内存 | `2,4,8,16` GB |
| GPU mode | `off,on` |
| input scale | 自动规划，通常 6 档 |
| warmup / repeat | `2 / 5` |
| 每行 workload | 自动持续到累计 application latency 约 10 秒 |
| compute profiler | `both`：Torch eager + GPU 行 NCU |
| execution profiler | `none` |
| 企业微信通知 | 配置 Webhook 后自动启用；`--notify none` 可关闭 |

主实验行数约为：

```text
CPU 数 × 内存数 × GPU mode 数 × input scale 数 × (warmup + repeat)
```

默认即 `4 × 4 × 2 × 6 × 7 = 1,344` 行。大矩阵开始前应同时评估运行时间和 profiler artifact 的磁盘占用。

### 常用变体

```bash
# 手动指定输入规模
python run.py --model google-bert/bert-base-uncased \
  --input-scales 64,128,256,512

# 时间序列模型
python run.py --model amazon/chronos-bolt-base \
  --task-family timeseries --backend chronos

# 复用已有镜像
python run.py --model google-bert/bert-base-uncased --skip-build

# 查看全部参数
python run.py --help
```

## 查看结果

结果目录名会把模型 ID 中的 `/` 替换为 `--`。例如 `google-bert/bert-base-uncased` 会写入 `google-bert--bert-base-uncased/`。

| 文件或目录 | 用途 |
| --- | --- |
| `result_all.csv` | 动态测量结果；每行对应一个资源配置、input scale 和一次 warmup/repeat window，包括延迟波动、归一化吞吐/能效、PCAP 网络字节、冷启动分解、cgroup memory/stat/PID/throttling/events/PSI、swap 与块 I/O 增量。 |
| `static_meta.json` | 模型 revision、参数量与参数 payload、模型 cache、精度、量化、许可证、输入输出格式、GPU/主机 RAM、主机 swap、Docker 存储、cgroup 版本/采集模式与实验命令。 |
| `collection_history.json` | 补采、超时重试和质量重采等数据修复过程的 provenance；不与静态元数据混放。 |
| `input_scale_plan.json` | 本次实际执行的 scale 和 payload 计划。 |
| `compute_profile_plan.json` | Torch / NCU 的 per-scale 结果与错误。 |
| `execution_profile_plan.json` | Massif / Nsys 的采样来源、复用 provenance、per-resource/per-scale 结果与错误。 |
| `compute_profiles/`、`execution_profiles/` | 启用对应 profiler 时默认保留的原始 artifacts。 |
| `tmux_all.log` | 从 tmux pane 启动时自动保存的完整终端输出。 |
| `cpu/`、`gpu/`、`gpu+cpu/` | `plot.py` 生成的指标图。 |
| `latency_model/` | `plot.py` 生成的延迟拟合报告、残差和诊断图。 |

阅读延迟列时注意：

- `latency_app_s` 是 client 在 `requests.post()` 外层测得的 application latency。
- `latency_s` 来自 `tcpdump + tshark` 的 packet-level latency。
- `input_units_per_request = effective_input_scale × batch_size`；这里的 input unit 沿用任务族的 `input_scale` 单位，CV 中是缩放倍率而不是像素数。
- `memory.peak`、`pids.peak` 是新建容器 cgroup 自创建以来的峰值；page fault、refault、I/O 操作和 PID max event 则是当前 workload window 的首尾增量。
- PCAP 的 protocol overhead 是 captured frame bytes 减去 TCP payload，只表示捕获到的 L2/L3/L4 开销，不含 TCP payload 内的 HTTP header/body 拆分。
- `warmup=1` 的行不会进入默认图表；`status=error` 的占位行会被性能图和延迟模型排除，但会保留在 `resource_feasibility_heatmap.png` 中展示 OOM、timeout 和其他失败边界。
- `gpu_mode=off` 时 GPU 指标为 `nan`、未启用 execution profiler 时 Massif/Nsys 指标为 `nan`，都属于预期行为。

完整字段字典、功率口径、FLOP 口径和延迟模型说明见[完整参考](REFERENCE.md)。

## 选择性能分析器

FLOP profiling 和主 latency / energy workload 相互独立：

| 选项 | 采集内容 | 适合场景 |
| --- | --- | --- |
| `--compute-profile-tool none` | 不运行 compute probe | smoke test、先完成主矩阵 |
| `torch` | Torch eager 逻辑 FLOP | 只关心模型算子形状对应的理论工作量 |
| `ncu` | GPU 实际执行的 Tensor / Scalar FLOP | 单独诊断 NVIDIA GPU |
| `both` | Torch eager；GPU 行再运行 NCU | 默认完整采集 |

Execution profiling 默认关闭。显式启用后采用缩减采样，并把来源记录到 plan 与静态元数据：

- `--execution-profile-tool massif`：CPU-only 的 process-lifetime 内存峰值。
- `--execution-profile-tool nsys`：GPU 的 CUDA API、kernel 和 memcpy timeline。
- `--execution-profile-tool both`：同时启用两者。
- Massif 默认 `--massif-sampling per-scale`：最大 CPU/内存 × 每个 input scale。
- Nsys 默认 `--nsys-sampling per-cpu-scale`：全部 CPU × 最大内存 × 每个 input scale。

需要严格采完整资源矩阵时显式传入：

```bash
python run.py --model google-bert/bert-base-uncased \
  --execution-profile-tool both \
  --massif-sampling full --nsys-sampling full
```

代表资源默认取本次 `--cpus` / `--mems` 中的最大值，也可用
`--massif-reference-cpu`、`--massif-reference-mem`、
`--nsys-reference-cpu`、`--nsys-reference-mem` 显式选择。

也可以先完成主矩阵，再安全地补采缺失指标：

```bash
# 只检查计划，不启动 profiler、不修改文件
python profile.py results/google-bert--bert-base-uncased --dry-run

# 只补采指定工具
python profile.py results/google-bert--bert-base-uncased --tools torch,ncu
```

`profile.py` 不会重跑 latency、energy、MIPS、resource usage 或 packet latency；写入前会备份旧结果并进行原子替换。NCU 和 Massif 会按 input scale 保存 checkpoint：NCU 可复用完整 CSV 或从遗留 `.ncu-rep` 恢复，Massif 可复用已完成的 `.out`，因此重启后只重新采集缺失尺度；`--force-reprofile` 会强制全部重采。详细行为见[完整参考中的补采说明](REFERENCE.md#posthoc-profiling)。

## 常见问题

### `[infra][ERROR]`

确认当前环境不是 WSL，并且 Docker 使用 `unix:///var/run/docker.sock`：

```bash
unset DOCKER_HOST DOCKER_CONTEXT
docker context use default
docker info
```

### `[sniff][ERROR]` 或 `latency_s` 无法合并

检查 `tcpdump`、`tshark`、capability 和 bridge：

```bash
command -v tcpdump tshark
getcap "$(command -v tcpdump)"
ip link show docker0
```

如果 Docker daemon 修改过默认 bridge，传入 `--sniff-iface <实际网卡>`。

### `[cpu-energy][ERROR]`

AC-Prof 要求 RAPL energy counter 可读。按错误信息检查 `/sys/class/powercap/*/energy_uj` 的存在性和权限；不要用 TDP 或 CPU utilization 伪造缺失功耗。

### `[cgroup][ERROR]`

正式实验要求统一 cgroup v2。检查 `test -f /sys/fs/cgroup/cgroup.controllers` 和 `cat /proc/self/cgroup`；修复主机启动/systemd 配置并重启后再采集。`--allow-cgroup-v1` 只用于旧环境诊断，运行会标记为 `legacy_compatible`，不要与正式 v2 数据合并分析。如果结果目录留有不同版本或版本未知的 `result_case_*.csv`，程序会拒绝续写；请换用新的 `--output-dir` 或先归档旧部分结果。

### `[mips][ERROR]`

先运行：

```bash
perf stat -e instructions -- true
cat /proc/sys/kernel/perf_event_paranoid
```

若权限不足，`run.py` 会输出适合当前主机的修复步骤。修好权限后用普通用户运行 AC-Prof，不要使用 `sudo python run.py`，以免结果文件归 root 所有。

### `container_oom_killed during startup`

模型加载时超过了 `--mems` 指定的 cgroup 限制。增大内存上限，或使用更小/量化模型。失败 case 会保留 `status=error` 占位行，不会进入性能图和延迟模型，但会进入资源可行性热力图。

### 运行中还没有 `result_all.csv`

这是正常的：矩阵执行期间先写 `result_case_*.csv`，全部 case 完成后才合并为 `result_all.csv`。如果在 tmux 中运行，可查看结果目录里的 `tmux_all.log`。

### `--skip-build` 后接口报错

本机可能仍是旧镜像。去掉 `--skip-build` 重新构建一次。

更多诊断，包括 idle baseline 波动、Profiler `nan`、GPU energy 和 PMU event 问题，见[完整故障排查](REFERENCE.md#troubleshooting)。

## 项目结构与开发

```text
acprof/
├── cli/          # run / plot CLI
├── host/         # 模型检测、容器编排、client、compute profile
├── container/    # 容器内 server、模型下载与 task handlers
├── workloads/    # 各任务族 workload generator
├── monitors/     # GPU / CPU / resource / perf side-channel monitors
└── packet/       # packet latency 解析与合并

dockerfiles/      # 各任务族及 profiler 镜像
run.py            # 主实验入口
plot.py           # 绘图入口
profile.py        # 已有结果的 profiler 补采入口
```

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q acprof run.py plot.py profile.py
```

更深入的 CLI 参数、input scale 规则、音频 workload、所有输出字段、时间估算和扩展方式，请阅读 [REFERENCE.md](REFERENCE.md)。

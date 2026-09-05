# AC-Prof

AC-Prof 是一个面向 Hugging Face 推理服务的零侵入运行时分析工具。给它一个模型 ID，它会构建包含模型权重的 Docker 镜像，在不同 CPU、内存、GPU 和输入规模下运行同一组 workload，最后输出可复现的 CSV、静态元数据和图表。

`Hugging Face 模型 → Docker 镜像 → 资源矩阵实验 → 延迟 / 能耗 / 资源 / FLOP 指标 → CSV 与图表`

[快速开始](#快速开始) · [正式实验](#运行正式实验) · [查看结果](#查看结果) · [高级分析](#选择性能分析器) · [排查问题](#常见问题) · [完整参考](REFERENCE.md)

## 先看这两点

> **运行环境：** AC-Prof 只支持原生 Linux 主机和本机 Docker Engine，正式采集默认强制使用统一 cgroup v2。WSL、Docker Desktop、远程 Docker daemon、Windows 和 macOS 不能作为实验采集环境。当前推荐并验证的是 Ubuntu 24.04。

> **时间成本：** 默认 6 档 input scale 时，完整命令会运行 1,344 行主实验。按默认每行约 10 秒 workload、20 秒 Idle 基线和 5 秒前置冷却估算，仅主测量窗口就约 13 小时，且还不包含模型下载、镜像构建、case 切换，以及显式启用计算分析器时的额外耗时。第一次使用请先跑下面的最小 smoke test。

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

### 可选：使用交互式终端界面

不想反复输入长命令时，可以从项目根目录启动全屏 TUI：

```bash
./acprof-tui
```

也可以预填模型并直接加载最小 smoke 配置：

```bash
./acprof-tui --model google-bert/bert-base-uncased --preset smoke
```

TUI 分为“实验配置”“运行监控”“结果工具”和“设置”四页。实验页集中填写模型、资源矩阵和
输入规模，提供三种预设及自动命令预览；底部固定显示环境检查、最大输入探测和开始采集按钮。
表单会随终端宽度切换排列。监控页显示 case 级进度和日志，结果页提供摘要、绘图和 profiler
补采入口。开始探测、采集和执行补采前都会显示确认页。

`F2` 或 `/settings` 打开设置，`F5` 开始采集，`F6` 执行环境检查，`Ctrl+X` 安全终止当前任务。
底部输入框还支持 `/run`、`/probe`、`/check`、`/status`、`/stop`、`/plot`、`/profile` 和 `/help`
等快捷命令。

设置页只放界面偏好：八种主题（深海蓝、纸白、石墨灰、松林绿、暮紫、琥珀、暖砂、雾蓝）、
日志保留行数（500 / 1000 / 3000 / 10000）、新日志自动换行，以及是否显示底部快捷命令框。
修改当次生效，点击“保存设置”后，下次启动沿用。
换行设置作用于新日志；日志行数上限在后续写入时执行裁剪。“恢复界面默认”先恢复当次显示，
再点击保存即可保留。

采集参数留在实验页。点击“高级参数”，切换到 batch size、warmup/repeat、请求窗口、采样
频率、请求超时、分析器和任务覆盖参数等选项；点击“返回基本配置”回到模型和资源矩阵。
高级参数内的“记住实验配置”会保存当前实验表单，下次打开此项目时自动填入。
命令行显式提供的 `--model` 或 `--preset` 优先于保存的配置。运行任务期间，配置控件暂时锁定。

设置文件按项目目录隔离，保存在 `$XDG_CONFIG_HOME/acprof/<项目路径哈希>/tui.json`；
未设置有效的 `XDG_CONFIG_HOME` 时使用 `~/.config/acprof/<项目路径哈希>/tui.json`。
设置页会显示完整保存位置。文件仅包含界面偏好和显式记住的实验默认参数；凭据仍由本地环境
配置管理。设置文件损坏时，界面提示并使用默认值，原文件保留到下一次主动保存。

“探测最大输入”会选择最小 CPU，并在可选时优先使用 `GPU=off`；内存列表按从小到大
作为候选值逐档实测。输入规模留空时取自动规划结果的最大档，手动填写时取最大值。
每档使用全新容器并最多运行一次 `/predict`；遇到明确的启动或运行期主机内存 OOM 时
自动尝试下一档，第一个成功完成最大输入请求的值才是最低可用内存。界面随后显示该档
容器冷启动、单次请求以及两者合计耗时。探测结果位于独立的 `probes/` 目录，不会写入
或修改正式实验 CSV。

界面不会重写采集逻辑，而是启动现有 `run.py`、`probe.py`、`plot.py` 和 `profile.py`。为了降低
对能耗与延迟实验的影响，正式 workload 窗口内停止常规日志重绘，不运行实时绘图，
也不轮询正在写入的 CSV；状态仅从已有进程输出中事件驱动更新。TUI 内运行时还会
禁用子进程的 tmux pane 捕获，避免把全屏 ANSI 重绘写进 `tmux_all.log`。论文复现仍可
直接复制界面显示的完整命令，在普通 CLI 或自动化脚本中执行。

同一功能也可以直接从 CLI 运行：

```bash
.venv/bin/python probe.py --model google-bert/bert-base-uncased --skip-build
```

例如传入 `--cpus 1,2,4 --mems 2,4,8 --gpus off,on` 时，固定使用
`CPU=1, GPU=off`，再依次尝试 `2GB → 4GB → 8GB`，找到第一档成功值后停止。
CUDA OOM 属于显存不足，增加这里的主机内存上限无效，因此不会继续；请求超时或其他
非 OOM 错误也会停止，避免把未验证的内存档误报为最低值。探测请求默认不设超时，
会一直等待模型返回或发生明确错误；自动化任务如需限制时长，可显式传入
`--timeout-seconds <正数>`。每次运行写入新的
`results/<model-dir>/probes/largest_scale_<timestamp>/largest_scale_probe.json`；其中
`memory_probe.attempts` 记录每档的 OOM/成功证据，`memory_probe.minimum_viable_mem_gb`
记录最低成功值；`timing.request_s` 是该成功档请求的端到端耗时，`cold_start.total_s`
是该档全新容器到 ready 的耗时，`timing.ready_plus_request_s` 是两者之和。
镜像构建和输入规划等完整命令开销另记在 `timing.command_s`。该功能不启动 idle、
能耗、抓包或 profiler 采集，因此是运行时间预估，不替代正式实验行。

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
启动后的累计耗时。启用对应 profiler 时，CPU Torch、GPU Torch、NCU、Massif、
Nsys 各自的全部采样结束后，会分别发送一次阶段通知（兼容 vendor 模式的 CPU
Advisor 也适用）。通知包含工具名称、阶段采样耗时、累计耗时、采样项总数与失败数；
明确区分成功、部分失败、失败和无结果，关闭或不适用的工具不发送阶段通知。
采样项按工具实际采样的资源配置 × input scale 计数，不按 repeat 或矩阵复用次数
重复计数。此行为也适用于 TUI 启动的 `run.py`，沿用同一 `--notify` 设置。
每个 CPU × 内存 × GPU 资源 case 完成且对应容器、监控器和
抓包进程停止后，再同步发送一次进度，包含累计耗时、已完成 case/总 case、百分比、
刚完成的资源配置以及当前 case 的结果行/异常行；全部结果合并和 tmux 日志收尾后
发送最终总结。阶段通知在对应 profiler 容器退出后、下一阶段开始前同步发送；
为了不干扰实验测量，input scale、warmup 和 repeat 窗口内部不发送
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

上面两个例子先完成主矩阵，之后可用 `profile.py` 补采计算指标。计算分析器现在默认关闭；若希望在矩阵开始前直接采集 Torch / NCU，请显式传入 `--compute-profile-tool both`。`--execution-profile-tool` 默认也是 `none`。

启动 OOM 剪枝默认开启。程序先按内存从小到大完整采集最低 CPU；只有 Docker 明确报告 `OOMKilled` 且这些失败构成连续低内存前缀时，才在后续更高 CPU 中跳过同 GPU mode、同内存上限的 case。运行期 OOM、CUDA OOM、普通启动失败和请求超时不会触发剪枝。可运行 case 的 warmup、repeat、监控器和指标口径完全不变；跳过的 case 仍写入 `status=error` 占位行，并在 `startup_oom_pruning.json` 中记录推断依据，不能作为实测性能值使用。论文若要求每个资源格都独立启动验证，传入 `--no-prune-startup-oom`。

如果已有对应模型镜像，可加 `--skip-build`。该选项只复用当前本机 Docker image store 中已存在的镜像。

正式矩阵的每个 `/predict` 请求默认最多等待 300 秒。长耗时模型可显式调整，例如
`--request-timeout-seconds 1800` 表示单个请求最多等待 30 分钟；它不限制整条命令或整个
资源矩阵的总运行时间。TUI 的“单请求超时秒”会同步写入完整命令。

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
| 单个 `/predict` 请求超时 | `300` 秒 |
| Idle 基线 / 前置冷却 | `20 / 5` 秒 |
| compute profiler | `none`（关闭；需要时显式启用或后续补采） |
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

# 单个推理请求最多等待 30 分钟
python run.py --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --request-timeout-seconds 1800

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
| `startup_oom_pruning.json` | 使用 `--prune-startup-oom` 时的参考 CPU、实测启动 OOM 前缀、推断跳过 case 和适用假设。 |
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
- `warmup=1` 的行不会进入默认图表；`status=error` 的占位行会被性能图和延迟模型排除，但会保留在 `resource_feasibility_heatmap.png` 中展示 OOM、timeout 和其他失败边界。剪枝推断的启动 OOM 会单独显示为 `P-OOM`，不会冒充实测 `OOM-S`。
- `gpu_mode=off` 时 GPU 指标为 `nan`、未启用 execution profiler 时 Massif/Nsys 指标为 `nan`，都属于预期行为。

完整字段字典、功率口径、FLOP 口径和延迟模型说明见[完整参考](REFERENCE.md)。

## 选择性能分析器

FLOP profiling 和主 latency / energy workload 相互独立：

| 选项 | 采集内容 | 适合场景 |
| --- | --- | --- |
| `--compute-profile-tool none` | 不运行 compute probe | 默认；smoke test、先完成主矩阵 |
| `torch` | Torch eager 逻辑 FLOP | 只关心模型算子形状对应的理论工作量 |
| `ncu` | GPU 实际执行的 Tensor / Scalar FLOP | 单独诊断 NVIDIA GPU |
| `both` | Torch eager；GPU 行再运行 NCU | 显式启用完整采集 |

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

默认启用的启动 OOM 剪枝只把最低 CPU 上连续实测的启动 OOM 前缀外推到更高 CPU，并保留 `startup_oom_pruning.json`；论文中应将 `P-OOM` 表述为基于资源单调性假设的不可行配置推断，而不是独立实测样本。需要逐格验证完整矩阵时传入 `--no-prune-startup-oom`。

### `container_runtime_oom`

容器已通过 `/ready`，但在 workload 期间触达 memory cgroup 上限并被 Docker 标记为 `OOMKilled`。程序会保留 OOM 前已经完成的测量行，为未完成的计划行写入 `status=error` 和 Docker 退出诊断，然后继续下一个资源 case。运行期 OOM 不参与启动 OOM 剪枝；已有成功行仍是实测值，错误占位行不会进入性能图或延迟模型。

### 运行中还没有 `result_all.csv`

这是正常的：矩阵执行期间先写 `result_case_*.csv`，全部 case 完成后才合并为 `result_all.csv`。如果在 tmux 中运行，可查看结果目录里的 `tmux_all.log`。

### `--skip-build` 后接口报错

本机可能仍是旧镜像。去掉 `--skip-build` 重新构建一次。

更多诊断，包括 idle baseline 波动、Profiler `nan`、GPU energy 和 PMU event 问题，见[完整故障排查](REFERENCE.md#troubleshooting)。

## 项目结构与开发

```text
acprof/
├── cli/          # run / probe / plot / posthoc CLI 与 TUI
├── host/         # 模型检测、容器编排、client、compute profile
├── container/    # 容器内 server、模型下载与 task handlers
├── workloads/    # 各任务族 workload generator
├── monitors/     # GPU / CPU / resource / perf side-channel monitors
└── packet/       # packet latency 解析与合并

dockerfiles/      # 各任务族及 profiler 镜像
run.py            # 主实验入口
plot.py           # 绘图入口
profile.py        # 已有结果的 profiler 补采入口
tui.py            # Textual 交互式终端界面入口
acprof-tui         # 自动使用项目 .venv 的便捷启动器
```

终端界面的交互逻辑位于 `acprof/cli/tui.py`，布局与主题样式位于 `acprof/cli/tui.tcss`；
`tui_core.py` 负责命令、验证和进度解析，`tui_settings.py` 负责本地设置的验证与原子保存。

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q acprof run.py plot.py profile.py tui.py
```

更深入的 CLI 参数、input scale 规则、音频 workload、所有输出字段、时间估算和扩展方式，请阅读 [REFERENCE.md](REFERENCE.md)。

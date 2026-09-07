# AC-Prof

AC-Prof 是一个面向 Hugging Face 推理服务的零侵入运行时分析工具。给它一个模型 ID，它会构建包含模型权重的 Docker 镜像，在不同 CPU、内存、GPU 和输入规模下运行同一组 workload，最后输出可复现的 CSV、静态元数据和图表。

`Hugging Face 模型 → Docker 镜像 → 资源矩阵实验 → 延迟 / 能耗 / 资源 / FLOP 指标 → CSV 与图表`

[快速开始](#快速开始) · [正式实验](#运行正式实验) · [终端界面](#交互式终端界面) · [企业通知](#企业微信通知) · [查看结果](#查看结果) · [性能分析](#选择性能分析器) · [排查问题](#常见问题) · [实验与指标参考](REFERENCE.md)

## 先看这两点

> **运行环境：** AC-Prof 只支持原生 Linux 主机和本机 Docker Engine，正式采集默认强制使用统一 cgroup v2。WSL、Docker Desktop、远程 Docker daemon、Windows 和 macOS 不能作为实验采集环境。当前推荐并验证的是 Ubuntu 24.04。

> **时间成本：** 默认 6 档 input scale 时，完整矩阵计划生成 1,344 行主实验（含 warmup）。按默认每行约 10 秒 workload、20 秒 Idle 基线和 5 秒前置冷却估算，仅主测量窗口就约 13 小时，且还不包含模型下载、镜像构建、case 切换，以及显式启用各类分析器时的额外耗时。第一次使用请先跑下面的最小 smoke test；具体公式见[时间成本估算](REFERENCE.md#结果行数和时间成本估算)。

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
# 可选：仅在 host 侧 perf/tcpdump 需要 sudo 且 sudo -n 不可用时设置
ACPROF_SUDO_PASSWORD=your_sudo_password
```

`.env.local` 已被 Git 忽略，可用 `chmod 600 .env.local` 限制读取权限。程序自动读取
`.env` 和 `.env.local`；令牌只用于主机检测和构建时的 BuildKit secret，正式推理容器
从镜像内的 `/models/model-snapshot` 离线加载模型，不接收令牌或在运行中下载权重。

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

### 先探测最大输入

完整矩阵开始前，可先扫描候选内存上限，确认最大输入能否完成一次请求：

```bash
python probe.py --model google-bert/bert-base-uncased \
  --cpus 1,2,4 --mems 2,4,8 --gpus off,on --skip-build
```

这个例子固定使用最小 CPU `1`，优先选择 `GPU=off`，按 `2GB → 4GB → 8GB` 实测。
输入尺度留空时取自动规划结果的最大档；手动传入 `--input-scales` 时取其中最大值。
每档使用全新容器，最多发送一次 `/predict`，第一个成功值是这些候选中的最低可用内存。
明确的启动或运行期主机内存 OOM 会推进到下一档；CUDA OOM、超时、尺度不一致和其他
错误会停止，尚未验证的更大内存不会被报告为可行。

探测请求默认不设超时，需要限制时传入 `--timeout-seconds <正数>`。
这与正式矩阵默认 `--request-timeout-seconds 300` 相互独立。结果同时报告成功档的
容器冷启动、单次请求及两者合计耗时，写入独立的 `probes/` 目录；该流程不采集能耗、
PMU 或网络指标，也不写正式 CSV。字段见[探测输出](REFERENCE.md#最大输入探测结果)。
TUI 的“探测最大输入”调用同一入口。

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

若实际规划出 6 档输入，完整矩阵包含 384 行 warmup 和 960 行正式测量。
行数、请求数与端到端耗时的区别见[时间成本估算](REFERENCE.md#结果行数和时间成本估算)。
大矩阵开始前也应检查 profiler artifacts 的磁盘占用。

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

### 镜像复用、超时与失败处理

启动 OOM 剪枝默认开启。程序先按内存从小到大完整采集最低 CPU；只有 Docker 明确报告 `OOMKilled` 且这些失败构成连续低内存前缀时，才在后续更高 CPU 中跳过同 GPU mode、同内存上限的 case。运行期 OOM、CUDA OOM、普通启动失败和请求超时不会触发剪枝。可运行 case 的 warmup、repeat、监控器和指标口径完全不变；跳过的 case 仍写入 `status=error` 占位行，并在 `startup_oom_pruning.json` 中记录推断依据，不能作为实测性能值使用。论文若要求每个资源格都独立启动验证，传入 `--no-prune-startup-oom`。

传入 `--skip-build`，或在 TUI 勾选“复用现有镜像”后，采集与探测都会提前检查本机 Docker image store 中的目标模型镜像：存在就跳过构建并复用；不存在则在日志中提示，并自动构建后继续任务。Docker 查询失败（如连接或权限错误）会明确报错，不会被当作镜像缺失。未勾选时仍执行正常构建。

正式矩阵的每个 `/predict` 请求默认最多等待 300 秒。长耗时模型可显式调整，例如
`--request-timeout-seconds 1800` 表示单个请求最多等待 30 分钟；它不限制整条命令或整个
资源矩阵的总运行时间。TUI 的“单请求超时秒”会同步写入完整命令。

## 交互式终端界面

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
点击界面右上角的“×”或按 `Ctrl+Q` 退出；任务运行中会提示先安全终止任务。
底部输入框还支持 `/run`、`/probe`、`/check`、`/status`、`/stop`、`/plot`、`/profile` 和 `/help`
等快捷命令。

点击“环境检查”或按 `F6` 后，界面立即切换到“运行监控”显示日志，无需等待检查结束。
环境检查不会启动模型或正式采集。`perf instructions` 与正式启动共用权限探测，依次尝试
普通用户、免交互 sudo，以及通过 `ACPROF_SUDO_PASSWORD` 配置的 sudo；读到有效指令计数
才显示通过，并注明使用的方式。失败时日志保留各次尝试的错误。该检查不会修改系统权限设置，
正式启动时仍会执行完整预检。

输入框使用细竖线插入光标，停下输入后亮 0.5 秒、灭 0.5 秒；键入、移动或点击后立即显示。
TUI 直接控制光标亮灭，不依赖终端的闪烁设置，也不为闪烁重绘界面。离开输入框后隐藏并停止
计时，正式测量窗口内暂停闪烁。光标形状需要终端支持 DECSCUSR；退出或挂起 TUI 时重置样式。

操作按钮统一使用透明底色和细线圆角边框，配色随主题切换；主操作、警告和终止操作用不同颜色区分。
悬停或键盘聚焦时突出文字与边框，禁用时淡化显示；日志工具栏和确认弹窗也使用同一风格。

设置页只放界面偏好：界面语言（简体中文 / English）、八种主题（深海蓝、纸白、石墨灰、松林绿、暮紫、琥珀、暖砂、雾蓝）、
日志保留行数（500 / 1000 / 3000 / 10000）、日志自动换行，以及是否显示底部快捷命令框。
修改当次生效，点击“保存设置”后，下次启动沿用。
按 `F2` 打开设置，在“界面语言 / Language”选择 `English` 即可切换为英文；
页面、表单、提示、确认弹窗和监控状态同步切换，当前输入、预设、日志选区和已读取摘要保留。
日志中的原始子进程输出保留原文，模型 ID、路径、命令和实验数据不做翻译。
默认使用简体中文；已有设置文件缺少语言字段时也使用简体中文。
换行设置会重新排布已有日志；行数上限按原始文本行计算，调整后立即裁剪较旧内容。
“恢复界面默认”先恢复当次显示，再点击保存即可保留。

采集参数留在实验页。点击“高级参数”，切换到 batch size、warmup/repeat、请求窗口、采样
频率、请求超时、分析器和任务覆盖参数等选项；点击“返回基本配置”回到模型和资源矩阵。
高级参数内的“记住实验配置”会保存当前实验表单，下次打开此项目时自动填入。
模型 ID 会在确认启动采集或最大输入探测时自动记住，无需点击保存；即使任务失败或被终止，
下次打开仍会填入该模型。仅修改输入、预览命令或取消确认不会更新记录。
模型的填入顺序为：命令行 `--model` → 最近启动的模型 → 手动保存的实验配置。
`--preset` 优先于保存的其他实验参数，并保留按上述顺序选出的模型。运行任务期间，配置控件暂时锁定。

设置文件按项目目录隔离，保存在 `$XDG_CONFIG_HOME/acprof/<项目路径哈希>/tui.json`；
未设置有效的 `XDG_CONFIG_HOME` 时使用 `~/.config/acprof/<项目路径哈希>/tui.json`。
设置页会显示完整保存位置。文件仅包含界面偏好、显式记住的实验默认参数和最近启动的模型 ID；
凭据仍由本地环境配置管理。设置文件损坏时，界面提示并使用默认值，自动记忆不会覆盖原文件；
下一次主动保存设置后恢复自动记忆。模型 ID 保存失败会提示，但不阻止任务启动。

监控页优先显示日志，资源矩阵默认折叠，点击标题或输入 `/matrix` 可展开。
日志是只读文本：鼠标拖动选择，双击选词，`Ctrl+A` 全选，`Ctrl+C` 或“复制选区”复制。
复制保留原始文本，屏幕上的自动换行不会变成额外换行；系统剪贴板通过终端的 OSC 52 通道写入，
需要终端允许该功能。点击“放大日志”、按 `F8` 或输入 `/log` 可铺满终端，`Esc` 或“返回监控”恢复。
放大时仍可复制、清空和终止任务。滚动查看历史或选择文字后暂停跟随，新增日志继续保留，
点击“回到最新”恢复自动跟随。滚动条使用与主题一致的整格滑块，避免半格字形形成黑色断带。

界面不会重写采集逻辑，而是启动现有 `run.py`、`probe.py`、`plot.py` 和 `profile.py`。为了降低
对能耗与延迟实验的影响，正式 workload 窗口内停止常规日志重绘，不运行实时绘图，
也不轮询正在写入的 CSV；状态仅从已有进程输出中事件驱动更新。TUI 内运行时还会
禁用子进程的 tmux pane 捕获，避免把全屏 ANSI 重绘写进 `tmux_all.log`。论文复现仍可
直接复制界面显示的完整命令，在普通 CLI 或自动化脚本中执行。

## 企业微信通知

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

`--notify` 默认为 `auto`（检测到 Webhook 自动启用）；传入 `--notify wecom` 显式启用，传入 `--notify none` 临时关闭：

```bash
python run.py --model google-bert/bert-base-uncased --notify none
```

通知覆盖实验开始、已启用 profiler 的各工具阶段完成、每个资源 case 完成和最终总结。
CPU Torch、GPU Torch、NCU、Massif、Nsys 各自汇总实际采样项、失败数、阶段耗时和累计耗时；
兼容 vendor 模式的 CPU Advisor 同样适用。阶段状态区分成功、部分失败、失败和无结果。
关闭或不适用的工具不发阶段通知，代表资源复用不重复计数。最终总结区分成功、部分成功、
无结果、失败和用户取消。TUI 启动的 `run.py` 使用同一设置，独立 `profile.py` 不发送这些通知。

通知在测量窗口外发送：profiler 阶段返回后、下一个阶段前，或 case 的容器、监控器与抓包
全部停止后。input scale、warmup 和 repeat 窗口内部不发送网络通知。每次发送超时 5 秒，
最多尝试两次；失败只产生警告，不改变测量结果或原退出码。Webhook 只保存在本地环境配置中。

## 查看结果

结果目录为 `<output-dir>/<model-dir>/`，模型 ID 中的 `/` 替换为 `--`。
例如 `google-bert/bert-base-uncased` 对应 `google-bert--bert-base-uncased/`。

| 阅读目的 | 入口 |
| --- | --- |
| 查看测量值 | `result_all.csv`；正式性能分析筛选 `status=ok` 且 `warmup=0`。 |
| 复现实验对象和输入 | `static_meta.json` 与 `input_scale_plan.json`。 |
| 追踪补采或修复 | `collection_history.json` 与对应 profiler plan。 |
| 查看图表和拟合 | `cpu/`、`gpu/`、`gpu+cpu/` 与 `latency_model/`。 |

`latency_app_s` 是客户端应用层计时，`latency_s` 是抓包解析得到的 packet-level 计时。
关闭 GPU 或未启用某个 profiler 时，对应字段为 `nan` 属于预期结果。
运行中先写 `result_case_*.csv`，矩阵完成后才合并为 `result_all.csv`。

完整说明集中在[输出文件](REFERENCE.md#输出文件)、[CSV 字段字典](REFERENCE.md#result_allcsv-字段解释)
和[常见判断](REFERENCE.md#常见判断)。

## 选择性能分析器

FLOP profiling 和主 latency / energy workload 相互独立：

| 选项 | 采集内容 | 适合场景 |
| --- | --- | --- |
| `--compute-profile-tool none` | 不运行 compute probe | 默认；smoke test、先完成主矩阵 |
| `torch` | Torch eager 逻辑 FLOP | 只关心模型算子形状对应的理论工作量 |
| `ncu` | GPU 实际执行的 Tensor / Scalar FLOP | 单独诊断 NVIDIA GPU |
| `both` | Torch eager；GPU 行再运行 NCU | 显式启用完整采集 |

Torch probe 会强制并验证 eager attention，正式请求仍使用正常运行时的 attention 实现。
NCU 需要主机上的 `ncu` 和可用的 GPU 性能计数器，可用 `--ncu-root` 指定工具位置。

Execution profiling 默认关闭。显式启用后采用缩减采样，并把来源记录到 plan 与静态元数据：

- `--execution-profile-tool massif`：CPU-only 的 process-lifetime 内存峰值。
- `--execution-profile-tool nsys`：GPU 的 CUDA API、kernel 和 memcpy timeline。
- `--execution-profile-tool both`：同时启用两者。
- Massif 默认 `--massif-sampling per-scale`：最大 CPU/内存 × 每个 input scale。
- Nsys 默认 `--nsys-sampling per-cpu-scale`：全部 CPU × 最大内存 × 每个 input scale。

新构建的模型共享 `acprof-base` 中预装的 Valgrind 和 Nsys 运行库；启用分析时直接使用模型镜像，无需为每个新模型再构建 Massif / Nsys 镜像。Nsys 主程序仍从宿主机挂载，可用 `--nsys-root` 指定；host 无需安装 Valgrind。两个工具只在独立分析探针中运行。已有旧模型镜像无需重新下载权重：首次使用时按需构建兼容镜像，以后实际模型镜像 ID 和分析 Dockerfile 均未改变时直接复用，跳过 `docker build`。

需要严格采完整资源矩阵时显式传入：

```bash
python run.py --model google-bert/bert-base-uncased \
  --execution-profile-tool both \
  --massif-sampling full --nsys-sampling full
```

代表资源默认取本次 `--cpus` / `--mems` 中的最大值，也可用
`--massif-reference-cpu`、`--massif-reference-mem`、
`--nsys-reference-cpu`、`--nsys-reference-mem` 显式选择。

### 补采已有结果

完成主矩阵后，结果目录需同时具有 `result_all.csv`、`static_meta.json` 和
`input_scale_plan.json`。先检查计划，再执行补采：

```bash
python profile.py results/google-bert--bert-base-uncased --dry-run
python profile.py results/google-bert--bert-base-uncased --tools torch,ncu
```

不传 `--tools` 时，默认补齐适用且尚未成功的 `torch,ncu,nsys,massif`。
Torch 匹配已有 CPU/GPU 数据，NCU/Nsys 只用于 GPU 行，Massif 只用于 CPU-only 行。
补采沿用前述 Massif/Nsys 采样策略，也支持 `--massif-sampling full`、
`--nsys-sampling per-scale` 或 `--nsys-sampling full`。

NCU 和 Massif 会按 input scale 保存 checkpoint。NCU 可复用匹配的 CSV 或从已有
`.ncu-rep` 恢复，Massif 可复用匹配的 `.out`；模型 revision、镜像、资源、repeat 和
NCU metrics（适用时）必须匹配，才能恢复旧报告。分析固定使用不可变镜像 ID，
旧 tag 形式的 Massif checkpoint 与新 ID 不匹配时会重采。完整成功的已有 plan 也可复用。
默认保留已有成功 CSV 值；`--force-reprofile` 强制重新采集并替换所选 profiler 字段。

写入前把旧文件备份到 `posthoc_backups/<timestamp>/`，验证临时文件后原子替换
`result_all.csv`、`static_meta.json` 和 `collection_history.json`，失败时从备份恢复。
操作记录追加到 `posthoc_profile_history`，原始实验命令和非 profiler 字段保持原样。
旧结果首次成功补采时会创建历史文件并迁移已有记录；报告与补采 plan 位于 `posthoc_profiles/`。
同一结果目录若仍被采集或分析进程使用，补采会拒绝启动。

### 从已有计划生成派生 CSV

如果 latency 已采集完成、之后才生成 `compute_profile_plan.json`，可以写出一份带 FLOP/MFLOPS 的新 CSV：

```bash
python -m acprof.cli.backfill_compute \
  results/google-bert--bert-base-uncased/result_all.csv \
  results/google-bert--bert-base-uncased/compute_profile_plan.json \
  --output results/google-bert--bert-base-uncased/result_all.with_compute.csv
```

工具按 GPU mode 和 input scale 匹配已有计划，生成带 Torch/NCU 字段的派生 CSV。
输出采用原子写入，默认拒绝覆盖已有文件；确需替换显式输出路径时追加 `--overwrite`。
FLOP/MFLOPS 的单位、延迟分母和缺失值规则见[计算指标字典](REFERENCE.md#torch-与-ncu-计算指标)。

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

这是正常的：矩阵执行期间先写 `result_case_*.csv`，全部 case 完成后才合并为 `result_all.csv`。如果在 tmux 中运行，采集期间查看对应终端；`tmux_all.log` 在命令结束或报错退出时落盘。

### `--skip-build` 后接口报错

本机可能仍是旧镜像。去掉 `--skip-build` 重新构建一次。

更多诊断，包括 idle baseline 波动、Profiler `nan`、GPU energy 和 PMU event 问题，见[常见判断](REFERENCE.md#常见判断)。

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
probe.py          # 最大输入与最低候选内存探测
plot.py           # 绘图入口
profile.py        # 已有结果的 profiler 补采入口
tui.py            # Textual 交互式终端界面入口
acprof-tui         # 自动使用项目 .venv 的便捷启动器
```

终端界面的交互逻辑位于 `acprof/cli/tui.py`，布局与主题样式位于 `acprof/cli/tui.tcss`；
`tui_core.py` 负责命令、验证和进度解析，`tui_settings.py` 负责本地设置的验证与原子保存，
`tui_i18n.py` 集中管理界面文案与语言选项。

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q acprof run.py probe.py plot.py profile.py tui.py
```

同一任务族中的新模型通常由 `acprof/host/detect.py` 自动识别；确需新增任务族时，要同步
补齐 `acprof/config.py` 的检测映射与尺度定义、`acprof/container/handlers/` 的模型处理、
`acprof/workloads/` 的确定性输入、`dockerfiles/` 的离线镜像构建，并覆盖检测、尺度、
离线加载和编排测试。

修改输出或指标时，同步维护 [REFERENCE.md](REFERENCE.md) 的字段来源、单位、适用条件与历史兼容说明。

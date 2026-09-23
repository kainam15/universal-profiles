# 安装与运行指南

首次使用可先按[首页的快速开始](../README.md#快速开始)跑通一个 basic CPU 实验。
本文保留完整主机检查、认证配置，以及 full、GPU、ONNX 和资源矩阵示例。
下文保留源码目录和 `.venv` 的开发方式；安装后的 `acprof` 命令可从任意工作目录执行。
隔离工具安装、standalone 下载及发布方式见[发行包说明](Distribution.md)。

[文档导航](README.md) · [CLI 参数](CLI_Reference.md) · [运行排障](Troubleshooting.md)

## 快速开始

### 1. 检查主机环境

AC-Prof 只支持原生 Linux 主机和本机 Docker Engine，正式采集强制使用统一 cgroup v2。
WSL、Docker Desktop、远程 Docker daemon、Windows 和 macOS 不能作为实验采集环境。
当前推荐并验证的是 Ubuntu 24.04。

必需条件：

- Python 3.10+。
- 当前用户可以直接访问 `unix:///var/run/docker.sock`，无需使用 `sudo docker`。
- Host 使用统一 cgroup v2；`/sys/fs/cgroup/cgroup.controllers` 必须存在。
- Hugging Face Hub 可访问；私有或 gated 模型还需要 `HF_TOKEN`。
- `full` 模式要求 Linux RAPL powercap 可读。
- `full` 模式要求 Linux `perf` 可以访问硬件 `instructions` 事件。
- `full` 模式要求 `tcpdump`、`tshark` 和本机 Docker bridge 可用。
- 运行 `--gpus on` 时，还需要 NVIDIA driver 和 NVIDIA Container Toolkit。

安装 AC-Prof 后可先运行只读检查，查看当前缺项及处理建议：

```bash
acprof doctor --profiling-mode basic
acprof doctor --profiling-mode full --gpus on
```

standalone 不要求目标机安装 Python。`doctor` 不自动安装软件、更改权限或启动容器；
通过仅表示所选模式的前置检查通过，仍需运行最小实验验证。

先确认 Docker 指向本机 daemon。下面的命令会清除当前 shell 的 Docker 覆盖变量，
并切换默认 context；确认你要使用本机 Docker 后再执行：

```bash
unset DOCKER_HOST DOCKER_CONTEXT
docker context use default
docker context inspect default --format '{{(index .Endpoints "docker").Host}}'
docker info --format 'OperatingSystem={{.OperatingSystem}}'
test -f /sys/fs/cgroup/cgroup.controllers
cat /proc/self/cgroup
```

`docker context inspect` 应输出 `unix:///var/run/docker.sock`。`full` 模式还需检查采集工具：

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

首次获取源码：

```bash
git clone https://github.com/kainam15/universal-profiles.git
cd universal-profiles
```

在仓库根目录安装主机依赖；已有 `.venv` 时直接激活并安装：

```bash
# 仅在 .venv 不存在时执行下一行
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
```

容器运行依赖由独立的平台和完整制品锁管理：保留 7 个任务族，37 个逻辑 profile 共享为
24 个依赖环境，其中 `onnxruntime-cpu` 完全不安装 Torch。镜像按需构建和复用；只读检查可运行 `python scripts/compile_locks.py --check`，
分层及锁更新命令见[运行兼容](Runtime_Compatibility.md#当前配置)。

### Hugging Face 认证

私有或 gated 模型可在当前工作目录创建 `.env.local`（从源码根目录启动时就是项目根目录）：

```env
HF_TOKEN=hf_xxx
# 可选：仅在 host 侧 perf/tcpdump 需要 sudo 且 sudo -n 不可用时设置
ACPROF_SUDO_PASSWORD=your_sudo_password
```

`.env.local` 已被 Git 忽略，可用 `chmod 600 .env.local` 限制读取权限。程序自动读取
`.env` 和 `.env.local`；令牌只用于主机检测和构建时的 BuildKit secret，正式推理容器
从镜像内的 `/models/model-snapshot` 离线加载模型，不接收令牌或在运行中下载权重。
地址、令牌的优先级与空白值处理见 [主机环境与 Hugging Face 认证](CLI_Reference.md#主机环境与-hugging-face-认证)。

### 3. 跑一个最小 smoke test

下面使用默认 `full` 模式，只运行一个 CPU、一个内存限制、一个输入尺度和一个请求，
并关闭独立 profiler。首次跑通流程可使用[首页的 basic 示例](../README.md#快速开始)；
本例还要求前述 RAPL、perf 和抓包条件。

```bash
python run.py --model google-bert/bert-base-uncased \
  --cpus 1 --mems 4 --gpus off \
  --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none \
  --execution-profile-tool none \
  --output-dir results/smoke
```

Stable Diffusion 建议先做单 GPU、单分辨率 smoke test（同样使用 `full` 模式）：

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

### 无 Torch 的 ONNX Runtime CPU 示例

`basic` 只采集 application latency、吞吐、容器 CPU 和内存；仍要求原生 Linux、本机 Docker、
cgroup v2。RAPL、perf、抓包不参与此模式，结果明确记录模式和能力状态。默认 `full` 保留原有严格条件。

```bash
.venv/bin/python run.py --model Ritual-Net/iris-classification \
  --task tabular-classification --backend onnxruntime \
  --workload-spec examples/onnxruntime/iris.json --input-scales 1 \
  --cpus 1 --mems 1 --gpus off --batch-size 1 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --profiling-mode basic --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/onnx-basic
```

此模型输入固定为 `[1,4]`，更大的输入或 batch 会明确拒绝。模型 revision 会解析并写入结果；
任务 sanity check 不等于分类准确率评测。接口与声明格式见[扩展声明](Runtime_Compatibility.md#扩展声明与按需加载)。

离线容器回归（含真实服务、CPU/内存采集、落盘和审计）：

```bash
.venv/bin/python scripts/check_runtime.py --profile onnxruntime-cpu --basic-e2e \
  --output-dir internal-testing/onnx-runtime
```

图像分类和多输入文本分类复用同一 ONNX 环境与现有任务生成器；
支持边界及预训练小模型的复现步骤见[无 Torch 验收](Testing.md#无-torch-运行时验收)。

## 查看结果

用 `python plot.py <result_all.csv 路径>` 生成图表；[结果阅读指南](Metrics.md#从结果目录开始)说明输出目录、分析过滤、只读审计与统计入口。

## 运行正式实验

建议先逐步扩大规模：最小 smoke test → 单个资源配置的全部 input scale → 不带 profiler 的目标资源矩阵 → 最后补采高开销 profiler。

中断后使用原命令加 `--resume`，或在 TUI 高级参数勾选“恢复未完成实验”。系统会核对原参数、
镜像和输入计划，保留完成的 case，并备份后重新测量中断的 case。新实验应选择新的输出目录；
已有产物不会被默认覆盖。符合当前产物协议的实验没有恢复状态文件时仍可绘图、补采，主实验恢复约定见
[结果完整性与断点续跑](Profiling_Protocol.md#结果完整性与断点续跑)。

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
PMU 或网络指标，也不写正式 CSV。字段见[探测输出](Profiling_Protocol.md#最大输入探测结果)。
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

上面两个例子使用默认 `full` 模式，先完成主矩阵，之后可用 `profile.py` 补采计算指标。计算分析器现在默认关闭；若希望在矩阵开始前直接采集 Torch / NCU，请显式传入 `--compute-profile-tool both`。`--execution-profile-tool` 默认也是 `none`。

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
行数、请求数与端到端耗时的区别见[时间成本估算](Profiling_Protocol.md#结果行数和时间成本估算)。
默认每行约 10 秒 workload、20 秒 Idle 基线和 5 秒前置冷却，仅主测量窗口就约 13 小时；
模型下载、镜像构建、case 切换与显式启用的 profiler 还需额外时间。
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

- 模型文件筛选、依赖/模型/代码分层及缓存含义见[运行兼容说明](Runtime_Compatibility.md#模型文件选择规则)。
- `--skip-build` 核验匹配后复用，不存在则构建；镜像与独立推理验证的失败边界见[构建、复用和验证](Runtime_Compatibility.md#构建复用和验证)。
- 启动 OOM、剪枝推断及部分结果处理见[排障说明](Troubleshooting.md#启动-oom-与剪枝占位)。
- 长请求可设置 `--request-timeout-seconds 1800`；它限制单次请求，不限制整个矩阵。默认值与适用阶段见[CLI 参数](CLI_Reference.md#请求窗口与采样)。

通知配置见[企业微信通知](CLI_Reference.md#企业微信通知)，补采流程见[补采已有结果](Profilers.md#补采已有结果)。

# AC-Prof Universal Profiler

AC-Prof 是一个面向 containerized HuggingFace inference service 的运行时 profiling 工具。它会把模型权重 bake 进 Docker image，在不同 CPU / Memory / GPU 资源限制和不同 input scale 下运行推理 workload，并输出 latency、throughput、cold start、GPU power / energy、packet-level latency 等指标。

本项目是 loose Python modules 结构，没有 `pyproject.toml` / `setup.py`。入口是 [run.py](run.py)，绘图入口是 [plot.py](plot.py)。

## 1. 环境要求

基础要求：

- Python 3.10+
- Docker
- Hugging Face Hub 网络访问，或可用的 `HF_TOKEN`
- `pip install -r requirements.txt`

可选但会影响字段完整性：

- NVIDIA GPU + NVIDIA Container Toolkit：用于 `--gpus on` 和 GPU energy metrics。
- Linux RAPL powercap（`/sys/class/powercap/*/energy_uj`）：用于 CPU package power / energy 和 estimated vCPU energy metrics。
- `tcpdump` + `tshark`：用于填充 `result_all.csv` 的 `latency_s` packet-level latency。
- Linux / WSL native Docker 的 `docker0` bridge：packet sniffing 默认监听 `docker0`。如果抓包不可用，`latency_s` 会保留为 `nan`，但 `latency_app_s` 仍然可用。

Hugging Face token 可以放在项目根目录 `.env.local`：

```env
HF_TOKEN=hf_xxx
```

`run.py` 会自动读取 `.env` 和 `.env.local`，并把 `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` 传给 Docker build 和 runtime。

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
- repeat-in-window: 20 requests per row

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

### 采集 `latency_s` 的推荐启动方式

如果只需要 host-side application latency，看 `latency_app_s` 即可，普通 `python run.py ...` 可以使用。  
如果要稳定采集 `latency_s`，推荐从 WSL 启动，并显式使用 WSL native Docker daemon，不要落回 Docker Desktop daemon：

```bash
wsl.exe sh -lc "cd /mnt/d/DOR/universal-profiles && DOCKER_HOST=unix:///var/run/docker-native.sock python3 run.py --model google-bert/bert-base-uncased --skip-build --output-dir results/test"
```

这条路径用于让 `tcpdump` 在 WSL Docker 的 `docker0` bridge 上抓到容器流量。前提是：

- WSL native Docker daemon 已启动，并监听 `unix:///var/run/docker-native.sock`。
- WSL 内已安装 `tcpdump` 和 `tshark`。
- `--skip-build` 只有在该 native daemon 的 image store 里已经有对应 image 时才可用；Docker Desktop 和 WSL native Docker 的 image store 不是同一个。
- 实验需要完整跑到 merge 阶段，`latency_s` 才会从 `nan` 更新为 packet-level latency。

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
- `avg_power_vs_scale.png`
- `energy_vs_scale.png`
- `cpu_avg_power_vs_scale.png`
- `cpu_energy_vs_scale.png`
- `vcpu_avg_power_vs_scale.png`
- `vcpu_energy_vs_scale.png`
- `throughput_vs_scale.png`
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
| `--repeat-in-window` | `20` | 每一行内部连续发送的 `/predict` request 数量。 |
| `--sample-hz` | `20.0` | GPU power sampling rate，单位 Hz。 |
| `--idle-seconds` | `3.0` | GPU idle baseline 测量时长。 |
| `--input-scales` | auto | 手动覆盖 input scale 列表。未提供时自动规划 6 档。 |
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
| `*.png` | `plot.py` 生成的图表。 |

中间文件 `result_case_*.csv`、`lat_case_*.json`、`sniff_case_*.pcap` 会在 `result_all.csv` 成功 merge 后自动清理。若运行被中断，这些中间文件可能保留。

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
| `sniff_group_id` | 用于 packet latency merge 的 request group ID。`client.py` 会把它写入 HTTP `X-Req-Id`。 |
| `repeat_in_window` | 本行内部连续发送的 request 数量。`latency_app_s` 和 `latency_s` 都是该 window 内 request 的平均值。 |
| `latency_s` | packet-level latency，来自 `tcpdump` PCAP + `tshark` 解析 + `merge_packet_latency.py` merge。抓包不可用、PCAP 为空或运行中断时为 `nan`。 |
| `latency_app_s` | host-side application latency。`client.py` 用 `requests.post()` 外层 `time.perf_counter()` 测得，通常比 `latency_s` 更容易稳定产出。 |
| `throughput_samples_per_s` | 吞吐量，约等于 `batch_size / latency`。如果 `latency_s` 成功 merge，会优先按 `latency_s` 更新；否则按 `latency_app_s` 计算。 |
| `idle_power_w` | GPU idle baseline power，单位 W。仅 `gpu_mode=on` 且 NVML 可用时有值。 |
| `energy_iters` | GPU energy measurement 内部采样窗口中的 iteration 数。 |
| `avg_power_total_w` | 测量窗口内 GPU total average power，单位 W。 |
| `peak_power_total_w` | 测量窗口内 GPU total peak power，单位 W。 |
| `energy_total_j` | 本行平均到单 request 的 total GPU energy，单位 J。 |
| `avg_power_eff_w` | 扣除 idle baseline 后的 effective average power，单位 W。 |
| `peak_power_eff_w` | 扣除 idle baseline 后的 effective peak power，单位 W。 |
| `energy_eff_j` | 本行平均到单 request 的 effective GPU energy，单位 J。 |
| `cpu_idle_power_w` | CPU package idle baseline power，单位 W。仅 Linux/WSL 暴露 RAPL `/sys/class/powercap/*/energy_uj` 时有值。 |
| `cpu_energy_iters` | CPU package energy measurement 采样窗口中的 sample 数。 |
| `cpu_avg_power_total_w` | 测量窗口内 host CPU package total average power，单位 W。 |
| `cpu_peak_power_total_w` | 测量窗口内 host CPU package total peak power，单位 W。 |
| `cpu_energy_total_j` | 本行平均到单 request 的 total CPU package energy，单位 J。 |
| `cpu_avg_power_eff_w` | 扣除 CPU idle baseline 后的 CPU package effective average power，单位 W。 |
| `cpu_peak_power_eff_w` | 扣除 CPU idle baseline 后的 CPU package effective peak power，单位 W。 |
| `cpu_energy_eff_j` | 本行平均到单 request 的 effective CPU package energy，单位 J。 |
| `vcpu_cpu_share` | container cgroup CPU time delta / host active CPU time delta，用于估算本 container 占 host active CPU 的比例。 |
| `vcpu_cpu_time_s` | 本行平均到单 request 的 container CPU time，单位秒。 |
| `vcpu_avg_power_total_w` | 按 `vcpu_cpu_share` 分摊后的 estimated vCPU total average power，单位 W。 |
| `vcpu_peak_power_total_w` | 按 `vcpu_cpu_share` 分摊后的 estimated vCPU total peak power，单位 W。 |
| `vcpu_energy_total_j` | 本行平均到单 request 的 estimated vCPU total energy，单位 J。 |
| `vcpu_avg_power_eff_w` | 按 `vcpu_cpu_share` 分摊后的 estimated vCPU effective average power，单位 W。 |
| `vcpu_peak_power_eff_w` | 按 `vcpu_cpu_share` 分摊后的 estimated vCPU effective peak power，单位 W。 |
| `vcpu_energy_eff_j` | 本行平均到单 request 的 estimated vCPU effective energy，单位 J。 |
| `cold_start_s` | 当前 container 从 `docker run` 到 `/ready` 成功的时间，单位秒。 |
| `status` | `ok`、`warn` 或 `error`。`warn` 常用于可继续分析但存在异常值的行。 |
| `error` | 错误或 warning 文本。正常行为空。 |

### `latency_s` 和 `latency_app_s` 的区别

- `latency_app_s` 是 client 侧应用层计时，只要 `/predict` 请求成功，一般就能写出。
- `latency_s` 是 packet-level 计时，需要完整完成 `tcpdump` capture、`sniff_parse_pcap.py` parse、`merge_packet_latency.py` merge。
- 因此 `latency_s=nan` 不一定表示推理失败；先看 `status`、`error`、`latency_app_s`，再判断是否是抓包路径问题。

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
| `model_weight_bytes` | Docker image 内 `/models/hf` 下 Hugging Face cache artifacts 的总字节数，不是严格的单一权重文件大小。 |
| `docker_image_bytes` | `docker image inspect <image_tag> --format "{{.Size}}"` 返回的本地 image size，单位 bytes。 |
| `environment` | 自动检测的运行环境标签，例如 `windows11+wsl`、`ubuntu24.04+wsl`、`ubuntu24.04`、`macos15`。 |
| `cpu_power_source` | CPU package 功耗来源。`rapl` 表示使用 Linux RAPL powercap 真实计数器；`unavailable` 表示当前环境没有可用 RAPL。 |
| `vcpu_power_method` | estimated vCPU 功耗计算方法。`rapl_cgroup_cpu_share` 表示用 RAPL package energy 乘以 container cgroup CPU share；`unavailable` 表示无法估算。 |

## 10. 结果行数估算

`result_all.csv` 行数大致为：

```text
len(cpus) * len(mems) * len(gpus) * len(input_scales) * (warmup + repeat)
```

每一行内部还会发送：

```text
repeat_in_window
```

个 `/predict` request。因此默认完整 run 的 request 数较大：

```text
4 CPU * 4 MEM * 2 GPU * 6 scales * (2 warmup + 5 repeat) * 20 requests = 26880 requests
```

## 11. 常见判断

`status=ok` 但 `latency_s=nan`：

- 推理请求成功，但 packet sniffing 没有成功 merge。
- 检查 `tcpdump`、`tshark`、`--sniff-iface`、Docker bridge，以及运行是否被中断。
- 如果只需要应用层 latency，可以使用 `latency_app_s`。

GPU energy 字段全是 `nan`：

- `gpu_mode=off` 时这是正常结果。
- `gpu_mode=on` 时检查 NVIDIA driver、NVIDIA Container Toolkit、`pynvml` 和容器 GPU 可见性。

CPU / vCPU energy 字段全是 `nan`：

- 这是当前环境没有暴露 RAPL `/sys/class/powercap/*/energy_uj` 时的正常结果，不会影响 latency / throughput 采集。
- `cpu_*` 字段是 host CPU package 的真实 RAPL 测量值；`vcpu_*` 字段是在同一窗口内按 container cgroup CPU share 分摊出来的估计值。
- 本项目不会用 TDP 或 CPU utilization 造功耗值；没有 RAPL 时保持 `nan`。

`--skip-build` 后 `/scale_meta` 或 `/probe` 报错：

- 可能使用了旧 image。重新运行一次不带 `--skip-build` 的 build。

## 12. 扩展新模型或任务族

通常不需要为新 Hugging Face model 写代码。`detect.py` 会尽量自动识别任务和 backend。

新增 task family 时，需要同时补齐：

- `handlers/`：容器内 model load / preprocess / predict / postprocess。
- `workloads/`：host 侧 workload generator。
- `dockerfiles/`：对应 task family 的 Dockerfile。
- `config.py`：`PIPELINE_TAG_TO_FAMILY` 和 `SCALING_DIMENSIONS`。

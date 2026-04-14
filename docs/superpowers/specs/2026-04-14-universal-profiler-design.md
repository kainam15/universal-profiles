# AC-Prof Universal HuggingFace Model Profiler -- Design Spec

## Context

AC-Prof 已有一套针对 Amazon Chronos 时间序列模型的专用采集代码（`example-code/`）。该代码采用 Control-Execution-Monitor 分离架构，通过 Docker 资源约束 + 侧信道监控（NVML 能耗 + tcpdump 网络延迟）实现零侵入式性能采集。

**本次目标**：将这套框架泛化为一个通用工具，用户只需提供 HuggingFace 模型 ID，即可自动完成模型类型检测、Docker 容器构建、资源矩阵扫描和数据采集。

---

## 架构总览

采用 **按任务族分 Docker 镜像 + 可插拔 Handler + Runtime Backend 抽象** 的方案。

```
┌──────────────────────────────────────────────────────────────┐
│  run.py (入口)                                                │
│  python run.py --model bert-base-uncased                     │
└──────────────┬───────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  detect.py           │  HF Hub API → pipeline_tag → task_family + runtime_backend
    │  (任务自动检测)       │  支持 --task / --task-family / --backend 人工覆盖
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  orchestrator.py     │  Docker 生命周期 · 资源约束 · tcpdump · 冷启动计时
    │  (编排器)             │  替代 run_case.sh / run_matrix.sh
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────────────────────────────┐
    │  Docker Container                                        │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ server.py (通用 Flask 服务)                          │ │
    │  │ TASK_FAMILY + TASK_TYPE + RUNTIME_BACKEND env vars  │ │
    │  │     ↓                                                │ │
    │  │ Handler Registry → handler.load() → handler.predict()│ │
    │  └─────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐       ┌────────────────────────┐
    │  client.py (通用客户端)│ ←── │  workloads/ (输入生成器) │
    │  负载发送 · 延迟计时   │       │  nlp / cv / audio / ts  │
    │  能耗采集 · CSV 写入   │       └────────────────────────┘
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────────────────────┐
    │  Monitor (侧信道, 复用 example-code)             │
    │  energy_nvml.py · sniff_parse_pcap.py            │
    │  merge_packet_latency.py                         │
    └─────────────────────────────────────────────────┘
```

---

## 1. 三层抽象：Task Family / Runtime Backend / Handler

### 1.1 Task Family（任务族）

将 HuggingFace 模型分为 4 个任务族，每族一个 Dockerfile：

| 任务族 | 覆盖的 pipeline_tag | 输入缩放维度 |
|--------|---------------------|-------------|
| **nlp** | text-generation, text2text-generation, text-classification, token-classification, question-answering, summarization, translation, fill-mask, feature-extraction, zero-shot-classification, sentence-similarity | seq_length (64, 128, 256, 512, 1024, 2048) |
| **cv** | image-classification, object-detection, image-segmentation, depth-estimation, image-to-text, zero-shot-image-classification | resolution_scale (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0) |
| **audio** | automatic-speech-recognition, audio-classification, text-to-speech, audio-to-audio | duration_s (1, 2, 5, 10, 20, 30) |
| **timeseries** | time-series-forecasting | context_length (64, 128, 256, 512, 1024, 2048) |

### 1.2 Runtime Backend（运行时后端）

在 task_family 之下增加 runtime_backend 层，解耦模型加载方式：

| backend | 说明 | 适用场景 |
|---------|------|---------|
| `transformers_pipeline` | 使用 `transformers.pipeline()` 统一接口 | 大多数 NLP/CV/Audio 模型 |
| `transformers_model` | 直接加载 `AutoModel` + `AutoProcessor` | pipeline 不支持的模型 |
| `chronos` | 使用 `ChronosBoltPipeline` / `ChronosPipeline` | 时间序列预测 |
| `diffusers` | 使用 `DiffusionPipeline` | 图像生成 (扩展预留) |
| `custom` | 用户提供自定义加载脚本 | 特殊模型 (扩展预留) |

检测逻辑：先根据 `library_name` 确定 backend，再根据 `pipeline_tag` 确定 task_family。

### 1.3 Handler 标准接口

每个 handler 实现统一的四阶段接口：

```python
class BaseHandler:
    def load(self, model_id: str, task_type: str, backend: str, device: str) -> dict:
        """加载模型，返回 model_ctx（包含 model, tokenizer, processor, device 等）"""

    def preprocess(self, model_ctx: dict, raw_input: dict) -> Any:
        """将原始请求数据转换为模型输入张量/格式"""

    def predict(self, model_ctx: dict, processed_input: Any) -> Any:
        """执行推理，返回原始输出"""

    def postprocess(self, model_ctx: dict, raw_output: Any) -> dict:
        """将模型输出转换为标准化响应（仅返回 shape/metadata，不返回完整输出）"""
```

server.py 调用链：`preprocess → predict → postprocess`，全部在 `torch.inference_mode()` 下执行。

---

## 2. 任务自动检测（detect.py）

### 检测流程（三级兜底）

```
Level 1: HF Hub API
    model_info(model_id) → pipeline_tag + library_name + tags
    ↓ 成功？→ 映射到 (task_family, runtime_backend)
    
Level 2: 模型配置推断
    AutoConfig.from_pretrained(model_id) → model_type + architectures
    → 根据 architecture 名称推断 task (e.g., ForCausalLM → text-generation)
    ↓ 成功？→ 映射到 (task_family, runtime_backend)
    
Level 3: 人工指定（CLI 覆盖）
    --task text-generation --task-family nlp --backend transformers_pipeline
    → 任何级别的手动覆盖都优先于自动检测
```

### 返回结构

```python
@dataclass
class TaskInfo:
    model_id: str                # e.g., "bert-base-uncased"
    pipeline_tag: str            # e.g., "text-classification"
    task_family: str             # e.g., "nlp"
    runtime_backend: str         # e.g., "transformers_pipeline"
    library_name: str            # e.g., "transformers"
    model_revision: str          # e.g., "main" or commit hash
    detection_method: str        # "hub_api" / "config_infer" / "manual"
```

---

## 3. 通用 Server（server.py）

### 端点

| 路径 | 方法 | 功能 |
|------|------|------|
| `/ready` | GET | 健康检查，返回 "ok" |
| `/predict` | POST | 推理端点，接受 JSON，返回结果 |
| `/meta` | GET | 返回模型元信息（model_id, task, device, 加载耗时） |

### 环境变量

```
MODEL_ID          模型 ID
TASK_FAMILY       任务族 (nlp/cv/audio/timeseries)
TASK_TYPE         具体任务 (text-generation, image-classification, ...)
RUNTIME_BACKEND   运行时后端 (transformers_pipeline, chronos, ...)
USE_GPU           是否启用 GPU (0/1)
MODEL_CACHE_DIR   模型缓存目录
```

### 路由逻辑

```python
# server.py 启动时
handler = HandlerRegistry.get(task_family, runtime_backend)
model_ctx = handler.load(MODEL_ID, TASK_TYPE, RUNTIME_BACKEND, device)

# /predict 请求处理
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    processed = handler.preprocess(model_ctx, data)
    with torch.inference_mode():
        output = handler.predict(model_ctx, processed)
    result = handler.postprocess(model_ctx, output)
    return jsonify(result)
```

---

## 4. Handler 实现

### 4.1 handlers/nlp.py

- **load**: `transformers.pipeline(task_type, model=model_id, device_map=device)`
- **preprocess**: 提取 `data["text"]`，按 task 类型组装参数
- **predict**: 调用 pipeline，传入文本和 generation/classification 参数
- **postprocess**: 返回 `{"output_type": "...", "output_shape": ..., "task": "..."}`

支持的 task_type：
- `text-generation`: 输入文本 → 生成输出
- `text-classification`: 输入文本 → 分类标签
- `token-classification`: 输入文本 → NER 标注
- `fill-mask`: 输入带 [MASK] 文本 → 填充
- `question-answering`: 输入 question + context → 答案
- `summarization` / `translation`: 输入文本 → 输出文本

### 4.2 handlers/cv.py

- **load**: `AutoImageProcessor` + `AutoModel` 或 `pipeline(task_type, model=model_id)`
- **preprocess**: base64 解码图像 → PIL Image → processor 处理
- **predict**: 模型前向传播
- **postprocess**: 返回 `{"output_type": "...", "output_shape": ..., "num_classes": ...}`

支持的 task_type：
- `image-classification`: 输入图像 → 分类
- `object-detection`: 输入图像 → 检测框
- `image-segmentation`: 输入图像 → 分割掩码

### 4.3 handlers/audio.py

- **load**: `pipeline(task_type, model=model_id)` 或 `AutoModel` + `AutoProcessor`
- **preprocess**: 解码音频采样 → numpy array
- **predict**: 模型推理
- **postprocess**: 返回 `{"output_type": "...", "output_shape": ...}`

### 4.4 handlers/timeseries.py

- **load**: 直接移植 example-code 的 `ChronosBoltPipeline.from_pretrained()` 逻辑
- **preprocess**: 解析 context 数组 → tensor
- **predict**: pipeline.predict()
- **postprocess**: 返回 `{"forecast_shape": [...]}`

---

## 5. 输入负载生成（workloads/）

每种任务族一个生成器，实现统一接口：

```python
class WorkloadGenerator:
    def __init__(self, model_id: str, task_type: str, batch_size: int):
        ...
    def generate(self, scale_value: float) -> dict:
        """返回可序列化为 JSON 的请求 payload"""
    def scale_label(self, scale_value: float) -> str:
        """返回用于 sniff_group_id 的缩放标签，如 'seq256' / 'res0.5'"""
```

### 各任务族实现

| 任务族 | 输入生成策略 | scale 含义 |
|--------|------------|-----------|
| nlp | 固定种子生成目标 token 数的文本（重复已知语料句子填充到目标长度） | seq_length (tokens) |
| cv | 生成合成 RGB 图像，分辨率 = 模型基础分辨率 × scale_factor（从 processor config 推断基础分辨率，如 ViT 为 224） | resolution_scale (倍数) |
| audio | 生成合成正弦波音频信号，采样率 16kHz | duration_s (秒) |
| timeseries | 固定种子生成随机浮点数序列（与 example-code 一致） | context_length (步数) |

---

## 6. CSV 输出格式

在原 23 字段基础上泛化，增加复现字段：

```
model_id              HF 模型 ID (e.g., "bert-base-uncased")
model_revision        模型版本 (commit hash 或 "main")
task_family           任务族 (nlp/cv/audio/timeseries)
pipeline_tag          具体任务 (text-generation, image-classification, ...)
runtime_backend       运行时后端 (transformers_pipeline, chronos, ...)
image_digest          Docker 镜像 SHA256 摘要
cpu_cores             CPU 核心数
mem_cap_gb            内存上限 (GB)
gpu_mode              GPU 模式 (on/off)
batch_size            批次大小
input_scale           输入规模值 (具体含义由 input_scale_type 决定)
input_scale_type      输入规模类型 (seq_length/resolution_scale/duration_s/context_length)
task_param            任务特定参数 (如 max_new_tokens/prediction_length)
repeat_idx            重复索引
warmup                是否为预热 (0/1)
sniff_group_id        嗅探分组 ID
repeat_in_window      窗口内重复次数
latency_s             网络包级延迟 (秒, 由 merge 回填)
latency_app_s         应用级延迟 (秒)
throughput_samples_per_s  吞吐量
idle_power_w          GPU 空闲功耗 (W)
energy_iters          能耗采样次数
avg_power_total_w     平均总功耗 (W)
peak_power_total_w    峰值总功耗 (W)
energy_total_j        总能耗 (J)
avg_power_eff_w       平均有效功耗 (W)
peak_power_eff_w      峰值有效功耗 (W)
energy_eff_j          有效能耗 (J)
cold_start_s          冷启动时间 (秒)
status                状态 (ok/warn/error)
error                 错误信息
```

新增字段：`model_id`, `model_revision`, `task_family`, `pipeline_tag`, `runtime_backend`, `image_digest`, `input_scale`, `input_scale_type`, `task_param`

---

## 7. Docker 镜像策略

### 基础镜像 (base.Dockerfile)

```dockerfile
FROM python:3.10-slim
# flask + huggingface_hub + numpy + torch (CPU/GPU)
```

### 任务族镜像

每个任务族从 base 扩展：

| Dockerfile | 额外依赖 |
|-----------|---------|
| nlp.Dockerfile | transformers, tokenizers, sentencepiece, accelerate |
| cv.Dockerfile | transformers, torchvision, Pillow |
| audio.Dockerfile | transformers, torchaudio, librosa, soundfile |
| timeseries.Dockerfile | chronos-forecasting |

### 构建流程

1. `run.py` 检测到 task_family
2. 检查本地是否有对应的基础镜像 `acprof-{family}:base`
3. 如果没有，构建基础镜像
4. 在基础镜像上叠加模型下载层，生成 `acprof-{family}-{model_hash}:latest`
5. 镜像缓存：同 family + 同 model 跳过构建

### 镜像摘要记录

构建完成后通过 `docker inspect --format='{{.Id}}'` 获取 image_digest，写入 CSV 的 `image_digest` 字段。

---

## 8. 编排器（orchestrator.py）

Python 替代 shell 脚本，核心函数：

```python
def build_image(task_info: TaskInfo) -> ImageInfo:
    """构建 Docker 镜像，返回 image_tag + image_digest"""

def run_single_case(task_info, cpu, mem, gpu, image_info, output_dir, ...) -> str:
    """
    单次实验：
    1. docker run 带资源约束 (--cpus, --memory, --gpus)
    2. 轮询 /ready，计时 cold_start_s
    3. 启动 tcpdump (subprocess.Popen)
    4. 运行 client.py 负载
    5. 停止 tcpdump，sleep flush
    6. sniff_parse_pcap.py 解析 PCAP
    7. merge_packet_latency.py 合并延迟
    8. docker stop + rm
    返回结果 CSV 路径
    """

def run_matrix(task_info, image_info, cpu_list, mem_list, gpu_list, ...):
    """遍历资源矩阵，对每个组合调用 run_single_case"""
```

### 关键不变量（从 example-code 继承）

- `X-Req-Id: {sniff_group_id}:{k}` 请求追踪头
- `Connection: close` 强制每请求独立 TCP 流
- `sniff_group_id` 格式：`{case_name}_scale{value}_rep{idx}`
- tcpdump 捕获 docker0 + 端口 8002
- 冷启动计时：docker run → /ready 响应的毫秒级差值

### 网络接口检测

- Linux: 默认 `docker0`
- macOS/Windows: 使用 `--sniff-iface` 指定或自动检测 Docker Desktop 虚拟接口
- 不支持 tcpdump 时：跳过 packet-level latency，仅用 `latency_app_s`

---

## 9. 入口（run.py）

```
python run.py --model <hf_model_id> [options]

必选:
  --model         HuggingFace 模型 ID

可选 (检测覆盖):
  --task          强制指定 pipeline_tag (e.g., text-generation)
  --task-family   强制指定任务族 (nlp/cv/audio/timeseries)
  --backend       强制指定 runtime backend (transformers_pipeline/chronos/...)

可选 (实验参数):
  --cpus          CPU 列表 (默认 1,2,4,8)
  --mems          内存列表 (默认 2,4,8,16)
  --gpus          GPU 模式列表 (默认 off,on)
  --batch-size    批次大小 (默认 1)
  --warmup        预热次数 (默认 2)
  --repeat        重复次数 (默认 5)
  --repeat-in-window  窗口内重复数 (默认 20)
  --sniff-iface   网络抓包接口 (默认 docker0)
  --output-dir    输出目录 (默认 results/)
```

---

## 10. 监控层复用

三个监控文件从 example-code 直接复用，无需修改：

| 文件 | 复用原因 |
|------|---------|
| `energy_nvml.py` | `measure_energy_threaded(fn)` 接口完全通用，接受任意 callable |
| `sniff_parse_pcap.py` | 基于 X-Req-Id + /predict 端点过滤，与通用 server 完全兼容 |
| `merge_packet_latency.py` | 基于 sniff_group_id 合并，与通用 CSV schema 兼容 |

---

## 11. 文件结构

```
d:\DOR\universal-profiles\
├── run.py                          # 入口
├── config.py                       # 常量、映射、缩放维度、CSV schema
├── detect.py                       # HF Hub 任务自动检测（三级兜底）
├── orchestrator.py                 # Docker 生命周期编排（替代 shell）
├── client.py                       # 通用客户端（可插拔负载生成）
├── server.py                       # 通用 Flask 服务（Handler 路由）
├── download_model.py               # 模型下载器（复用）
├── energy_nvml.py                  # GPU 能耗监控（复用）
├── sniff_parse_pcap.py             # 网络延迟捕获（复用）
├── merge_packet_latency.py         # 延迟合并（复用）
├── handlers/
│   ├── __init__.py                 # HandlerRegistry + BaseHandler 定义
│   ├── nlp.py                      # NLP handler (load/preprocess/predict/postprocess)
│   ├── cv.py                       # CV handler
│   ├── audio.py                    # Audio handler
│   └── timeseries.py              # Time-series handler
├── workloads/
│   ├── __init__.py                 # WorkloadGenerator 基类 + registry
│   ├── nlp.py                      # 文本负载生成
│   ├── cv.py                       # 图像负载生成
│   ├── audio.py                    # 音频负载生成
│   └── timeseries.py              # 时间序列负载生成
├── dockerfiles/
│   ├── base.Dockerfile             # 公共基础镜像
│   ├── nlp.Dockerfile              # NLP 镜像
│   ├── cv.Dockerfile               # CV 镜像
│   ├── audio.Dockerfile            # Audio 镜像
│   └── timeseries.Dockerfile      # Time-series 镜像
├── plot.py                         # 绘图工具（适配通用 CSV）
├── results/                        # 默认输出目录
├── example-code/                   # 参考代码（不修改）
└── docs/
    └── superpowers/specs/          # 设计文档
```

---

## 12. 验证方案

### 端到端验证流程

1. **NLP 验证**: `python run.py --model bert-base-uncased --cpus 1,2 --mems 4 --gpus off`
   - 验证：自动检测为 nlp/fill-mask/transformers_pipeline
   - 验证：Docker 构建成功，server /ready 返回 ok
   - 验证：CSV 输出包含所有字段，input_scale_type = "seq_length"

2. **CV 验证**: `python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4 --gpus off`
   - 验证：自动检测为 cv/image-classification/transformers_pipeline
   - 验证：合成图像正确生成，分辨率缩放正确

3. **Time-series 验证**: `python run.py --model amazon/chronos-bolt-base --cpus 1,2 --mems 4 --gpus off`
   - 验证：结果与 example-code 产出一致（回归测试）

4. **检测兜底验证**: `python run.py --model <no-pipeline-tag-model> --task text-generation --task-family nlp`
   - 验证：手动覆盖正确生效

5. **错误处理验证**: `python run.py --model bert-base-uncased --cpus 1 --mems 2 --gpus off`
   - 验证：内存不足时 CSV 写入 error 行，不中断矩阵扫描

# 命令行、输入清单与界面设置

查 CLI 参数、workload 清单或 TUI 持久化契约时查阅。交互操作与运行示例见 [README](../README.md)，实现依据为当前入口的 `--help`。文中的命令从仓库根目录执行。

[文档导航](README.md)

## TUI 本地设置

`acprof/tui/settings.py` 管理项目隔离的 `tui.json`，当前版本为 v4；
路径与操作方式见 [README 的 TUI 说明](../README.md#交互式终端界面)。
`ui.language` 是字符串，仅接受 `zh`（简体中文，默认）和 `en`（English），不使用系统 locale 自动推断。
兼容读取 v1、v2、v3 设置，缺少语言字段时使用中文；加载时不改写文件，下次保存时写入 v4。
未知语言值或错误类型遵循现有校验规则：提示、使用默认设置，并保留原文件，直到用户主动保存。

切换语言仅更新当次界面，点击“保存设置”后持久化；“恢复界面默认”将当次语言恢复为中文。
自动记住模型 ID、结果路径或显式记住实验配置时，不会顺带保存尚未保存的界面偏好。
语言只影响 TUI 文案，原始子进程日志、命令参数、结果文件和进度解析状态值保持原有语义。
切换时复用已挂载控件和已读取摘要，不重新读取结果 CSV，也不启动定时刷新；任务运行期间语言控件随其他偏好锁定。

v4 新增顶层字符串 `last_result_dir` 和 `last_result_csv`，分别保存最近使用的结果目录和 CSV 路径。
TUI 中，结果目录位于“补采工具”页，结果 CSV 与摘要位于“绘图工具”页；切换页面保留输入草稿和工具勾选。
两者默认均为 `""`；旧文件缺少字段时保持空值，不从模型草稿推测历史输出目录。
TUI 保存实际使用的绝对路径，相对输入以项目根目录为基准，支持 `~`、空格和中文。
确认采集时，两者与 `last_model` 一次性原子保存；采集结束后采用进度解析得到的合并 CSV，
未提供时沿用已确认配置中的输出路径。它们表示最近一次采集的目标位置，不保证任务成功或文件仍然存在。
读取摘要成功或启动绘图只更新 CSV 字段，启动补采（含 dry-run）只更新目录字段；取消确认和无效输入不更新。
恢复路径不扫描目录、不读取 CSV、不检查文件存在性；实际读取摘要、绘图或补采时再验证。
自动写入复用现有原子替换和错误隔离，保留已保存的 UI 偏好、实验默认参数及未涉及的历史字段。

## CLI 参数

以下参数表对应 `run.py`。示例命令见 [README](../README.md#运行正式实验)，
默认值与实际选项以当前入口的 `--help` 和 [acprof/config.py](../acprof/config.py) 为准。

[run.py](#runpy) · [probe.py](#probepy) · [profile.py](#profilepy) · [其他入口](#其他入口) · [输入规模与音频清单](#输入规模与音频清单)

### `run.py`

#### 模型与资源矩阵

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | required | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `--task` | auto | 覆盖 `pipeline_tag`，例如 `fill-mask`、`text-generation`。 |
| `--task-family` | auto | 覆盖任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`、`multimodal`、`structured`。 |
| `--backend` | auto | 覆盖 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`。 |
| `--cpus` | `1,2,4,8` | CPU core 限制列表。 |
| `--mems` | `2,4,8,16` | Memory cap GB 列表。 |
| `--gpus` | `off,on` | GPU mode 列表。`on` 会用 Docker `--gpus all`。 |
| `--prune-startup-oom` / `--no-prune-startup-oom` | enabled | 默认以最低选中 CPU 为参考，按内存升序完整采集；仅把 Docker 明确 `OOMKilled` 的连续低内存启动失败前缀推断到后续更高 CPU。跳过 case 保留占位行和独立 provenance。运行期/CUDA OOM、timeout 与普通启动失败不触发剪枝。使用 `--no-prune-startup-oom` 可恢复逐格独立尝试。 |

#### 请求窗口与采样

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
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

#### 输入

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-scales` | auto | 手动覆盖 input scale 列表；未提供时通常自动规划 6 档，自定义音频清单按声明档数。 |
| `--workload-spec` | task default | cv、audio、multimodal、diffusion 或 structured 的 workload 清单 JSON。读取音频的任务默认复用内置 LibriSpeech 语音；文本到音频使用确定性文本。结构化清单声明输入宽度、尺度与种子。各任务须使用对应的清单格式。 |

#### 计算分析器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
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

#### 执行分析器

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
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

#### 输出与运行环境

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sniff-iface` | `docker0` | 本机 Docker 默认 bridge 对应的 `tcpdump` 抓包网卡。只有 daemon 改过 bridge 名时才覆盖。 |
| `--output-dir` | `results` | 输出根目录。最终还会追加 model name 子目录。 |
| `--resume` | false | 使用原参数和目录恢复实验；核对运行身份、保留完成 case，并备份后重测中断 case。已完成实验不重测。 |
| `--skip-build` | false | 核验构建指纹和环境清单后复用镜像；不存在时自动构建，不匹配时退出。 |
| `--model-download-policy` | `auto` | `auto` 按已覆盖的加载器规则筛选文件，未知结构保留完整快照并记录原因；`full` 下载固定 commit 的完整仓库。策略进入镜像指纹，不能相互误复用。采集和探测入口均支持。 |
| `--notify` | `auto` | `auto` 在配置 Webhook 后启用企业微信；`none` 关闭，`wecom` 显式选择企业微信。配置见 [README](../README.md#企业微信通知)。 |
| `--help` | — | 显示此入口的全部公开参数后退出。 |
| `--allow-cgroup-v1` | false | 仅用于旧主机诊断的兼容开关。默认正式模式要求 cgroup v2；启用后允许 v1，但会记录 `legacy_compatible`，且 memory peak/stat、I/O 操作数、PID、memory events 与 per-cgroup PSI 不具备同等口径。 |

结果目录存在异常中断留下的 `result_case_*.csv` 时，`run.py` 会先读取同目录 `static_meta.json/cgroup_version`。只有版本与当前 host 一致才允许续写；版本不同、缺失或元数据不可读时会退出，避免把 v1/v2 窗口合并到同一结果文件。

### 输入规模与音频清单

`input_scale` 是每个任务族的主输入尺度，语义由 `static_meta.json` 的 `input_scale_type` 决定：

| task family | `input_scale_type` | 含义 |
| --- | --- | --- |
| `nlp` 文本任务 | `seq_length` | 输入 token length；问答取 context，检索／排序取候选文本 token 数的最大值，固定 query。 |
| `nlp` 表格问答 | `table_rows` | 每张表的行数；query 和列结构固定。 |
| `cv` | `resolution_scale` | 图像／视频帧基础边长 224 像素的缩放倍率；视频帧数固定。 |
| `audio` 读取音频 | `duration_s` | 输入音频时长，单位秒。 |
| `audio` 文本到语音／音频 | `seq_length` | tokenizer 实测输入 token 数；不是生成音频的秒数。 |
| `timeseries` | `context_length` | 时间序列 context length。 |
| `diffusion` | `resolution_px` 或 `denoising_steps` | 图像／视频生成是方形输出边长（像素），帧数固定；无条件图像与 Shap-E 3D 为去噪步数，分辨率／网格解码设置固定。以输入计划为准。 |
| `multimodal` 图像理解／问答／文档检索 | `resolution_px` | 输入图像边长，单位像素；模型 processor 可能重新缩放、切块或固定尺寸。 |
| `multimodal` 音频理解 | `duration_s` | 输入 WAV 的实际样本数 / 采样率，单位秒。 |
| `multimodal` 视频理解 | `frame_count` | 输入 PNG 帧数；FPS 和帧分辨率固定并写入计划。 |
| `multimodal` Any-to-Any | 由 `scale_modality` 决定 | 单一输入模态随尺度变化，其余固定；默认改变音频秒数。 |
| `structured` 表格 | `table_rows` | 每个 batch 项的表格行数，特征宽度固定。 |
| `structured` 策略 | `observation_count` | 每个 batch 项的独立向量观测数，观测宽度固定；不是交互时间步。 |
| `structured` 图 | `node_count` | 每张图的节点数，特征宽度固定，默认双向环有 2 × 节点数条边。 |

未提供 `--input-scales` 时，当前内置 workload/legacy 配置通常会为一次 profiling run 规划 6 档 input scale；自定义音频清单则使用清单中声明的档数：

- `nlp` 文本任务会启动容器读取 tokenizer / handler 的可用最大输入长度，最后一档尽量贴近有效上限。Decoder-only 生成额外预留 `max_new_tokens`；encoder-decoder 不从 encoder 输入预算扣除 decoder 输出长度。表格问答按 `1,2,4,8,16,32` 行规划，不进入 token 二分搜索；超出模型容量时明确失败。
- 读取音频的任务从 workload 清单读取默认尺度；内置英文语音清单为 `1,2,5,10,20,30` 秒。文本到语音／音频进入同一 token 规划器；tokenizer 没有有限上限时采用显式 512-token 采集上限，该值不是模型最大容量，Bark 使用自身 semantic 输入限制。`cv` 使用 generator 最大尺度；`timeseries` 读取已加载 Chronos 的 context limit，并与 workload 上限取较小值，拒绝静默截断。
- `structured` 表格／策略默认 `1,8,32,128` 行／观测，图默认 `8,32,128,512` 节点；可用清单或 CLI 覆盖，当前生成器限制非图不超过 4096、图不超过 2048。该限制是 workload 的输入大小限制，不是模型容量。
- `diffusion` 图像／视频任务默认使用 `128,192,256,320,384,512` 像素输出边长；提示词、随机种子、guidance scale 和去噪步数在各尺度间保持不变。`unconditional-image-generation`、`text-to-3d`、`image-to-3d` 改为扫描 `1,2,4,8,16,20` 个去噪步；Shap-E 使用真实 mesh 解码，渲染图片尺寸不作为网格计算规模。
- `multimodal` 从任务 workload 读取默认尺度：图像边长 `224,336,448`；音频 `1,2,5,10` 秒；视频 `2,4,8` 帧。清单 `input_scales` 可覆盖默认值，CLI `--input-scales` 优先。`static_meta.input_scale_type` 在计划完成后取实际 workload 的单位，不能将所有多模态任务统一解释为 token 数。
- 同一次 run 的所有资源配置共用同一组 scale。
- 所有任务族都会把已确定尺度的 payload 写入唯一的 `input_scale_plan.json`；主采集与 compute profiler 共同读取该文件，保证实际执行 payload、FLOP profiling 和 CSV 中记录的 `input_scale` 一致。
- 手动传入 `--input-scales` 时以手动值为准；workload 会在 sweep 前验证合法性（图像／视频生成分辨率至少为 64 且必须是 8 的倍数；去噪步数是正整数；CV 倍率为有限正数）。

#### 真实音频 workload

`automatic-speech-recognition` 默认使用 `assets/audio/librispeech-clean-test-en-30s/source.json`。该清单引用 LibriSpeech `clean/test` 中同一说话人、同一章节的三条连续语音，按固定顺序拼接后截取前 30 秒；素材是单声道 16 kHz PCM16 WAV，许可证为 CC BY 4.0。每一档输入都从同一个 30 秒基准音频取前缀，不做逐档归一化、补全或循环。

音频分类、Encodec／DAC 音频重建和 Silero VAD 默认复用相同语音前缀。codec 按模型需要在预处理阶段重采样，输入规模仍按源音频时长记录。生成／重建响应的 `audio_num_samples`、`audio_sample_rate`、`audio_duration_s` 描述输出波形，不能写入文字 token 计数；VAD 的 `segments` 是秒为单位的连续阈值帧区间，阈值与分帧策略随响应记录，不是识别文本。Silero 每个请求重置状态，profiler 重复调用也不会延续上一请求的隐藏状态。

新增任务沿用现有 CSV 字段、静态 schema v7 与输入计划 schema v2，旧文件无需迁移。`input_units_per_request = effective_input_scale × batch_size`：表格／策略是总行数／观测数，图是总节点数。结构化 `input_num_samples` 对表格／策略记总行数／观测数，对图记图数量，另外在计划记录总节点数与边数。NLP 检索／排序的单位仍为候选文本尺度乘 batch，不再乘候选数量；固定 query、候选数及重复编码成本属于该请求，比较实验时必须保持一致。零样本 NLI 的候选标签推理成本同样包含在请求中。

结构化输入为固定种子的合成矩阵／环图，保存特征宽度、种子、结构、清单 SHA256 及实际 payload；各资源组合和 profiler 使用同一计划。模型缺少 SafeTensors 元数据时，参数量／权重字节数保持 `null`，不能以输入大小代替。skops 与 Silero 的 GPU 配置失败不会生成虚假的 GPU 指标；TorchScript 不支持加载时更换 attention implementation，eager FLOP 采集明确失败并保留工具状态，不能把缺失 FLOP 当作 0。

音频请求采用 JSON 内的 Base64 WAV；handler 仍能读取历史 `audio_samples` 浮点数组。短音频模式会读取模型 feature extractor 的约束并拒绝超过 receptive field 的尺度。对于 Whisper，30 秒是音频 receptive field；当前 feature extractor 会把接受的短音频补齐为固定的 480,000 samples / 3,000 frames，`/scale_meta` 会显式记录这一点。`max_target_positions=448` 是解码器输出 token 上限，不是音频输入上限，因此框架不会把 latency 必须随 `duration_s` 单调增加作为正确性条件。

自定义素材时可复制内置 `source.json`，设置新的 `workload_id`，再修改相对素材路径、SHA256、provenance 和 inference 字段。自定义 provenance 可以描述单条录音或既有重采样/增益流程；运行时仍会严格验证派生 WAV 本身是单声道、16 kHz、PCM16 且哈希匹配。然后传入：

```bash
python run.py --model openai/whisper-large-v3 \
  --workload-spec /path/to/source.json
```

当前音频 request 只实现 `batch_size=1` 和 `short_form`。清单会拒绝非空的 `chunk_length_s` / `stride_length_s`；长音频 sequential/chunked 应使用独立 workload，不能通过把本清单尺度直接扩展到 30 秒以上来混测。

视觉、多模态与图像条件生成也接受 `--workload-spec`，清单和支持边界见 [README 视觉任务](Runtime_Compatibility.md#视觉任务) 与 [多模态任务](Runtime_Compatibility.md#多模态任务)。输入计划沿用 schema v2，新增信息写在扩展的 `workload`、`input_metadata` 和 `payload` object 内；CSV 未增加列，历史文件无需填补新字段。`workload` 保存素材路径／SHA256、清单 SHA256、提示词、参数、尺度单位和固定条件；实际序列化 payload 及计划 SHA256 是重放依据。

CV 每请求一个图片／视频样本，`input_num_samples=1`；视频帧数及每帧 SHA256 单独记录在 `input_metadata`。零样本标签、VitPose 的人物框和参数也随 payload 重放，不增加隐藏的人物检测请求。CV 响应按任务区分 classification、detection、caption、depth、segmentation、masks、features、keypoints；大张量／掩码／深度图只返回摘要。未产生文本的任务不填写输出 token 数，相关 CSV 指标保持 `NaN`。

无条件图像与 3D 的 `input_units_per_request` 单位为去噪步；其 `task_param.num_inference_steps` 随输入尺度变化。Shap-E 响应 `output_type="mesh"`，输出长度表示网格数量，顶点和三角面数量另行记录，不冒充文本长度、像素数或 token 数。该任务的 profiler 覆盖去噪及真实网格解码；图像条件预处理仍属于请求应用延迟。此扩展不改变历史图像生成的分辨率单位，读取时应依据每次实验的 `input_scale_type`，不能只按 diffusion 任务族判断单位。

多模态 `input_num_samples=1` 表示一个请求样本；其中可含图像、音频、视频或 query＋文档。音频 PCM 样本数与视频帧数分别保存在 `input_metadata.audio_num_samples` / `video_num_frames`。`input_units_per_request` 继续等于有效 `input_scale × batch_size`，因此其单位取决于上表，不可横跨不同尺度类型直接比较每单位延迟。

文字生成／问答的 `output_length_avg` 是返回文字的字符数窗口均值；`output_token_count_avg` 是对输出文字重新分词的 token 数，不代表所有解码步或 Omni 音频 token。Omni 音频摘要另含 24000 Hz 采样率、PCM 样本数和秒数，不混入文字长度。图像生成 `output_length` 为图像数，视频生成为总帧数；检索仅返回 query-by-document 分数矩阵，不产生文字长度／token 字段，其对应 CSV 值保持 `NaN`。

主请求延迟包含输入预处理、模型推理和结果摘要。后置 profiler 测量 handler 的 `predict()`：多模态理解、视频分类和关键点的直接模型路径不包含在 `preprocess()` 中运行的 processor；使用 Transformers pipeline 的 CV 路径仍包含 pipeline 内部预处理和后处理。Base64 解码与 handler 的摘要处理均在 profiler 的预测段之外；Diffusers 的完整生成／媒体解码、检索的两个编码器 forward 与 MaxSim 则在预测段内。Omni GPU 采用 thinker/talker FP16 和 Token2Wav FP32；其 eager FLOP 请求明确不支持，NCU/Nsys 仍可测完整推理。Shap-E 及 Diffusers Transformer 视频架构在没有可验证 eager 替换时同样报告工具错误，不伪填 FLOP。生成媒体不会作为图片／音频／视频／网格响应体返回，因此网络指标描述当前摘要服务协议。

### `probe.py`

复用 `--model`、`--task`、`--task-family`、`--backend`、`--batch-size`、`--workload-spec`、
`--output-dir`、`--skip-build` 和 `--allow-cgroup-v1` 的参数及默认值。
资源列表与超时的用途如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--cpus` | `1,2,4,8` | 只选择列表中的最小 CPU。 |
| `--mems` | `2,4,8,16` | 从低到高实测的候选内存上限。 |
| `--gpus` | `off,on` | 包含 `off` 时优先 CPU-only，否则使用 `on`。 |
| `--input-scales` | 自动规划 | 只探测已确定尺度中的最大值。 |
| `--timeout-seconds` | 不设超时 | 单次探测请求的等待上限；显式值必须有限且大于 0。 |

`probe.py` 不接收 `run.py` 的 `--request-timeout-seconds`、warmup/repeat、能耗采样或 profiler 参数。
详细用法见 [README](../README.md#先探测最大输入)。

### `profile.py`

位置参数 `result_dir` 是已完成的模型结果目录。操作和恢复规则见
[README 补采说明](../README.md#补采已有结果)。

TUI 使用四项复选框选择补采工具（初始勾选 `torch`、`ncu`），将勾选结果传给
`--tools`；未勾选任何工具时不启动补采。工具适用范围、采样策略和指标口径与 CLI 相同。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tools` | `torch,ncu,nsys,massif` | 选择需要补齐的工具，逗号分隔。 |
| `--dry-run` | 关闭 | 验证已有文件并展示计划，不启动分析器或改写结果。 |
| `--force-reprofile` | 关闭 | 强制重采并替换所选工具已经成功的值。 |
| `--massif-sampling` | `per-scale` | 可选 `per-scale`、`full`。 |
| `--nsys-sampling` | `per-cpu-scale` | 可选 `per-cpu-scale`、`per-scale`、`full`。 |
| `--massif-reference-cpu` / `--massif-reference-mem` | 结果矩阵最大值 | 在已有 CPU-only 资源配置中选择代表值。 |
| `--nsys-reference-cpu` / `--nsys-reference-mem` | 结果矩阵最大值 | 在已有 GPU 资源配置中选择代表值；`per-cpu-scale` 只用代表内存。 |
| `--torch-profiler-repeat` / `--torch-repeat` | `1` | 同一参数的两个名称；控制 Torch probe 内推理次数。 |
| `--ncu-repeat` / `--nsys-repeat` / `--massif-repeat` | `1` | 对应工具的 probe 内推理次数，归一化口径同 `run.py`。 |
| `--ncu-root` / `--nsys-root` | 自动检测 | Host 工具安装目录或可执行文件。 |
| `--compute-profile-cpus` / `--compute-profile-mem` | host 逻辑 CPU / 75% host memory | 临时 compute profiler 的 CPU/内存上限，内存单位 GB。 |

### 其他入口

`plot.py` 接收结果 CSV 路径，`tui.py` 可用 `--model` 预填模型、用 `--preset` 选择预设。
`audit.py <目录或 CSV>` 只读校验结果；`--json` 输出报告，`--require-complete --require-ok`
用于验收新实验。`stats.py <目录或 CSV>` 按测量窗口计算置信区间，支持重复 `--metric`、
`--confidence`、`--resamples`、`--seed`、`--block-size` 和新的 `--output` 文件；定义见[结果分析](Metrics.md)。
TUI“统计报告”页的“计算统计”使用 `stats.py` 默认参数，并在源 CSV 旁的 `analysis/` 保存唯一命名的 JSON。
`/stats [csv/dir]` 与按钮等价；`/report [json]` 或“查看报告”读取已有窗口统计、监测开销或 CLI/TUI 对照报告。
这些操作需要 TUI 空闲；开销实验仍通过独立脚本显式运行。报告展示与路径带入方式见 [TUI 说明](../README.md#交互式终端界面)。
`/images` 打开“镜像管理”页，手动“刷新”后可搜索、勾选或“选择同模型”，再确认删除所选镜像的全部标签。
镜像列表分列显示 `Repository` 和 `Tag`；无标签镜像显示短 image ID，`Tag` 显示 `—`，完整标签可在行详情查看。
筛选与勾选只在本次会话保留；每次刷新清空选择。删除范围和保留原镜像的要求见[镜像管理与清理](Runtime_Compatibility.md#镜像管理与清理)。
各入口的完整帮助可直接运行：

```bash
.venv/bin/python run.py --help
.venv/bin/python probe.py --help
.venv/bin/python profile.py --help
.venv/bin/python plot.py --help
.venv/bin/python audit.py --help
.venv/bin/python stats.py --help
.venv/bin/python tui.py --help
```

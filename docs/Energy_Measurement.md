# 能耗测量与归因

解释 GPU、CPU package、估算 vCPU 能耗或 idle baseline 时查阅。请求窗口与对照生命周期见 [采集协议](Profiling_Protocol.md#每行测量窗口)。

[文档导航](README.md)

## 采样与归因边界

GPU 功率来自 NVML device 0，能量对带时间戳的功率样本积分；它不是逐容器独立能量计。
CPU package 能量来自 RAPL 根域计数器差值，避免重复累加 core 子域。估算 vCPU 使用同一 RAPL 轨迹，
按每个有效采样区间的容器/主机 CPU 时间比例归因。实现见
[`energy_nvml.py`](../acprof/monitors/energy_nvml.py)和 [`energy_cpu.py`](../acprof/monitors/energy_cpu.py) 的 `_result_from_samples()`。

```text
share_i = clamp(container_cpu_delta_i / host_active_cpu_delta_i, 0, 1)
vcpu_total_window_j = Σ(package_energy_delta_i × share_i)
vcpu_effective_window_j = Σ((package_energy_delta_i - idle_power_w × dt_i) × share_i)
vcpu_average_power_w = 对应归因窗口能量 / 完整测量时长
vcpu_energy_per_request_j = 对应归因窗口能量 / repeat_in_window
```

CPU 计数缺失、host delta 非正或 container delta 为负的区间不参与 vCPU 归因。
汇总 `vcpu_cpu_share` 是有效区间 CPU delta 的总量之比，通常不能乘整段 package energy 来重建上述积分。
effective 区间能量允许为负；不均匀加权和有效区间筛选可能使 vCPU effective 大于 package effective，
不能单凭这一关系判定 CSV 算错。派生能效对负值的处理以对应字段定义为准。
CPU quota 节流、计数更新粒度与采样起点可能影响估算重复性；这些值不代表插座侧整机能耗。

## GPU 功率与能耗

| 字段 | 含义 |
| --- | --- |
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

## CPU package 功率与能耗

| 字段 | 含义 |
| --- | --- |
| `cpu_idle_power_w` | CPU package matched-control baseline power，单位 W。仅主机暴露可读的 Linux RAPL `/sys/class/powercap/*/energy_uj` 时有值；由与 workload 相同 monitor 生命周期下整段 control window 的 RAPL 能耗差 / 实际 duration 得到。 |
| `cpu_idle_measured_at` | `--idle-debug` 开启时，CPU idle baseline 测量完成时的本地 ISO-8601 时间戳；未开启时为 `nan`。 |
| `cpu_idle_rel_range_so_far` | `--idle-debug` 开启时，截至本行为止当前 case 内有效 `cpu_idle_power_w` 的相对极差，公式为 `(max - min) / mean`；`0.05` 表示 5%。未开启时为 `nan`。 |
| `cpu_energy_iters` | CPU package energy measurement 采样窗口中的 sample 数。 |
| `cpu_avg_power_total_w` | 测量窗口内 host CPU package total average power，单位 W。 |
| `cpu_peak_power_total_w` | 测量窗口内 host CPU package total peak power，单位 W。peak 取相邻 RAPL 采样区间功率的最大值，低于 `0.5 / sample_hz` 的尾部短区间不参与 peak 计算。 |
| `cpu_energy_total_j` | 本行平均到单 request 的 total CPU package energy，单位 J。 |
| `cpu_avg_power_eff_w` | 扣除 CPU idle baseline 后的 CPU package effective average power，单位 W。 |
| `cpu_peak_power_eff_w` | 扣除 CPU idle baseline 后的 CPU package effective peak power，单位 W。peak 口径同 `cpu_peak_power_total_w`。 |
| `cpu_energy_eff_j` | 本行平均到单 request 的 effective CPU package energy，单位 J。 |

## 估算 vCPU 能耗与派生能效

| 字段 | 含义 |
| --- | --- |
| `vcpu_cpu_share` | container cgroup CPU time delta / host active CPU time delta，用于估算本 container 占 host active CPU 的比例。 |
| `vcpu_cpu_time_s` | 本行平均到单 request 的 container CPU time，单位秒。 |
| `vcpu_avg_power_total_w` | 逐有效采样区间归因的 total energy 除以完整测量时长，单位 W；不能简单乘汇总 `vcpu_cpu_share` 重建。 |
| `vcpu_peak_power_total_w` | 按相邻采样区间 container cgroup CPU share 分摊后的 estimated vCPU total peak power，单位 W。低于 `0.5 / sample_hz` 的尾部短区间不参与 peak 计算。 |
| `vcpu_energy_total_j` | 本行平均到单 request 的 estimated vCPU total energy，单位 J。 |
| `vcpu_avg_power_eff_w` | 逐有效采样区间扣除 idle 并归因的 effective energy 除以完整测量时长，单位 W。 |
| `vcpu_peak_power_eff_w` | 按相邻采样区间 container cgroup CPU share 分摊后的 estimated vCPU effective peak power，单位 W。peak 口径同 `vcpu_peak_power_total_w`。 |
| `vcpu_energy_eff_j` | 本行平均到单 request 的 estimated vCPU effective energy，单位 J。 |
| `container_attributed_energy_eff_j` | 不增加采集轮次的派生值。CPU-only 行等于 estimated `vcpu_energy_eff_j`；GPU 行等于 `vcpu_energy_eff_j + gpu_energy_eff_j`。任一必需分量缺失或为负时为 `nan`。它只覆盖已采集并归因的 vCPU/GPU 分量，不代表 wall-plug system energy。 |
| `container_attributed_samples_per_j` | `batch_size / container_attributed_energy_eff_j`，单位 samples/J；能量不为正或缺失时为 `nan`。 |
| `container_attributed_edp_app_js` | `container_attributed_energy_eff_j * latency_app_s`，即 application-latency energy-delay product，单位 J·s/request。 |
| `output_tokens_per_s_app` | `output_token_count_avg / latency_app_s`。只在 handler 能可靠返回正数 output token count 时有值，适用于 ASR 和图像描述；分母包含完整请求的图像/音频预处理、推理、后处理和传输，不是纯解码速度。 |
| `container_attributed_j_per_output_token` | `container_attributed_energy_eff_j / output_token_count_avg`；没有可靠 output token count 时为 `nan`。 |
| `container_attributed_j_per_input_unit` | `container_attributed_energy_eff_j / input_units_per_request`；沿用上述任务族 input unit 语义，不增加能耗采集窗口。 |
| `container_attributed_j_per_input_megapixel` / `container_attributed_j_per_output_megapixel` | `container_attributed_energy_eff_j × 1,000,000 / 对应的 pixels_per_request`，单位 J/Mpixel；能量归因和 idle 扣除口径与分子完全一致。原始能量缺失、为负或像素数无效时为 `nan`；有效零能量仍为 `0`。 |

### GPU energy 字段全是 `nan`

- `gpu_mode=off` 时这是正常结果。
- `gpu_mode=on` 时检查 NVIDIA driver、NVIDIA Container Toolkit、`pynvml` 和容器 GPU 可见性。

### CPU / vCPU energy 字段全是 `nan`

- 当前版本会在 task detection 前检查 RAPL；计数器不存在或不可读时，`run.py` 会退出并给出权限修复步骤，不会继续生成新的完整结果。
- 历史结果或中断产生的 CSV 仍可能包含 `nan`。这类缺失值不应使用 TDP 或 CPU utilization 猜测回填。
- `cpu_*` 字段是 host CPU package/root domain 的真实 RAPL 测量值，不累加 `intel-rapl:*:*` 这类 core 子 domain；`vcpu_*` 字段是在同一窗口内按 container cgroup CPU share 分摊出来的估计值。

### CPU idle baseline 波动 warning

- `cpu_idle_power_w` 的 case 级相对极差达到或超过 5% 时，case 结束后会输出 warning，实验继续运行。常见原因是 host 后台进程、IDE/远程桌面、Docker 其他容器、系统索引或 CPU 温度/频率策略变化。
- 可先增大 `--idle-cooldown-seconds` 让上一轮 workload 的瞬态结束；如果单次 idle 窗口本身仍然抖动，再增大 `--idle-seconds`。同时关闭非必要后台进程，并检查 `static_meta.json` 中的 `cpu_governor` / `cpu_boost` 是否符合实验设置。
- 需要定位具体时间点和进程时，加 `--idle-debug` 重跑；优先查看 `debug_idle_diag/result_case_*.csv.idle_diag.jsonl` 中同一 row 的 `rapl_trace.top_power_windows`、`idle_host_active_delta_s`、`idle_container_cpu_delta_s`，再结合 control window 结束后采集的 loadavg、top CPU processes 和 Docker 快照。`idle_proc_cpu_top` 在 matched control 模式下保持为空，避免 `/proc` 全量遍历进入 baseline 能量。

### GPU idle baseline 波动 warning

- `gpu_idle_power_w` 的 case 级相对极差达到或超过 5% 时，case 结束后会输出 warning，实验继续运行。常见原因是桌面显示栈、其他 GPU 进程、P-state/clock 调整、温度或电源管理状态变化。
- 可先增大 `--idle-cooldown-seconds`，等待上一轮 workload 后的 GPU clock/power 状态回落；如果 idle trace 内部仍抖动，再增大 `--idle-seconds`。
- 需要定位具体时间点和进程时，加 `--idle-debug` 重跑；优先查看同一 row 的 `gpu_idle_power_samples`、`nvidia_smi_gpu`、`nvidia_smi_pmon` 和 `nvidia_smi_compute_apps`。

### CPU / vCPU peak power 看起来异常

- `cpu_peak_power_*` 和 `vcpu_peak_power_*` 是相邻采样区间功率的最大值，不是整段平均功率。增大 `--sample-hz` 会缩短区间、提高捕捉短峰值的能力，也可能让峰值更敏感。
- 为避免停止监控时的极短尾部区间放大 peak，当前实现会排除短于半个采样周期的 peak 区间；avg power 和 energy 仍按完整测量窗口计算。

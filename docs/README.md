# AC-Prof: Automated inference-serving containers (containerized tools) run-time profiling dataset & framework, featuring scaled resource specifications.

> **A high-fidelity dataset along with reproducible profiling framework for characterizing the run-time behavior of containerized AI tools under constrained resource specifications.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#requirements) 


---

## 📖 Overview

**AC-Prof (Automated inference-serving Containers/Containerized tools run-time Profiling)** addresses the lack of behavior data that reflect how an inference container, of varied AI models and input scale, respond to different budgets of resources (e.g., GPU, CPUs) allocated to the runtime. Unlike general-purpose monitoring tools, AC-Prof is specifically architected for Deep Learning (DL) inference services. Unlike MLPerf, AC-Prof focuses on the sensitivity of performance to resources and covers more metrics, e.g., power/energy, apart from inference delay.

It provides **two core assets** for the research community:
1.  **The Dataset**: A comprehensive collection of performance metrics covering cold-starts and runtime behaviors under strict resource limits (CPU/GPU/Memory) and input variations.
2.  **The Framework**: A decoupled, side-channel profiling tool that captures **Network Latency** (via packet sniffing), **GPU Energy** (via NVML integration), Linux/WSL **CPU package / estimated vCPU energy** (via RAPL powercap plus cgroup CPU share), **resource utilization** (container CPU/memory plus NVML GPU/VRAM usage), and vendor-tool **MFLOPS** probes (Intel Advisor for CPU, NVIDIA Nsight Compute `ncu` for GPU).

## 🌟 Key Features

* **🕵️ Zero-Intrusion Architecture**: Profiles AI containers as black-boxes by monitoring external application-level signals and hardware states (GPU Polling) without modifying any model-server source code.
* **🧩 Modularity & Extensibility**: Features a decoupled monitor architecture. Easily extend profiling capabilities with custom probes without altering the core experiment orchestrator.
* **📦 Reproducible Environments**: Leveraging standard Docker runtimes and PyTorch Hub models to ensure a deterministic execution environment. This framework enables researchers to reproduce the profiling workflow and comparative analysis across different hardware setups.

## 🏗️ System Architecture

The framework adopts a strict Control-Execution-Monitor separation principle to facilitate modular extensibility and reproducible orchestration.



| Component | Responsibility |
| :--- | :--- |
| **Controller** | Orchestrates the experiment workflow (Warm-up $\rightarrow$ Input Scaling $\rightarrow$ Batch Loop $\rightarrow$ Cool-down). |
| **Client** | Generates workloads and handles data serialization. Supports variable input scales (e.g., image resolution). |
| **Server** | The black-box AI container (Flask/TorchServe) executing the inference logic. |
| **Monitor** | **Side-channel Collector**: <br>1. **Sniffer**: Captures TCP packets on `docker0` bridge to measure physical transport latency. <br>2. **GPU Energy**: Polls NVIDIA NVML at 20Hz to integrate total GPU power usage. <br>3. **CPU / vCPU Energy**: Reads Linux RAPL package counters and attributes estimated vCPU energy by container cgroup CPU share when available. <br>4. **Resource Usage**: Samples container cgroup CPU/memory usage and NVML device-level GPU utilization / VRAM usage in the same workload window. <br>5. **Compute Throughput**: Runs separate profiler containers with Intel Advisor for CPU FLOPs and NVIDIA Nsight Compute `ncu` for GPU FLOP/tensor counters, then reports MFLOPS using normal AC-Prof latencies. |

## 📊 Dataset Specifications

We perform a comprehensive sweep across multiple resource dimensions to construct the dataset.

### Resource Matrix
| Dimension | Configuration Space |
| :--- | :--- |
| **Compute (CPU)** | 1, 2, 4, 8 vCPUs |
| **Memory Caps** | 2 GB, 4 GB, 8 GB, 16 GB |
| **Accelerator** | NVIDIA GeForce RTX 3090 (ON / OFF) |
| **Input Scaling** | Task-specific granularity (e.g., Image resolution $0.1\times$ to $2.0\times$) |

### Collected Metrics
* **End-to-End Latency**: latency (seconds).
* **Energy Consumption**: Total GPU energy, CPU package energy, and estimated vCPU energy per inference (Joules).
* **Power Draw**: Average and Peak GPU board power, CPU package power, and estimated vCPU power (Watts).
* **Resource Utilization**: Container CPU utilization, container memory usage/cap percentage, device-level GPU utilization, and device-level VRAM usage/cap percentage.
* **Compute Throughput**: `model_mflop_per_request`, `compute_mflops_app`, and packet-latency-adjusted `compute_mflops` from the default PyTorch profiler path, or from Intel Advisor / `ncu` when `--compute-profile-tool vendor` is selected.
* **Static Meta**: Model weight size, Docker image download volume.

### Output Files
Each model run writes two top-level CSV artifacts under `results/<model>/`:

* **result_all.csv**: Dynamic profiling measurements across the full resource matrix. It contains per-run fields only, such as resource settings, `input_scale`, timing, power, energy, resource utilization, and status columns. By default, AC-Prof now auto-plans exactly 6 `input_scale` levels before profiling starts. For `nlp`, the last point is chosen to stay as close as possible to the tokenizer's usable maximum length, and the CSV keeps recording the effective input scale actually executed.
* **static_meta.csv**: One-row static metadata summary. `model_name` stores the HuggingFace model ID, and the file also carries `model_revision`, `task_family`, `pipeline_tag`, `runtime_backend`, `image_tag`, `batch_size`, `input_scale_type`, `model_download_url`, `gpu`, `gpu_mem_total_bytes`, `model_weight_bytes`, `docker_image_bytes`, `environment`, `cpu_power_source`, and `vcpu_power_method`.
* **compute_profile_plan.json**: Per-scale FLOP profile data used to fill compute columns in `result_all.csv`. The default path uses PyTorch profiler; vendor mode uses Advisor/ncu. Profiler failures record diagnostics here and keep compute values as `nan`.

Static metadata is collected on the host after the model image is ready and before the profiling matrix starts. The byte fields use these exact measurement rules:

* `model_weight_bytes`: Total bytes of downloaded Hugging Face cache artifacts stored under `/models/hf` inside the built model image.
* `docker_image_bytes`: Local Docker image size reported by `docker image inspect` for the model image.
* `gpu`: Host GPU model name for device 0, or `unknown` when the machine does not expose an NVIDIA GPU.
* `gpu_mem_total_bytes`: Host GPU total VRAM for device 0 in bytes, or empty when unavailable.
* `environment`: Short host execution environment label such as `windows11+wsl`, `ubuntu24.04`, or `macos15`.
* `cpu_power_source`: `rapl` when Linux RAPL powercap counters are available, otherwise `unavailable`.
* `vcpu_power_method`: `rapl_cgroup_cpu_share` when estimated vCPU energy can be derived from RAPL package energy and container cgroup CPU share, otherwise `unavailable`.
* `input_scale_type`: The semantic name of `result_all.csv/input_scale`, for example `seq_length` or `resolution_scale`.
* CPU / vCPU power fields remain `nan` when RAPL `/sys/class/powercap/*/energy_uj` is not exposed. AC-Prof does not synthesize CPU power from TDP or utilization-only estimates.
* Container CPU and memory utilization fields are read from Docker cgroups for the measured container. GPU utilization and VRAM fields are NVML device-level measurements for device 0, not strict per-container process attribution, and remain `nan` when `gpu_mode=off` or NVML is unavailable.
* Scale planning:
  When `--input-scales` is not provided, AC-Prof auto-generates 6 scale levels for each run.
  `nlp` uses the container-side tokenizer metadata to estimate the maximum usable input length, then derives 6 legal sequence lengths with the final point near that maximum.
  `cv`, `audio`, and `timeseries` also default to 6 automatically planned levels based on their family-specific maximum scale.
  When `--input-scales` is provided, the manual values are used as-is; `nlp` and `timeseries` still validate that those values are legal before the sweep starts.

## Measurement Environment
- OS: Ubuntu 24.04.6 LTS  
- Container runtime: Docker 27.5.1  
- Drivers/Libraries: CUDA 12.1, cuDNN 9.1  
- Language/Framework: Python 3.12, PyTorch 2.5.1+cu121

## Data Sources
This dataset is collected with reference to the **APIBench** dataset methodology. External model APIs are sourced from three popular ML model repositories:
- **TorchHub**: https://pytorch.org/hub/
- **TensorFlow Hub**: https://www.tensorflow.org/hub
- **HuggingFace Models**: https://huggingface.co/models

## 📈 Benchmark Results (Preview)

*The following plots demonstrate the non-linear relationship between input scale, latency, and energy consumption captured by AC-Prof.*

![Latency-Energy Tradeoff](docs/container_runtime_example.png)
*(Figure: FCN-ResNet50 performance profile on RTX 3090. Note the linear power consumption vs. non-linear energy accumulation.)*





## Modeling Guidance
- After collecting measurements for a container, fit a simple parametric or piecewise model (e.g., least squares) for latency as a function of resources and input size, and report goodness of fit and residuals. Keep train/test splits separate for each container–task pair.


## Contribution
New contributors are welcome. Please open an issue to discuss your idea before submitting a pull request. Follow the code style and ensure tests pass. 
## License
This project is released under the Apache-2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements
This dataset is part of the DOR project (https://github.com/wingter562/DISTINT_open_data) by Dr. Wentai Wu, Jinan University, with primary contribution by Dr. Shenghai Li, South China University of Technology.

**List of contributors:**
- Wentai Wu, JNU
- Shenghai Li, SCUT
- Qinan Wu, JNU
- Kaizhe Song, JNU
- Yukai Wang, JNU

Project contact: wentaiwu[at]jnu[dot]edu[dot]cn | lishenghai2022[at]foxmail[dot]com

Issues and feature requests: please open a GitHub Issue

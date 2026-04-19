# AC-Prof: Automated inference-serving containers (containerized tools) run-time profiling dataset & framework, featuring scaled resource specifications.

> **A high-fidelity dataset along with reproducible profiling framework for characterizing the run-time behavior of containerized AI tools under constrained resource specifications.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#requirements) 


---

## 📖 Overview

**AC-Prof (Automated inference-serving Containers/Containerized tools run-time Profiling)** addresses the lack of behavior data that reflect how an inference container, of varied AI models and input scale, respond to different budgets of resources (e.g., GPU, CPUs) allocated to the runtime. Unlike general-purpose monitoring tools, AC-Prof is specifically architected for Deep Learning (DL) inference services. Unlike MLPerf, AC-Prof focuses on the sensitivity of performance to resources and covers more metrics, e.g., power/energy, apart from inference delay.

It provides **two core assets** for the research community:
1.  **The Dataset**: A comprehensive collection of performance metrics covering cold-starts and runtime behaviors under strict resource limits (CPU/GPU/Memory) and input variations.
2.  **The Framework**: A decoupled, side-channel profiling tool that captures **Network Latency** (via packet sniffing) and **GPU Energy** (via NVML integration) with **zero code intrusion**.

## 🌟 Key Features

* **🕵️ Zero-Intrusion Architecture**: Profiles AI containers as black-boxes by monitoring external application-level signals and hardware states (GPU Polling) without modifying any model-server source code.
* **🧩 Modularity & Extensibility**: Features a decoupled monitor architecture. Easily extend profiling capabilities with custom probes (e.g., CPU utilization, Memory footprint) without altering the core experiment orchestrator.
* **📦 Reproducible Environments**: Leveraging standard Docker runtimes and PyTorch Hub models to ensure a deterministic execution environment. This framework enables researchers to reproduce the profiling workflow and comparative analysis across different hardware setups.

## 🏗️ System Architecture

The framework adopts a strict Control-Execution-Monitor separation principle to facilitate modular extensibility and reproducible orchestration.



| Component | Responsibility |
| :--- | :--- |
| **Controller** | Orchestrates the experiment workflow (Warm-up $\rightarrow$ Input Scaling $\rightarrow$ Batch Loop $\rightarrow$ Cool-down). |
| **Client** | Generates workloads and handles data serialization. Supports variable input scales (e.g., image resolution). |
| **Server** | The black-box AI container (Flask/TorchServe) executing the inference logic. |
| **Monitor** | **Side-channel Collector**: <br>1. **Sniffer**: Captures TCP packets on `docker0` bridge to measure physical transport latency. <br>2. **Energy**: Polls NVIDIA NVML at 20Hz to integrate total GPU power usage. |

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
* **Energy Consumption**: Total GPU energy per inference (Joules).
* **Power Draw**: Average and Peak GPU board power (Watts).
* **Static Meta**: Model weight size, Docker image download volume.

### Output Files
Each model run writes two top-level CSV artifacts under `results/<model>/`:

* **result_all.csv**: Dynamic profiling measurements across the full resource matrix. It contains per-run fields only, such as resource settings, `input_scale`, timing, power, energy, and status columns. By default, AC-Prof now auto-plans exactly 6 `input_scale` levels before profiling starts. For `nlp`, the last point is chosen to stay as close as possible to the tokenizer's usable maximum length, and the CSV keeps recording the effective input scale actually executed.
* **static_meta.csv**: One-row static metadata summary. `model_name` stores the HuggingFace model ID, and the file also carries `model_revision`, `task_family`, `pipeline_tag`, `runtime_backend`, `image_tag`, `batch_size`, `input_scale_type`, `model_download_url`, `gpu`, `model_weight_bytes`, and `docker_image_bytes`.

Static metadata is collected on the host after the model image is ready and before the profiling matrix starts. The byte fields use these exact measurement rules:

* `model_weight_bytes`: Total bytes of downloaded Hugging Face cache artifacts stored under `/models/hf` inside the built model image.
* `docker_image_bytes`: Local Docker image size reported by `docker image inspect` for the model image.
* `gpu`: Host GPU model name for device 0, or `unknown` when the machine does not expose an NVIDIA GPU.
* `input_scale_type`: The semantic name of `result_all.csv/input_scale`, for example `seq_length` or `resolution_scale`.
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

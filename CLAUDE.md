# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AC-Prof** (Automated inference-serving Containers run-time Profiling) — a zero-intrusion profiling framework that characterizes runtime behavior (latency, GPU energy, cold start) of containerized HuggingFace AI models under constrained CPU/Memory/GPU resources. Python 3.10+, no package manager — loose modules executed directly.

## Running the Profiler

```bash
# Install host-side dependencies
pip install -r requirements.txt

# Profile a model (full resource matrix sweep)
python run.py --model bert-base-uncased
python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
python run.py --model amazon/chronos-bolt-base --task-family timeseries --backend chronos

# Skip Docker rebuild if image exists
python run.py --model bert-base-uncased --skip-build

# Plot results
python plot.py results/<model>/result_all.csv
```

There is no test suite, no linter configuration, and no build step. The project has no `pyproject.toml` or `setup.py` — all modules are imported directly.

## Architecture

**Control-Execution-Monitor separation** across host and container:

```
Host side                          Container side (Docker)
─────────────────────────          ──────────────────────────
run.py          CLI entry          server.py      Flask API
  │                                  ├─ handlers/   Model load/infer
  ├─ detect.py  HF task detect       │   ├─ nlp.py
  │                                  │   ├─ cv.py
  ├─ orchestrator.py                 │   ├─ audio.py
  │   Docker build (2-stage)         │   └─ timeseries.py
  │   Container lifecycle            └─ /ready, /predict endpoints
  │   tcpdump start/stop
  │
  ├─ client.py  Workload gen      Monitors (side-channel)
  │   ├─ workloads/                ──────────────────────────
  │   │   ├─ nlp.py               energy_nvml.py     GPU power @ 20Hz
  │   │   ├─ cv.py                sniff_parse_pcap.py  TCP latency
  │   │   ├─ audio.py             merge_packet_latency.py
  │   │   └─ timeseries.py
  │   └─ HTTP → /predict
  │
  └─ results/ → CSV output
```

### Key data flow

1. `run.py` parses CLI args → `detect.py` resolves model's task family & backend (3-level fallback: HF API → config inference → CLI override)
2. `orchestrator.py` builds Docker image (base + task-family layer with model weights baked in), then sweeps the CPU x MEM x GPU resource matrix
3. For each resource config: start container → poll `/ready` → optionally start tcpdump → run `client.py` as subprocess → collect energy via NVML thread → stop → write CSV
4. `client.py` generates task-specific workloads via `workloads/` registry, iterates over input scales (warmup + repeat), measures latency
5. Results merge into `results/<model>/result_all.csv` (31 fields defined in `config.CSV_FIELDS`)

### Extension points

- **New task family**: Add handler in `handlers/`, workload generator in `workloads/`, Dockerfile in `dockerfiles/`, and register the pipeline tags in `config.PIPELINE_TAG_TO_FAMILY`
- **New monitor**: Side-channel design means monitors are independent — add alongside `energy_nvml.py` without touching orchestrator core
- **Handler interface** (`handlers/__init__.py`): `load()` → `preprocess()` → `predict()` → `postprocess()`

## Configuration

All defaults live in `config.py`. Key constants:
- Resource matrix: `DEFAULT_CPU_LIST`, `DEFAULT_MEM_LIST`, `DEFAULT_GPU_LIST`
- Experiment: `DEFAULT_WARMUP=2`, `DEFAULT_REPEAT=5`, `DEFAULT_REPEAT_IN_WINDOW=20`
- Scaling dimensions per task family in `SCALING_DIMENSIONS` dict
- `SERVER_PORT=8002`, container port mapping: `8002 + cpu*100 + mem`

## Infrastructure Requirements

- Docker with optional NVIDIA GPU support (`--gpus all`)
- `tcpdump` + `tshark` for packet-level latency (optional, gracefully degrades)
- `sudo` required for tcpdump on `docker0` bridge
- HuggingFace Hub access for model detection and weight download during Docker build
- China network support via `HF_MIRROR_ENDPOINT` and `PYPI_MIRROR_INDEX` in config.py

## Code Conventions

- Comments and docstrings are a mix of Chinese and English
- Environment variables are the interface between `orchestrator.py` (host) and `client.py` (subprocess) — not function arguments
- Model weights are always baked into Docker images (no volume mounts) for reproducibility
- Workload generators use deterministic seeded RNG

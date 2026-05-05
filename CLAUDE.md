# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AC-Prof** (Automated inference-serving Containers run-time Profiling) — a zero-intrusion profiling framework that characterizes runtime behavior (latency, GPU energy, cold start) of containerized HuggingFace AI models under constrained CPU/Memory/GPU resources. Python 3.10+, core code under `acprof/` with root-level compatibility wrappers for the legacy script commands.

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

Tests use the standard-library `unittest` runner: `.venv/bin/python -m unittest discover -s tests -v`. There is still no linter configuration, no build step, and no `pyproject.toml` or `setup.py`.

## Architecture

**Control-Execution-Monitor separation** across host and container:

```
Host side                                  Container side (Docker)
───────────────────────────────            ───────────────────────────────
run.py wrapper → acprof/cli/run.py         acprof/container/server.py
  │                                          ├─ handlers/ model load/infer
  ├─ acprof/host/detect.py                   │   ├─ nlp.py
  ├─ acprof/host/orchestrator.py             │   ├─ cv.py
  │   Docker build + container lifecycle     │   ├─ audio.py
  │   tcpdump start/stop + CSV merge         │   └─ timeseries.py
  │                                          └─ /ready, /predict endpoints
  ├─ acprof/host/client.py
  │   ├─ acprof/workloads/
  │   └─ HTTP → /predict
  │
  ├─ acprof/monitors/        GPU/CPU/resource/perf side channels
  ├─ acprof/packet/          packet latency parse/merge
  └─ results/ → CSV output
```

### Key data flow

1. `run.py` parses CLI args → `acprof/host/detect.py` resolves model's task family & backend (3-level fallback: HF API → config inference → CLI override)
2. `acprof/host/orchestrator.py` builds Docker image (base + task-family layer with model weights baked in), then sweeps the CPU x MEM x GPU resource matrix
3. For each resource config: start container → poll `/ready` → start tcpdump → run `acprof.host.client` as subprocess → collect monitors → stop → write CSV
4. `acprof/host/client.py` generates task-specific workloads via `acprof/workloads/` registry, iterates over input scales (warmup + repeat), measures latency
5. Results merge into `results/<model>/result_all.csv` for dynamic measurements, while one-row static metadata is written to `results/<model>/static_meta.csv`

### Extension points

- **New task family**: Add handler in `acprof/container/handlers/`, workload generator in `acprof/workloads/`, Dockerfile in `dockerfiles/`, and register the pipeline tags in `acprof.config.PIPELINE_TAG_TO_FAMILY`
- **New monitor**: Side-channel design means monitors are independent — add alongside `acprof/monitors/energy_nvml.py` without touching orchestrator core
- **Handler interface** (`acprof/container/handlers/__init__.py`): `load()` → `preprocess()` → `predict()` → `postprocess()`

## Configuration

All defaults live in `acprof/config.py`. Key constants:
- Resource matrix: `DEFAULT_CPU_LIST`, `DEFAULT_MEM_LIST`, `DEFAULT_GPU_LIST`
- Experiment: `DEFAULT_WARMUP=2`, `DEFAULT_REPEAT=5`, `DEFAULT_REPEAT_IN_WINDOW=0`
- Scaling dimensions per task family in `SCALING_DIMENSIONS` dict
- `SERVER_PORT=8002`, container port mapping: `8002 + cpu*100 + mem`

## Infrastructure Requirements

- Docker with optional NVIDIA GPU support (`--gpus all`)
- `tcpdump` + `tshark` for packet-level latency (optional, gracefully degrades)
- `sudo` required for tcpdump on `docker0` bridge
- HuggingFace Hub access for model detection and weight download during Docker build
- China network support via `HF_MIRROR_ENDPOINT` and `PYPI_MIRROR_INDEX` in `acprof/config.py`

## Code Conventions

- Comments and docstrings are a mix of Chinese and English
- Environment variables are the interface between `acprof/host/orchestrator.py` and `acprof/host/client.py` — not function arguments
- Model weights are always baked into Docker images (no volume mounts) for reproducibility
- Workload generators use deterministic seeded RNG

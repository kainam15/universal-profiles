# syntax=docker/dockerfile:1
# Shared dependency layer; contains no model weights or AC-Prof source tree.
ARG PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.10-slim
FROM ${PYTHON_BASE_IMAGE}
ARG REQUIREMENTS_LOCK
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN apt-get update \
    && if apt-cache show libdw1t64 >/dev/null 2>&1; then runtime_libdw=libdw1t64; else runtime_libdw=libdw1; fi \
    && apt-get install -y --no-install-recommends valgrind "${runtime_libdw}" build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY ${REQUIREMENTS_LOCK} /opt/acprof/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r /opt/acprof/requirements.lock \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --extra-index-url ${TORCH_INDEX_URL} \
    && pip check
LABEL org.acprof.execution-profile.massif="1" org.acprof.execution-profile.nsys="1"
ENV HF_HOME=/models/hf MODEL_CACHE_DIR=/models MODEL_LOCAL_PATH=/models/model-snapshot \
    TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
WORKDIR /app
EXPOSE 8002

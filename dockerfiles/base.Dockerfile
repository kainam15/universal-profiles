ARG PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.10-slim
FROM ${PYTHON_BASE_IMAGE}

# Install the execution profilers' runtime once, below all model/code layers.
# Nsys itself is mounted from the host. These packages start no services and
# are only invoked by the isolated execution-profile probes.
RUN apt-get update \
    && if apt-cache show libdw1t64 >/dev/null 2>&1; then \
         elfutils_runtime=libdw1t64; \
       else \
         elfutils_runtime=libdw1; \
       fi \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         valgrind "${elfutils_runtime}" \
    && rm -rf /var/lib/apt/lists/*

LABEL org.acprof.execution-profile.massif="1" \
      org.acprof.execution-profile.nsys="1"

WORKDIR /app

ARG HF_ENDPOINT=https://hf-mirror.com
ARG HF_FALLBACK_ENDPOINTS=https://huggingface.co
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYPI_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Cache directories
ENV MODEL_CACHE_DIR=/models
ENV HF_HOME=/models/hf
ENV SENTENCE_TRANSFORMERS_HOME=/models
ENV MODEL_LOCAL_PATH=/models/model-snapshot
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

# Mirror for China network stability (optional)
ENV HF_ENDPOINT=${HF_ENDPOINT}
ENV HF_FALLBACK_ENDPOINTS=${HF_FALLBACK_ENDPOINTS}

# Base dependencies
RUN pip install --no-cache-dir -i ${PYPI_INDEX_URL} \
    --trusted-host ${PYPI_TRUSTED_HOST} \
    flask==3.0.2 \
    'huggingface_hub>=0.23.0' \
    'numpy>=1.26'

# Application code is copied only after dependency and model layers.

EXPOSE 8002

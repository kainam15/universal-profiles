# syntax=docker/dockerfile:1

# BASE_IMAGE must be provided explicitly with --build-arg.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_PACKAGE_SPEC=torch>=2.7

# NLP-specific dependencies
# Step 1: Install PyTorch from a host-compatible CUDA wheel index.
# Default remains cu128 for RTX 50-series hosts, but orchestrator can downgrade
# to cu124 when the local driver only supports CUDA 12.4.
RUN pip install --no-cache-dir \
    "${TORCH_PACKAGE_SPEC}" \
    --index-url ${TORCH_INDEX_URL}

# Step 2: Install remaining deps from Tsinghua mirror (without touching torch)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'transformers==4.57.6' \
    'sentence-transformers==5.1.2' \
    'pandas==2.3.3' \
    sentencepiece \
    accelerate \
    protobuf

FROM runtime AS model
ARG MODEL_ID
ARG MODEL_REVISION=main
ARG TASK_FAMILY=nlp
ARG RUNTIME_BACKEND=transformers_pipeline
ARG MODEL_ADAPTER=family-default
ARG MODEL_DOWNLOAD_POLICY=auto
ENV MODEL_ID=${MODEL_ID} MODEL_REVISION=${MODEL_REVISION}
ENV TASK_FAMILY=${TASK_FAMILY} RUNTIME_BACKEND=${RUNTIME_BACKEND} MODEL_ADAPTER=${MODEL_ADAPTER}
ENV MODEL_DOWNLOAD_POLICY=${MODEL_DOWNLOAD_POLICY}
COPY acprof/container/download_model.py acprof/container/model_files.py /opt/acprof/
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
    python /opt/acprof/download_model.py

COPY acprof/ /app/acprof/
CMD ["python", "-m", "acprof.container.server"]

# syntax=docker/dockerfile:1

ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

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
    'transformers>=4.40' \
    sentencepiece \
    accelerate \
    protobuf

# Download model weights (baked into image layer)
ARG MODEL_ID
ARG MODEL_REVISION=main
ENV MODEL_ID=${MODEL_ID}
ENV MODEL_REVISION=${MODEL_REVISION}
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then \
        export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
    fi; \
    python -m acprof.container.download_model

CMD ["python", "-m", "acprof.container.server"]

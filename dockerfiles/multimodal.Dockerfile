# syntax=docker/dockerfile:1

ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_PACKAGE_SPEC=torch>=2.7

# Keep the torch/vision/audio wheel set on the host-compatible CUDA index.
RUN pip install --no-cache-dir \
    "${TORCH_PACKAGE_SPEC}" torchvision torchaudio \
    --index-url ${TORCH_INDEX_URL}

# 4.57.6 provides native Qwen2-Audio, Qwen2.5-Omni, ColPali/ColQwen2,
# video processors and the QA pipeline phases used by this adapter.
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'transformers==4.57.6' \
    Pillow librosa soundfile accelerate sentencepiece protobuf

# Eager GPU inference may compile a Triton launcher on first use.
# DocQA consumes supplied word boxes; no OCR process runs in predict.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG MODEL_ID
ARG MODEL_REVISION=main
ENV MODEL_ID=${MODEL_ID}
ENV MODEL_REVISION=${MODEL_REVISION}
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then \
        export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
    fi; \
    python -m acprof.container.download_model

COPY acprof/ acprof/

CMD ["python", "-m", "acprof.container.server"]

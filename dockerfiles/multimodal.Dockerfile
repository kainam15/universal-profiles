# syntax=docker/dockerfile:1

# BASE_IMAGE must be provided explicitly with --build-arg.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime

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

FROM runtime AS model
ARG MODEL_ID
ARG MODEL_REVISION=main
ARG TASK_FAMILY=multimodal
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

# syntax=docker/dockerfile:1

# BASE_IMAGE must be provided explicitly with --build-arg.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime

# Install a host-driver-compatible CUDA wheel. The orchestrator selects the
# same tested wheel index used by the NLP image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_PACKAGE_SPEC=torch>=2.7
RUN pip install --no-cache-dir \
    "${TORCH_PACKAGE_SPEC}" \
    --index-url ${TORCH_INDEX_URL}

# Image/video diffusion dependencies (preserve the selected torch build).
# Diffusers 0.40+ requires huggingface_hub 1.x while Transformers 4.x still
# requires huggingface_hub <1, so keep this image on the compatible 0.39/4.57
# release line until both libraries share the same Hub major version.
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'huggingface_hub==0.36.2' \
    'diffusers==0.39.0' \
    'transformers==4.57.6' \
    'accelerate==1.14.0' \
    'safetensors==0.8.0' \
    sentencepiece \
    protobuf \
    Pillow

# T5/UMT5 tokenizers used by native image/video pipelines need SentencePiece
# and protobuf. Outputs remain decoded PIL frames; no video encoder is needed.

FROM runtime AS model
ARG MODEL_ID
ARG MODEL_REVISION=main
ARG TASK_FAMILY=diffusion
ARG RUNTIME_BACKEND=diffusers
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

# syntax=docker/dockerfile:1

ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

# Install a host-driver-compatible CUDA wheel. The orchestrator selects the
# same tested wheel index used by the NLP image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_PACKAGE_SPEC=torch>=2.7
RUN pip install --no-cache-dir \
    "${TORCH_PACKAGE_SPEC}" \
    --index-url ${TORCH_INDEX_URL}

# Text-to-image dependencies (without replacing the selected torch build).
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
    Pillow

# Download the complete Diffusers repository snapshot into the image.
ARG MODEL_ID
ARG MODEL_REVISION=main
ENV MODEL_ID=${MODEL_ID}
ENV MODEL_REVISION=${MODEL_REVISION}
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then \
        export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
    fi; \
    python -m acprof.container.download_model

# Keep handler changes in a cheap layer above model weights.
COPY acprof/ acprof/

CMD ["python", "-m", "acprof.container.server"]

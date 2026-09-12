# syntax=docker/dockerfile:1
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE}
ARG MODEL_ID
ARG MODEL_REVISION
ARG HF_ENDPOINT=https://hf-mirror.com
ENV MODEL_ID=${MODEL_ID} MODEL_REVISION=${MODEL_REVISION} HF_ENDPOINT=${HF_ENDPOINT}
ENV HF_FALLBACK_ENDPOINTS=https://huggingface.co
COPY acprof/container/download_model.py /opt/acprof/download_model.py
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
    python /opt/acprof/download_model.py

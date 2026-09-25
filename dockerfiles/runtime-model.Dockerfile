# syntax=docker/dockerfile:1
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE}
ARG MODEL_ID
ARG MODEL_REVISION
ARG HF_ENDPOINT=https://huggingface.co
ARG HF_FALLBACK_ENDPOINTS=
ARG TASK_FAMILY
ARG RUNTIME_BACKEND
ARG MODEL_ADAPTER=family-default
ARG MODEL_DOWNLOAD_POLICY=auto
ARG MODEL_FILES_KEY
ARG MODEL_DEPENDENCIES_B64=
ENV MODEL_ID=${MODEL_ID} MODEL_REVISION=${MODEL_REVISION} HF_ENDPOINT=${HF_ENDPOINT}
ENV TASK_FAMILY=${TASK_FAMILY} RUNTIME_BACKEND=${RUNTIME_BACKEND} MODEL_ADAPTER=${MODEL_ADAPTER}
ENV MODEL_DOWNLOAD_POLICY=${MODEL_DOWNLOAD_POLICY}
ENV ACPROF_MODEL_DEPENDENCIES_B64=${MODEL_DEPENDENCIES_B64}
ENV HF_HUB_CACHE=/models/hf
LABEL org.acprof.model-files-key=${MODEL_FILES_KEY} org.acprof.image-kind="weights"
ENV HF_FALLBACK_ENDPOINTS=${HF_FALLBACK_ENDPOINTS}
COPY acprof/container/download_model.py acprof/container/model_files.py /opt/acprof/
COPY acprof/model_spec.py /opt/acprof/acprof/model_spec.py
COPY acprof/hf_endpoints.py /opt/acprof/acprof/hf_endpoints.py
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
    python /opt/acprof/download_model.py

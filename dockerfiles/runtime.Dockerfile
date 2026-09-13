# syntax=docker/dockerfile:1
# Dependency environment only; no model weights or AC-Prof business source.
ARG PLATFORM_IMAGE
FROM ${PLATFORM_IMAGE}
ARG ENVIRONMENT_ID
ARG ENVIRONMENT_BUILD_FINGERPRINT
COPY requirements.lock expectation.json /opt/acprof/
RUN --mount=type=bind,source=environment_tools.py,target=/build/environment_tools.py \
    --mount=type=bind,source=dependency_locks.py,target=/build/dependency_locks.py \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python /build/environment_tools.py environment
LABEL org.acprof.environment=${ENVIRONMENT_ID} \
      org.acprof.environment-build-fingerprint=${ENVIRONMENT_BUILD_FINGERPRINT} \
      org.acprof.image-kind="environment" \
      org.acprof.execution-profile.massif="1" org.acprof.execution-profile.nsys="1"
ENV HF_HOME=/models/hf MODEL_CACHE_DIR=/models MODEL_LOCAL_PATH=/models/model-snapshot \
    TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
WORKDIR /app
EXPOSE 8002

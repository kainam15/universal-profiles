# syntax=docker/dockerfile:1

ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE} AS runtime

# Time-series specific dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'torch>=2.2' \
    "chronos-forecasting>=1.5.0"

FROM runtime AS model
ARG MODEL_ID
ARG MODEL_REVISION=main
ARG TASK_FAMILY=timeseries
ARG RUNTIME_BACKEND=chronos
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

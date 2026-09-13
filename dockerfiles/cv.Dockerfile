# syntax=docker/dockerfile:1

# BASE_IMAGE must be provided explicitly with --build-arg.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime

# CV-specific dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'torch>=2.2' \
    'transformers==4.57.6' \
    torchvision \
    Pillow \
    scipy \
    accelerate

# Triton compiles its GPU launcher on first use; the slim base has no compiler.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

FROM runtime AS model
ARG MODEL_ID
ARG MODEL_REVISION=main
ARG TASK_FAMILY=cv
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

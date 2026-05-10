ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

# Time-series specific dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'torch>=2.2' \
    "chronos-forecasting>=1.5.0"

# Download model weights (baked into image layer)
ARG MODEL_ID
ARG MODEL_REVISION=main
ARG HF_TOKEN
ENV MODEL_ID=${MODEL_ID}
ENV MODEL_REVISION=${MODEL_REVISION}
RUN HF_TOKEN="${HF_TOKEN}" python -m acprof.container.download_model

CMD ["python", "-m", "acprof.container.server"]

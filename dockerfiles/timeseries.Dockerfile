ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

# Time-series specific dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    torch>=2.2 \
    "chronos-forecasting>=1.5.0"

# Download model weights (baked into image layer)
ARG MODEL_ID
ENV MODEL_ID=${MODEL_ID}
RUN python download_model.py

CMD ["python", "server.py"]

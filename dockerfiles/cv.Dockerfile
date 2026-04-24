ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

# CV-specific dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'torch>=2.2' \
    'transformers>=4.40' \
    torchvision \
    Pillow \
    accelerate

# Download model weights (baked into image layer)
ARG MODEL_ID
ARG HF_TOKEN
ENV MODEL_ID=${MODEL_ID}
RUN HF_TOKEN="${HF_TOKEN}" python download_model.py

CMD ["python", "server.py"]

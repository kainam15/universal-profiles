ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

# NLP-specific dependencies
# Step 1: Install PyTorch with CUDA 12.8 from PyTorch index.
# RTX 50-series GPUs expose sm_120, which is not supported by cu124 wheels.
RUN pip install --no-cache-dir \
    torch>=2.7 \
    --index-url https://download.pytorch.org/whl/cu128

# Step 2: Install remaining deps from Tsinghua mirror (without touching torch)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    transformers>=4.40 \
    sentencepiece \
    accelerate \
    protobuf

# Download model weights (baked into image layer)
ARG MODEL_ID
ARG HF_TOKEN
ENV MODEL_ID=${MODEL_ID}
RUN HF_TOKEN="${HF_TOKEN}" python download_model.py

CMD ["python", "server.py"]

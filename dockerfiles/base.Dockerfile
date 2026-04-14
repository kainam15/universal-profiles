FROM python:3.10-slim

WORKDIR /app

# Cache directories
ENV MODEL_CACHE_DIR=/models
ENV HF_HOME=/models/hf
ENV SENTENCE_TRANSFORMERS_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

# Mirror for China network stability (optional)
ENV HF_ENDPOINT=https://hf-mirror.com

# Base dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    flask==3.0.2 \
    huggingface_hub>=0.23.0 \
    numpy>=1.26

# Copy shared code
COPY download_model.py .
COPY server.py .
COPY handlers/ handlers/

EXPOSE 8002

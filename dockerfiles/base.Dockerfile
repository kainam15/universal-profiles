FROM python:3.10-slim

WORKDIR /app

ARG HF_ENDPOINT=https://hf-mirror.com
ARG HF_FALLBACK_ENDPOINTS=https://huggingface.co
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYPI_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Cache directories
ENV MODEL_CACHE_DIR=/models
ENV HF_HOME=/models/hf
ENV SENTENCE_TRANSFORMERS_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

# Mirror for China network stability (optional)
ENV HF_ENDPOINT=${HF_ENDPOINT}
ENV HF_FALLBACK_ENDPOINTS=${HF_FALLBACK_ENDPOINTS}

# Base dependencies
RUN pip install --no-cache-dir -i ${PYPI_INDEX_URL} \
    --trusted-host ${PYPI_TRUSTED_HOST} \
    flask==3.0.2 \
    huggingface_hub>=0.23.0 \
    numpy>=1.26

# Copy shared code
COPY download_model.py .
COPY server.py .
COPY handlers/ handlers/

EXPOSE 8002
